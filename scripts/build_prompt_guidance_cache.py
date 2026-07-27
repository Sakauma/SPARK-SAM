#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from sparksam.models.prompt_estimator import decode_auto_prompt, ir_prior_stack_from_path, load_auto_prompt_model  # noqa: E402
from sparksam.protocols.reproduction import (  # noqa: E402
    reproduction_protocol_enabled,
    git_record,
    lineage_path_for,
    read_lineage,
    resolve_project_path,
    sha256_file,
)
from scripts.train_sparksam import _load_samples_for_role, _read_yaml, _write_json  # noqa: E402


def _safe_id(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text))[:180]


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _tensor_from_prior(prior: np.ndarray, device: str, dtype: torch.dtype):
    return torch.from_numpy(prior[None]).to(device=device, dtype=dtype, non_blocking=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate dense prompt-estimator objectness supervision for SPARK-SAM.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--roles", default="train,validation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = _read_yaml(args.config)
    source = _read_yaml(Path(str(cfg["source_config"])))
    roles = [x.strip() for x in args.roles.split(",") if x.strip()]
    if "test" in roles and not args.allow_test:
        raise RuntimeError("Refusing to generate test dense cache without --allow-test.")
    out_root = args.output_root if args.output_root.is_absolute() else (PROJECT_ROOT / args.output_root).resolve()
    records_root = out_root / "records"
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else (PROJECT_ROOT / args.checkpoint).resolve()
    manifest_path = out_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Prompt-teacher cache already exists: {manifest_path}")
    strict = reproduction_protocol_enabled(cfg)
    generator_code = git_record()
    if strict and (not str(generator_code.get("revision", "")).strip() or bool(generator_code.get("dirty", False))):
        raise RuntimeError("Clean cache generation requires a clean, revisioned benchmark worktree.")
    source_lineage = None
    split_manifest = None
    if strict:
        if roles != ["train"]:
            raise RuntimeError(f"Clean prompt-teacher cache requires --roles train exactly, got {roles}")
        if args.allow_test or args.overwrite:
            raise RuntimeError("Clean prompt-teacher caches are immutable and never allow test access/overwrite.")
        protocol = cfg.get("reproduction_protocol", {})
        split_manifest = resolve_project_path(protocol.get("split_manifest") or cfg.get("split_policy", {}).get("split_manifest"))
        source_lineage = read_lineage(checkpoint, required=True, expected_split_manifest=split_manifest)
    model, meta = load_auto_prompt_model(checkpoint, device=args.device)
    model.eval()
    mcfg = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    param_dtype = next(model.parameters()).dtype
    torch_dtype = param_dtype if param_dtype in {torch.float16, torch.bfloat16} else torch.float32
    entries = []
    counts = {}
    started = time.time()
    with torch.no_grad():
        for role in roles:
            samples = _load_samples_for_role(cfg, source, role)
            if args.max_samples:
                samples = samples[: args.max_samples]
            for idx, s in enumerate(samples, 1):
                prior = ir_prior_stack_from_path(
                    Path(s.image_path),
                    use_local_contrast=bool(mcfg.get("use_local_contrast", True)),
                    use_top_hat=bool(mcfg.get("use_top_hat", True)),
                    use_dog_filter=bool(mcfg.get("use_dog_filter", False)),
                    use_fft_highpass=bool(mcfg.get("use_fft_highpass", False)),
                )
                x = _tensor_from_prior(prior, args.device, torch_dtype)
                outputs = model(x)
                logits = outputs["objectness_logits"].detach().float().cpu().numpy()[0]
                if logits.ndim == 3 and logits.shape[0] == 1:
                    logits = logits[0]
                prob = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
                box = outputs["box_size"].detach().float().cpu().numpy()[0].astype(np.float32)
                decoded = decode_auto_prompt(
                    objectness_logits=outputs["objectness_logits"],
                    box_size=outputs["box_size"],
                    confidence_logit=outputs.get("confidence_logits"),
                    image_width=int(s.width),
                    image_height=int(s.height),
                    min_box_side=float(mcfg.get("min_box_side", 2.0)),
                    negative_ring=False,
                    negative_ring_offset=float(mcfg.get("negative_ring_offset", 4.0)),
                    top_k=int(cfg.get("prompt_head", {}).get("top_k", 5)),
                    point_budget=int(cfg.get("prompt_head", {}).get("point_budget", 1)),
                    response_threshold=0.15,
                    nms_radius=int(cfg.get("prompt_head", {}).get("nms_radius", 4)),
                    border_suppression_px=int(cfg.get("prompt_head", {}).get("border_suppression_px", 4)),
                )
                rec_dir = records_root / _safe_id(s.dataset_key) / role
                rec_dir.mkdir(parents=True, exist_ok=True)
                rec_path = rec_dir / f"{_safe_id(s.sample_id)}.npz"
                if args.dtype == "float16":
                    np.savez_compressed(
                        rec_path,
                        objectness_prob=prob.astype(np.float16),
                        objectness_logits=logits.astype(np.float16),
                        box_size=box.astype(np.float16),
                    )
                else:
                    np.savez_compressed(rec_path, objectness_prob=prob, objectness_logits=logits, box_size=box)
                entries.append(
                    {
                        "dataset": s.dataset_key,
                        "sample_id": s.sample_id,
                        "role": role,
                        "split": "train" if role == "train" else ("val" if role == "validation" else role),
                        "path": _rel(rec_path, out_root),
                        "sha256": sha256_file(rec_path),
                        "size_bytes": rec_path.stat().st_size,
                        "height": int(s.height),
                        "width": int(s.width),
                        "objectness_shape": list(prob.shape),
                        "prompt_teacher_point": decoded.point,
                        "prompt_teacher_box": decoded.box,
                        "prompt_teacher_candidate_score": float(decoded.metadata.get("candidate_score", 0.0)),
                    }
                )
                counts[f"{role}:{s.dataset_key}"] = counts.get(f"{role}:{s.dataset_key}", 0) + 1
                if idx == 1 or idx % 100 == 0:
                    print(
                        json.dumps(
                            {"role": role, "dataset": s.dataset_key, "index": idx, "sample_id": s.sample_id, "path": str(rec_path)},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    manifest = {
        "schema_version": "prompt_guidance_cache_v1",
        "created_at": time.time(),
        "elapsed_s": time.time() - started,
        "generator_code": git_record(),
        "generator_code_at_start": generator_code,
        "source": "Prompt-teacher dense objectness logits/prob for SPARK-SAM training supervision only",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_checkpoint_lineage": str(lineage_path_for(checkpoint)) if source_lineage is not None else "",
        "checkpoint_metadata": meta.get("metadata", {}),
        "checkpoint_config": mcfg,
        "roles": roles,
        "datasets": sorted({e["dataset"] for e in entries}),
        "counts": counts,
        "dtype": args.dtype,
        "training_supervision_only": True,
        "inference_forbidden": True,
        "required": True,
        "split_manifest": str(split_manifest) if split_manifest is not None else "",
        "split_manifest_sha256": sha256_file(split_manifest) if split_manifest is not None else "",
        "entries": entries,
        "audit": {
            "status": "passed",
            "test_included": ("test" in roles),
            "train_supervision_only": True,
            "inference_forbidden": True,
            "reproduction_protocol": strict,
        },
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "entries": len(entries), "counts": counts}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
