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

from sparksam.models.sam2 import load_image_rgb  # noqa: E402
from sparksam.protocols.reproduction import (  # noqa: E402
    reproduction_protocol_enabled,
    git_record,
    lineage_path_for,
    read_lineage,
    resolve_project_path,
    sha256_file,
    validate_selection_lock,
)
from scripts.training_common import _add_sam2_repo_to_path, _build_sam2_model  # noqa: E402
from scripts.train_sparksam import SPARKSAM, _load_samples_for_role, _read_yaml, _write_json  # noqa: E402


def _safe_id(sample_id: str) -> str:
    out = []
    for ch in str(sample_id):
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(out)[:220]


def _iou(prob: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    pred = prob >= float(threshold)
    target = gt > 0.5
    union = np.logical_or(pred, target).sum()
    if union <= 0:
        return 1.0
    return float(np.logical_and(pred, target).sum()) / float(union)


def _build_model(cfg: dict, source: dict, checkpoint: Path, device: torch.device) -> SPARKSAM:
    sam = _build_sam2_model(cfg["student"]["model"], cfg["student"]["checkpoint"], source, device, train=False)
    hc = cfg.get("prompt_head", {}) if isinstance(cfg.get("prompt_head"), dict) else {}
    model = SPARKSAM(
        sam,
        token_count=int(hc.get("learned_sparse_prompt_tokens", 2) or 2),
        min_box_side=float(hc.get("min_box_side", 2.0) or 2.0),
        max_box_fraction=float(hc.get("max_box_fraction", 0.05) or 0.05),
        temperature=float(hc.get("soft_argmax_temperature", 0.05) or 0.05),
        candidate_count=int(hc.get("candidate_count", hc.get("point_budget", 1)) or 1),
        decoder_point_count=int(hc.get("decoder_point_count", 0) or 0),
        candidate_mask_count=int(hc.get("candidate_mask_count", 0) or 0),
        decoder_mode=str(hc.get("decoder_mode", "top_points") or "top_points"),
        mask_select_temperature=float(hc.get("mask_select_temperature", 1.0) or 1.0),
        local_prompt_token_count=int(hc.get("local_prompt_token_count", 0) or 0),
        mask_selector_score_source=str(hc.get("mask_selector_score_source", "candidate_logits") or "candidate_logits"),
        mask_selector_sam_iou_weight=float(hc.get("mask_selector_sam_iou_weight", 1.0) or 1.0),
        prompt_gate_enabled=bool(hc.get("prompt_gate_enabled", False)),
        prompt_gate_strength=float(hc.get("prompt_gate_strength", 0.5) or 0.5),
        logit_calibration_enabled=bool(hc.get("logit_calibration_enabled", False)),
        logit_calibration_max_abs_bias=float(hc.get("logit_calibration_max_abs_bias", 2.0) or 2.0),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the joint-adaptation response cache used by the calibration phases.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--selection-lock", type=Path)
    ap.add_argument("--roles", default="train,validation")
    ap.add_argument("--teacher-name", default="SPARK-SAM-Joint-Adaptation-Validation-Selected")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args(argv)

    out_root = args.output_root
    manifest_path = out_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Calibration response manifest already exists: {manifest_path}")

    cfg = _read_yaml(args.config)
    source = _read_yaml(Path(str(cfg["source_config"])))
    sam2_repo = _add_sam2_repo_to_path(source)
    if args.checkpoint is None:
        if args.selection_lock is None:
            raise RuntimeError("--checkpoint or --selection-lock is required.")
        lock_payload = json.loads(args.selection_lock.read_text(encoding="utf-8"))
        args.checkpoint = resolve_project_path(lock_payload.get("selected_checkpoint", {}).get("path", ""), base=args.selection_lock.parent)
    roles = [x.strip() for x in str(args.roles).split(",") if x.strip()]
    if any(r == "test" for r in roles):
        raise RuntimeError("SPARK-SAM calibration response cache must not include test split.")
    strict = reproduction_protocol_enabled(cfg)
    source_lineage = None
    split_manifest = None
    selection_audit = None
    generator_code = git_record()
    if strict and (not str(generator_code.get("revision", "")).strip() or bool(generator_code.get("dirty", False))):
        raise RuntimeError("Leakage-resistant cache generation requires a clean, revisioned benchmark worktree.")
    if strict:
        if roles != ["train"]:
            raise RuntimeError(f"Leakage-resistant calibration response cache requires --roles train exactly, got {roles}")
        if args.overwrite:
            raise RuntimeError("Leakage-resistant calibration response caches are immutable; choose a new output root.")
        if args.selection_lock is None:
            raise RuntimeError("Leakage-resistant calibration response cache requires a joint-adaptation validation selection lock.")
        lock_payload = json.loads(args.selection_lock.read_text(encoding="utf-8"))
        selection_audit = validate_selection_lock(
            args.selection_lock,
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            threshold=float(lock_payload["threshold"]),
            config_path=args.config,
        )
        protocol = cfg.get("reproduction_protocol", {})
        split_manifest = resolve_project_path(protocol.get("split_manifest") or cfg.get("split_policy", {}).get("split_manifest"))
        source_lineage = read_lineage(args.checkpoint, required=True, expected_split_manifest=split_manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sam2.utils.transforms import SAM2Transforms

    model = _build_model(cfg, source, args.checkpoint, device)
    transforms = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0)

    entries, failures = [], []
    started = time.time()
    with torch.no_grad():
        for role in roles:
            samples = _load_samples_for_role(cfg, source, role)
            if args.max_samples:
                samples = samples[: args.max_samples]
            for idx, s in enumerate(samples, 1):
                try:
                    x = transforms(load_image_rgb(s.image_path))[None].to(device)
                    out = model(x, return_candidate_masks=True)
                    full_logits = transforms.postprocess_masks(out["low_res_logits"], (s.height, s.width))[0, 0]
                    prob = torch.sigmoid(full_logits).detach().cpu().numpy().astype(np.float32)
                    rec_dir = out_root / "records" / role / s.dataset_key
                    rec_dir.mkdir(parents=True, exist_ok=True)
                    rel = Path("records") / role / s.dataset_key / f"{_safe_id(s.sample_id)}.npz"
                    np.savez_compressed(out_root / rel, teacher_prob=prob)
                    entries.append(
                        {
                            "dataset": s.dataset_key,
                            "sample_id": s.sample_id,
                            "role": role,
                            "split": "val" if role == "validation" else role,
                            "path": str(rel),
                            "sha256": sha256_file(out_root / rel),
                            "size_bytes": (out_root / rel).stat().st_size,
                            "teacher_iou": _iou(prob, np.asarray(s.mask, dtype=np.float32), 0.5),
                            "height": int(s.height),
                            "width": int(s.width),
                        }
                    )
                    if idx == 1 or idx % 100 == 0:
                        print(
                            json.dumps({"role": role, "index": idx, "total": len(samples), "entries": len(entries)}, ensure_ascii=False),
                            flush=True,
                        )
                except Exception as exc:
                    failures.append({"dataset": s.dataset_key, "sample_id": s.sample_id, "role": role, "error": repr(exc)})
                    print(
                        json.dumps({"role": role, "index": idx, "failed": len(failures), "error": repr(exc)}, ensure_ascii=False),
                        flush=True,
                    )

    datasets = sorted({e["dataset"] for e in entries})
    manifest = {
        "schema_version": "calibration_response_cache_v1",
        "generator_code": git_record(),
        "generator_code_at_start": generator_code,
        "sam2_code": git_record(Path(sam2_repo)),
        "teacher_name": args.teacher_name,
        "teacher_checkpoint": str(args.checkpoint),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_checkpoint_lineage": str(lineage_path_for(args.checkpoint)) if source_lineage is not None else "",
        "source_selection_lock": str(args.selection_lock.resolve()) if args.selection_lock is not None else "",
        "source_selection_audit": selection_audit,
        "teacher_config": str(args.config),
        "datasets": datasets,
        "roles": roles,
        "training_supervision_only": True,
        "inference_forbidden": True,
        "split_manifest": str(split_manifest) if split_manifest is not None else "",
        "split_manifest_sha256": sha256_file(split_manifest) if split_manifest is not None else "",
        "required": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.time() - started,
        "record_count": len(entries),
        "failure_count": len(failures),
        "entries": entries,
        "failures": failures,
        "audit": {
            "status": "passed" if not failures else "failed",
            "no_test_split": True,
            "source": "validation-selected joint-adaptation response mask",
            "reproduction_protocol": strict,
        },
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps({"manifest": str(manifest_path), "record_count": len(entries), "failure_count": len(failures)}, ensure_ascii=False),
        flush=True,
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
