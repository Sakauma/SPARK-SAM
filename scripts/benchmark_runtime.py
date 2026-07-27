#!/usr/bin/env python3
"""Benchmark SPARK-SAM with synchronized, full-split, batch-1 timing boundaries."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sparksam.models.sam2 import load_image_rgb  # noqa: E402
from sparksam.protocols.reproduction import artifact_record, git_record, read_mapping, resolve_project_path, validate_selection_lock  # noqa: E402
from scripts.evaluate_sparksam import _build_model  # noqa: E402
from scripts.training_common import _add_sam2_repo_to_path  # noqa: E402
from scripts.train_sparksam import _load_samples_for_role, _read_yaml  # noqa: E402


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
        "std": float(array.std()),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    sam2_backbone_decoder = int(sum(parameter.numel() for parameter in model.sam2_model.parameters()))
    components = {
        "sam2_backbone_prompt_encoder_decoder": sam2_backbone_decoder,
        "prompt_head": int(sum(parameter.numel() for parameter in model.prompt_head.parameters())),
        "local_prompt_projector": int(
            sum(parameter.numel() for parameter in getattr(model, "local_prompt_projector", torch.nn.Identity()).parameters())
        ),
        "logit_calibration_head": int(
            sum(parameter.numel() for parameter in getattr(model, "logit_calibration_head", torch.nn.Identity()).parameters())
        ),
    }
    return {
        "total": total,
        "sam2_preserved_components": sam2_backbone_decoder,
        "added_components": total - sam2_backbone_decoder,
        "trainable_at_runtime_config": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        **{f"component_{name}": value for name, value in components.items()},
    }


def _verify_test_result(path: Path, lock_path: Path, checkpoint: Path) -> dict[str, Any]:
    result = read_mapping(path)
    if result.get("role") != "test":
        raise RuntimeError(f"Runtime benchmark requires a completed locked test result: {path}")
    if result.get("checkpoint", {}).get("sha256") != artifact_record(checkpoint)["sha256"]:
        raise RuntimeError("Test result checkpoint differs from runtime checkpoint")
    selection = result.get("selection_audit", {})
    if selection.get("lock", {}).get("sha256") != artifact_record(lock_path)["sha256"]:
        raise RuntimeError("Test result was not produced from the supplied selection lock")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--measure-flops", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = _read_yaml(args.config)
    benchmark_code = git_record()
    if not str(benchmark_code.get("revision", "")).strip() or bool(benchmark_code.get("dirty", False)):
        raise RuntimeError("Runtime benchmark requires a clean, revisioned benchmark worktree")
    if cfg.get("fixed_prompt_prior") or cfg.get("prompt_prior"):
        raise RuntimeError("Runtime benchmark forbids fixed/external prompts")
    lock_path = args.selection_lock.resolve()
    lock = read_mapping(lock_path)
    checkpoint = resolve_project_path(lock.get("selected_checkpoint", {}).get("path", ""), base=lock_path.parent)
    threshold = float(lock["threshold"])
    selection_audit = validate_selection_lock(
        lock_path,
        cfg=cfg,
        checkpoint_path=checkpoint,
        threshold=threshold,
        config_path=args.config,
    )
    _verify_test_result(args.test_result.resolve(), lock_path, checkpoint)
    source = _read_yaml(Path(str(cfg["source_config"])))
    sam2_repo = _add_sam2_repo_to_path(source)
    sam2_code = git_record(Path(sam2_repo))
    if not str(sam2_code.get("revision", "")).strip() or bool(sam2_code.get("dirty", False)):
        raise RuntimeError("Runtime benchmark requires a clean, revisioned SAM2 dependency")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Paper runtime benchmark requires CUDA")
    from sam2.utils.transforms import SAM2Transforms

    model = _build_model(cfg, source, checkpoint, device)
    transforms = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0)
    samples = _load_samples_for_role(cfg, source, "test")
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No test samples loaded")
    amp = args.precision == "amp"
    warmup_sample = samples[0]
    warmup_rgb = load_image_rgb(warmup_sample.image_path)
    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            tensor = transforms(warmup_rgb)[None].to(device)
            with torch.autocast(device_type="cuda", enabled=amp):
                output = model(tensor, fixed_prompt=None)
                full = transforms.postprocess_masks(output["low_res_logits"], (warmup_sample.height, warmup_sample.width))[0, 0]
                _ = (torch.sigmoid(full) >= threshold).to("cpu")
        _sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    preprocess_ms: list[float] = []
    forward_ms: list[float] = []
    postprocess_ms: list[float] = []
    end_to_end_ms: list[float] = []
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for repeat in range(max(1, args.repeats)):
            for sample in samples:
                rgb = load_image_rgb(sample.image_path)  # Disk IO is intentionally outside the timed boundary.
                _sync(device)
                total_start = time.perf_counter()
                preprocess_start = total_start
                tensor = transforms(rgb)[None].to(device)
                _sync(device)
                forward_start = time.perf_counter()
                preprocess_value = (forward_start - preprocess_start) * 1000.0
                with torch.autocast(device_type="cuda", enabled=amp):
                    output = model(tensor, fixed_prompt=None)
                _sync(device)
                postprocess_start = time.perf_counter()
                forward_value = (postprocess_start - forward_start) * 1000.0
                full = transforms.postprocess_masks(output["low_res_logits"], (sample.height, sample.width))[0, 0]
                cpu_mask = (torch.sigmoid(full) >= threshold).to("cpu").numpy()
                _sync(device)
                end = time.perf_counter()
                postprocess_value = (end - postprocess_start) * 1000.0
                total_value = (end - total_start) * 1000.0
                preprocess_ms.append(preprocess_value)
                forward_ms.append(forward_value)
                postprocess_ms.append(postprocess_value)
                end_to_end_ms.append(total_value)
                rows.append(
                    {
                        "repeat": repeat,
                        "dataset": sample.dataset_key,
                        "sample_id": sample.sample_id,
                        "height": int(sample.height),
                        "width": int(sample.width),
                        "preprocess_h2d_ms": preprocess_value,
                        "model_forward_ms": forward_value,
                        "postprocess_d2h_ms": postprocess_value,
                        "in_memory_input_to_cpu_mask_ms": total_value,
                        "predicted_pixels": int(cpu_mask.sum()),
                    }
                )
    flops: dict[str, Any] = {"status": "not_requested", "total": None}
    if args.measure_flops:
        try:
            from fvcore.nn import FlopCountAnalysis

            sample_tensor = transforms(warmup_rgb)[None].to(device)
            analysis = FlopCountAnalysis(model, sample_tensor)
            flops = {"status": "measured", "total": int(analysis.total()), "unsupported_ops": analysis.unsupported_ops()}
        except Exception as exc:
            flops = {"status": "unavailable", "total": None, "reason": repr(exc)}
    payload = {
        "schema_version": "spark_runtime_benchmark_v1",
        "code": benchmark_code,
        "sam2_code": sam2_code,
        "timing_contract": {
            "batch_size": 1,
            "precision": args.precision,
            "warmup_iterations": int(args.warmup),
            "repeats": int(args.repeats),
            "dataset_scope": "complete locked test split" if not args.max_samples else "diagnostic subset; not paper-reportable",
            "disk_io_included": False,
            "cuda_synchronization": True,
            "preprocess_h2d_ms": "RGB array to normalized GPU tensor",
            "model_forward_ms": "SPARK model forward only",
            "postprocess_d2h_ms": "resize logits, sigmoid, threshold, and CPU transfer",
            "in_memory_input_to_cpu_mask_ms": "sum of the three measured boundaries",
        },
        "config": artifact_record(args.config.resolve()),
        "checkpoint": artifact_record(checkpoint),
        "selection_audit": selection_audit,
        "test_result": artifact_record(args.test_result.resolve()),
        "samples": len(samples),
        "measurements": len(rows),
        "input_resolution": int(model.image_size),
        "parameter_counts": _parameter_counts(model),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "flops": flops,
        "summary_ms": {
            "preprocess_h2d": _stats(preprocess_ms),
            "model_forward": _stats(forward_ms),
            "postprocess_d2h": _stats(postprocess_ms),
            "in_memory_input_to_cpu_mask": _stats(end_to_end_ms),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary_ms": payload["summary_ms"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
