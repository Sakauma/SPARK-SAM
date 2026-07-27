#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sparksam.evaluation.segmentation_metrics import (  # noqa: E402
    METRIC_SCHEMA_VERSION,
    aggregate_binary_mask_rows,
    binary_mask_row,
    compatibility_aliases,
)
from sparksam.models.sam2 import load_image_rgb  # noqa: E402
from sparksam.protocols.reproduction import (  # noqa: E402
    artifact_record,
    reproduction_protocol_enabled,
    git_record,
    read_lineage,
    validate_selection_lock,
)
from scripts.training_common import _add_sam2_repo_to_path, _build_sam2_model  # noqa: E402
from scripts.train_sparksam import (  # noqa: E402
    SPARKSAM,
    _load_samples_for_role,
    _read_yaml,
    _write_json,
)


def _hit(point: np.ndarray | list[float], target: np.ndarray) -> float:
    x = int(np.clip(round(float(point[0])), 0, target.shape[1] - 1))
    y = int(np.clip(round(float(point[1])), 0, target.shape[0] - 1))
    return 1.0 if target[y, x] > 0.5 else 0.0


def _candidate_stats(points: np.ndarray, target: np.ndarray) -> dict[str, float]:
    hits = [_hit(point, target) for point in points]
    any_hit = float(any(value > 0 for value in hits))
    return {
        "PromptHitRateTop1": float(hits[0] if hits else 0.0),
        "CandidateAnyHitRate": any_hit,
        "TopKHitRate": any_hit,
        "CandidateHitCount": float(sum(hits)),
        "PromptCandidateCount": float(len(hits)),
    }


def _box_coverage(box: np.ndarray, target: np.ndarray) -> float:
    foreground = target > 0.5
    if not foreground.any():
        return 1.0
    x0, y0, x1, y1 = [float(value) for value in box]
    yy, xx = np.nonzero(foreground)
    inside = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    return float(inside.sum() / max(1, len(xx)))


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _mask_quality_summary(rows: list[dict[str, object]]) -> dict[str, float | None]:
    if not rows:
        return {
            "MaskQualityScoreMean": None,
            "MaskQualityTargetIoUMean": None,
            "MaskQualityMAE": None,
            "MaskQualityPearson": None,
            "MaskQualitySpearman": None,
            "MaskQualityRejectionRate": None,
            "MaskQualityRejectBelow": None,
        }
    predicted = np.asarray([float(row["MaskQualityScore"]) for row in rows], dtype=np.float64)
    target = np.asarray([float(row["MaskQualityTargetIoU"]) for row in rows], dtype=np.float64)
    return {
        "MaskQualityScoreMean": float(predicted.mean()),
        "MaskQualityTargetIoUMean": float(target.mean()),
        "MaskQualityMAE": float(np.abs(predicted - target).mean()),
        "MaskQualityPearson": _correlation(predicted, target),
        "MaskQualitySpearman": _correlation(_rankdata(predicted), _rankdata(target)),
        "MaskQualityRejectionRate": float(np.mean([float(row.get("MaskQualityRejected", 0.0)) for row in rows])),
        "MaskQualityRejectBelow": float(rows[0].get("MaskQualityRejectBelow", 0.0)),
    }


def _summarize_rows(rows: list[dict[str, object]], thresholds: list[float]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for threshold in thresholds:
        subset = [row for row in rows if float(row["threshold"]) == threshold]
        if not subset:
            continue
        aggregate = compatibility_aliases(aggregate_binary_mask_rows(subset))
        aggregate.update(
            {
                "threshold": threshold,
                "R50": _mean(subset, "R50"),
                "PromptHitRate": _mean(subset, "PromptHitRate"),
                "PromptHitRateTop1": _mean(subset, "PromptHitRateTop1"),
                "CandidateAnyHitRate": _mean(subset, "CandidateAnyHitRate"),
                "TopKHitRate": _mean(subset, "TopKHitRate"),
                "PromptBoxCoverage": _mean(subset, "PromptBoxCoverage"),
                "CandidateHitCount": _mean(subset, "CandidateHitCount"),
                "PromptCandidateCount": _mean(subset, "PromptCandidateCount"),
            }
            | _mask_quality_summary(subset)
        )
        summaries.append(aggregate)
    return summaries


def _build_model(cfg: dict, source: dict, checkpoint: Path, device: torch.device) -> SPARKSAM:
    sam = _build_sam2_model(cfg["student"]["model"], cfg["student"]["checkpoint"], source, device, train=False)
    head = cfg.get("prompt_head", {}) if isinstance(cfg.get("prompt_head"), dict) else {}
    model = SPARKSAM(
        sam,
        token_count=int(head.get("learned_sparse_prompt_tokens", 2) or 2),
        min_box_side=float(head.get("min_box_side", 2.0) or 2.0),
        max_box_fraction=float(head.get("max_box_fraction", 0.05) or 0.05),
        temperature=float(head.get("soft_argmax_temperature", 0.05) or 0.05),
        candidate_count=int(head.get("candidate_count", head.get("point_budget", 1)) or 1),
        decoder_point_count=int(head.get("decoder_point_count", 0) or 0),
        candidate_mask_count=int(head.get("candidate_mask_count", 0) or 0),
        decoder_mode=str(head.get("decoder_mode", "top_points") or "top_points"),
        mask_select_temperature=float(head.get("mask_select_temperature", 1.0) or 1.0),
        local_prompt_token_count=int(head.get("local_prompt_token_count", 0) or 0),
        mask_selector_score_source=str(head.get("mask_selector_score_source", "candidate_logits") or "candidate_logits"),
        mask_selector_sam_iou_weight=float(head.get("mask_selector_sam_iou_weight", 1.0) or 1.0),
        prompt_gate_enabled=bool(head.get("prompt_gate_enabled", False)),
        prompt_gate_strength=float(head.get("prompt_gate_strength", 0.5) or 0.5),
        box_parameterization=str(head.get("box_parameterization", "point_centered_width_height") or "point_centered_width_height"),
        box_extent_init_fraction=float(head.get("box_extent_init_fraction", 0.05) or 0.05),
        box_edge_temperature=float(head.get("box_edge_temperature", 0.1) or 0.1),
        box_edge_decode_mode=str(head.get("box_edge_decode_mode", "expectation") or "expectation"),
        box_edge_outer_quantile=float(head.get("box_edge_outer_quantile", 0.1) or 0.1),
        box_occupancy_threshold=float(head.get("box_occupancy_threshold", 0.5) or 0.5),
        point_offset_enabled=bool(head.get("point_offset_enabled", False)),
        point_offset_max_cell_fraction=float(head.get("point_offset_max_cell_fraction", 0.5) or 0.5),
        prompt_box_scale=float(head.get("prompt_box_scale", 1.0) or 1.0),
        logit_calibration_enabled=bool(head.get("logit_calibration_enabled", False)),
        logit_calibration_max_abs_bias=float(head.get("logit_calibration_max_abs_bias", 2.0) or 2.0),
        dense_mask_refinement_enabled=bool(head.get("dense_mask_refinement_enabled", False)),
        dense_mask_refinement_hidden_dim=int(head.get("dense_mask_refinement_hidden_dim", 32) or 32),
        dense_mask_refinement_max_abs_residual=float(head.get("dense_mask_refinement_max_abs_residual", 2.0) or 2.0),
        highres_prompt_refinement_enabled=bool(head.get("highres_prompt_refinement_enabled", False)),
        highres_prompt_refinement_hidden_dim=int(head.get("highres_prompt_refinement_hidden_dim", 32) or 32),
        highres_prompt_refinement_scale=float(head.get("highres_prompt_refinement_scale", 1.0) or 1.0),
        highres_prompt_recenter_box=bool(head.get("highres_prompt_recenter_box", True)),
        highres_prompt_candidate_mode=str(head.get("highres_prompt_candidate_mode", "replace") or "replace"),
        highres_prompt_base_candidate_count=int(head.get("highres_prompt_base_candidate_count", 4) or 0),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def _parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Duplicate thresholds are not allowed")
    return thresholds


def _save_mask(mask_root: Path, dataset: str, sample_id: str, prediction: np.ndarray) -> str:
    safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(sample_id))
    path = mask_root / dataset / f"{safe_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((prediction.astype(np.uint8) * 255), mode="L").save(path)
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a self-prompt Tiny checkpoint under an explicit data role.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--role", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--thresholds", default="0.5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--forbid-teacher", action="store_true")
    parser.add_argument("--forbid-cache", action="store_true")
    parser.add_argument("--forbid-external-prompt", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--box-edge-decode-mode", choices=["expectation", "argmax", "outer_quantile"])
    parser.add_argument("--box-edge-outer-quantile", type=float)
    parser.add_argument("--box-edge-temperature", type=float)
    parser.add_argument("--box-occupancy-threshold", type=float)
    parser.add_argument("--decoder-point-count", type=int)
    parser.add_argument("--prompt-box-scale", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.forbid_teacher and args.forbid_cache and args.forbid_external_prompt):
        raise RuntimeError("Evaluation requires --forbid-teacher --forbid-cache --forbid-external-prompt")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation artifact: {args.output}")
    cfg = _read_yaml(args.config)
    evaluation_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation", {}), dict) else {}
    quality_reject_below = float(evaluation_cfg.get("mask_quality_reject_below", 0.0) or 0.0)
    if not 0.0 <= quality_reject_below <= 1.0:
        raise ValueError("evaluation.mask_quality_reject_below must be in [0,1]")
    diagnostic_overrides = {}
    if args.box_edge_decode_mode is not None:
        diagnostic_overrides["box_edge_decode_mode"] = args.box_edge_decode_mode
    if args.box_edge_outer_quantile is not None:
        diagnostic_overrides["box_edge_outer_quantile"] = args.box_edge_outer_quantile
    if args.box_edge_temperature is not None:
        diagnostic_overrides["box_edge_temperature"] = args.box_edge_temperature
    if args.box_occupancy_threshold is not None:
        diagnostic_overrides["box_occupancy_threshold"] = args.box_occupancy_threshold
    if args.decoder_point_count is not None:
        if args.decoder_point_count < 1:
            raise ValueError("--decoder-point-count must be positive")
        diagnostic_overrides["decoder_point_count"] = args.decoder_point_count
    if args.prompt_box_scale is not None:
        if args.prompt_box_scale <= 0.0:
            raise ValueError("--prompt-box-scale must be positive")
        diagnostic_overrides["prompt_box_scale"] = args.prompt_box_scale
    if diagnostic_overrides and args.role == "test" and args.selection_lock is None:
        raise RuntimeError("Test inference overrides require an exact validation selection lock")
    if diagnostic_overrides and args.role not in {"validation", "test"}:
        raise RuntimeError("Inference overrides are restricted to validation or locked test evaluation")
    if diagnostic_overrides:
        cfg.setdefault("prompt_head", {}).update(diagnostic_overrides)
    benchmark_code = git_record()
    if reproduction_protocol_enabled(cfg) and (not str(benchmark_code.get("revision", "")).strip() or bool(benchmark_code.get("dirty", False))):
        raise RuntimeError("Strict evaluation requires a clean, revisioned benchmark worktree")
    source = _read_yaml(Path(str(cfg["source_config"])))
    sam2_repo = _add_sam2_repo_to_path(source)
    sam2_code = git_record(Path(sam2_repo))
    if reproduction_protocol_enabled(cfg) and (not str(sam2_code.get("revision", "")).strip() or bool(sam2_code.get("dirty", False))):
        raise RuntimeError("Strict evaluation requires a clean, revisioned SAM2 dependency")
    thresholds = _parse_thresholds(args.thresholds)
    if cfg.get("fixed_prompt_prior"):
        raise RuntimeError("Config contains fixed_prompt_prior but --forbid-external-prompt is active")
    checkpoint = args.checkpoint.resolve()
    selection_audit = None
    if reproduction_protocol_enabled(cfg):
        read_lineage(checkpoint, required=True)
        if args.role == "test":
            if args.selection_lock is None:
                raise RuntimeError("Strict test evaluation requires --selection-lock")
            if len(thresholds) != 1:
                raise RuntimeError("Strict test evaluation accepts exactly one validation-selected threshold")
            selection_audit = validate_selection_lock(
                args.selection_lock,
                cfg=cfg,
                checkpoint_path=checkpoint,
                threshold=thresholds[0],
                config_path=args.config,
                inference_overrides=diagnostic_overrides,
            )
        elif args.selection_lock is not None:
            raise RuntimeError("Selection locks are consumed only by test evaluation")
    if args.save_masks and len(thresholds) != 1:
        raise RuntimeError("Mask export requires exactly one threshold")
    mask_root = (args.mask_root or args.output.parent / f"{args.output.stem}_masks").resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sam2.utils.transforms import SAM2Transforms

    model = _build_model(cfg, source, checkpoint, device)
    transforms = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0)
    samples, data_access_ledgers = _load_samples_for_role(cfg, source, args.role, return_ledgers=True)
    debug_overfit = cfg.get("debug_overfit", {}) if isinstance(cfg.get("debug_overfit"), dict) else {}
    if args.role == "train" and int(debug_overfit.get("max_train_samples", 0) or 0) > 0:
        train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
        base_seed = int(train_cfg.get("seed", 42) or 42)
        random.Random(int(debug_overfit.get("sample_seed", base_seed) or base_seed)).shuffle(samples)
        samples = samples[: int(debug_overfit.get("max_train_samples", 0) or 0)]
    if args.max_samples:
        samples = samples[: args.max_samples]

    rows: list[dict[str, object]] = []
    exported_masks: list[dict[str, str]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for sample in samples:
            image = transforms(load_image_rgb(sample.image_path))[None].to(device)
            output = model(image, fixed_prompt=None)
            full_logits = transforms.postprocess_masks(output["low_res_logits"], (sample.height, sample.width))[0, 0]
            probability = torch.sigmoid(full_logits).detach().cpu().numpy().astype(np.float32)
            target = (np.asarray(sample.mask, dtype=np.float32) > 0.5).astype(np.float32)
            points = output["point_coords"][0].detach().cpu().numpy()
            quality_raw = float(output["iou"].float().mean().detach().cpu())
            quality_score = float(np.clip(quality_raw, 0.0, 1.0))
            quality_rejected = float(quality_reject_below > 0.0 and quality_score < quality_reject_below)
            effective_probability = np.zeros_like(probability) if quality_rejected > 0.5 else probability
            points_original = np.stack(
                [points[:, 0] / model.image_size * sample.width, points[:, 1] / model.image_size * sample.height], axis=1
            ).astype(np.float32)
            box = output["box_coords"][0].detach().cpu().numpy().astype(np.float32)
            box_original = np.asarray(
                [
                    box[0] / model.image_size * sample.width,
                    box[1] / model.image_size * sample.height,
                    box[2] / model.image_size * sample.width,
                    box[3] / model.image_size * sample.height,
                ],
                dtype=np.float32,
            )
            diagnostics = {**_candidate_stats(points_original, target), "PromptBoxCoverage": _box_coverage(box_original, target)}
            diagnostics["PromptHitRate"] = diagnostics["PromptHitRateTop1"]
            for threshold in thresholds:
                raw_metric_row = binary_mask_row(probability, target, threshold)
                metric_row = binary_mask_row(effective_probability, target, threshold)
                metric_row.update(
                    {
                        "dataset": sample.dataset_key,
                        "sample_id": sample.sample_id,
                        "R50": float(metric_row["IoU_image"] >= 0.5),
                        **diagnostics,
                        "MaskQualityScoreRaw": quality_raw,
                        "MaskQualityScore": quality_score,
                        "MaskQualityTargetIoU": float(raw_metric_row["IoU_image"]),
                        "MaskQualityAbsoluteError": abs(quality_score - float(raw_metric_row["IoU_image"])),
                        "MaskQualityRejected": quality_rejected,
                        "MaskQualityRejectBelow": quality_reject_below,
                    }
                )
                rows.append(metric_row)
            if args.save_masks:
                path = _save_mask(mask_root, sample.dataset_key, sample.sample_id, effective_probability >= thresholds[0])
                exported_masks.append({"dataset": sample.dataset_key, "sample_id": sample.sample_id, "path": path})
    elapsed = time.perf_counter() - started

    summaries = _summarize_rows(rows, thresholds)
    target_present_rows = [row for row in rows if float(row["TargetPresent"]) > 0.5]
    target_absent_rows = [row for row in rows if float(row["TargetPresent"]) <= 0.5]
    summary_by_target_presence = {
        "target_present": _summarize_rows(target_present_rows, thresholds),
        "target_absent": _summarize_rows(target_absent_rows, thresholds),
    }
    payload = {
        "protocol": "sparksam_image_only_evaluation"
        if reproduction_protocol_enabled(cfg)
        else "sparksam_image_only_evaluation",
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "code": benchmark_code,
        "sam2_code": sam2_code,
        "metric_notes": {
            "global_IoU": "dataset-level TP/(TP+FP+FN)",
            "nIoU": "mean per-image IoU",
            "global_F1": "dataset-level 2TP/(2TP+FP+FN)",
            "Dice_image_mean": "mean per-image Dice; intentionally distinct from global_F1",
            "FApxMP_global": "total false-alarm pixels divided by total evaluated megapixels",
            "FApxMP_image_mean": "mean of per-image false-alarm pixels per megapixel",
            "target_absent_image_FPR": "fraction of ground-truth-empty images with any predicted foreground",
            "wall_clock_ms_per_image": "includes image IO, preprocessing, inference, metric computation, and optional mask export; not a latency benchmark",
            "MaskQualityScore": "SAM2 mask-decoder quality prediction, averaged over decoder masks and clamped to [0,1]",
            "MaskQualityTargetIoU": "per-image prediction/GT mask IoU at the evaluated threshold",
            "MaskQualityRejected": "1 when the config-locked mask-quality threshold replaces the prediction with an empty mask",
            "MaskQualityRejectBelow": "quality-only rejection threshold declared in the immutable experiment config; no area rule is used",
        },
        "config": artifact_record(args.config.resolve()),
        "mask_quality_rejection": {"enabled": quality_reject_below > 0.0, "reject_below": quality_reject_below, "uses_area": False},
        "validation_diagnostic_overrides": diagnostic_overrides,
        "locked_test_inference_overrides": diagnostic_overrides if args.role == "test" else {},
        "checkpoint": artifact_record(checkpoint),
        "role": args.role,
        "selection_audit": selection_audit,
        "fixed_prompt_prior": None,
        "teacher_loaded": False,
        "cache_loaded": False,
        "external_prompt_loaded": False,
        "data_access_ledgers": data_access_ledgers,
        "elapsed_s": elapsed,
        "wall_clock_ms_per_image": elapsed * 1000.0 / max(1, len(samples)),
        "summary": summaries,
        "summary_by_target_presence": summary_by_target_presence,
        "rows": rows,
        "exported_masks": exported_masks,
    }
    _write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "summary": summaries}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
