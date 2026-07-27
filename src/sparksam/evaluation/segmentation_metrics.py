"""Canonical pixel metrics for the three-dataset clean protocol.

The module deliberately exposes both dataset-global and image-mean quantities.
This prevents the historical ambiguity where a global IoU was reported beside an
image-mean Dice value under the generic label ``F1``.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


METRIC_SCHEMA_VERSION = "segmentation_pixel_metrics_v2"


def binary_mask_row(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    probability = np.asarray(probability, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if probability.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {probability.shape} != {target.shape}")
    prediction = probability >= float(threshold)
    foreground = target > 0.5
    tp = float(np.logical_and(prediction, foreground).sum())
    fp = float(np.logical_and(prediction, ~foreground).sum())
    fn = float(np.logical_and(~prediction, foreground).sum())
    pred_area = float(prediction.sum())
    gt_area = float(foreground.sum())
    union = tp + fp + fn
    denom_f1 = 2.0 * tp + fp + fn
    image_mp = float(target.size) / 1_000_000.0
    target_present = gt_area > 0.0
    prediction_present = pred_area > 0.0
    return {
        "threshold": float(threshold),
        "PixelTP": tp,
        "PixelFP": fp,
        "PixelFN": fn,
        "PredAreaPixels": pred_area,
        "GTAreaPixels": gt_area,
        "ImageMP": image_mp,
        "IoU_image": tp / union if union > 0 else 1.0,
        "Dice_image": 2.0 * tp / denom_f1 if denom_f1 > 0 else 1.0,
        "Precision_image": tp / pred_area if pred_area > 0 else (1.0 if not target_present else 0.0),
        "Recall_image": tp / gt_area if gt_area > 0 else 1.0,
        "FApxMP_image": fp / max(image_mp, 1e-12),
        "TargetPresent": float(target_present),
        "PredictionPresent": float(prediction_present),
        "TargetAbsentFalsePositive": float((not target_present) and prediction_present),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def aggregate_binary_mask_rows(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    materialized = list(rows)
    tp = sum(float(row.get("PixelTP", 0.0)) for row in materialized)
    fp = sum(float(row.get("PixelFP", 0.0)) for row in materialized)
    fn = sum(float(row.get("PixelFN", 0.0)) for row in materialized)
    image_mp = sum(float(row.get("ImageMP", 0.0)) for row in materialized)
    positive = [row for row in materialized if float(row.get("TargetPresent", 0.0)) > 0.5]
    negative = [row for row in materialized if float(row.get("TargetPresent", 0.0)) <= 0.5]
    global_union = tp + fp + fn
    global_f1_denom = 2.0 * tp + fp + fn
    return {
        "samples": len(materialized),
        "target_present_samples": len(positive),
        "target_absent_samples": len(negative),
        "global_IoU": tp / global_union if global_union > 0 else 1.0,
        "nIoU": _mean(materialized, "IoU_image"),
        "positive_nIoU": _mean(positive, "IoU_image"),
        "global_F1": 2.0 * tp / global_f1_denom if global_f1_denom > 0 else 1.0,
        "Dice_image_mean": _mean(materialized, "Dice_image"),
        "Precision_global": tp / (tp + fp) if (tp + fp) > 0 else (1.0 if (tp + fn) == 0 else 0.0),
        "Recall_global": tp / (tp + fn) if (tp + fn) > 0 else 1.0,
        "FApxMP_global": fp / max(image_mp, 1e-12),
        "FApxMP_image_mean": _mean(materialized, "FApxMP_image"),
        "target_absent_image_FPR": _mean(negative, "TargetAbsentFalsePositive"),
        "PixelTP": tp,
        "PixelFP": fp,
        "PixelFN": fn,
        "ImageMP": image_mp,
    }


def compatibility_aliases(summary: dict[str, Any]) -> dict[str, Any]:
    """Return explicit compatibility aliases while keeping their aggregation visible."""
    return {
        **summary,
        "IoU": summary["global_IoU"],
        "F1": summary["global_F1"],
        "Dice": summary["Dice_image_mean"],
        "FalseAlarmPixelsPerMP": summary["FApxMP_global"],
        "FApxMP": summary["FApxMP_global"],
    }
