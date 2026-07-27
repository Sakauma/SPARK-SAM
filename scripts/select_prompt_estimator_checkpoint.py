#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sparksam.config import load_app_config  # noqa: E402
from sparksam.data import build_dataset_adapter  # noqa: E402
from sparksam.data.masks import sample_mask_array  # noqa: E402
from sparksam.data.prompt_synthesis import connected_components  # noqa: E402
from sparksam.evaluation.prompt_metrics import prompt_metrics  # noqa: E402
from sparksam.models import load_auto_prompt_model, predict_learned_auto_prompt_from_path  # noqa: E402


PROMPT_METRICS = (
    "TopKComponentRecall",
    "TopKAllComponentsHit",
    "PrimaryMatchedComponentBoxIoU",
    "PromptHitRate",
    "TargetRecallIoU25",
    "PromptTopKHitRate",
    "PromptBoxCoverage",
    "FalseAlarmPixelsPerMP",
)
BBOX_METRICS = ("PromptPointInBBox", "PromptTopKInBBox", "PromptBoxBBoxIoU")
COMPONENT_SELECTION_METRICS = (
    "TopKComponentRecall",
    "TopKAllComponentsHit",
    "PromptHitRate",
    "PrimaryMatchedComponentBoxIoU",
)
COMPONENT_SELECTION_RULE = (
    "maximize TopKComponentRecall",
    "maximize TopKAllComponentsHit",
    "maximize PromptHitRate",
    "maximize PrimaryMatchedComponentBoxIoU",
    "prefer earlier epoch",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalise_lower_is_better(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        return 0.0
    return (value - min_v) / (max_v - min_v)


def _score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    false_alarm_values = [_as_float(row.get("FalseAlarmPixelsPerMP")) for row in rows]
    scored: list[dict[str, Any]] = []
    for row in rows:
        false_alarm_norm = _normalise_lower_is_better(false_alarm_values, _as_float(row.get("FalseAlarmPixelsPerMP")))
        if any(metric in row for metric in ("PromptHitRate", "TargetRecallIoU25", "PromptBoxCoverage")):
            score = (
                0.35 * _as_float(row.get("PromptHitRate"))
                + 0.25 * _as_float(row.get("TargetRecallIoU25"))
                + 0.20 * _as_float(row.get("PromptTopKHitRate"))
                + 0.10 * _as_float(row.get("PromptBoxCoverage"))
                - 0.10 * false_alarm_norm
            )
        else:
            score = (
                0.45 * _as_float(row.get("PromptPointInBBox"))
                + 0.35 * _as_float(row.get("PromptTopKInBBox"))
                + 0.20 * _as_float(row.get("PromptBoxBBoxIoU"))
            )
        scored.append({**row, "PromptSelectionScore": score, "FalseAlarmPixelsPerMPNorm": false_alarm_norm})
    return scored


def _rows_from_metrics_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in metrics CSV: {path}")
    return rows


def _rows_from_train_summary(train_dir: Path) -> list[dict[str, Any]]:
    summary_path = train_dir / "train_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing train_summary.json: {summary_path}")
    summary = _read_json(summary_path)
    rows: list[dict[str, Any]] = []
    for record in summary.get("checkpoint_history", []):
        if not isinstance(record, dict):
            continue
        checkpoint_path = record.get("checkpoint_path")
        if not checkpoint_path:
            continue
        metric_name = str(record.get("metric_name", "loss"))
        metric_value = _as_float(record.get("metric_value"))
        rows.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "epoch": record.get("epoch", ""),
                "source_metric_name": metric_name,
                "source_metric_value": metric_value,
                "PromptSelectionScore": -metric_value,
            }
        )
    if not rows and summary.get("selected_checkpoint_path"):
        rows.append(
            {
                "checkpoint_path": str(summary["selected_checkpoint_path"]),
                "epoch": summary.get("best_checkpoint_epoch", ""),
                "source_metric_name": summary.get("best_metric_name", "selected"),
                "source_metric_value": summary.get("best_metric_value", 0.0),
                "PromptSelectionScore": 0.0,
            }
        )
    if not rows:
        raise ValueError(f"No checkpoint candidates found in {summary_path}")
    return rows


def _bbox_iou(box_a: list[float] | None, box_b: list[float] | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in box_b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0.0 else 0.0


def _point_in_box(point: object, box: list[float] | None) -> float:
    if box is None or not isinstance(point, list) or len(point) < 2:
        return 0.0
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    return 1.0 if x1 <= x <= x2 and y1 <= y <= y2 else 0.0


def _component_box(component: Any) -> list[float] | None:
    import numpy as np

    ys, xs = np.where(component > 0.5)
    if xs.size == 0 or ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _component_index_for_point(point: object, components: list[Any]) -> int:
    if not isinstance(point, list) or len(point) < 2:
        return -1
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    for index, component in enumerate(components):
        height, width = component.shape
        if 0 <= x < width and 0 <= y < height and component[y, x] > 0.5:
            return index
    return -1


def _component_prompt_metrics(prompt: Any, gt_mask: Any) -> dict[str, float]:
    components = connected_components(gt_mask)
    candidate_points = prompt.metadata.get("candidate_points", [])
    if not isinstance(candidate_points, list):
        candidate_points = []
    hit_components = {
        _component_index_for_point(point, components)
        for point in candidate_points
        if isinstance(point, list)
    }
    hit_components.discard(-1)
    primary_index = _component_index_for_point(prompt.point, components)
    primary_box = _component_box(components[primary_index]) if primary_index >= 0 else None
    return {
        "TopKComponentRecall": float(len(hit_components) / max(1, len(components))),
        "TopKAllComponentsHit": float(bool(components) and len(hit_components) == len(components)),
        "PrimaryMatchedComponentBoxIoU": _bbox_iou(prompt.box, primary_box),
    }


def _target_recall_from_prompt_box(prompt_box: list[float] | None, gt_box: list[float] | None, threshold: float = 0.25) -> float:
    return 1.0 if _bbox_iou(prompt_box, gt_box) >= float(threshold) else 0.0


def _checkpoint_candidates_from_summary(train_dir: Path) -> list[dict[str, Any]]:
    rows = _rows_from_train_summary(train_dir)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        path = Path(str(row["checkpoint_path"]))
        if not path.is_absolute():
            path = train_dir / path
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        candidates.append({**row, "checkpoint_path": key})
    return candidates


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _evaluate_checkpoint_prompt_metrics(
    *,
    checkpoint_path: Path,
    dataset_config_paths: list[Path],
    device: str,
    max_samples: int,
    top_k: int,
    point_budget: int,
    response_threshold: float,
    nms_radius: int,
    border_suppression_px: int,
) -> dict[str, Any]:
    model, metadata = load_auto_prompt_model(checkpoint_path, device=device)
    cfg = metadata.get("config", {})
    metric_values: dict[str, list[float]] = {}
    access_ledgers: list[dict[str, Any]] = []
    sample_count = 0
    for dataset_config_path in dataset_config_paths:
        app_config = load_app_config(dataset_config_path)
        split_name = str(getattr(app_config.dataset, "split_name", "") or "").strip().lower()
        if split_name not in {"val", "validation", "valid", "dev"}:
            raise ValueError(
                "Checkpoint selection is validation-only; "
                f"dataset config {dataset_config_path} declares split_name={split_name!r}."
            )
        adapter = build_dataset_adapter(app_config)
        loaded = adapter.load(app_config)
        manifest = loaded.manifest.to_dict()
        if manifest.get("physical_file_policy") != "split_before_decode":
            raise RuntimeError(
                "Checkpoint selection requires an audited split-before-decode adapter; "
                f"dataset={app_config.dataset.dataset_id!r}, policy={manifest.get('physical_file_policy')!r}."
            )
        requested_count = int(manifest.get("requested_frame_count", 0) or 0)
        if requested_count <= 0 or int(manifest.get("opened_frame_count", 0) or 0) != requested_count:
            raise RuntimeError(
                "Checkpoint selection must physically open every requested validation frame; "
                f"dataset={app_config.dataset.dataset_id!r}, requested={requested_count}, "
                f"opened={manifest.get('opened_frame_count')}."
            )
        if int(manifest.get("opened_image_count", 0) or 0) != requested_count or int(
            manifest.get("opened_mask_count", 0) or 0
        ) != requested_count:
            raise RuntimeError(
                "Checkpoint selection validation image/mask access is incomplete; "
                f"dataset={app_config.dataset.dataset_id!r}."
            )
        access_ledgers.append(manifest)
        samples = loaded.samples
        for sample in samples:
            if sample.bbox_tight is None:
                continue
            prompt = predict_learned_auto_prompt_from_path(
                model=model,
                image_path=sample.image_path,
                device=device,
                min_box_side=float(cfg.get("min_box_side", 2.0)),
                negative_ring_offset=float(cfg.get("negative_ring_offset", 4.0)),
                top_k=top_k,
                point_budget=point_budget,
                response_threshold=response_threshold,
                nms_radius=nms_radius,
                border_suppression_px=border_suppression_px,
                use_local_contrast=bool(cfg.get("use_local_contrast", True)),
                use_top_hat=bool(cfg.get("use_top_hat", True)),
            )
            prompt_dict = {"point": prompt.point, "box": prompt.box, **prompt.metadata}
            mask = sample_mask_array(sample)
            if mask is not None:
                for key, value in prompt_metrics(prompt_dict, mask).items():
                    metric_values.setdefault(key, []).append(float(value))
                for key, value in _component_prompt_metrics(prompt, mask).items():
                    metric_values.setdefault(key, []).append(float(value))
                metric_values.setdefault("TargetRecallIoU25", []).append(_target_recall_from_prompt_box(prompt.box, sample.bbox_tight, threshold=0.25))
            else:
                metric_values.setdefault("PromptPointInBBox", []).append(_point_in_box(prompt.point, sample.bbox_tight))
                candidates = prompt.metadata.get("candidate_points", [])
                if isinstance(candidates, list):
                    hit = any(_point_in_box(candidate, sample.bbox_tight) >= 1.0 for candidate in candidates if isinstance(candidate, list))
                    metric_values.setdefault("PromptTopKInBBox", []).append(1.0 if hit else 0.0)
                metric_values.setdefault("PromptBoxBBoxIoU", []).append(_bbox_iou(prompt.box, sample.bbox_tight))
            sample_count += 1
            if max_samples > 0 and sample_count >= max_samples:
                break
        if max_samples > 0 and sample_count >= max_samples:
            break
    output = {key: _mean(values) for key, values in sorted(metric_values.items())}
    output["SelectionSampleCount"] = float(sample_count)
    output["ValidationDataAccessLedgerJson"] = json.dumps(
        access_ledgers,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return output


def _rows_from_prompt_validation(
    *,
    train_dir: Path,
    dataset_config_paths: list[Path],
    device: str,
    max_samples: int,
    top_k: int,
    point_budget: int,
    response_threshold: float,
    nms_radius: int,
    border_suppression_px: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in _checkpoint_candidates_from_summary(train_dir):
        checkpoint_path = Path(str(candidate["checkpoint_path"]))
        metrics = _evaluate_checkpoint_prompt_metrics(
            checkpoint_path=checkpoint_path,
            dataset_config_paths=dataset_config_paths,
            device=device,
            max_samples=max_samples,
            top_k=top_k,
            point_budget=point_budget,
            response_threshold=response_threshold,
            nms_radius=nms_radius,
            border_suppression_px=border_suppression_px,
        )
        rows.append({**candidate, **metrics})
    if not rows:
        raise ValueError(f"No prompt-validation checkpoint rows were produced for {train_dir}")
    return rows


def _epoch_number(row: dict[str, Any]) -> float:
    try:
        return float(row.get("epoch"))
    except (TypeError, ValueError):
        return float("inf")


def _resolve_selection_policy(requested: str, rows: list[dict[str, Any]]) -> str:
    policy = str(requested or "auto").strip().lower()
    if policy not in {"auto", "component_lexicographic", "weighted_validation_score"}:
        raise ValueError(f"Unsupported selection policy: {requested!r}")
    if policy != "auto":
        return policy
    if rows and all(metric in row for row in rows for metric in COMPONENT_SELECTION_METRICS):
        return "component_lexicographic"
    return "weighted_validation_score"


def _sort_rows_for_selection(
    rows: list[dict[str, Any]],
    *,
    selection_policy: str,
) -> list[dict[str, Any]]:
    if selection_policy == "component_lexicographic":
        missing = sorted(
            {
                metric
                for row in rows
                for metric in COMPONENT_SELECTION_METRICS
                if metric not in row
            }
        )
        if missing:
            raise ValueError(
                "component_lexicographic selection requires validation localization metrics; "
                f"missing: {', '.join(missing)}"
            )
        for row in rows:
            selection_tuple = [_as_float(row.get(metric)) for metric in COMPONENT_SELECTION_METRICS]
            selection_tuple.append(-_epoch_number(row))
            row["SelectionTuple"] = json.dumps(selection_tuple, separators=(",", ":"))
        ordered = sorted(
            rows,
            key=lambda row: (
                *(_as_float(row.get(metric)) for metric in COMPONENT_SELECTION_METRICS),
                -_epoch_number(row),
            ),
            reverse=True,
        )
    else:
        if any("PromptSelectionScore" not in row for row in rows):
            rows = _score_rows(rows)
        for row in rows:
            row["PromptSelectionScore"] = _as_float(row.get("PromptSelectionScore"))
        ordered = sorted(rows, key=lambda item: _as_float(item.get("PromptSelectionScore")), reverse=True)
    for rank, row in enumerate(ordered, start=1):
        row["SelectionPolicy"] = selection_policy
        row["SelectionRank"] = rank
    return ordered


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for preferred in (
        "checkpoint_path",
        "epoch",
        "SelectionPolicy",
        "SelectionRank",
        "SelectionTuple",
        "PromptSelectionScore",
        *PROMPT_METRICS,
        "FalseAlarmPixelsPerMPNorm",
        "ValidationDataAccessLedgerJson",
    ):
        if any(preferred in row for row in rows) and preferred not in keys:
            keys.append(preferred)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_checkpoint(
    *,
    train_dir: Path,
    metrics_csv: Path | None,
    dataset_config_paths: list[Path] | None,
    device: str,
    max_samples: int,
    top_k: int,
    point_budget: int,
    response_threshold: float,
    nms_radius: int,
    border_suppression_px: int,
    output_name: str,
    report_name: str,
    selection_policy: str = "auto",
) -> dict[str, Any]:
    if dataset_config_paths:
        rows = _rows_from_prompt_validation(
            train_dir=train_dir,
            dataset_config_paths=dataset_config_paths,
            device=device,
            max_samples=max_samples,
            top_k=top_k,
            point_budget=point_budget,
            response_threshold=response_threshold,
            nms_radius=nms_radius,
            border_suppression_px=border_suppression_px,
        )
    else:
        rows = _rows_from_metrics_csv(metrics_csv) if metrics_csv is not None else _rows_from_train_summary(train_dir)
    resolved_policy = _resolve_selection_policy(selection_policy, rows)
    rows = _sort_rows_for_selection(rows, selection_policy=resolved_policy)
    selected = rows[0]
    checkpoint_path = Path(str(selected.get("checkpoint_path", "")))
    if not checkpoint_path.is_absolute():
        checkpoint_path = train_dir / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint_path}")
    output_path = train_dir / output_name
    if checkpoint_path.resolve() != output_path.resolve():
        shutil.copy2(checkpoint_path, output_path)
    report_path = train_dir / report_name
    _write_report(report_path, rows)
    access_ledgers: list[dict[str, Any]] = []
    raw_access_ledger = selected.get("ValidationDataAccessLedgerJson")
    if isinstance(raw_access_ledger, str) and raw_access_ledger:
        parsed = json.loads(raw_access_ledger)
        if isinstance(parsed, list):
            access_ledgers = [dict(item) for item in parsed if isinstance(item, dict)]
    summary = {
        "selected_checkpoint": str(output_path),
        "source_checkpoint": str(checkpoint_path),
        "report_path": str(report_path),
        "selection_policy": resolved_policy,
        "selection_rule": list(COMPONENT_SELECTION_RULE) if resolved_policy == "component_lexicographic" else ["maximize PromptSelectionScore"],
        "selection_tuple": selected.get("SelectionTuple"),
        "score": _as_float(selected.get("PromptSelectionScore")) if "PromptSelectionScore" in selected else None,
        "validation_data_access_ledgers": access_ledgers,
        "evaluation_protocol": {
            "mode": "validation_inference" if dataset_config_paths else "precomputed_metrics_or_train_summary",
            "dataset_configs": [_file_record(path) for path in (dataset_config_paths or [])],
            "device": device,
            "max_samples": max_samples,
            "top_k": top_k,
            "point_budget": point_budget,
            "response_threshold": response_threshold,
            "nms_radius": nms_radius,
            "border_suppression_px": border_suppression_px,
        },
        "row": selected,
    }
    (train_dir / "checkpoint_selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Select an auto-prompt checkpoint for prompt estimator.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--dataset-config", action="append", type=Path, default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--point-budget", type=int, default=1)
    parser.add_argument("--response-threshold", type=float, default=0.15)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--border-suppression-px", type=int, default=4)
    parser.add_argument("--output-name", default="checkpoint_selected_prompt_estimator.pt")
    parser.add_argument("--report-name", default="checkpoint_selection_report.csv")
    parser.add_argument(
        "--selection-policy",
        choices=("auto", "component_lexicographic", "weighted_validation_score"),
        default="auto",
    )
    args = parser.parse_args()
    summary = select_checkpoint(
        train_dir=args.train_dir.resolve(),
        metrics_csv=args.metrics_csv.resolve() if args.metrics_csv else None,
        dataset_config_paths=[path.resolve() for path in args.dataset_config],
        device=args.device,
        max_samples=args.max_samples,
        top_k=args.top_k,
        point_budget=args.point_budget,
        response_threshold=args.response_threshold,
        nms_radius=args.nms_radius,
        border_suppression_px=args.border_suppression_px,
        output_name=args.output_name,
        report_name=args.report_name,
        selection_policy=args.selection_policy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
