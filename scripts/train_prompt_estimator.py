#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from sparksam.training import train_auto_prompt_from_config  # noqa: E402
from sparksam.protocols.reproduction import audit_auto_prompt_training_config, finalize_prompt_teacher_lineage  # noqa: E402
from scripts.select_prompt_estimator_checkpoint import select_checkpoint  # noqa: E402


def _resolve_from_config(config_path: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()
    config_candidate = (config_path.parent / path).resolve()
    return project_candidate if project_candidate.exists() else config_candidate


def _post_training_selection(config_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    selection = payload.get("post_training_checkpoint_selection", {})
    if not isinstance(selection, dict) or not bool(selection.get("enabled", False)):
        return summary
    raw_dataset_configs = selection.get("dataset_configs", [])
    if not isinstance(raw_dataset_configs, list) or not raw_dataset_configs:
        raise ValueError("post_training_checkpoint_selection.dataset_configs must be a non-empty list")
    train_dir = Path(str(summary.get("output_dir", ""))).expanduser().resolve()
    optimization_selection = {
        "selected_checkpoint_path": summary.get("selected_checkpoint_path"),
        "best_checkpoint_path": summary.get("best_checkpoint_path"),
        "best_checkpoint_epoch": summary.get("best_checkpoint_epoch"),
        "best_metric_name": summary.get("best_metric_name"),
        "best_metric_value": summary.get("best_metric_value"),
    }
    selected = select_checkpoint(
        train_dir=train_dir,
        metrics_csv=None,
        dataset_config_paths=[_resolve_from_config(config_path, item) for item in raw_dataset_configs],
        device=str(selection.get("device", (payload.get("train", {}) or {}).get("device", "cuda"))),
        max_samples=int(selection.get("max_samples", 0) or 0),
        top_k=int(selection.get("top_k", 8) or 8),
        point_budget=int(selection.get("point_budget", 1) or 1),
        response_threshold=float(selection.get("response_threshold", 0.15) or 0.15),
        nms_radius=int(selection.get("nms_radius", 4) or 4),
        border_suppression_px=int(selection.get("border_suppression_px", 4) or 4),
        output_name=str(selection.get("output_name", "checkpoint_best.pt")),
        report_name=str(selection.get("report_name", "checkpoint_selection_report.csv")),
        selection_policy=str(selection.get("selection_policy", "component_lexicographic")),
    )
    row = selected.get("row", {}) if isinstance(selected.get("row"), dict) else {}
    try:
        selected_epoch = int(float(row.get("epoch")))
    except (TypeError, ValueError):
        selected_epoch = None
    updated = {
        **summary,
        "optimization_checkpoint_selection": optimization_selection,
        "selected_checkpoint_path": selected["selected_checkpoint"],
        "best_checkpoint_path": selected["selected_checkpoint"],
        "best_checkpoint_epoch": selected_epoch,
        "best_metric_name": "val_component_lexicographic",
        "best_metric_value": row.get("TopKComponentRecall"),
        "checkpoint_selection": selected,
    }
    (train_dir / "train_summary.json").write_text(
        json.dumps(updated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the learned infrared prompt-estimation model.")
    parser.add_argument("--config", required=True, type=Path, help="Path to auto prompt training YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Parse arguments without training.")
    parser.add_argument("--preflight-only", action="store_true", help="Audit strict train/validation roles without training.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps({"config": str(args.config), "dry_run": True}, ensure_ascii=False, indent=2))
        return 0
    if args.preflight_only:
        report = audit_auto_prompt_training_config(args.config.resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    config_path = args.config.resolve()
    summary = train_auto_prompt_from_config(config_path)
    summary = _post_training_selection(config_path, summary)
    lineage_path = finalize_prompt_teacher_lineage(config_path, summary)
    if lineage_path is not None:
        summary = {**summary, "artifact_lineage": str(lineage_path)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
