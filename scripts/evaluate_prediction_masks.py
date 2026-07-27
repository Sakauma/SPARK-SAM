#!/usr/bin/env python3
"""Recompute canonical three-dataset metrics from saved binary/probability masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
from sparksam.protocols.reproduction import artifact_record, git_record, read_mapping, sha256_file  # noqa: E402
from scripts.train_sparksam import _load_samples_for_role, _read_yaml  # noqa: E402


def _prediction_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy"}:
            index.setdefault(path.stem, []).append(path)
    return index


def _find_prediction(index: dict[str, list[Path]], dataset: str, sample_id: str) -> Path:
    raw = str(sample_id)
    base = raw.split("::", 1)[0]
    aliases = {raw, Path(raw).name, Path(raw).stem, base, Path(base).name, Path(base).stem}
    aliases.update("".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value) for value in list(aliases))
    candidates = [path for alias in aliases for path in index.get(Path(alias).stem, [])]
    dataset_token = "".join(ch for ch in dataset.lower() if ch.isalnum())
    preferred = [path for path in candidates if dataset_token in "".join(ch for ch in str(path.parent).lower() if ch.isalnum())]
    candidates = preferred or candidates
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(f"Expected one prediction for {dataset}/{sample_id}, found {len(unique)}: {unique[:5]}")
    return unique[0]


def _load_probability(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        array = np.asarray(Image.open(path).convert("L"))
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Prediction must be 2D: {path} shape={array.shape}")
    if float(array.max(initial=0.0)) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=["validation", "test"], required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-oracle-diagnostic", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = read_mapping(args.baseline_manifest.resolve())
    audit = read_mapping(args.baseline_audit.resolve())
    if audit.get("status") != "passed":
        raise RuntimeError("Baseline protocol audit did not pass")
    if str((audit.get("manifest", {}) or {}).get("sha256", "")) != sha256_file(args.baseline_manifest.resolve()):
        raise RuntimeError("Baseline audit was created for a different baseline manifest")
    if not audit.get("eligible_for_main_accuracy_table") and not args.allow_oracle_diagnostic:
        raise RuntimeError("Oracle-prompt baseline may be evaluated only with --allow-oracle-diagnostic")
    selection = manifest.get("selection", {}) if isinstance(manifest.get("selection"), dict) else {}
    threshold = selection.get("threshold")
    if threshold is None:
        raise RuntimeError("Baseline manifest must record its validation-selected threshold")
    threshold = float(threshold)
    cfg = _read_yaml(args.config)
    source = _read_yaml(Path(str(cfg["source_config"])))
    samples = _load_samples_for_role(cfg, source, args.role)
    prediction_root = args.prediction_root.resolve()
    index = _prediction_index(prediction_root)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        prediction_path = _find_prediction(index, sample.dataset_key, sample.sample_id)
        probability = _load_probability(prediction_path)
        target = (np.asarray(sample.mask, dtype=np.float32) > 0.5).astype(np.float32)
        if probability.shape != target.shape:
            raise RuntimeError(f"Prediction shape mismatch for {sample.dataset_key}/{sample.sample_id}: {probability.shape} != {target.shape}")
        row = binary_mask_row(probability, target, threshold)
        row.update({"dataset": sample.dataset_key, "sample_id": sample.sample_id, "prediction_path": str(prediction_path)})
        rows.append(row)
    summary = compatibility_aliases(aggregate_binary_mask_rows(rows))
    payload = {
        "schema_version": "segmentation_saved_mask_evaluation_v1",
        "code": git_record(),
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "role": args.role,
        "threshold": threshold,
        "track": audit.get("track"),
        "eligible_for_main_accuracy_table": bool(audit.get("eligible_for_main_accuracy_table")),
        "config": artifact_record(args.config.resolve()),
        "baseline_manifest": artifact_record(args.baseline_manifest.resolve()),
        "baseline_audit": artifact_record(args.baseline_audit.resolve()),
        "prediction_root": str(prediction_root),
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
