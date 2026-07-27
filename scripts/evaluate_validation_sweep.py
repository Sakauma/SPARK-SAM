#!/usr/bin/env python3
"""Evaluate every saved checkpoint on validation and create one selection lock."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sparksam.protocols.reproduction import artifact_record, read_lineage, read_mapping, reproduction_protocol_enabled  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    parser.add_argument("--checkpoint-glob", default="checkpoint_epoch_*_sparksam.pt")
    parser.add_argument("--python", default=sys.executable)
    return parser


def _existing_report_matches(path: Path, checkpoint: Path) -> bool:
    if not path.exists():
        return False
    payload = read_mapping(path)
    return payload.get("role") == "validation" and payload.get("checkpoint", {}).get("sha256") == artifact_record(checkpoint)["sha256"]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = read_mapping(args.config.resolve())
    strict = reproduction_protocol_enabled(cfg)
    thresholds = [float(value) for value in (cfg.get("evaluation", {}) or {}).get("threshold_sweep", [])]
    if not thresholds:
        raise RuntimeError("Config has no evaluation.threshold_sweep")
    checkpoints = sorted(args.train_dir.resolve().glob(args.checkpoint_glob))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoint_glob!r} in {args.train_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[Path] = []
    for checkpoint in checkpoints:
        read_lineage(checkpoint, required=strict)
        report = args.output_dir / f"{checkpoint.stem}_validation.json"
        reports.append(report)
        if _existing_report_matches(report, checkpoint):
            continue
        if report.exists():
            raise RuntimeError(f"Existing validation report does not match checkpoint: {report}")
        command = [
            args.python,
            str(PROJECT_ROOT / "scripts" / "evaluate_sparksam.py"),
            "--config",
            str(args.config.resolve()),
            "--checkpoint",
            str(checkpoint),
            "--role",
            "validation",
            "--thresholds",
            ",".join(str(value) for value in thresholds),
            "--output",
            str(report),
            "--forbid-teacher",
            "--forbid-cache",
            "--forbid-external-prompt",
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if args.output_lock.exists():
        print(json.dumps({"status": "selection_lock_exists", "selection_lock": str(args.output_lock)}, indent=2))
        return 0
    select_command = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "select_operating_point.py"),
        "--config",
        str(args.config.resolve()),
        "--reports",
        *[str(path) for path in reports],
        "--output-lock",
        str(args.output_lock.resolve()),
    ]
    subprocess.run(select_command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
