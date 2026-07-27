#!/usr/bin/env python3
"""Consume one validation selection lock and evaluate the sealed test split once."""

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

from sparksam.protocols.reproduction import (  # noqa: E402
    SELECTION_LOCK_SCHEMA_VERSION,
    artifact_record,
    git_record,
    normalize_locked_inference_overrides,
    read_mapping,
    resolve_project_path,
    sha256_file,
    utc_now,
    validate_selection_lock,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--retry-failed-claim",
        action="store_true",
        help="Retry only an exact, technically failed claim that produced no result artifact.",
    )
    return parser


def _write_claim(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lock_path = args.selection_lock.resolve()
    lock = read_mapping(lock_path)
    if lock.get("schema_version") != SELECTION_LOCK_SCHEMA_VERSION:
        raise RuntimeError(f"Invalid selection lock: {lock_path}")
    checkpoint = resolve_project_path(lock.get("selected_checkpoint", {}).get("path", ""), base=lock_path.parent)
    threshold = float(lock["threshold"])
    inference_overrides = normalize_locked_inference_overrides(lock.get("inference_overrides"))
    config_path = args.config.resolve()
    cfg = read_mapping(config_path)
    selection_audit = validate_selection_lock(
        lock_path,
        cfg=cfg,
        checkpoint_path=checkpoint,
        threshold=threshold,
        config_path=config_path,
        inference_overrides=inference_overrides,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Test result already exists: {output}")
    claim_path = lock_path.with_name(lock_path.stem + ".test_claim.json")
    previous_claim = read_mapping(claim_path) if claim_path.exists() else None
    if previous_claim is not None:
        if not args.retry_failed_claim:
            raise RuntimeError(f"This validation lock has already been claimed for test: {claim_path}")
        if str(previous_claim.get("status", "")) != "failed":
            raise RuntimeError("--retry-failed-claim accepts only a claim whose status is exactly 'failed'")
        expected = {
            "selection_lock": artifact_record(lock_path)["sha256"],
            "config": artifact_record(config_path)["sha256"],
            "checkpoint": artifact_record(checkpoint)["sha256"],
        }
        for field, digest in expected.items():
            if str((previous_claim.get(field, {}) or {}).get("sha256", "")) != digest:
                raise RuntimeError(f"Failed claim {field} differs from the sealed retry request")
        if abs(float(previous_claim.get("threshold")) - threshold) > 1e-12:
            raise RuntimeError("Failed claim threshold differs from the sealed retry request")
        if Path(str(previous_claim.get("output", ""))).resolve() != output:
            raise RuntimeError("Failed claim output path differs from the sealed retry request")
    claim = {
        "status": "claimed",
        "code": git_record(),
        "selection_lock": artifact_record(lock_path),
        "selection_audit": selection_audit,
        "config": artifact_record(config_path),
        "checkpoint": artifact_record(checkpoint),
        "threshold": threshold,
        "inference_overrides": inference_overrides,
        "output": str(output),
        "claimed_at": utc_now(),
        "attempt_count": int((previous_claim or {}).get("attempt_count", 0)) + 1,
        "retry_of_failed_claim_sha256": sha256_file(claim_path) if previous_claim is not None else None,
    }
    if previous_claim is None:
        _write_claim(claim_path, claim)
    else:
        claim_path.write_text(json.dumps(claim, ensure_ascii=False, indent=2), encoding="utf-8")
    command = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "evaluate_sparksam.py"),
        "--config",
        str(args.config.resolve()),
        "--checkpoint",
        str(checkpoint),
        "--role",
        "test",
        "--thresholds",
        str(threshold),
        "--output",
        str(output),
        "--selection-lock",
        str(lock_path),
        "--forbid-teacher",
        "--forbid-cache",
        "--forbid-external-prompt",
        "--save-masks",
    ]
    override_flags = {
        "box_edge_decode_mode": "--box-edge-decode-mode",
        "box_edge_outer_quantile": "--box-edge-outer-quantile",
        "box_edge_temperature": "--box-edge-temperature",
        "box_occupancy_threshold": "--box-occupancy-threshold",
        "decoder_point_count": "--decoder-point-count",
        "prompt_box_scale": "--prompt-box-scale",
    }
    for name, value in inference_overrides.items():
        command.extend([override_flags[name], str(value)])
    if args.mask_root is not None:
        command.extend(["--mask-root", str(args.mask_root.resolve())])
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    claim.update(
        {
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "completed_at": utc_now(),
            "result": artifact_record(output) if output.exists() else None,
        }
    )
    claim_path.write_text(json.dumps(claim, ensure_ascii=False, indent=2), encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
