#!/usr/bin/env python3
"""Run a no-training SPARK clean-protocol preflight and write its audit record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sparksam.protocols.reproduction import audit_auto_prompt_training_config, audit_spark_training_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=["student", "prompt_teacher"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-cache-record-hashes",
        action="store_true",
        help="Read every cache record and verify its declared SHA-256; slower but suitable for the final evidence audit.",
    )
    args = parser.parse_args(argv)
    report = (
        audit_spark_training_config(
            args.config.resolve(),
            verify_cache_record_hashes=args.verify_cache_record_hashes,
        )
        if args.kind == "student"
        else audit_auto_prompt_training_config(args.config.resolve())
    )
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
