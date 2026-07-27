#!/usr/bin/env python3
"""Select one SPARK-SAM checkpoint and threshold from validation reports and seal them."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sparksam.protocols.reproduction import (  # noqa: E402
    SELECTION_LOCK_SCHEMA_VERSION,
    ProtocolViolation,
    artifact_record,
    canonical_role,
    reproduction_protocol,
    reproduction_protocol_enabled,
    git_record,
    normalize_locked_inference_overrides,
    read_lineage,
    read_mapping,
    resolve_project_path,
    sha256_file,
    utc_now,
)


def _as_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation(f"Candidate metric {name!r} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ProtocolViolation(f"Candidate metric {name!r} is non-finite: {result}")
    return result


def _report_candidates(
    report_path: Path,
    config_path: Path,
    expected_thresholds: set[float],
    split_path: Path,
    *,
    strict: bool,
    allow_validation_diagnostic_overrides: bool = False,
) -> list[dict[str, Any]]:
    report = read_mapping(report_path)
    if canonical_role(report.get("role")) != "validation":
        raise ProtocolViolation(f"Operating-point report is not validation-only: {report_path}")
    if report.get("fixed_prompt_prior") is not None or bool(report.get("teacher_loaded")) or bool(report.get("cache_loaded")):
        raise ProtocolViolation(f"Validation report used a forbidden inference dependency: {report_path}")
    if bool(report.get("external_prompt_loaded")):
        raise ProtocolViolation(f"Validation report used an external prompt: {report_path}")
    report_code = report.get("code", {})
    if strict and (
        not isinstance(report_code, dict)
        or not str(report_code.get("revision", "")).strip()
        or bool(report_code.get("dirty", False))
    ):
        raise ProtocolViolation(f"Validation report has no clean code revision: {report_path}")
    sam2_code = report.get("sam2_code", {})
    if strict and (
        not isinstance(sam2_code, dict)
        or not str(sam2_code.get("revision", "")).strip()
        or bool(sam2_code.get("dirty", False))
    ):
        raise ProtocolViolation(f"Validation report has no clean SAM2 revision: {report_path}")
    config_record = report.get("config", {})
    if str(config_record.get("sha256", "")) != sha256_file(config_path):
        raise ProtocolViolation(f"Validation report config hash mismatch: {report_path}")
    checkpoint_record = report.get("checkpoint", {})
    checkpoint = resolve_project_path(checkpoint_record.get("path", ""), base=report_path.parent)
    if str(checkpoint_record.get("sha256", "")) != sha256_file(checkpoint):
        raise ProtocolViolation(f"Validation report checkpoint hash mismatch: {report_path}")
    checkpoint_lineage = read_lineage(checkpoint, required=strict, expected_split_manifest=split_path)
    training_code_revision = str((checkpoint_lineage or {}).get("code", {}).get("revision", ""))
    if strict and not allow_validation_diagnostic_overrides and training_code_revision != str(report_code.get("revision", "")):
        raise ProtocolViolation(f"Validation report code revision differs from checkpoint lineage: {report_path}")
    checkpoint_sam2 = (checkpoint_lineage or {}).get("sam2_code", {})
    if strict and checkpoint_sam2 and str(checkpoint_sam2.get("revision", "")) != str(sam2_code.get("revision", "")):
        raise ProtocolViolation(f"Validation report SAM2 revision differs from checkpoint lineage: {report_path}")
    summaries = report.get("summary", [])
    if not isinstance(summaries, list) or not summaries:
        raise ProtocolViolation(f"Validation report has no summary rows: {report_path}")
    observed_thresholds = {_as_float(row.get("threshold"), "threshold") for row in summaries if isinstance(row, dict)}
    inference_overrides = normalize_locked_inference_overrides(report.get("validation_diagnostic_overrides"))
    if inference_overrides and not allow_validation_diagnostic_overrides:
        raise ProtocolViolation(f"Validation report contains unsealed diagnostic overrides: {report_path}")
    if not inference_overrides and allow_validation_diagnostic_overrides:
        raise ProtocolViolation(f"Diagnostic override selection requires an explicit override: {report_path}")
    if not allow_validation_diagnostic_overrides and observed_thresholds != expected_thresholds:
        raise ProtocolViolation(
            f"Validation threshold grid differs from the predeclared grid in {report_path}: "
            f"observed={sorted(observed_thresholds)} expected={sorted(expected_thresholds)}"
        )
    return [
        {
            "report": artifact_record(report_path),
            "checkpoint": artifact_record(checkpoint),
            "code_revision": str(report_code.get("revision", "")),
            "training_code_revision": training_code_revision,
            "sam2_revision": str(sam2_code.get("revision", "")),
            "inference_overrides": inference_overrides,
            "threshold_grid": sorted(observed_thresholds),
            "threshold": _as_float(row.get("threshold"), "threshold"),
            "metrics": dict(row),
        }
        for row in summaries
        if isinstance(row, dict)
    ]


def _select(candidates: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    primary = str(policy.get("primary_metric", "global_IoU"))
    direction = str(policy.get("direction", "max")).lower()
    if direction not in {"max", "min"}:
        raise ProtocolViolation(f"Unsupported selection direction: {direction}")
    fa_metric = str(policy.get("false_alarm_metric", "FApxMP_global"))
    fa_budget = policy.get("false_alarm_budget")
    eligible = list(candidates)
    if fa_budget is not None:
        budget = float(fa_budget)
        eligible = [candidate for candidate in eligible if _as_float(candidate["metrics"].get(fa_metric), fa_metric) <= budget]
        if not eligible:
            raise ProtocolViolation(f"No validation candidate satisfies {fa_metric} <= {budget}")
    for candidate in eligible:
        _as_float(candidate["metrics"].get(primary), primary)
        _as_float(candidate["metrics"].get(fa_metric), fa_metric)
        _as_float(candidate["metrics"].get("nIoU"), "nIoU")
    sign = 1.0 if direction == "max" else -1.0
    eligible.sort(
        key=lambda candidate: (
            sign * _as_float(candidate["metrics"].get(primary), primary),
            -_as_float(candidate["metrics"].get(fa_metric), fa_metric),
            _as_float(candidate["metrics"].get("nIoU"), "nIoU"),
            -float(candidate["threshold"]),
            str(candidate["checkpoint"]["sha256"]),
        ),
        reverse=True,
    )
    return eligible[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    parser.add_argument(
        "--allow-validation-diagnostic-overrides",
        action="store_true",
        help="Seal one validation-only inference override and its observed threshold grid for one locked test.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    cfg = read_mapping(config_path)
    strict = reproduction_protocol_enabled(cfg)
    if args.output_lock.exists():
        raise FileExistsError(f"Refusing to overwrite immutable selection lock: {args.output_lock}")
    evaluation = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}
    threshold_values = evaluation.get("threshold_sweep", [])
    expected_thresholds = {float(value) for value in threshold_values}
    if not expected_thresholds:
        raise ProtocolViolation("evaluation.threshold_sweep must be predeclared")
    protocol = reproduction_protocol(cfg)
    policy = protocol.get("selection", {}) if isinstance(protocol.get("selection"), dict) else {}
    split_value = protocol.get("split_manifest") or (cfg.get("split_policy", {}) or {}).get("split_manifest")
    split_path = resolve_project_path(split_value)
    candidates: list[dict[str, Any]] = []
    for report_path in args.reports:
        candidates.extend(
            _report_candidates(
                report_path.resolve(),
                config_path,
                expected_thresholds,
                split_path,
                strict=strict,
                allow_validation_diagnostic_overrides=args.allow_validation_diagnostic_overrides,
            )
        )
    selected = _select(candidates, policy)
    selection_code = git_record()
    if strict and (not str(selection_code.get("revision", "")).strip() or bool(selection_code.get("dirty", False))):
        raise ProtocolViolation("Selection requires a clean, revisioned benchmark worktree")
    validation_code_revisions = {str(candidate.get("code_revision", "")) for candidate in candidates}
    if strict and not args.allow_validation_diagnostic_overrides and validation_code_revisions != {str(selection_code["revision"])}:
        raise ProtocolViolation("Benchmark code changed between validation evaluation and selection")
    if strict and (len(validation_code_revisions) != 1 or not next(iter(validation_code_revisions))):
        raise ProtocolViolation("Validation reports do not share one explicit benchmark revision")
    training_code_revisions = {str(candidate.get("training_code_revision", "")) for candidate in candidates}
    if strict and (len(training_code_revisions) != 1 or not next(iter(training_code_revisions))):
        raise ProtocolViolation("Validation reports do not share one checkpoint training revision")
    inference_overrides = {json.dumps(candidate.get("inference_overrides", {}), sort_keys=True) for candidate in candidates}
    threshold_grids = {json.dumps(candidate.get("threshold_grid", [])) for candidate in candidates}
    if len(inference_overrides) != 1 or len(threshold_grids) != 1:
        raise ProtocolViolation("Validation diagnostic reports do not share one override and threshold grid")
    sealed_overrides = json.loads(next(iter(inference_overrides)))
    sealed_threshold_grid = json.loads(next(iter(threshold_grids)))
    sam2_revisions = {str(candidate.get("sam2_revision", "")) for candidate in candidates}
    if strict and (len(sam2_revisions) != 1 or not next(iter(sam2_revisions))):
        raise ProtocolViolation("Validation reports do not share one explicit SAM2 revision")
    validation_code_revision = next(iter(validation_code_revisions)) if len(validation_code_revisions) == 1 else ""
    training_code_revision = next(iter(training_code_revisions)) if len(training_code_revisions) == 1 else ""
    sam2_revision = next(iter(sam2_revisions)) if len(sam2_revisions) == 1 else ""
    payload = {
        "schema_version": SELECTION_LOCK_SCHEMA_VERSION,
        "anonymous_release_hash_lock": not strict,
        "selection_role": "validation",
        "test_accessed": False,
        "code": selection_code,
        "training_code_revision": training_code_revision,
        "validation_code_revision": validation_code_revision,
        "sam2_revision": sam2_revision,
        "config": artifact_record(config_path),
        "split_manifest": artifact_record(split_path),
        "stage": protocol.get("stage") or (cfg.get("metadata", {}) or {}).get("stage"),
        "ablation": protocol.get("ablation", "full"),
        "seed": int((cfg.get("train", {}) or {}).get("seed", 0)),
        "datasets": [str(item) for item in (cfg.get("datasets", {}) or {}).get("train", [])],
        "selection_policy": {
            "primary_metric": str(policy.get("primary_metric", "global_IoU")),
            "direction": str(policy.get("direction", "max")),
            "false_alarm_metric": str(policy.get("false_alarm_metric", "FApxMP_global")),
            "false_alarm_budget": policy.get("false_alarm_budget"),
            "tie_breakers": ["lower_false_alarm", "higher_nIoU", "lower_threshold", "checkpoint_sha256"],
        },
        "threshold_grid": sealed_threshold_grid if args.allow_validation_diagnostic_overrides else sorted(expected_thresholds),
        "inference_overrides": sealed_overrides,
        "selection_scope": (
            "validation_diagnostic_override" if args.allow_validation_diagnostic_overrides else "predeclared_validation_grid"
        ),
        "selected_checkpoint": selected["checkpoint"],
        "threshold": selected["threshold"],
        "validation_metrics": selected["metrics"],
        "selected_from_report": selected["report"],
        "validation_reports": [artifact_record(path.resolve()) for path in args.reports],
        "candidate_count": len(candidates),
        "created_at": utc_now(),
    }
    args.output_lock.parent.mkdir(parents=True, exist_ok=True)
    args.output_lock.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selection_lock": str(args.output_lock), "selected": selected}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
