"""Leakage-resistant protocol guards for the SPARK-SAM rerun.

Configurations without a ``reproduction_protocol`` block remain usable and are
present.  Once strict mode is enabled, every training-time artifact must be tied
to the same split manifest, may contain train records only, and must carry a
lineage sidecar.  Validation creates an immutable operating-point lock; test
evaluation is accepted only when checkpoint and threshold match that lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA_VERSION = "reproduction_protocol_v1"
LINEAGE_SCHEMA_VERSION = "spark_artifact_lineage_v1"
SELECTION_LOCK_SCHEMA_VERSION = "spark_validation_selection_lock_v1"

LOCKED_INFERENCE_OVERRIDE_NAMES = frozenset(
    {
        "box_edge_decode_mode",
        "box_edge_outer_quantile",
        "box_edge_temperature",
        "box_occupancy_threshold",
        "decoder_point_count",
        "prompt_box_scale",
    }
)

ROLE_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
}
SPLIT_NAME_BY_ROLE = {"train": "train", "validation": "val", "test": "test"}


class ProtocolViolation(RuntimeError):
    """Raised when a clean experiment could access forbidden information."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_role(value: object) -> str:
    token = str(value or "").strip().lower()
    return ROLE_ALIASES.get(token, token)


def _normalise_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def resolve_project_path(value: str | Path, *, base: Path | None = None, require_resolved_env: bool = True) -> Path:
    raw = os.path.expanduser(os.path.expandvars(str(value)))
    if require_resolved_env and ("${" in raw or re.search(r"%[^%]+%", raw)):
        raise ProtocolViolation(f"Unresolved environment token in path: {value}")
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return ((base or PROJECT_ROOT) / path).resolve()


def read_mapping(path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    text = resolved.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if resolved.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(payload, dict):
        raise ProtocolViolation(f"Expected a mapping in {resolved}")
    return payload


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_record(path: str | Path, *, allow_missing: bool = False) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        if allow_missing:
            return {"path": str(resolved), "exists": False, "sha256": "", "size_bytes": 0}
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "exists": True,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def git_record(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    status = run("status", "--short")
    return {"root": str(root), "revision": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status.splitlines()}


def reproduction_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("reproduction_protocol", {})
    return value if isinstance(value, dict) else {}


def reproduction_protocol_enabled(cfg: dict[str, Any]) -> bool:
    protocol = reproduction_protocol(cfg)
    return bool(protocol.get("strict", False))


def normalize_locked_inference_overrides(value: object) -> dict[str, Any]:
    """Validate and canonicalize inference-only values sealed by validation."""

    if value in (None, {}):
        return {}
    if not isinstance(value, dict):
        raise ProtocolViolation("Locked inference overrides must be a mapping")
    unknown = sorted(set(value) - LOCKED_INFERENCE_OVERRIDE_NAMES)
    if unknown:
        raise ProtocolViolation(f"Unsupported locked inference overrides: {unknown}")
    result: dict[str, Any] = {}
    for name, raw in value.items():
        if name == "box_edge_decode_mode":
            token = str(raw)
            if token not in {"expectation", "argmax", "outer_quantile"}:
                raise ProtocolViolation(f"Unsupported box-edge decode mode: {token}")
            result[name] = token
        elif name == "decoder_point_count":
            number = int(raw)
            if number < 1 or float(raw) != float(number):
                raise ProtocolViolation("decoder_point_count must be a positive integer")
            result[name] = number
        else:
            number = float(raw)
            if not (number > 0.0):
                raise ProtocolViolation(f"{name} must be positive")
            result[name] = number
    return result


def _assert_clean_code_record(record: dict[str, Any], label: str) -> None:
    if not str(record.get("revision", "")).strip():
        raise ProtocolViolation(f"{label} has no Git revision")
    if bool(record.get("dirty", False)):
        raise ProtocolViolation(f"{label} worktree is dirty")


def _source_dependency_records(cfg: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source_value = cfg.get("source_config")
    if not source_value:
        return None, None
    source_path = resolve_project_path(source_value)
    source_record = artifact_record(source_path)
    source_cfg = read_mapping(source_path)
    paths = source_cfg.get("paths", {}) if isinstance(source_cfg.get("paths"), dict) else {}
    sam2_paths = paths.get("sam2", {}) if isinstance(paths.get("sam2"), dict) else {}
    sam2_repo_value = sam2_paths.get("repo")
    if not sam2_repo_value:
        return source_record, None
    sam2_repo = resolve_project_path(sam2_repo_value, base=source_path.parent)
    return source_record, git_record(sam2_repo)


def _dataset_entry(manifest: dict[str, Any], dataset_key: str) -> dict[str, Any]:
    datasets = manifest.get("datasets", manifest)
    if not isinstance(datasets, dict):
        raise ProtocolViolation("Split manifest has no datasets mapping")
    wanted = _normalise_token(dataset_key)
    for key, value in datasets.items():
        if not isinstance(value, dict):
            continue
        candidates = {_normalise_token(key)}
        candidates.update(_normalise_token(value.get(name)) for name in ("dataset_key", "dataset_id", "name", "root"))
        if wanted in candidates:
            return value
    raise ProtocolViolation(f"Dataset {dataset_key!r} is absent from split manifest")


def _split_ids(entry: dict[str, Any], role: str) -> set[str]:
    splits = entry.get("splits", entry)
    if not isinstance(splits, dict):
        raise ProtocolViolation("Dataset split entry is not a mapping")
    split_name = SPLIT_NAME_BY_ROLE[canonical_role(role)]
    values = splits.get(split_name)
    if not isinstance(values, list):
        raise ProtocolViolation(f"Split {split_name!r} is missing or is not a list")
    return {str(value) for value in values}


def _id_aliases(value: object) -> set[str]:
    text = str(value or "").strip()
    aliases: set[str] = set()

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if not candidate:
            return
        path = Path(candidate)
        aliases.update({candidate, path.name, path.stem})

    add(text)
    # Generic mask adapters append ``::<category>::<annotation protocol>`` to
    # the frame id.  Split manifests intentionally store the underlying frame
    # id, so keep that mapping explicit instead of relying on substring tests.
    if "::" in text:
        add(text.split("::", 1)[0])
    return aliases


def _id_in_split(sample_id: object, split_ids: set[str]) -> bool:
    aliases = _id_aliases(sample_id)
    return any(bool(aliases & _id_aliases(frame_id)) for frame_id in split_ids)


def audit_split_manifest(path: str | Path, dataset_keys: Iterable[str]) -> dict[str, Any]:
    manifest_path = resolve_project_path(path)
    manifest = read_mapping(manifest_path)
    datasets: dict[str, Any] = {}
    violations: list[str] = []
    for dataset_key in [str(item) for item in dataset_keys]:
        entry = _dataset_entry(manifest, dataset_key)
        split_sets = {role: _split_ids(entry, role) for role in ("train", "validation", "test")}
        overlaps = {
            "train_validation": sorted(split_sets["train"] & split_sets["validation"]),
            "train_test": sorted(split_sets["train"] & split_sets["test"]),
            "validation_test": sorted(split_sets["validation"] & split_sets["test"]),
        }
        for name, values in overlaps.items():
            if values:
                violations.append(f"{dataset_key}:{name} overlap={len(values)}")
        datasets[dataset_key] = {
            "counts": {role: len(ids) for role, ids in split_sets.items()},
            "id_hashes": {role: json_sha256(sorted(ids)) for role, ids in split_sets.items()},
            "overlap_counts": {name: len(values) for name, values in overlaps.items()},
        }
    if violations:
        raise ProtocolViolation("Split audit failed: " + "; ".join(violations))
    return {
        "status": "passed",
        "manifest": artifact_record(manifest_path),
        "datasets": datasets,
    }


def _records_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("records", "entries"):
        values = manifest.get(key)
        if isinstance(values, list):
            return [dict(item) for item in values if isinstance(item, dict)]
    return []


def _record_dataset(record: dict[str, Any]) -> str:
    return str(record.get("dataset") or record.get("dataset_key") or record.get("dataset_id") or "")


def _record_sample(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("frame_id") or record.get("image_id") or "")


def _record_role(record: dict[str, Any]) -> str:
    return canonical_role(record.get("role") or record.get("split"))


def lineage_path_for(artifact_path: str | Path) -> Path:
    path = resolve_project_path(artifact_path)
    return path.with_name(path.name + ".lineage.json")


def read_lineage(
    artifact_path: str | Path,
    *,
    required: bool = True,
    expected_split_manifest: str | Path | None = None,
) -> dict[str, Any] | None:
    artifact = resolve_project_path(artifact_path)
    sidecar = lineage_path_for(artifact)
    if not sidecar.exists():
        if required:
            raise ProtocolViolation(f"Missing artifact lineage sidecar: {sidecar}")
        return None
    payload = read_mapping(sidecar)
    if payload.get("schema_version") != LINEAGE_SCHEMA_VERSION:
        raise ProtocolViolation(f"Unsupported lineage schema in {sidecar}: {payload.get('schema_version')!r}")
    declared = payload.get("artifact", {})
    declared_path = resolve_project_path(declared.get("path", ""), base=sidecar.parent) if isinstance(declared, dict) else None
    if declared_path != artifact or declared.get("sha256") != sha256_file(artifact):
        raise ProtocolViolation(f"Artifact hash does not match lineage: {artifact_path}")
    if bool(payload.get("test_accessed", False)):
        raise ProtocolViolation(f"Training artifact lineage declares test access: {sidecar}")
    training_roles = {canonical_role(item) for item in payload.get("training_roles", [])}
    selection_roles = {canonical_role(item) for item in payload.get("selection_roles", [])}
    if training_roles - {"train"}:
        raise ProtocolViolation(f"Lineage contains non-train training roles: {sorted(training_roles)}")
    if selection_roles - {"validation"}:
        raise ProtocolViolation(f"Lineage contains non-validation selection roles: {sorted(selection_roles)}")
    code = payload.get("code", {})
    if not isinstance(code, dict) or not str(code.get("revision", "")).strip():
        raise ProtocolViolation(f"Lineage has no code revision: {sidecar}")
    if bool(code.get("dirty", False)):
        raise ProtocolViolation(f"Lineage was produced from a dirty worktree: {sidecar}")
    sam2_code = payload.get("sam2_code")
    if sam2_code is not None:
        if not isinstance(sam2_code, dict) or not str(sam2_code.get("revision", "")).strip():
            raise ProtocolViolation(f"Lineage has no SAM2 dependency revision: {sidecar}")
        if bool(sam2_code.get("dirty", False)):
            raise ProtocolViolation(f"Lineage used a dirty SAM2 dependency: {sidecar}")
    split_record = payload.get("split_manifest", {})
    if not isinstance(split_record, dict) or not split_record.get("path") or not split_record.get("sha256"):
        raise ProtocolViolation(f"Lineage has no complete split-manifest record: {sidecar}")
    declared_split = resolve_project_path(split_record["path"], base=sidecar.parent)
    if not declared_split.exists() or str(split_record["sha256"]) != sha256_file(declared_split):
        raise ProtocolViolation(f"Lineage split-manifest hash mismatch: {sidecar}")
    if expected_split_manifest is not None:
        expected_split = resolve_project_path(expected_split_manifest)
        if declared_split != expected_split or str(split_record["sha256"]) != sha256_file(expected_split):
            raise ProtocolViolation(f"Lineage uses a different split manifest: {sidecar}")
    return payload


def audit_cache_manifest(
    manifest_path: str | Path,
    *,
    split_manifest_path: str | Path,
    dataset_keys: Iterable[str],
    allowed_roles: Iterable[str] = ("train",),
    expected_prompt_modes: Iterable[str] | None = None,
    require_lineage: bool = True,
    verify_record_hashes: bool = False,
) -> dict[str, Any]:
    resolved = resolve_project_path(manifest_path)
    manifest = read_mapping(resolved)
    split_path = resolve_project_path(split_manifest_path)
    split_manifest = read_mapping(split_path)
    allowed = {canonical_role(item) for item in allowed_roles}
    records = _records_from_manifest(manifest)
    if not records:
        raise ProtocolViolation(f"Strict cache manifest has no inline records/entries: {resolved}")
    declared_split_hash = str(manifest.get("split_manifest_sha256", ""))
    actual_split_hash = sha256_file(split_path)
    if declared_split_hash != actual_split_hash:
        raise ProtocolViolation(f"Cache split hash mismatch: {resolved}")
    if not bool(manifest.get("training_supervision_only", False)) or not bool(manifest.get("inference_forbidden", False)):
        raise ProtocolViolation(f"Cache must declare training_supervision_only and inference_forbidden: {resolved}")
    expected_modes = {str(item) for item in (expected_prompt_modes or [])}
    requested_datasets = [str(item) for item in dataset_keys]
    violations: list[str] = []
    generator_code = manifest.get("generator_code", {})
    if not isinstance(generator_code, dict) or not str(generator_code.get("revision", "")).strip():
        violations.append("cache manifest has no generator code revision")
    elif bool(generator_code.get("dirty", False)):
        violations.append("cache was generated from a dirty worktree")
    generator_code_at_start = manifest.get("generator_code_at_start", {})
    if not isinstance(generator_code_at_start, dict) or generator_code_at_start.get("revision") != generator_code.get("revision"):
        violations.append("cache generator code revision changed during generation")
    elif bool(generator_code_at_start.get("dirty", False)):
        violations.append("cache generation started from a dirty worktree")
    if str(manifest.get("schema_version", "")) in {"response_guidance_cache_v1", "calibration_response_cache_v1"}:
        sam2_code = manifest.get("sam2_code", {})
        if not isinstance(sam2_code, dict) or not str(sam2_code.get("revision", "")).strip():
            violations.append("SAM2-dependent cache has no SAM2 code revision")
        elif bool(sam2_code.get("dirty", False)):
            violations.append("SAM2-dependent cache used a dirty SAM2 worktree")
    if int(manifest.get("failure_count", 0) or 0) > 0 or manifest.get("failures"):
        violations.append(f"cache declares generation failures={int(manifest.get('failure_count', 0) or 0)}")
    cache_audit = manifest.get("audit", {})
    if isinstance(cache_audit, dict) and str(cache_audit.get("status", "passed")).lower() != "passed":
        violations.append(f"cache generator audit status={cache_audit.get('status')!r}")
    seen: dict[str, set[str]] = {key: set() for key in requested_datasets}
    observed_roles: set[str] = set()
    for record in records:
        role = _record_role(record)
        observed_roles.add(role)
        if role not in allowed:
            violations.append(f"forbidden role={role or '<missing>'} sample={_record_sample(record)}")
            continue
        dataset = _record_dataset(record)
        matches = [key for key in requested_datasets if _normalise_token(key) == _normalise_token(dataset)]
        if not matches:
            continue
        dataset_key = matches[0]
        mode = str(record.get("prompt_mode") or record.get("mode") or "")
        if expected_modes and mode not in expected_modes:
            continue
        record_value = record.get("path") or record.get("npz")
        if not record_value:
            violations.append(f"cache record has no artifact path: {_record_sample(record)}")
            continue
        record_path = resolve_project_path(str(record_value), base=resolved.parent)
        if not record_path.exists():
            violations.append(f"cache record artifact is missing: {record_path}")
            continue
        record_hash = str(record.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", record_hash):
            violations.append(f"cache record has no valid SHA-256 declaration: {record_path}")
            continue
        if int(record.get("size_bytes", -1)) != record_path.stat().st_size:
            violations.append(f"cache record size mismatch: {record_path}")
            continue
        if verify_record_hashes and record_hash != sha256_file(record_path):
            violations.append(f"cache record hash mismatch: {record_path}")
            continue
        sample_id = _record_sample(record)
        train_ids = _split_ids(_dataset_entry(split_manifest, dataset_key), "train")
        if not _id_in_split(sample_id, train_ids):
            violations.append(f"record outside train split: {dataset_key}/{sample_id}")
        else:
            for frame_id in train_ids:
                if _id_in_split(sample_id, {frame_id}):
                    seen[dataset_key].add(frame_id)
                    break
    for dataset_key in requested_datasets:
        train_ids = _split_ids(_dataset_entry(split_manifest, dataset_key), "train")
        missing = train_ids - seen[dataset_key]
        if missing:
            violations.append(f"cache missing {len(missing)} train ids for {dataset_key}")
    source_checkpoint = manifest.get("source_checkpoint") or manifest.get("checkpoint") or manifest.get("teacher_checkpoint")
    official_teacher = manifest.get("teacher_checkpoint")
    if isinstance(official_teacher, dict):
        teacher_value = official_teacher.get("path")
        teacher_hash = str(official_teacher.get("sha256", ""))
        if not teacher_value or not teacher_hash:
            violations.append("official teacher checkpoint record is incomplete")
        else:
            teacher_path = resolve_project_path(str(teacher_value), base=resolved.parent)
            size_matches = int(official_teacher.get("size_bytes", -1)) == teacher_path.stat().st_size if teacher_path.exists() else False
            hash_declared = bool(re.fullmatch(r"[0-9a-f]{64}", teacher_hash))
            hash_matches = not verify_record_hashes or teacher_hash == sha256_file(teacher_path)
            if not teacher_path.exists() or not size_matches or not hash_declared or not hash_matches:
                violations.append("official teacher checkpoint hash mismatch")
    if require_lineage:
        if not source_checkpoint:
            raise ProtocolViolation(f"Cache has no learned source_checkpoint: {resolved}")
        source_path = resolve_project_path(str(source_checkpoint), base=resolved.parent)
        if not source_path.exists():
            raise ProtocolViolation(f"Cache source checkpoint does not exist: {source_path}")
        declared_source_hash = str(manifest.get("source_checkpoint_sha256", ""))
        if not declared_source_hash or declared_source_hash != sha256_file(source_path):
            violations.append("source checkpoint hash mismatch")
        lineage_value = manifest.get("source_checkpoint_lineage")
        if not lineage_value:
            raise ProtocolViolation(f"Cache has no source_checkpoint_lineage: {resolved}")
        lineage_file = resolve_project_path(str(lineage_value), base=resolved.parent)
        if not lineage_file.exists():
            raise ProtocolViolation(f"Cache source lineage does not exist: {lineage_file}")
        if lineage_file != lineage_path_for(source_path):
            violations.append("source checkpoint lineage path is not the checkpoint sidecar")
        else:
            try:
                source_lineage = read_lineage(source_path, required=True, expected_split_manifest=split_path)
                if str((source_lineage or {}).get("code", {}).get("revision", "")) != str(generator_code.get("revision", "")):
                    violations.append("cache generator revision differs from source checkpoint lineage")
                lineage_datasets = {_normalise_token(item) for item in (source_lineage or {}).get("datasets", [])}
                expected_lineage_datasets = {_normalise_token(item) for item in requested_datasets}
                if lineage_datasets != expected_lineage_datasets:
                    violations.append("source checkpoint lineage belongs to different datasets")
            except (OSError, ProtocolViolation, ValueError) as exc:
                violations.append(f"invalid source checkpoint lineage: {exc}")
    if violations:
        preview = "; ".join(violations[:20])
        raise ProtocolViolation(f"Cache audit failed for {resolved}: {preview}")
    return {
        "status": "passed",
        "manifest": artifact_record(resolved),
        "record_count": len(records),
        "roles": sorted(observed_roles),
        "datasets": requested_datasets,
        "prompt_modes": sorted(expected_modes),
        "record_hash_verification": "content_sha256" if verify_record_hashes else "declared_sha256_and_file_size",
    }


def _positive_loss(cfg: dict[str, Any], name: str) -> bool:
    losses = cfg.get("losses", {})
    entry = losses.get(name, {}) if isinstance(losses, dict) else {}
    return isinstance(entry, dict) and float(entry.get("weight", 0.0) or 0.0) > 0.0


def _training_dataset_keys(cfg: dict[str, Any]) -> list[str]:
    datasets = cfg.get("datasets", {}) if isinstance(cfg.get("datasets"), dict) else {}
    return [str(item) for item in datasets.get("train", [])]


def _protocol_split_path(cfg: dict[str, Any]) -> Path:
    protocol = reproduction_protocol(cfg)
    split_policy = cfg.get("split_policy", {}) if isinstance(cfg.get("split_policy"), dict) else {}
    value = protocol.get("split_manifest") or split_policy.get("split_manifest")
    if not value:
        raise ProtocolViolation("reproduction_protocol.split_manifest is required")
    return resolve_project_path(str(value))


def audit_spark_training_config(
    config_path: str | Path,
    cfg: dict[str, Any] | None = None,
    *,
    verify_cache_record_hashes: bool = False,
) -> dict[str, Any]:
    resolved_config = resolve_project_path(config_path)
    cfg = cfg or read_mapping(resolved_config)
    if not reproduction_protocol_enabled(cfg):
        return {"status": "legacy_skipped", "strict": False, "config": artifact_record(resolved_config)}
    protocol = reproduction_protocol(cfg)
    if str(protocol.get("schema_version", PROTOCOL_SCHEMA_VERSION)) != PROTOCOL_SCHEMA_VERSION:
        raise ProtocolViolation(f"Unsupported clean protocol schema: {protocol.get('schema_version')}")
    code_record = git_record()
    if bool(protocol.get("require_clean_code", False)):
        _assert_clean_code_record(code_record, "Benchmark code")
    source_config_record, sam2_code_record = _source_dependency_records(cfg)
    if bool(protocol.get("require_clean_sam2_repo", False)):
        if source_config_record is None or sam2_code_record is None:
            raise ProtocolViolation("Strict student training requires source_config and a declared SAM2 repository")
        _assert_clean_code_record(sam2_code_record, "SAM2 dependency")
    dataset_keys = _training_dataset_keys(cfg)
    if not dataset_keys:
        raise ProtocolViolation("Strict training config has no train datasets")
    split_path = _protocol_split_path(cfg)
    split_audit = audit_split_manifest(split_path, dataset_keys)
    split_policy = cfg.get("split_policy", {}) if isinstance(cfg.get("split_policy"), dict) else {}
    role_names = split_policy.get("role_split_names", {}) if isinstance(split_policy.get("role_split_names"), dict) else {}
    expected_roles = {"train": "train", "validation": "val", "test": "test"}
    if {key: str(role_names.get(key, "")) for key in expected_roles} != expected_roles:
        raise ProtocolViolation(f"Strict role_split_names must equal {expected_roles}")
    if not bool(split_policy.get("strict_no_overlap", False)):
        raise ProtocolViolation("split_policy.strict_no_overlap must be true")
    if cfg.get("fixed_prompt_prior"):
        raise ProtocolViolation("fixed_prompt_prior is forbidden in clean training")
    if cfg.get("prompt_prior"):
        raise ProtocolViolation("prompt_prior is forbidden in clean training; use initialization with a validation selection lock")
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    debug_overfit = cfg.get("debug_overfit", {}) if isinstance(cfg.get("debug_overfit"), dict) else {}
    if any(int(debug_overfit.get(name, 0) or 0) > 0 for name in ("max_train_samples", "max_validation_samples", "max_test_samples")):
        raise ProtocolViolation("Strict paper runs forbid debug_overfit sample limits")
    if bool(cfg.get("smoke_test", False)) or int(cfg.get("max_samples", 0) or 0) > 0:
        raise ProtocolViolation("Strict paper runs forbid smoke-test/sample-limit settings")
    student = cfg.get("student", {}) if isinstance(cfg.get("student"), dict) else {}
    if student.get("trainable"):
        raise ProtocolViolation("student.trainable is ambiguous in strict mode; use train.module_policy only")
    module_policy = train_cfg.get("module_policy")
    if module_policy is not None and train_cfg.get("freeze_modules"):
        raise ProtocolViolation("Use train.module_policy or freeze_modules, not both")
    if not isinstance(module_policy, dict):
        raise ProtocolViolation("Strict training requires an explicit train.module_policy mapping")
    allowed_modules = {
        "image_encoder",
        "prompt_encoder",
        "mask_decoder",
        "prompt_head",
        "local_prompt_projector",
        "logit_calibration_head",
        "dense_mask_refinement_head",
        "highres_prompt_refinement_head",
    }
    unknown_modules = set(module_policy) - allowed_modules
    if unknown_modules:
        raise ProtocolViolation(f"Unknown train.module_policy keys: {sorted(unknown_modules)}")
    invalid_states = {str(value).lower() for value in module_policy.values()} - {"train", "trainable", "frozen", "freeze"}
    if invalid_states:
        raise ProtocolViolation(f"Invalid train.module_policy states: {sorted(invalid_states)}")
    prompt_losses = (
        "prompt_objectness_distillation",
        "prompt_box_loss",
        "prompt_point_loss",
        "prompt_ranking_loss",
        "prompt_teacher_dense_objectness_distillation",
        "candidate_score_loss",
        "candidate_pairwise_ranking_loss",
        "candidate_multi_positive_loss",
        "candidate_coverage_loss",
        "candidate_objectness_loss",
        "candidate_mask_aware_score_loss",
        "candidate_mask_aware_pairwise_loss",
        "candidate_mask_quality_regression_loss",
        "candidate_oracle_mask_response_loss",
        "point_gate_loss",
    )
    frozen_states = {"frozen", "freeze"}
    prompt_supervision_modules = (
        ("prompt_head", "train"),
        ("highres_prompt_refinement_head", "frozen"),
    )
    prompt_supervision_is_frozen = all(
        str(module_policy.get(name, default_state)).lower() in frozen_states
        for name, default_state in prompt_supervision_modules
    )
    if any(_positive_loss(cfg, name) for name in prompt_losses) and prompt_supervision_is_frozen:
        raise ProtocolViolation(
            "Prompt losses are enabled while prompt_head and highres_prompt_refinement_head are frozen"
        )
    if _positive_loss(cfg, "logit_calibration_band_loss") and str(module_policy.get("logit_calibration_head", "frozen")).lower() in {
        "frozen",
        "freeze",
    }:
        raise ProtocolViolation("Logit-calibration loss is enabled while logit_calibration_head is frozen")
    cache_audits: dict[str, Any] = {}
    teacher_needed = any(_positive_loss(cfg, name) for name in ("teacher_mask_distillation", "boundary_distillation", "iou_distillation"))
    if teacher_needed:
        spec = cfg.get("teacher_cache")
        if not isinstance(spec, dict) or not spec.get("manifest"):
            raise ProtocolViolation("Teacher-dependent losses require teacher_cache.manifest")
        cache_audits["teacher_cache"] = audit_cache_manifest(
            spec["manifest"],
            split_manifest_path=split_path,
            dataset_keys=dataset_keys,
            expected_prompt_modes=("box_point",),
            verify_record_hashes=verify_cache_record_hashes,
        )
    dense_needed = _positive_loss(cfg, "prompt_teacher_dense_objectness_distillation")
    if dense_needed:
        spec = cfg.get("prompt_teacher_dense_objectness_cache")
        if not isinstance(spec, dict) or not spec.get("manifest"):
            raise ProtocolViolation("Dense prompt supervision requires prompt_teacher_dense_objectness_cache.manifest")
        cache_audits["prompt_teacher_cache"] = audit_cache_manifest(
            spec["manifest"],
            split_manifest_path=split_path,
            dataset_keys=dataset_keys,
            verify_record_hashes=verify_cache_record_hashes,
        )
    calibration_response_spec = cfg.get("calibration_response_cache")
    if isinstance(calibration_response_spec, dict) and calibration_response_spec.get("manifest"):
        cache_audits["calibration_response_cache"] = audit_cache_manifest(
            calibration_response_spec["manifest"],
            split_manifest_path=split_path,
            dataset_keys=dataset_keys,
            verify_record_hashes=verify_cache_record_hashes,
        )
    initialization = cfg.get("initialization", {}) if isinstance(cfg.get("initialization"), dict) else {}
    stage = str(protocol.get("stage", "joint_adaptation"))
    initialization_record: dict[str, Any]
    if stage == "joint_adaptation":
        if initialization.get("kind") != "official_sam2" or initialization.get("selection_lock") or initialization.get("checkpoint"):
            raise ProtocolViolation("joint_adaptation must initialize only from the declared official SAM2 checkpoint")
        student = cfg.get("student", {}) if isinstance(cfg.get("student"), dict) else {}
        checkpoint_spec = student.get("checkpoint", {}) if isinstance(student.get("checkpoint"), dict) else {}
        if not checkpoint_spec.get("path"):
            raise ProtocolViolation("joint_adaptation has no student.checkpoint.path for official SAM2 initialization")
        initialization_record = {"kind": "official_sam2", "checkpoint": artifact_record(checkpoint_spec["path"])}
    else:
        if initialization.get("kind") != "validation_selection_lock" or not initialization.get("selection_lock") or initialization.get("checkpoint"):
            raise ProtocolViolation(f"{stage} must initialize only from initialization.selection_lock")
        initialization_record = {"kind": "validation_selection_lock", "lock": artifact_record(initialization["selection_lock"])}
    report = {
        "status": "passed",
        "strict": True,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "config": artifact_record(resolved_config),
        "code": code_record,
        "source_config": source_config_record,
        "sam2_code": sam2_code_record,
        "split": split_audit,
        "stage": stage,
        "ablation": str(protocol.get("ablation", "full")),
        "seed": int((cfg.get("train", {}) or {}).get("seed", 0)),
        "datasets": dataset_keys,
        "module_policy": module_policy,
        "initialization": initialization_record,
        "cache_audits": cache_audits,
        "test_accessed": False,
        "audited_at": utc_now(),
    }
    return report


def _dataset_loader_record(config_path: Path, expected_role: str, split_hash: str) -> dict[str, Any]:
    payload = read_mapping(config_path)
    dataset = payload.get("dataset", {}) if isinstance(payload.get("dataset"), dict) else {}
    split_name = str(dataset.get("split_name", ""))
    expected_split = SPLIT_NAME_BY_ROLE[expected_role]
    if split_name != expected_split:
        raise ProtocolViolation(f"{config_path} uses split={split_name!r}; expected {expected_split!r}")
    split_path = resolve_project_path(str(dataset.get("split_manifest", "")), base=config_path.parent)
    if sha256_file(split_path) != split_hash:
        raise ProtocolViolation(f"Dataset loader split hash mismatch: {config_path}")
    return {"config": artifact_record(config_path), "dataset_id": dataset.get("dataset_id"), "split_name": split_name}


def audit_auto_prompt_training_config(config_path: str | Path, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = resolve_project_path(config_path)
    cfg = cfg or read_mapping(resolved)
    if not reproduction_protocol_enabled(cfg):
        return {"status": "legacy_skipped", "strict": False, "config": artifact_record(resolved)}
    protocol = reproduction_protocol(cfg)
    code_record = git_record()
    if bool(protocol.get("require_clean_code", False)):
        _assert_clean_code_record(code_record, "Benchmark code")
    split_path = _protocol_split_path(cfg)
    split_hash = sha256_file(split_path)
    dataset_keys = [str(item) for item in protocol.get("datasets", [])]
    if not dataset_keys:
        raise ProtocolViolation("Clean prompt-teacher config must declare reproduction_protocol.datasets")
    split_audit = audit_split_manifest(split_path, dataset_keys)
    train_configs = cfg.get("light_cache_dataset_configs", [])
    val_configs = cfg.get("validation_light_cache_dataset_configs", cfg.get("validation_dataset_configs", []))
    if not train_configs or not val_configs:
        raise ProtocolViolation("Clean prompt-teacher training requires separate train and validation dataset configs")
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    if not bool(train_cfg.get("select_best_checkpoint", False)) or not str(train_cfg.get("selection_metric", "")).startswith("val_"):
        raise ProtocolViolation("Prompt teacher must select checkpoints using a val_* metric")
    checkpoint_selection = cfg.get("post_training_checkpoint_selection", {})
    if not isinstance(checkpoint_selection, dict) or not bool(checkpoint_selection.get("enabled", False)):
        raise ProtocolViolation("Clean prompt teacher requires post-training validation localization selection")
    selection_policy = str(checkpoint_selection.get("selection_policy", "")).strip()
    if selection_policy != "component_lexicographic":
        raise ProtocolViolation("Clean prompt teacher must use component_lexicographic checkpoint selection")
    declared_policy = str(protocol.get("checkpoint_selection_policy", "")).strip()
    if declared_policy != selection_policy:
        raise ProtocolViolation("reproduction_protocol checkpoint selection policy does not match the executable selection policy")
    selection_configs = checkpoint_selection.get("dataset_configs", [])
    if not isinstance(selection_configs, list) or not selection_configs:
        raise ProtocolViolation("Clean prompt teacher selection must declare validation dataset configs")
    resolved_val_configs = sorted(str(resolve_project_path(item, base=resolved.parent)) for item in val_configs)
    resolved_selection_configs = sorted(str(resolve_project_path(item, base=resolved.parent)) for item in selection_configs)
    if resolved_selection_configs != resolved_val_configs:
        raise ProtocolViolation("Checkpoint selector dataset configs must exactly match validation loaders")
    if int(checkpoint_selection.get("max_samples", 0) or 0) != 0:
        raise ProtocolViolation("Clean checkpoint selection must evaluate the complete validation split")
    if int(checkpoint_selection.get("top_k", 0) or 0) <= 0:
        raise ProtocolViolation("Clean checkpoint selection requires a positive top_k")
    expected_output_name = str(train_cfg.get("best_checkpoint_name", "checkpoint_best.pt"))
    if str(checkpoint_selection.get("output_name", "")) != expected_output_name:
        raise ProtocolViolation("Checkpoint selector output must be the downstream prompt-teacher checkpoint")
    loaders = {
        "train": [_dataset_loader_record(resolve_project_path(item, base=resolved.parent), "train", split_hash) for item in train_configs],
        "validation": [
            _dataset_loader_record(resolve_project_path(item, base=resolved.parent), "validation", split_hash) for item in val_configs
        ],
    }
    for role, records in loaders.items():
        observed = {_normalise_token(record.get("dataset_id")) for record in records}
        expected = {_normalise_token(value) for value in dataset_keys}
        if observed != expected:
            raise ProtocolViolation(f"Prompt-teacher {role} loaders do not match reproduction_protocol.datasets")
    if {canonical_role(item) for item in protocol.get("training_roles", [])} != {"train"}:
        raise ProtocolViolation("Prompt-teacher reproduction_protocol.training_roles must be [train]")
    if {canonical_role(item) for item in protocol.get("selection_roles", [])} != {"validation"}:
        raise ProtocolViolation("Prompt-teacher reproduction_protocol.selection_roles must be [validation]")
    return {
        "status": "passed",
        "strict": True,
        "config": artifact_record(resolved),
        "code": code_record,
        "split": split_audit,
        "loaders": loaders,
        "training_roles": ["train"],
        "selection_roles": ["validation"],
        "checkpoint_selection": {
            "policy": selection_policy,
            "dataset_configs": resolved_selection_configs,
            "max_samples": 0,
            "top_k": int(checkpoint_selection.get("top_k")),
            "output_name": expected_output_name,
        },
        "test_accessed": False,
        "audited_at": utc_now(),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_checkpoint_lineage(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    cfg: dict[str, Any],
    protocol_audit: dict[str, Any],
    artifact_type: str = "spark_student_checkpoint",
    extra: dict[str, Any] | None = None,
) -> Path | None:
    if not reproduction_protocol_enabled(cfg):
        return None
    start_code = protocol_audit.get("code", {}) if isinstance(protocol_audit, dict) else {}
    current_code = git_record()
    _assert_clean_code_record(start_code, "Training-start benchmark code")
    _assert_clean_code_record(current_code, "Checkpoint-save benchmark code")
    if start_code.get("revision") != current_code.get("revision"):
        raise ProtocolViolation("Benchmark code revision changed while training")
    checkpoint = resolve_project_path(checkpoint_path)
    split_path = _protocol_split_path(cfg)
    parents: list[dict[str, Any]] = []
    initialization = cfg.get("initialization", {}) if isinstance(cfg.get("initialization"), dict) else {}
    if initialization.get("selection_lock"):
        lock_path = resolve_project_path(initialization["selection_lock"])
        parents.append({"relation": "initialization_selection_lock", **artifact_record(lock_path)})
    elif initialization.get("checkpoint"):
        parents.append({"relation": "initialization_checkpoint", **artifact_record(initialization["checkpoint"])})
    elif initialization.get("kind") == "official_sam2":
        student = cfg.get("student", {}) if isinstance(cfg.get("student"), dict) else {}
        checkpoint_spec = student.get("checkpoint", {}) if isinstance(student.get("checkpoint"), dict) else {}
        parents.append({"relation": "official_sam2_initialization", **artifact_record(checkpoint_spec.get("path", ""))})
    for key in ("teacher_cache", "prompt_teacher_dense_objectness_cache", "calibration_response_cache"):
        spec = cfg.get(key)
        if isinstance(spec, dict) and spec.get("manifest"):
            parents.append({"relation": key, **artifact_record(spec["manifest"])})
    payload = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "artifact": artifact_record(checkpoint),
        "config": artifact_record(resolve_project_path(config_path)),
        "code": start_code,
        "code_at_checkpoint_save": current_code,
        "sam2_code": protocol_audit.get("sam2_code"),
        "split_manifest": artifact_record(split_path),
        "datasets": _training_dataset_keys(cfg),
        "training_roles": ["train"],
        "selection_roles": [],
        "test_accessed": False,
        "stage": reproduction_protocol(cfg).get("stage"),
        "ablation": reproduction_protocol(cfg).get("ablation"),
        "seed": int((cfg.get("train", {}) or {}).get("seed", 0)),
        "parents": parents,
        "protocol_audit_sha256": json_sha256(protocol_audit),
        "created_at": utc_now(),
        **(extra or {}),
    }
    sidecar = lineage_path_for(checkpoint)
    _write_json(sidecar, payload)
    return sidecar


def finalize_prompt_teacher_lineage(config_path: str | Path, summary: dict[str, Any]) -> Path | None:
    config = resolve_project_path(config_path)
    cfg = read_mapping(config)
    if not reproduction_protocol_enabled(cfg):
        return None
    audit = audit_auto_prompt_training_config(config, cfg)
    start_audit = summary.get("reproduction_protocol_audit_at_start", {})
    if not isinstance(start_audit, dict) or start_audit.get("status") != "passed":
        raise ProtocolViolation("Prompt-teacher summary has no passed training-start protocol audit")
    start_code = start_audit.get("code", {})
    current_code = git_record()
    _assert_clean_code_record(start_code, "Prompt-teacher start code")
    _assert_clean_code_record(current_code, "Prompt-teacher finalize code")
    if start_code.get("revision") != current_code.get("revision"):
        raise ProtocolViolation("Benchmark code revision changed while training the prompt teacher")
    selected = summary.get("selected_checkpoint_path") or summary.get("best_checkpoint_path") or summary.get("checkpoint_path")
    if not selected:
        raise ProtocolViolation("Prompt-teacher summary has no selected validation checkpoint")
    checkpoint = resolve_project_path(str(selected), base=config.parent)
    selection = summary.get("checkpoint_selection", {})
    if not isinstance(selection, dict) or selection.get("selection_policy") != "component_lexicographic":
        raise ProtocolViolation("Prompt-teacher summary has no component-aware validation checkpoint selection")
    access_ledgers = selection.get("validation_data_access_ledgers", [])
    if not isinstance(access_ledgers, list) or not access_ledgers:
        raise ProtocolViolation("Prompt-teacher checkpoint selection has no physical validation access ledger")
    expected_split_hash = sha256_file(_protocol_split_path(cfg))
    for ledger in access_ledgers:
        if not isinstance(ledger, dict):
            raise ProtocolViolation("Prompt-teacher validation access ledger is malformed")
        if canonical_role(ledger.get("split_name")) != "validation":
            raise ProtocolViolation("Prompt-teacher checkpoint selector accessed a non-validation split")
        if str(ledger.get("split_manifest_sha256", "")) != expected_split_hash:
            raise ProtocolViolation("Prompt-teacher checkpoint selector used a different split manifest")
        if str(ledger.get("physical_file_policy", "")) != "split_before_decode":
            raise ProtocolViolation("Prompt-teacher checkpoint selector lacks split-before-decode evidence")
        if not str(ledger.get("opened_frame_ids_sha256", "")):
            raise ProtocolViolation("Prompt-teacher checkpoint selector has no opened-frame audit hash")
        requested_count = int(ledger.get("requested_frame_count", 0) or 0)
        if requested_count <= 0 or int(ledger.get("opened_frame_count", 0) or 0) != requested_count:
            raise ProtocolViolation("Prompt-teacher checkpoint selector did not cover the complete validation split")
        if int(ledger.get("opened_image_count", 0) or 0) != requested_count or int(
            ledger.get("opened_mask_count", 0) or 0
        ) != requested_count:
            raise ProtocolViolation("Prompt-teacher validation image/mask access counts are incomplete")
    evaluation_protocol = selection.get("evaluation_protocol", {})
    if not isinstance(evaluation_protocol, dict) or evaluation_protocol.get("mode") != "validation_inference":
        raise ProtocolViolation("Prompt-teacher selection has no validation inference protocol record")
    configured_selection = cfg.get("post_training_checkpoint_selection", {})
    if not isinstance(configured_selection, dict):
        raise ProtocolViolation("Prompt-teacher config has no executable checkpoint selection section")
    if int(evaluation_protocol.get("max_samples", -1)) != 0:
        raise ProtocolViolation("Prompt-teacher selection protocol did not use the complete validation split")
    for key in ("top_k", "point_budget", "nms_radius", "border_suppression_px"):
        if int(evaluation_protocol.get(key, -1)) != int(configured_selection.get(key, -2)):
            raise ProtocolViolation(f"Prompt-teacher selection protocol mismatch for {key}")
    if abs(float(evaluation_protocol.get("response_threshold", -1.0)) - float(configured_selection.get("response_threshold", -2.0))) > 1e-12:
        raise ProtocolViolation("Prompt-teacher selection protocol mismatch for response_threshold")
    expected_dataset_configs: dict[str, str] = {}
    for item in configured_selection.get("dataset_configs", []):
        dataset_config = resolve_project_path(item, base=config.parent)
        expected_dataset_configs[str(dataset_config)] = str(artifact_record(dataset_config)["sha256"])
    observed_dataset_configs = {
        str(record.get("path", "")): str(record.get("sha256", ""))
        for record in evaluation_protocol.get("dataset_configs", [])
        if isinstance(record, dict)
    }
    if observed_dataset_configs != expected_dataset_configs:
        raise ProtocolViolation("Prompt-teacher selection dataset-config hashes do not match the clean config")
    report_path = resolve_project_path(str(selection.get("report_path", "")), base=config.parent)
    payload = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "artifact_type": "spark_prompt_teacher_checkpoint",
        "artifact": artifact_record(checkpoint),
        "config": artifact_record(config),
        "code": start_code,
        "code_at_finalize": current_code,
        "split_manifest": artifact_record(_protocol_split_path(cfg)),
        "datasets": [str(item) for item in reproduction_protocol(cfg).get("datasets", [])],
        "training_roles": ["train"],
        "selection_roles": ["validation"],
        "test_accessed": False,
        "selection_metric": summary.get("best_metric_name"),
        "selection_metric_value": summary.get("best_metric_value"),
        "selection_epoch": summary.get("best_checkpoint_epoch"),
        "selection_policy": selection.get("selection_policy"),
        "selection_rule": selection.get("selection_rule"),
        "selection_tuple": selection.get("selection_tuple"),
        "selection_source_checkpoint": artifact_record(selection.get("source_checkpoint", "")),
        "selection_report": artifact_record(report_path),
        "selection_evaluation_protocol": evaluation_protocol,
        "validation_data_access_ledgers": access_ledgers,
        "protocol_audit_sha256": json_sha256(start_audit),
        "final_protocol_audit_sha256": json_sha256(audit),
        "created_at": utc_now(),
    }
    sidecar = lineage_path_for(checkpoint)
    _write_json(sidecar, payload)
    return sidecar


def resolve_initialization_checkpoint(cfg: dict[str, Any], *, config_path: str | Path) -> tuple[Path | None, dict[str, Any]]:
    initialization = cfg.get("initialization", {}) if isinstance(cfg.get("initialization"), dict) else {}
    if not initialization or initialization.get("kind") == "official_sam2":
        return None, {"kind": "official_sam2"}
    if initialization.get("checkpoint"):
        if reproduction_protocol_enabled(cfg):
            raise ProtocolViolation("Strict reproduction runs must initialize through a validation selection lock.")
        checkpoint = resolve_project_path(initialization["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(f"Initialization checkpoint does not exist: {checkpoint}")
        return checkpoint, {"kind": "checkpoint", "checkpoint": artifact_record(checkpoint)}
    if initialization.get("selection_lock"):
        lock_path = resolve_project_path(initialization["selection_lock"])
        lock = read_mapping(lock_path)
        if lock.get("schema_version") != SELECTION_LOCK_SCHEMA_VERSION:
            raise ProtocolViolation(f"Invalid initialization selection lock: {lock_path}")
        lock_code = lock.get("code", {})
        if not isinstance(lock_code, dict) or not str(lock_code.get("revision", "")).strip() or bool(lock_code.get("dirty", False)):
            raise ProtocolViolation(f"Initialization lock has no clean code revision: {lock_path}")
        checkpoint_record = lock.get("selected_checkpoint", {})
        checkpoint = resolve_project_path(checkpoint_record.get("path", ""), base=lock_path.parent)
        if checkpoint_record.get("sha256") != sha256_file(checkpoint):
            raise ProtocolViolation(f"Initialization checkpoint hash mismatch in {lock_path}")
        expected_split = _protocol_split_path(cfg)
        checkpoint_lineage = read_lineage(checkpoint, required=True, expected_split_manifest=expected_split)
        if str((checkpoint_lineage or {}).get("code", {}).get("revision", "")) != str(lock_code.get("revision", "")):
            raise ProtocolViolation("Initialization lock code revision differs from checkpoint lineage")
        checkpoint_sam2 = (checkpoint_lineage or {}).get("sam2_code", {})
        if checkpoint_sam2 and str(checkpoint_sam2.get("revision", "")) != str(lock.get("sam2_revision", "")):
            raise ProtocolViolation("Initialization lock SAM2 revision differs from checkpoint lineage")
        expected_split_hash = sha256_file(expected_split)
        if str(lock.get("split_manifest", {}).get("sha256", "")) != expected_split_hash:
            raise ProtocolViolation("Initialization lock uses a different split manifest")
        if canonical_role(lock.get("selection_role")) != "validation" or bool(lock.get("test_accessed", False)):
            raise ProtocolViolation("Initialization lock is not a validation-only lock")
        protocol = reproduction_protocol(cfg)
        code_revision_migration = None
        if bool(protocol.get("require_clean_code", False)):
            current_code = git_record()
            _assert_clean_code_record(current_code, "Current benchmark code")
            if current_code.get("revision") != lock_code.get("revision"):
                migration = initialization.get("code_revision_migration")
                if not isinstance(migration, dict) or not bool(migration.get("enabled", False)):
                    raise ProtocolViolation("Benchmark code changed between chained stages")
                expected_source_revision = str(migration.get("expected_source_revision", "")).strip()
                if expected_source_revision != str(lock_code.get("revision", "")):
                    raise ProtocolViolation("Code migration source revision does not match initialization lock")
                expected_checkpoint_sha256 = str(migration.get("expected_source_checkpoint_sha256", "")).strip()
                if expected_checkpoint_sha256 != str(checkpoint_record.get("sha256", "")):
                    raise ProtocolViolation("Code migration source checkpoint hash does not match initialization lock")
                reason = str(migration.get("reason", "")).strip()
                if len(reason) < 20:
                    raise ProtocolViolation("Code migration requires a substantive reason")
                expected_target_revision = str(migration.get("expected_target_revision", "")).strip()
                if expected_target_revision and expected_target_revision != str(current_code.get("revision", "")):
                    raise ProtocolViolation("Code migration target revision does not match current benchmark code")
                code_revision_migration = {
                    "enabled": True,
                    "source_revision": expected_source_revision,
                    "target_revision": str(current_code.get("revision", "")),
                    "source_checkpoint_sha256": expected_checkpoint_sha256,
                    "reason": reason,
                }
        if bool(protocol.get("require_clean_sam2_repo", False)):
            _, current_sam2 = _source_dependency_records(cfg)
            if current_sam2 is None:
                raise ProtocolViolation("Current stage has no declared SAM2 dependency")
            _assert_clean_code_record(current_sam2, "Current SAM2 dependency")
            if str(current_sam2.get("revision", "")) != str(lock.get("sam2_revision", "")):
                raise ProtocolViolation("SAM2 dependency changed between chained stages")
        stage = str(protocol.get("stage", ""))
        expected_parent_stage = {"response_calibration": "joint_adaptation", "false_alarm_calibration": "response_calibration", "high_resolution_refinement": "false_alarm_calibration"}.get(stage)
        if expected_parent_stage and str(lock.get("stage", "")) != expected_parent_stage:
            raise ProtocolViolation(f"{stage} must initialize from a {expected_parent_stage} selection lock")
        if str(lock.get("ablation", "")) != str(protocol.get("ablation", "")):
            raise ProtocolViolation("Initialization lock belongs to a different ablation")
        if int(lock.get("seed", -1)) != int((cfg.get("train", {}) or {}).get("seed", 0)):
            raise ProtocolViolation("Initialization lock belongs to a different seed")
        expected_datasets = sorted(str(item) for item in (cfg.get("datasets", {}) or {}).get("train", []))
        if sorted(str(item) for item in lock.get("datasets", [])) != expected_datasets:
            raise ProtocolViolation("Initialization lock belongs to different datasets")
        return checkpoint, {
            "kind": "validation_selection_lock",
            "lock": artifact_record(lock_path),
            "checkpoint": checkpoint_record,
            "code_revision_migration": code_revision_migration,
        }
    checkpoint = resolve_project_path(initialization.get("checkpoint", ""))
    read_lineage(checkpoint, required=True, expected_split_manifest=_protocol_split_path(cfg))
    return checkpoint, {"kind": "checkpoint", "checkpoint": artifact_record(checkpoint)}


def validate_selection_lock(
    lock_path: str | Path,
    *,
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    threshold: float,
    config_path: str | Path | None = None,
    inference_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve_project_path(lock_path)
    lock = read_mapping(resolved)
    if lock.get("schema_version") != SELECTION_LOCK_SCHEMA_VERSION:
        raise ProtocolViolation(f"Unsupported selection lock: {resolved}")
    portable_hash_lock = bool(lock.get("anonymous_release_hash_lock", False))
    lock_code = lock.get("code", {})
    if not portable_hash_lock and (
        not isinstance(lock_code, dict)
        or not str(lock_code.get("revision", "")).strip()
        or bool(lock_code.get("dirty", False))
    ):
        raise ProtocolViolation("Selection lock has no clean code revision")
    if canonical_role(lock.get("selection_role")) != "validation" or bool(lock.get("test_accessed", False)):
        raise ProtocolViolation("Selection lock must originate from validation and declare no test access")
    checkpoint = resolve_project_path(checkpoint_path)
    selected = lock.get("selected_checkpoint", {})
    if resolve_project_path(selected.get("path", ""), base=resolved.parent) != checkpoint:
        raise ProtocolViolation("Requested checkpoint differs from validation-selected checkpoint")
    if str(selected.get("sha256", "")) != sha256_file(checkpoint):
        raise ProtocolViolation("Selected checkpoint hash mismatch")
    if not portable_hash_lock:
        checkpoint_lineage = read_lineage(checkpoint, required=True, expected_split_manifest=_protocol_split_path(cfg))
        training_code_revision = str(lock.get("training_code_revision") or lock_code.get("revision", ""))
        if not training_code_revision:
            raise ProtocolViolation("Selection lock has no training code revision")
        if str((checkpoint_lineage or {}).get("code", {}).get("revision", "")) != training_code_revision:
            raise ProtocolViolation("Selection lock code revision differs from checkpoint lineage")
        checkpoint_sam2 = (checkpoint_lineage or {}).get("sam2_code", {})
        if checkpoint_sam2 and str(checkpoint_sam2.get("revision", "")) != str(lock.get("sam2_revision", "")):
            raise ProtocolViolation("Selection lock SAM2 revision differs from checkpoint lineage")
    if abs(float(lock.get("threshold")) - float(threshold)) > 1e-12:
        raise ProtocolViolation("Requested test threshold differs from validation-selected threshold")
    declared_overrides = normalize_locked_inference_overrides(lock.get("inference_overrides"))
    requested_overrides = normalize_locked_inference_overrides(inference_overrides)
    if requested_overrides != declared_overrides:
        raise ProtocolViolation("Requested test inference overrides differ from validation-selected overrides")
    if str(lock.get("split_manifest", {}).get("sha256", "")) != sha256_file(_protocol_split_path(cfg)):
        raise ProtocolViolation("Selection lock split manifest differs from evaluation config")
    if config_path is not None:
        config = resolve_project_path(config_path)
        if str(lock.get("config", {}).get("sha256", "")) != sha256_file(config):
            raise ProtocolViolation("Selection lock was created for a different config")
    protocol = reproduction_protocol(cfg)
    if bool(protocol.get("require_clean_code", False)):
        current_code = git_record()
        _assert_clean_code_record(current_code, "Current benchmark code")
        if current_code.get("revision") != lock_code.get("revision"):
            raise ProtocolViolation("Selection lock code revision differs from current evaluation code")
    if bool(protocol.get("require_clean_sam2_repo", False)):
        _, current_sam2 = _source_dependency_records(cfg)
        if current_sam2 is None:
            raise ProtocolViolation("Evaluation config has no declared SAM2 dependency")
        _assert_clean_code_record(current_sam2, "Current SAM2 dependency")
        if str(current_sam2.get("revision", "")) != str(lock.get("sam2_revision", "")):
            raise ProtocolViolation("Selection lock SAM2 revision differs from current dependency")
    for field in ("stage", "ablation"):
        expected = protocol.get(field)
        if expected not in (None, "") and str(lock.get(field, "")) != str(expected):
            raise ProtocolViolation(f"Selection lock {field} differs from evaluation config")
    expected_seed = int((cfg.get("train", {}) or {}).get("seed", 0))
    if "seed" in lock and int(lock.get("seed", -1)) != expected_seed:
        raise ProtocolViolation("Selection lock seed differs from evaluation config")
    expected_datasets = sorted(str(item) for item in (cfg.get("datasets", {}) or {}).get("train", []))
    if expected_datasets and sorted(str(item) for item in lock.get("datasets", [])) != expected_datasets:
        raise ProtocolViolation("Selection lock datasets differ from evaluation config")
    return {
        "status": "passed",
        "lock_type": "anonymous_release_hash_lock" if portable_hash_lock else "revisioned_reproduction_lock",
        "lock": artifact_record(resolved),
        "selection": lock,
    }
