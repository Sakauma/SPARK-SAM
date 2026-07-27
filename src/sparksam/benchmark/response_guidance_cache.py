from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_VERSION = "response_guidance_cache_v1"
DEFAULT_TEACHER_NAME = "Teacher-Full-SPARK-SAM"
REQUIRED_WORKFLOW_COMPONENTS = ("auto_prompt", "candidate_refinement", "rerank", "feedback", "calibration")
FORBIDDEN_TEACHER_MARKERS = (
    "teacher-sam2l-prompt_estimator",
    "teacher_sam2l_prompt_estimator",
    "sam2l-prompt_estimator",
    "sam2_large_prompt_estimator",
    "prompt_estimator_only",
    "prompt-estimator-only",
    "prompt_only_teacher",
    "prompt-only-teacher",
    "frozen learned prompt estimator only",
)


def resolve_project_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if base is not None:
        return (base / path).resolve()
    return (PROJECT_ROOT / path).resolve()


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Teacher cache manifest must be a mapping: {path}")
    return payload


def _candidate_manifest_paths(path_or_root: str | Path) -> list[Path]:
    path = resolve_project_path(path_or_root)
    if path.is_file() or path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return [path]
    return [path / "manifest.json", path / "manifest.yaml", path / "manifest.yml"]


def load_teacher_cache_manifest(path_or_root: str | Path) -> tuple[Path, dict[str, Any]]:
    for candidate in _candidate_manifest_paths(path_or_root):
        if candidate.exists():
            return candidate, _read_mapping(candidate)
    searched = ", ".join(str(item) for item in _candidate_manifest_paths(path_or_root))
    raise FileNotFoundError(f"Response Guidance teacher cache manifest not found. Searched: {searched}")


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        output: list[str] = []
        for key, item in value.items():
            if bool(item):
                output.append(str(key))
            if isinstance(item, (str, int, float)):
                output.append(str(item))
        return output
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        output: list[str] = []
        for key, item in value.items():
            output.append(str(key))
            output.extend(_flatten_strings(item))
        return output
    if isinstance(value, (list, tuple, set)):
        output = []
        for item in value:
            output.extend(_flatten_strings(item))
        return output
    if value is None:
        return []
    return [str(value)]


def _workflow_tokens(manifest: dict[str, Any]) -> set[str]:
    workflow = manifest.get("workflow", {})
    components = manifest.get("workflow_components", manifest.get("components", []))
    tokens = _flatten_strings(components)
    if isinstance(workflow, dict):
        tokens.extend(_flatten_strings(workflow.get("components", [])))
        tokens.extend(_flatten_strings(workflow.get("workflow_components", [])))
    normalized = " ".join(item.lower().replace("-", "_").replace("/", " ") for item in tokens)
    found: set[str] = set()
    if "auto_prompt" in normalized or "auto prompt" in normalized or "learned_auto" in normalized:
        found.add("auto_prompt")
    if "candidate" in normalized and ("refinement" in normalized or "decode" in normalized or "decoding" in normalized):
        found.add("candidate_refinement")
    if "top_k" in normalized or "top k" in normalized:
        found.add("candidate_refinement")
    if "rerank" in normalized or "re_rank" in normalized:
        found.add("rerank")
    if "feedback" in normalized or "sam2_feedback" in normalized:
        found.add("feedback")
    if "calibration" in normalized or "calibrated" in normalized:
        found.add("calibration")
    return found


def _explicit_reject_head_state(manifest: dict[str, Any]) -> bool | None:
    keys = (
        "target_absent_reject_head",
        "target_absent_reject",
        "reject_head",
        "target_absent_head_enabled",
    )
    for key in keys:
        if key in manifest and isinstance(manifest[key], bool):
            return bool(manifest[key])
    workflow = manifest.get("workflow", {})
    if isinstance(workflow, dict):
        for key in keys:
            if key in workflow and isinstance(workflow[key], bool):
                return bool(workflow[key])
    components = manifest.get("workflow_components", {})
    if isinstance(components, dict):
        for key in keys:
            if key in components and isinstance(components[key], bool):
                return bool(components[key])
    return None


def _manifest_values_for_keys(manifest: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    output: list[str] = []
    for key in keys:
        value = manifest.get(key)
        output.extend(_as_string_list(value))
    teacher = manifest.get("teacher", {})
    if isinstance(teacher, dict):
        for key in keys:
            output.extend(_as_string_list(teacher.get(key)))
    workflow = manifest.get("workflow", {})
    if isinstance(workflow, dict):
        for key in keys:
            output.extend(_as_string_list(workflow.get(key)))
    return output


def _contains_forbidden_teacher_marker(manifest: dict[str, Any]) -> list[str]:
    fields = _manifest_values_for_keys(
        manifest,
        (
            "teacher_name",
            "teacher_role",
            "teacher_alias",
            "teacher_protocol",
            "teacher_definition",
            "experiment",
            "protocol",
            "source_experiment",
        ),
    )
    text = "\n".join(fields).lower()
    return [marker for marker in FORBIDDEN_TEACHER_MARKERS if marker in text]


def _declared_values(manifest: dict[str, Any], keys: tuple[str, ...]) -> set[str]:
    values = {item for key in keys for item in _as_string_list(manifest.get(key))}
    data = manifest.get("data", {})
    if isinstance(data, dict):
        values.update(item for key in keys for item in _as_string_list(data.get(key)))
    cache = manifest.get("cache", {})
    if isinstance(cache, dict):
        values.update(item for key in keys for item in _as_string_list(cache.get(key)))
    return {str(item) for item in values if str(item)}


def audit_teacher_cache_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None = None,
    required_teacher_name: str = DEFAULT_TEACHER_NAME,
    expected_datasets: list[str] | None = None,
    expected_prompt_modes: list[str] | None = None,
    expected_splits: list[str] | None = None,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    schema_version = str(manifest.get("schema_version", ""))
    if schema_version and schema_version != DEFAULT_SCHEMA_VERSION:
        warnings.append({"reason": "non_default_schema_version", "schema_version": schema_version})

    teacher_names = _manifest_values_for_keys(
        manifest,
        ("teacher_name", "teacher_role", "teacher_definition", "teacher_protocol"),
    )
    teacher_text = " ".join(teacher_names)
    if required_teacher_name not in teacher_text:
        violations.append(
            {
                "reason": "teacher_name_not_full_sparksam",
                "required_teacher_name": required_teacher_name,
                "observed": teacher_names,
            }
        )
    forbidden = _contains_forbidden_teacher_marker(manifest)
    if forbidden:
        violations.append({"reason": "forbidden_retired_teacher_protocol_marker", "markers": forbidden})

    found_components = _workflow_tokens(manifest)
    missing_components = [name for name in REQUIRED_WORKFLOW_COMPONENTS if name not in found_components]
    if missing_components:
        violations.append(
            {
                "reason": "missing_full_workflow_components",
                "missing": missing_components,
                "required": list(REQUIRED_WORKFLOW_COMPONENTS),
                "found": sorted(found_components),
            }
        )

    reject_head_state = _explicit_reject_head_state(manifest)
    if reject_head_state is None:
        violations.append(
            {
                "reason": "target_absent_reject_head_not_explicitly_recorded",
                "required": "boolean true/false in manifest, workflow, or workflow_components",
            }
        )

    if expected_datasets:
        declared = _declared_values(manifest, ("datasets", "dataset_keys", "dataset_ids"))
        missing = [item for item in expected_datasets if item not in declared]
        if declared and missing:
            violations.append({"reason": "teacher_cache_dataset_coverage_missing", "missing": missing, "declared": sorted(declared)})
        elif not declared:
            warnings.append({"reason": "teacher_cache_datasets_not_declared", "expected": expected_datasets})

    if expected_prompt_modes:
        declared = _declared_values(manifest, ("prompt_modes", "modes"))
        missing = [item for item in expected_prompt_modes if item not in declared]
        if declared and missing:
            violations.append({"reason": "teacher_cache_prompt_mode_coverage_missing", "missing": missing, "declared": sorted(declared)})
        elif not declared:
            warnings.append({"reason": "teacher_cache_prompt_modes_not_declared", "expected": expected_prompt_modes})

    if expected_splits:
        declared = _declared_values(manifest, ("splits", "split_names", "role_split_names"))
        normalized_declared = {str(item).lower() for item in declared}
        missing = [item for item in expected_splits if str(item).lower() not in normalized_declared]
        if declared and missing:
            violations.append({"reason": "teacher_cache_split_coverage_missing", "missing": missing, "declared": sorted(declared)})
        elif not declared:
            warnings.append({"reason": "teacher_cache_splits_not_declared", "expected": expected_splits})

    has_records = isinstance(manifest.get("records"), list) and bool(manifest.get("records"))
    has_index = bool(manifest.get("records_index") or manifest.get("index_path"))
    if not has_records and not has_index:
        violations.append({"reason": "teacher_cache_has_no_records_or_records_index"})

    return {
        "status": "passed" if not violations else "failed",
        "schema_version": schema_version,
        "manifest_path": "" if manifest_path is None else str(manifest_path),
        "required_teacher_name": required_teacher_name,
        "found_workflow_components": sorted(found_components),
        "target_absent_reject_head": reject_head_state,
        "violations": violations,
        "warnings": warnings,
    }


def teacher_cache_key(dataset: str, sample_id: str, prompt_mode: str) -> tuple[str, str, str]:
    return (str(dataset), str(sample_id), str(prompt_mode))


def _json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [dict(item) for item in payload["records"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported Response Guidance teacher cache JSON index schema: {path}")


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_index_rows(manifest: dict[str, Any], manifest_path: Path, root: Path) -> list[dict[str, Any]]:
    records = manifest.get("records")
    if isinstance(records, list):
        return [dict(item) for item in records if isinstance(item, dict)]
    raw_index = manifest.get("records_index") or manifest.get("index_path")
    if not raw_index:
        return []
    index_path = resolve_project_path(str(raw_index), base=root)
    if not index_path.exists():
        index_path = resolve_project_path(str(raw_index), base=manifest_path.parent)
    if not index_path.exists():
        raise FileNotFoundError(f"Response Guidance teacher cache records_index does not exist: {raw_index}")
    suffix = index_path.suffix.lower()
    if suffix == ".csv":
        return _csv_rows(index_path)
    if suffix == ".jsonl":
        return _jsonl_rows(index_path)
    if suffix == ".json":
        return _json_rows(index_path)
    raise ValueError(f"Unsupported Response Guidance teacher cache records_index extension: {index_path}")


def _entry_dataset(entry: dict[str, Any]) -> str:
    return str(entry.get("dataset") or entry.get("dataset_key") or entry.get("dataset_id") or "")


def _entry_sample_id(entry: dict[str, Any]) -> str:
    return str(entry.get("sample_id") or entry.get("sample") or entry.get("image_id") or entry.get("id") or "")


def _entry_prompt_mode(entry: dict[str, Any]) -> str:
    return str(entry.get("prompt_mode") or entry.get("mode") or entry.get("prompt") or "")


def _sample_aliases(sample_id: str) -> set[str]:
    value = str(sample_id)
    path = Path(value)
    return {value, path.name, path.stem}


@dataclass
class TeacherCache:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    audit: dict[str, Any]
    records: list[dict[str, Any]]
    index: dict[tuple[str, str, str], dict[str, Any]]

    def entry_for(self, *, dataset: str, sample_id: str, prompt_mode: str) -> dict[str, Any] | None:
        for alias in _sample_aliases(sample_id):
            hit = self.index.get(teacher_cache_key(dataset, alias, prompt_mode))
            if hit is not None:
                return hit
        normalized_dataset = str(dataset).replace("-", "_").lower()
        for alias in _sample_aliases(sample_id):
            for (entry_dataset, entry_sample, entry_mode), entry in self.index.items():
                if entry_mode != prompt_mode:
                    continue
                if entry_sample not in _sample_aliases(alias):
                    continue
                if entry_dataset == dataset or entry_dataset.replace("-", "_").lower() == normalized_dataset:
                    return entry
        return None

    def record_path(self, entry: dict[str, Any]) -> Path:
        raw_path = entry.get("path") or entry.get("record_path") or entry.get("npz_path") or entry.get("cache_path")
        if not raw_path:
            dataset = _entry_dataset(entry)
            sample_id = _entry_sample_id(entry)
            prompt_mode = _entry_prompt_mode(entry)
            raw_path = Path(dataset) / prompt_mode / f"{Path(sample_id).stem}.npz"
        path = resolve_project_path(str(raw_path), base=self.root)
        if not path.exists():
            path = resolve_project_path(str(raw_path), base=self.manifest_path.parent)
        return path

    def load_record(self, entry: dict[str, Any]) -> dict[str, Any]:
        path = self.record_path(entry)
        if not path.exists():
            raise FileNotFoundError(f"Response Guidance teacher cache record does not exist: {path}")
        payload = load_teacher_cache_record(path)
        payload.setdefault("dataset", _entry_dataset(entry))
        payload.setdefault("sample_id", _entry_sample_id(entry))
        payload.setdefault("prompt_mode", _entry_prompt_mode(entry))
        payload.setdefault("record_path", str(path))
        return payload


def load_teacher_cache(
    cache_cfg: dict[str, Any],
    *,
    expected_datasets: list[str] | None = None,
    expected_prompt_modes: list[str] | None = None,
    expected_splits: list[str] | None = None,
) -> TeacherCache:
    root_value = cache_cfg.get("root") or cache_cfg.get("path") or cache_cfg.get("cache_root")
    manifest_value = cache_cfg.get("manifest") or cache_cfg.get("manifest_path") or root_value
    if not manifest_value:
        raise RuntimeError("Response Guidance requires response_guidance.teacher_cache.root or response_guidance.teacher_cache.manifest.")
    manifest_path, manifest = load_teacher_cache_manifest(str(manifest_value))
    root = resolve_project_path(str(root_value), base=manifest_path.parent) if root_value else manifest_path.parent
    audit_cfg = cache_cfg.get("audit", {}) if isinstance(cache_cfg.get("audit"), dict) else {}
    required_teacher_name = str(
        cache_cfg.get("required_teacher_name")
        or audit_cfg.get("required_teacher_name")
        or DEFAULT_TEACHER_NAME
    )
    audit = audit_teacher_cache_manifest(
        manifest,
        manifest_path=manifest_path,
        required_teacher_name=required_teacher_name,
        expected_datasets=expected_datasets,
        expected_prompt_modes=expected_prompt_modes,
        expected_splits=expected_splits,
    )
    if audit["status"] != "passed":
        raise RuntimeError(f"Response Guidance teacher cache audit failed: {json.dumps(audit['violations'], ensure_ascii=False)}")
    records = _load_index_rows(manifest, manifest_path, root)
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in records:
        dataset = _entry_dataset(entry)
        sample_id = _entry_sample_id(entry)
        prompt_mode = _entry_prompt_mode(entry)
        if not dataset or not sample_id or not prompt_mode:
            continue
        for alias in _sample_aliases(sample_id):
            index[teacher_cache_key(dataset, alias, prompt_mode)] = entry
    if not index:
        raise RuntimeError("Response Guidance teacher cache loaded zero indexed records.")
    return TeacherCache(root=root, manifest_path=manifest_path, manifest=manifest, audit=audit, records=records, index=index)


def _first_array(payload: dict[str, Any], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in payload:
            value = payload[name]
            if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
                value = value.item()
            return np.asarray(value)
    return None


def _float_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return array.astype(np.float32)


def _optional_scalar(payload: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in payload:
            value = payload[name]
            try:
                array = np.asarray(value, dtype=np.float32)
                return float(array.reshape(-1)[0])
            except Exception:
                return None
    return None


def _optional_list(payload: dict[str, Any], names: tuple[str, ...]) -> list[float] | None:
    for name in names:
        if name in payload:
            value = np.asarray(payload[name], dtype=np.float32).reshape(-1).tolist()
            return [float(item) for item in value]
    return None


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_json_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Response Guidance teacher cache JSON record must be a mapping: {path}")
    return payload


def load_teacher_cache_record(path: Path) -> dict[str, Any]:
    raw = _load_npz(path) if path.suffix.lower() == ".npz" else _load_json_record(path)
    teacher_logits = _float_array(
        _first_array(raw, ("teacher_mask_logits_full_resolution", "full_logits", "logits", "teacher_logits", "mask_logits"))
    )
    teacher_prob = _float_array(
        _first_array(raw, ("teacher_prob", "full_prob", "prob", "probability", "mask_probability", "teacher_probability"))
    )
    if teacher_logits is None and teacher_prob is None:
        raise ValueError(f"Response Guidance teacher cache record has no teacher logits/probability: {path}")
    if teacher_prob is None and teacher_logits is not None:
        teacher_prob = 1.0 / (1.0 + np.exp(-np.clip(teacher_logits, -32.0, 32.0)))
    if teacher_logits is None and teacher_prob is not None:
        clipped = np.clip(teacher_prob, 1e-6, 1.0 - 1e-6)
        teacher_logits = np.log(clipped / (1.0 - clipped)).astype(np.float32)
    gt_mask = _float_array(_first_array(raw, ("gt_mask", "mask", "target_mask", "ground_truth_mask")))
    low_res_logits = _float_array(_first_array(raw, ("teacher_low_res_logits", "low_res_logits", "lowres_logits")))
    binary_mask = _float_array(_first_array(raw, ("teacher_binary_mask", "binary_mask", "mask_binary")))
    record: dict[str, Any] = {
        "teacher_logits": teacher_logits,
        "teacher_prob": teacher_prob,
        "gt_mask": gt_mask,
        "low_res_logits": low_res_logits,
        "binary_mask": binary_mask,
        "prompt_point": _optional_list(raw, ("selected_prompt_point", "prompt_point", "point", "point_coords")),
        "prompt_box": _optional_list(raw, ("selected_prompt_box", "prompt_box", "box", "box_xyxy")),
        "teacher_iou": _optional_scalar(raw, ("teacher_iou", "iou", "iou_prediction", "mask_score", "score")),
        "rerank_score": _optional_scalar(raw, ("rerank_score", "calibrated_score")),
        "sam2_feedback": _optional_scalar(raw, ("sam2_feedback", "feedback_score")),
        "latency_ms": _optional_scalar(raw, ("latency_ms", "teacher_latency_ms", "workflow_latency_ms")),
        "selected_candidate_id": str(raw.get("selected_candidate_id", raw.get("candidate_id", ""))),
        "hard_bucket": str(raw.get("hard_bucket", raw.get("difficulty_bucket", ""))),
    }
    return record
