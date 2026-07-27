#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sparksam.baselines.response_guidance import LearnedAutoPromptedSAM2  # noqa: E402
from sparksam.benchmark import artifact_utils  # noqa: E402
from sparksam.benchmark.response_guidance_cache import audit_teacher_cache_manifest  # noqa: E402
from sparksam.config import load_app_config  # noqa: E402
from sparksam.core.interfaces import InferenceMode  # noqa: E402
from sparksam.data import build_dataset_adapter  # noqa: E402
from sparksam.data.masks import sample_mask_array  # noqa: E402
from sparksam.models.sam2 import SAM2ModelAdapter, load_image_rgb  # noqa: E402
from sparksam.protocols.reproduction import (  # noqa: E402
    reproduction_protocol_enabled,
    git_record,
    lineage_path_for,
    read_lineage,
    resolve_project_path,
    sha256_file,
)
from scripts.training_common import _add_sam2_repo_to_path, _dataset_config_payload  # noqa: E402


MODE_TO_INFERENCE = {
    "point": InferenceMode.POINT,
    "box": InferenceMode.BOX,
    "box_point": InferenceMode.BOX_POINT,
}
APP_MODEL_KEYS = {
    "model_id",
    "cfg",
    "ckpt",
    "repo",
    "family",
    "prompt_mode",
    "deployment",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _split_csv(value: str | None, fallback: list[str]) -> list[str]:
    if not value:
        return list(fallback)
    return [item.strip() for item in value.split(",") if item.strip()]


def _section(raw: dict[str, Any]) -> dict[str, Any]:
    payload = raw.get("response_guidance", {})
    return payload if isinstance(payload, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "variants":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_config_with_source(path: Path) -> dict[str, Any]:
    """Merge a compact dataset override onto the shared anonymous source config."""
    override = _read_yaml(path)
    source_value = override.pop("source_config", None)
    if not source_value:
        return override
    source_path = Path(str(source_value))
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    return _deep_merge(_read_yaml(source_path), override)


def _dataset_list(section: dict[str, Any], key: str, fallback: list[str]) -> list[str]:
    value = section.get(key, fallback)
    return [str(item) for item in value] if isinstance(value, list) else list(fallback)


def _role_split_names(section: dict[str, Any]) -> dict[str, str]:
    value = section.get("role_split_names", {})
    merged = {"train": "train", "validation": "val", "calibration": "val", "test": "test", "eval_test": "test"}
    if isinstance(value, dict):
        merged.update({str(key): str(item) for key, item in value.items() if item is not None})
    return merged


def _workflow_settings(mode: str) -> dict[str, Any]:
    return {
        "rerank_candidates": True,
        "calibrate_box": mode in {"box", "box_point"},
        "use_negative_ring": mode == "box_point",
        "box_calibration_policy": "gated" if mode == "box_point" else "always",
    }


def _resolve_checkpoint(model: dict[str, Any], raw: dict[str, Any]) -> Path:
    paths = raw.get("paths", {}) if isinstance(raw.get("paths"), dict) else {}
    return artifact_utils.resolve_checkpoint(model, paths)


def _teacher_model(raw: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
    alias = str(section.get("teacher_model", "large"))
    model = dict(artifact_utils.require_model(raw, alias))
    model["repo"] = str(raw.get("paths", {}).get("sam2", {}).get("repo", model.get("repo", "")))
    model["ckpt"] = str(_resolve_checkpoint(model, raw))
    return model


def _app_model_payload(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model.items() if key in APP_MODEL_KEYS}


def _teacher_method(section: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    methods = raw.get("methods", {}) if isinstance(raw.get("methods"), dict) else {}
    teacher_method = methods.get("sparksam_full_teacher", {})
    method = dict(teacher_method.get("method", {}) if isinstance(teacher_method, dict) else {})
    override = section.get("teacher_method", {})
    if isinstance(override, dict):
        method = _deep_merge(method, override)
    for key in (
        "prompt_checkpoint",
        "prompt_top_k",
        "prompt_point_budget",
        "prompt_use_dog_filter",
        "prompt_use_fft_highpass",
        "prompt_fuse_filter_candidates",
        "prompt_filter_candidate_weight",
    ):
        if key in section:
            method[key] = section[key]
    prompt_checkpoint = method.get("prompt_checkpoint") or section.get("prompt_checkpoint")
    if prompt_checkpoint:
        method["prompt_checkpoint"] = str(artifact_utils.resolve_project_path(str(prompt_checkpoint)))
    method.setdefault("prompt_top_k", int(section.get("teacher_prompt_top_k", 10) or 10))
    method.setdefault("prompt_point_budget", 1)
    reranker = method.get("prompt_reranker", {})
    reranker = dict(reranker) if isinstance(reranker, dict) else {}
    method["prompt_reranker"] = reranker
    return method


def _app_config_payload(
    *,
    raw: dict[str, Any],
    section: dict[str, Any],
    dataset_key: str,
    role: str,
    mode: str,
    cache_root: Path,
    device: str,
) -> Path:
    dataset_loader_path = _dataset_config_payload(
        raw,
        dataset_key,
        {
            "artifact_root": str(cache_root),
            "split_policy": {
                "split_manifest": str(section.get("split_manifest", "")),
                "role_split_names": _role_split_names(section),
            },
            "smoke_test": False,
            "max_samples": 0,
        },
        cache_root / "generated_dataset_loaders",
        role=role,
    )
    loader_payload = _read_yaml(dataset_loader_path)
    model = _app_model_payload(_teacher_model(raw, section))
    method = _teacher_method(section, raw)
    method["name"] = "response_guidance_cache_generator"
    method["prompt_mode"] = mode
    workflow_settings = _workflow_settings(mode)
    method["response_guidance_rerank_candidates"] = bool(workflow_settings["rerank_candidates"])
    method["response_guidance_calibrate_box"] = bool(workflow_settings["calibrate_box"])
    method["response_guidance_use_negative_ring"] = bool(workflow_settings["use_negative_ring"])
    method["response_guidance_box_calibration_policy"] = str(workflow_settings["box_calibration_policy"])
    runtime_defaults = raw.get("runtime_defaults", {}) if isinstance(raw.get("runtime_defaults"), dict) else {}
    runtime = {
        "artifact_root": str(cache_root),
        "reference_results_root": "reference_results",
        "output_name": f"response_guidance_teacher_cache/{role}/{dataset_key}/{mode}",
        "device": device,
        "num_workers": 0,
        "smoke_test": False,
        "max_samples": 0,
        "max_images": 0,
        "save_visuals": False,
        "update_reference_results": False,
        "seeds": [42],
        "profile_eval": True,
        "image_batch_size": int(runtime_defaults.get("image_batch_size", 1) or 1),
        "reuse_image_embedding": bool(runtime_defaults.get("reuse_image_embedding", True)),
    }
    evaluation_defaults = raw.get("evaluation_defaults", {}) if isinstance(raw.get("evaluation_defaults"), dict) else {}
    interface_mode = MODE_TO_INFERENCE[mode].value
    evaluation = {
        **evaluation_defaults,
        "benchmark_version": "response_guidance_cache_v1",
        "track": "image_prompted_segmentation",
        "protocol": "response_guidance_cache",
        "inference_mode": interface_mode,
        "prompt_policy": {
            "name": "complete_response_guidance_selected_prompt",
            "prompt_type": interface_mode,
            "prompt_source": "synthesized",
            "prompt_budget": 2 if mode == "box_point" else 1,
            "notes": "Full SPARK-SAM workflow selected prompt cached for response-guided adaptation response distillation.",
        },
    }
    payload = {
        "model": model,
        "dataset": loader_payload["dataset"],
        "runtime": runtime,
        "evaluation": evaluation,
        "method": method,
    }
    path = cache_root / "generated_configs" / f"{dataset_key}_{role}_{mode}_teacher_cache.yaml"
    _write_yaml(path, payload)
    return path


def _load_samples(config_path: Path) -> list[Any]:
    app_config = load_app_config(config_path)
    return list(build_dataset_adapter(app_config).load(app_config).samples)


def _best_idx(result: dict[str, Any]) -> int:
    scores = np.asarray(result.get("scores", []), dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return 0
    return int(np.nanargmax(scores))


def _sigmoid(array: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(array.astype(np.float32), -32.0, 32.0)))).astype(np.float32)


def _prompt_arrays(prompt: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    point = prompt.get("point")
    box = prompt.get("box")
    point_array = np.asarray(point, dtype=np.float32).reshape(-1)[:2] if point is not None else None
    box_array = np.asarray(box, dtype=np.float32).reshape(-1)[:4] if box is not None else None
    return point_array, box_array


def _record_relative_path(dataset_key: str, role: str, mode: str, sample_id: str) -> Path:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(sample_id))
    return Path("records") / role / dataset_key / mode / f"{safe}.npz"


def _method_for_mode(app_config: Any, mode: str) -> LearnedAutoPromptedSAM2:
    adapter = SAM2ModelAdapter(app_config)
    method_cfg = getattr(app_config, "method", {})
    rerank_candidates = bool(method_cfg.get("response_guidance_rerank_candidates", True))
    calibrate_box = bool(method_cfg.get("response_guidance_calibrate_box", mode in {"box", "box_point"}))
    use_negative_ring = bool(method_cfg.get("response_guidance_use_negative_ring", mode == "box_point"))
    box_calibration_policy = str(method_cfg.get("response_guidance_box_calibration_policy", "gated" if mode == "box_point" else "always"))
    return LearnedAutoPromptedSAM2(
        adapter,
        app_config,
        prompt_mode=MODE_TO_INFERENCE[mode],
        use_negative_ring=use_negative_ring,
        rerank_candidates=rerank_candidates,
        calibrate_box=calibrate_box,
        box_calibration_policy=box_calibration_policy,
    )


def _cache_one(
    *,
    method: LearnedAutoPromptedSAM2,
    sample: Any,
    dataset_key: str,
    role: str,
    mode: str,
    cache_root: Path,
) -> dict[str, Any] | None:
    gt_mask = sample_mask_array(sample)
    if gt_mask is None:
        return None
    gt = (np.asarray(gt_mask, dtype=np.float32) > 0.5).astype(np.float32)
    image_rgb = load_image_rgb(sample.image_path)
    start = time.perf_counter()
    prediction = method.predict_sample(sample)
    prompt = dict(prediction.get("prompt", {}))
    kwargs = method._kwargs_from_prompt(prompt)
    result = method.adapter.predict_image(image_rgb, **kwargs)
    latency_ms = (time.perf_counter() - start) * 1000.0
    idx = _best_idx(result)
    full_logits = np.asarray(result["masks"][idx], dtype=np.float32)
    low_res = np.asarray(result.get("logits", result["masks"])[idx], dtype=np.float32)
    prob = _sigmoid(full_logits)
    point, box = _prompt_arrays(prompt)
    candidate_source = str(prompt.get("candidate_source", ""))
    frequency_prior_version = str(prompt.get("frequency_prior_version", ""))
    dog_score = prompt.get("dog_score")
    fft_score = prompt.get("fft_score")
    dog_score_value = float(dog_score) if dog_score is not None else float("nan")
    fft_score_value = float(fft_score) if fft_score is not None else float("nan")
    rel = _record_relative_path(dataset_key, role, mode, sample.sample_id)
    path = cache_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        teacher_mask_logits_full_resolution=full_logits.astype(np.float32),
        teacher_prob=prob.astype(np.float32),
        teacher_low_res_logits=low_res.astype(np.float32),
        teacher_binary_mask=(prob >= 0.5).astype(np.uint8),
        gt_mask=gt.astype(np.uint8),
        selected_prompt_point=np.asarray([] if point is None else point, dtype=np.float32),
        selected_prompt_box=np.asarray([] if box is None else box, dtype=np.float32),
        teacher_iou=np.asarray([float(np.asarray(result.get("scores", [0.0]), dtype=np.float32).reshape(-1)[idx])], dtype=np.float32),
        rerank_score=np.asarray([float(prompt.get("candidate_score", 0.0) or 0.0)], dtype=np.float32),
        sam2_feedback=np.asarray(
            [float(prompt.get("rerank_feedback_score", prompt.get("box_calibration_point_feedback_score", 0.0)) or 0.0)], dtype=np.float32
        ),
        latency_ms=np.asarray([latency_ms], dtype=np.float32),
        selected_candidate_id=np.asarray([str(prompt.get("rerank_selected_index", prompt.get("candidate_rank", "")))], dtype=object),
        candidate_source=np.asarray([candidate_source], dtype=object),
        frequency_prior_version=np.asarray([frequency_prior_version], dtype=object),
        dog_score=np.asarray([dog_score_value], dtype=np.float32),
        fft_score=np.asarray([fft_score_value], dtype=np.float32),
        hard_bucket=np.asarray([str(prompt.get("hard_bucket", ""))], dtype=object),
    )
    return {
        "dataset": dataset_key,
        "sample_id": str(sample.sample_id),
        "image_path": str(sample.image_path),
        "role": role,
        "split": role,
        "prompt_mode": mode,
        "path": str(rel),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "selected_candidate_id": str(prompt.get("rerank_selected_index", prompt.get("candidate_rank", ""))),
        "candidate_source": candidate_source,
        "frequency_prior_version": frequency_prior_version,
        "dog_score": None if dog_score is None else dog_score_value,
        "fft_score": None if fft_score is None else fft_score_value,
        "rerank_score": float(prompt.get("candidate_score", 0.0) or 0.0),
        "sam2_feedback": float(prompt.get("rerank_feedback_score", prompt.get("box_calibration_point_feedback_score", 0.0)) or 0.0),
        "latency_ms": float(latency_ms),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate response-guided adaptation immutable full-workflow response cache.")
    parser.add_argument("--config", type=Path, required=True, help="Dataset-specific response-guidance YAML.")
    parser.add_argument("--roles", default="train,validation")
    parser.add_argument("--datasets", help="Comma-separated dataset keys. Defaults to response_guidance train/validation datasets.")
    parser.add_argument("--prompt-modes", default="point,box,box_point")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--cache-root", type=Path, help="Override teacher cache root, useful for isolated smoke runs.")
    parser.add_argument("--manifest", type=Path, help="Override teacher cache manifest path.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    raw = _read_config_with_source(args.config)
    section = _section(raw)
    cache_cfg = section.get("teacher_cache", {}) if isinstance(section.get("teacher_cache"), dict) else {}
    cache_root_value = args.cache_root or cache_cfg.get("root", "artifacts/response_guidance")
    cache_root = artifact_utils.resolve_project_path(str(cache_root_value))
    manifest_value = args.manifest or cache_cfg.get("manifest", cache_root / "manifest.json")
    manifest_path = artifact_utils.resolve_project_path(str(manifest_value))
    if manifest_path.exists() and not args.overwrite:
        print(json.dumps({"status": "exists", "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
        return 0
    cache_root.mkdir(parents=True, exist_ok=True)
    sam2_repo = _add_sam2_repo_to_path(raw)
    roles = _split_csv(args.roles, ["train", "validation"])
    modes = _split_csv(args.prompt_modes, ["point", "box", "box_point"])
    datasets = _split_csv(
        args.datasets,
        sorted(set(_dataset_list(section, "train_datasets", []) + _dataset_list(section, "validation_datasets", []))),
    )
    strict = reproduction_protocol_enabled(raw)
    source_lineage = None
    split_manifest = None
    prompt_checkpoint = None
    generator_code = git_record()
    if strict and (not str(generator_code.get("revision", "")).strip() or bool(generator_code.get("dirty", False))):
        raise RuntimeError("Clean cache generation requires a clean, revisioned benchmark worktree.")
    if strict:
        if roles != ["train"]:
            raise RuntimeError(f"Clean response-teacher cache requires --roles train exactly, got {roles}")
        if args.overwrite:
            raise RuntimeError("Clean response-teacher caches are immutable; choose a new cache root.")
        prompt_checkpoint_value = _teacher_method(section, raw).get("prompt_checkpoint")
        if not prompt_checkpoint_value:
            raise RuntimeError("Clean response teacher requires an explicit train-only prompt checkpoint.")
        prompt_checkpoint = resolve_project_path(prompt_checkpoint_value)
        protocol = raw.get("reproduction_protocol", {})
        split_manifest = resolve_project_path(protocol.get("split_manifest") or section.get("split_manifest"))
        source_lineage = read_lineage(prompt_checkpoint, required=True, expected_split_manifest=split_manifest)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for role in roles:
        for dataset_key in datasets:
            for mode in modes:
                config_path = _app_config_payload(
                    raw=raw,
                    section=section,
                    dataset_key=dataset_key,
                    role=role,
                    mode=mode,
                    cache_root=cache_root,
                    device=args.device,
                )
                app_config = load_app_config(config_path)
                method = _method_for_mode(app_config, mode)
                samples = _load_samples(config_path)
                if args.max_samples:
                    samples = samples[: max(1, int(args.max_samples))]
                for index, sample in enumerate(samples, start=1):
                    try:
                        row = _cache_one(
                            method=method,
                            sample=sample,
                            dataset_key=dataset_key,
                            role=role,
                            mode=mode,
                            cache_root=cache_root,
                        )
                        if row is not None:
                            records.append(row)
                    except Exception as exc:
                        failures.append(
                            {
                                "dataset": dataset_key,
                                "role": role,
                                "prompt_mode": mode,
                                "sample_id": str(getattr(sample, "sample_id", "")),
                                "reason": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
                    if index == 1 or index % 25 == 0:
                        print(
                            json.dumps(
                                {
                                    "role": role,
                                    "dataset": dataset_key,
                                    "prompt_mode": mode,
                                    "index": index,
                                    "samples": len(samples),
                                    "records": len(records),
                                    "failures": len(failures),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                del method
    teacher = _teacher_model(raw, section)
    teacher_ckpt = Path(str(teacher.get("ckpt", "")))
    role_splits = _role_split_names(section)
    teacher_name = str(section.get("teacher_name", "Complete-Response-Guidance-SAM2") or "Complete-Response-Guidance-SAM2")
    teacher_role = str(section.get("teacher_role", teacher_name) or teacher_name)
    manifest = {
        "schema_version": "response_guidance_cache_v1",
        "generator_code": git_record(),
        "generator_code_at_start": generator_code,
        "sam2_code": git_record(Path(sam2_repo)),
        "experiment": "response-guided SAM2 adaptation",
        "teacher_name": teacher_name,
        "teacher_role": teacher_role,
        "teacher_model": teacher,
        "teacher_checkpoint": artifact_utils.file_record(teacher_ckpt),
        "source_checkpoint": str(prompt_checkpoint) if prompt_checkpoint is not None else "",
        "source_checkpoint_sha256": sha256_file(prompt_checkpoint) if prompt_checkpoint is not None else "",
        "source_checkpoint_lineage": str(lineage_path_for(prompt_checkpoint)) if source_lineage is not None else "",
        "workflow_components": [
            "learned_auto_prompt",
            "top_k_candidate_decoding",
            "candidate_refinement",
            "sam2_feedback",
            "rerank",
            "calibration",
        ],
        "workflow_configuration": "complete_response_guidance",
        "frequency_prior_version": str(section.get("frequency_prior_version", "")),
        "candidate_source": str(section.get("candidate_source", "")),
        "workflow": {
            "components": ["auto_prompt", "candidate_refinement", "rerank", "feedback", "calibration"],
            "target_absent_reject_head": False,
            "cache_generator": "scripts/build_response_guidance_cache.py",
        },
        "target_absent_reject_head": False,
        "datasets": datasets,
        "prompt_modes": modes,
        "split_names": sorted({role_splits.get(role, role) for role in roles}),
        "role_split_names": {role: role_splits.get(role, role) for role in roles},
        "training_supervision_only": True,
        "inference_forbidden": True,
        "split_manifest": str(split_manifest) if split_manifest is not None else str(section.get("split_manifest", "")),
        "split_manifest_sha256": sha256_file(split_manifest) if split_manifest is not None else "",
        "records": records,
        "record_count": len(records),
        "failures": failures,
        "failure_count": len(failures),
        "source_config": str(args.config.resolve()),
        "created_at": artifact_utils.utc_now(),
    }
    audit = audit_teacher_cache_manifest(
        manifest,
        manifest_path=manifest_path,
        required_teacher_name=str(cache_cfg.get("required_teacher_name") or teacher_name),
        expected_datasets=datasets,
        expected_prompt_modes=modes,
        expected_splits=sorted({role_splits.get(role, role) for role in roles}),
    )
    manifest["audit"] = audit
    _write_json(manifest_path, manifest)
    _write_json(cache_root / "generation_failures.json", failures)
    print(
        json.dumps(
            {"manifest": str(manifest_path), "records": len(records), "failures": len(failures), "audit": audit},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if audit["status"] == "passed" and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
