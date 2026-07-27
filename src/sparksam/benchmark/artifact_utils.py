from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file_optional(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False, "sha256": "", "size_bytes": ""}
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file_optional(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
    }


def models_by_alias(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = raw.get("models", raw.get("checkpoints", []))
    output: dict[str, dict[str, Any]] = {}
    for entry in models if isinstance(models, list) else []:
        if isinstance(entry, dict) and entry.get("alias"):
            output[str(entry["alias"])] = dict(entry)
    return output


def require_model(raw: dict[str, Any], alias: str) -> dict[str, Any]:
    models = models_by_alias(raw)
    if alias not in models:
        raise KeyError(f"Unknown model alias {alias!r}. Available aliases: {', '.join(sorted(models))}")
    return models[alias]


def resolve_checkpoint(model: dict[str, Any], paths: dict[str, Any]) -> Path:
    checkpoint = Path(str(model.get("ckpt", "")))
    if checkpoint.is_absolute():
        return checkpoint
    sam2_paths = paths.get("sam2", {}) if isinstance(paths.get("sam2"), dict) else {}
    checkpoint_root = sam2_paths.get("checkpoint_root")
    if checkpoint_root:
        return resolve_project_path(str(checkpoint_root)) / checkpoint.name
    return resolve_project_path(checkpoint)
