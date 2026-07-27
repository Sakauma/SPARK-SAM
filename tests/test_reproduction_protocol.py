from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from sparksam.protocols.reproduction import (
    LINEAGE_SCHEMA_VERSION,
    PROTOCOL_SCHEMA_VERSION,
    SELECTION_LOCK_SCHEMA_VERSION,
    ProtocolViolation,
    artifact_record,
    audit_cache_manifest,
    audit_spark_training_config,
    lineage_path_for,
    resolve_initialization_checkpoint,
    sha256_file,
    validate_selection_lock,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _split_manifest(path: Path, *, overlap: bool = False) -> None:
    _write_json(
        path,
        {
            "datasets": {
                "nuaa_sirst": {
                    "dataset_key": "nuaa_sirst",
                    "dataset_id": "NUAA-SIRST",
                    "splits": {
                        "train": ["a", "b"],
                        "val": ["b" if overlap else "c"],
                        "test": ["d"],
                    },
                }
            }
        },
    )


def _checkpoint_with_lineage(path: Path, split_path: Path) -> None:
    path.write_bytes(b"checkpoint")
    _write_json(
        lineage_path_for(path),
        {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "artifact_type": "unit_test_checkpoint",
            "artifact": artifact_record(path),
            "code": {"revision": "unit-test", "dirty": False, "status": []},
            "split_manifest": artifact_record(split_path),
            "datasets": ["nuaa_sirst"],
            "training_roles": ["train"],
            "selection_roles": ["validation"],
            "test_accessed": False,
        },
    )


class ReproductionProtocolTests(unittest.TestCase):
    def test_training_config_rejects_dirty_benchmark_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "joint_adaptation.yaml"
            config_path.write_text(
                yaml.safe_dump({"reproduction_protocol": {"strict": True, "require_clean_code": True}}),
                encoding="utf-8",
            )
            dirty_record = {"revision": "unit-test", "dirty": True, "status": [" M tracked.py"]}
            with patch("sparksam.protocols.reproduction.git_record", return_value=dirty_record):
                with self.assertRaises(ProtocolViolation):
                    audit_spark_training_config(config_path)

    def test_training_config_rejects_dirty_sam2_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sam2_repo = root / "sam2"
            sam2_repo.mkdir()
            source_path = root / "source.yaml"
            source_path.write_text(
                yaml.safe_dump({"paths": {"sam2": {"repo": str(sam2_repo)}}}),
                encoding="utf-8",
            )
            config_path = root / "joint_adaptation.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "source_config": str(source_path),
                        "reproduction_protocol": {"strict": True, "require_clean_sam2_repo": True},
                    }
                ),
                encoding="utf-8",
            )
            dirty_record = {"revision": "sam2-unit-test", "dirty": True, "status": [" M sam2.py"]}
            with patch("sparksam.protocols.reproduction.git_record", return_value=dirty_record):
                with self.assertRaises(ProtocolViolation):
                    audit_spark_training_config(config_path)

    def test_cache_frame_alias_matches_generic_adapter_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path)
            checkpoint = root / "teacher.pt"
            _checkpoint_with_lineage(checkpoint, split_path)
            cache_root = root / "cache"
            entries = []
            for frame_id in ("a", "b"):
                record_path = cache_root / f"{frame_id}.npz"
                record_path.parent.mkdir(parents=True, exist_ok=True)
                record_path.write_bytes(frame_id.encode("utf-8"))
                entries.append(
                    {
                        "dataset": "nuaa_sirst",
                        "sample_id": f"{frame_id}::foreground::generic_binary_mask",
                        "role": "train",
                        "path": record_path.name,
                        "sha256": sha256_file(record_path),
                        "size_bytes": record_path.stat().st_size,
                    }
                )
            manifest_path = cache_root / "manifest.json"
            _write_json(
                manifest_path,
                {
                    "generator_code": {"revision": "unit-test", "dirty": False, "status": []},
                    "generator_code_at_start": {"revision": "unit-test", "dirty": False, "status": []},
                    "training_supervision_only": True,
                    "inference_forbidden": True,
                    "split_manifest_sha256": sha256_file(split_path),
                    "source_checkpoint": str(checkpoint),
                    "source_checkpoint_sha256": sha256_file(checkpoint),
                    "source_checkpoint_lineage": str(lineage_path_for(checkpoint)),
                    "entries": entries,
                    "failure_count": 0,
                    "failures": [],
                    "audit": {"status": "passed"},
                },
            )
            report = audit_cache_manifest(
                manifest_path,
                split_manifest_path=split_path,
                dataset_keys=["nuaa_sirst"],
                verify_record_hashes=True,
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["record_hash_verification"], "content_sha256")

    def test_training_config_passes_without_teacher_for_segmentation_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path)
            official_checkpoint = root / "sam2_tiny.pt"
            official_checkpoint.write_bytes(b"official tiny")
            config_path = root / "joint_adaptation.yaml"
            config = {
                "datasets": {"train": ["nuaa_sirst"], "validation": ["nuaa_sirst"], "test": ["nuaa_sirst"]},
                "split_policy": {
                    "split_manifest": str(split_path),
                    "role_split_names": {"train": "train", "validation": "val", "test": "test"},
                    "strict_no_overlap": True,
                },
                "losses": {"gt_segmentation": {"weight": 1.0}},
                "train": {
                    "seed": 42,
                    "module_policy": {
                        "image_encoder": "train",
                        "prompt_encoder": "train",
                        "mask_decoder": "train",
                        "prompt_head": "train",
                        "local_prompt_projector": "train",
                    },
                },
                "student": {"checkpoint": {"path": str(official_checkpoint)}},
                "initialization": {"kind": "official_sam2"},
                "reproduction_protocol": {
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                    "strict": True,
                    "stage": "joint_adaptation",
                    "ablation": "seg_only",
                    "split_manifest": str(split_path),
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            report = audit_spark_training_config(config_path)
            self.assertEqual(report["status"], "passed")
            self.assertFalse(report["test_accessed"])

    def test_prompt_losses_can_train_highres_refinement_with_prompt_head_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path)
            official_checkpoint = root / "sam2_tiny.pt"
            official_checkpoint.write_bytes(b"official tiny")
            config_path = root / "joint_adaptation.yaml"
            config = {
                "datasets": {"train": ["nuaa_sirst"], "validation": ["nuaa_sirst"], "test": ["nuaa_sirst"]},
                "split_policy": {
                    "split_manifest": str(split_path),
                    "role_split_names": {"train": "train", "validation": "val", "test": "test"},
                    "strict_no_overlap": True,
                },
                "losses": {"prompt_objectness_distillation": {"weight": 1.0}},
                "train": {
                    "seed": 42,
                    "module_policy": {
                        "prompt_head": "frozen",
                        "highres_prompt_refinement_head": "train",
                    },
                },
                "student": {"checkpoint": {"path": str(official_checkpoint)}},
                "initialization": {"kind": "official_sam2"},
                "reproduction_protocol": {
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                    "strict": True,
                    "stage": "joint_adaptation",
                    "ablation": "highres_prompt_refinement",
                    "split_manifest": str(split_path),
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            report = audit_spark_training_config(config_path)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["module_policy"]["prompt_head"], "frozen")
            self.assertEqual(report["module_policy"]["highres_prompt_refinement_head"], "train")

            config["train"]["module_policy"]["highres_prompt_refinement_head"] = "frozen"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolViolation, "Prompt losses are enabled"):
                audit_spark_training_config(config_path)

    def test_split_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path, overlap=True)
            config_path = root / "joint_adaptation.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "datasets": {"train": ["nuaa_sirst"]},
                        "split_policy": {
                            "split_manifest": str(split_path),
                            "role_split_names": {"train": "train", "validation": "val", "test": "test"},
                            "strict_no_overlap": True,
                        },
                        "losses": {"gt_segmentation": {"weight": 1.0}},
                        "train": {"module_policy": {"mask_decoder": "train"}},
                        "reproduction_protocol": {"strict": True, "stage": "joint_adaptation", "split_manifest": str(split_path)},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ProtocolViolation):
                audit_spark_training_config(config_path)

    def test_cache_with_test_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path)
            checkpoint = root / "teacher.pt"
            _checkpoint_with_lineage(checkpoint, split_path)
            manifest_path = root / "cache" / "manifest.json"
            _write_json(
                manifest_path,
                {
                    "training_supervision_only": True,
                    "inference_forbidden": True,
                    "split_manifest_sha256": sha256_file(split_path),
                    "source_checkpoint": str(checkpoint),
                    "source_checkpoint_sha256": sha256_file(checkpoint),
                    "source_checkpoint_lineage": str(lineage_path_for(checkpoint)),
                    "entries": [
                        {"dataset": "nuaa_sirst", "sample_id": "a", "role": "train", "path": "a.npz"},
                        {"dataset": "nuaa_sirst", "sample_id": "b", "role": "train", "path": "b.npz"},
                        {"dataset": "nuaa_sirst", "sample_id": "d", "role": "test", "path": "d.npz"},
                    ],
                },
            )
            with self.assertRaises(ProtocolViolation):
                audit_cache_manifest(manifest_path, split_manifest_path=split_path, dataset_keys=["nuaa_sirst"])

    def test_cache_record_size_or_content_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path)
            checkpoint = root / "teacher.pt"
            _checkpoint_with_lineage(checkpoint, split_path)
            cache_root = root / "cache"
            entries = []
            record_paths = []
            for frame_id in ("a", "b"):
                record_path = cache_root / f"{frame_id}.npz"
                record_path.parent.mkdir(parents=True, exist_ok=True)
                record_path.write_bytes(frame_id.encode("utf-8"))
                record_paths.append(record_path)
                entries.append(
                    {
                        "dataset": "nuaa_sirst",
                        "sample_id": frame_id,
                        "role": "train",
                        "path": record_path.name,
                        "sha256": sha256_file(record_path),
                        "size_bytes": record_path.stat().st_size,
                    }
                )
            manifest_path = cache_root / "manifest.json"
            _write_json(
                manifest_path,
                {
                    "generator_code": {"revision": "unit-test", "dirty": False, "status": []},
                    "generator_code_at_start": {"revision": "unit-test", "dirty": False, "status": []},
                    "training_supervision_only": True,
                    "inference_forbidden": True,
                    "split_manifest_sha256": sha256_file(split_path),
                    "source_checkpoint": str(checkpoint),
                    "source_checkpoint_sha256": sha256_file(checkpoint),
                    "source_checkpoint_lineage": str(lineage_path_for(checkpoint)),
                    "entries": entries,
                    "failure_count": 0,
                    "failures": [],
                    "audit": {"status": "passed"},
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"][0]["size_bytes"] += 1
            _write_json(manifest_path, manifest)
            with self.assertRaises(ProtocolViolation):
                audit_cache_manifest(
                    manifest_path,
                    split_manifest_path=split_path,
                    dataset_keys=["nuaa_sirst"],
                )
            manifest["entries"][0]["size_bytes"] = record_paths[0].stat().st_size
            _write_json(manifest_path, manifest)
            record_paths[0].write_bytes(b"z")
            with self.assertRaises(ProtocolViolation):
                audit_cache_manifest(
                    manifest_path,
                    split_manifest_path=split_path,
                    dataset_keys=["nuaa_sirst"],
                    verify_record_hashes=True,
                )

    def test_selection_lock_requires_exact_checkpoint_and_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path)
            checkpoint = root / "student.pt"
            _checkpoint_with_lineage(checkpoint, split_path)
            config = {"reproduction_protocol": {"strict": True, "split_manifest": str(split_path)}}
            lock_path = root / "selection_lock.json"
            _write_json(
                lock_path,
                {
                    "schema_version": SELECTION_LOCK_SCHEMA_VERSION,
                    "code": {"revision": "unit-test", "dirty": False, "status": []},
                    "selection_role": "validation",
                    "test_accessed": False,
                    "split_manifest": artifact_record(split_path),
                    "selected_checkpoint": artifact_record(checkpoint),
                    "threshold": 0.35,
                },
            )
            report = validate_selection_lock(lock_path, cfg=config, checkpoint_path=checkpoint, threshold=0.35)
            self.assertEqual(report["status"], "passed")
            with self.assertRaises(ProtocolViolation):
                validate_selection_lock(lock_path, cfg=config, checkpoint_path=checkpoint, threshold=0.5)

    def test_selection_lock_seals_training_revision_and_inference_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path)
            checkpoint = root / "student.pt"
            _checkpoint_with_lineage(checkpoint, split_path)
            config = {"reproduction_protocol": {"strict": True, "split_manifest": str(split_path)}}
            lock_path = root / "selection_lock.json"
            _write_json(
                lock_path,
                {
                    "schema_version": SELECTION_LOCK_SCHEMA_VERSION,
                    "code": {"revision": "unit-test-evaluation", "dirty": False, "status": []},
                    "training_code_revision": "unit-test",
                    "validation_code_revision": "unit-test-validation",
                    "selection_role": "validation",
                    "test_accessed": False,
                    "split_manifest": artifact_record(split_path),
                    "selected_checkpoint": artifact_record(checkpoint),
                    "threshold": 0.22,
                    "inference_overrides": {"prompt_box_scale": 0.82},
                },
            )
            report = validate_selection_lock(
                lock_path,
                cfg=config,
                checkpoint_path=checkpoint,
                threshold=0.22,
                inference_overrides={"prompt_box_scale": 0.82},
            )
            self.assertEqual(report["status"], "passed")
            with self.assertRaises(ProtocolViolation):
                validate_selection_lock(
                    lock_path,
                    cfg=config,
                    checkpoint_path=checkpoint,
                    threshold=0.22,
                )
            with self.assertRaises(ProtocolViolation):
                validate_selection_lock(
                    lock_path,
                    cfg=config,
                    checkpoint_path=checkpoint,
                    threshold=0.22,
                    inference_overrides={"prompt_box_scale": 0.84},
                )

    def test_selection_lock_rejects_sam2_revision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path)
            checkpoint = root / "student.pt"
            _checkpoint_with_lineage(checkpoint, split_path)
            lineage_path = lineage_path_for(checkpoint)
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            lineage["sam2_code"] = {"revision": "sam2-a", "dirty": False, "status": []}
            _write_json(lineage_path, lineage)
            config = {"reproduction_protocol": {"strict": True, "split_manifest": str(split_path)}}
            lock_path = root / "selection_lock.json"
            _write_json(
                lock_path,
                {
                    "schema_version": SELECTION_LOCK_SCHEMA_VERSION,
                    "code": {"revision": "unit-test", "dirty": False, "status": []},
                    "sam2_revision": "sam2-b",
                    "selection_role": "validation",
                    "test_accessed": False,
                    "split_manifest": artifact_record(split_path),
                    "selected_checkpoint": artifact_record(checkpoint),
                    "threshold": 0.35,
                },
            )
            with self.assertRaises(ProtocolViolation):
                validate_selection_lock(lock_path, cfg=config, checkpoint_path=checkpoint, threshold=0.35)

    def test_training_phase_initialization_rejects_cross_ablation_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path)
            checkpoint = root / "student.pt"
            _checkpoint_with_lineage(checkpoint, split_path)
            lock_path = root / "selection_lock.json"
            _write_json(
                lock_path,
                {
                    "schema_version": SELECTION_LOCK_SCHEMA_VERSION,
                    "code": {"revision": "unit-test", "dirty": False, "status": []},
                    "selection_role": "validation",
                    "test_accessed": False,
                    "split_manifest": artifact_record(split_path),
                    "selected_checkpoint": artifact_record(checkpoint),
                    "threshold": 0.35,
                    "stage": "joint_adaptation",
                    "ablation": "full",
                    "seed": 42,
                    "datasets": ["nuaa_sirst"],
                },
            )
            config = {
                "datasets": {"train": ["nuaa_sirst"]},
                "train": {"seed": 42},
                "initialization": {"kind": "validation_selection_lock", "selection_lock": str(lock_path)},
                "reproduction_protocol": {
                    "strict": True,
                    "stage": "response_calibration",
                    "ablation": "minus_fa",
                    "split_manifest": str(split_path),
                },
            }
            with self.assertRaises(ProtocolViolation):
                resolve_initialization_checkpoint(config, config_path=root / "response_calibration.yaml")

    def test_high_resolution_refinement_initializes_only_from_false_alarm_calibration_selection_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path)
            checkpoint = root / "student.pt"
            _checkpoint_with_lineage(checkpoint, split_path)
            lock_path = root / "selection_lock.json"
            lock = {
                "schema_version": SELECTION_LOCK_SCHEMA_VERSION,
                "code": {"revision": "unit-test", "dirty": False, "status": []},
                "selection_role": "validation",
                "test_accessed": False,
                "split_manifest": artifact_record(split_path),
                "selected_checkpoint": artifact_record(checkpoint),
                "threshold": 0.35,
                "stage": "false_alarm_calibration",
                "ablation": "full",
                "seed": 42,
                "datasets": ["nuaa_sirst"],
            }
            _write_json(lock_path, lock)
            config = {
                "datasets": {"train": ["nuaa_sirst"]},
                "train": {"seed": 42},
                "initialization": {"kind": "validation_selection_lock", "selection_lock": str(lock_path)},
                "reproduction_protocol": {
                    "strict": True,
                    "stage": "high_resolution_refinement",
                    "ablation": "full",
                    "split_manifest": str(split_path),
                },
            }

            resolved, _ = resolve_initialization_checkpoint(config, config_path=root / "high_resolution_refinement.yaml")
            self.assertEqual(resolved, checkpoint)

            lock["stage"] = "response_calibration"
            _write_json(lock_path, lock)
            with self.assertRaisesRegex(ProtocolViolation, "high_resolution_refinement must initialize from a false_alarm_calibration"):
                resolve_initialization_checkpoint(config, config_path=root / "high_resolution_refinement.yaml")

    def test_explicit_code_revision_migration_requires_pinned_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_path = root / "split.json"
            _split_manifest(split_path)
            checkpoint = root / "student.pt"
            _checkpoint_with_lineage(checkpoint, split_path)
            lock_path = root / "selection_lock.json"
            _write_json(
                lock_path,
                {
                    "schema_version": SELECTION_LOCK_SCHEMA_VERSION,
                    "code": {"revision": "unit-test", "dirty": False, "status": []},
                    "selection_role": "validation",
                    "test_accessed": False,
                    "split_manifest": artifact_record(split_path),
                    "selected_checkpoint": artifact_record(checkpoint),
                    "threshold": 0.35,
                    "stage": "response_calibration",
                    "ablation": "full",
                    "seed": 42,
                    "datasets": ["nuaa_sirst"],
                },
            )
            config = {
                "datasets": {"train": ["nuaa_sirst"]},
                "train": {"seed": 42},
                "initialization": {
                    "kind": "validation_selection_lock",
                    "selection_lock": str(lock_path),
                    "code_revision_migration": {
                        "enabled": True,
                        "expected_source_revision": "unit-test",
                        "expected_source_checkpoint_sha256": sha256_file(checkpoint),
                        "reason": "Introduce an independently parameterized union-box prediction head.",
                    },
                },
                "reproduction_protocol": {
                    "strict": True,
                    "require_clean_code": True,
                    "stage": "box_repair",
                    "ablation": "full",
                    "split_manifest": str(split_path),
                },
            }
            current = {"revision": "new-revision", "dirty": False, "status": []}
            with patch("sparksam.protocols.reproduction.git_record", return_value=current):
                resolved, summary = resolve_initialization_checkpoint(config, config_path=root / "box_repair.yaml")
            self.assertEqual(resolved, checkpoint)
            self.assertEqual(summary["code_revision_migration"]["source_revision"], "unit-test")
            self.assertEqual(summary["code_revision_migration"]["target_revision"], "new-revision")

            config["initialization"]["code_revision_migration"]["expected_source_checkpoint_sha256"] = "0" * 64
            with patch("sparksam.protocols.reproduction.git_record", return_value=current):
                with self.assertRaises(ProtocolViolation):
                    resolve_initialization_checkpoint(config, config_path=root / "box_repair.yaml")


if __name__ == "__main__":
    unittest.main()
