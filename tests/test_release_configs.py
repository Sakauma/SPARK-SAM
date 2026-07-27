import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from sparksam.config import load_app_config
from sparksam.data.adapters import _resolve_split_manifest_path
from sparksam.models.prompt_estimator import AutoPromptModelConfig, build_infrared_prompt_estimator
from sparksam.protocols.reproduction import artifact_record, validate_selection_lock
from scripts.select_operating_point import main as select_operating_point


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleaseConfigTests(unittest.TestCase):
    def test_every_yaml_has_a_mapping_root(self):
        paths = sorted((PROJECT_ROOT / "configs").rglob("*.yaml"))
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(path=path):
                self.assertIsInstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)

    def test_dataset_configs_resolve_environment_roots_and_split_manifest(self):
        environment = {
            "NUAA_SIRST_ROOT": "/datasets/nuaa-sirst",
            "NUDT_SIRST_ROOT": "/datasets/nudt-sirst",
            "IRSTD_1K_ROOT": "/datasets/irstd-1k",
            "ARTIFACT_ROOT": "artifacts",
        }
        expected_split = (PROJECT_ROOT / "splits" / "paper_split.json").resolve()
        with patch.dict(os.environ, environment, clear=False):
            for path in sorted((PROJECT_ROOT / "configs" / "datasets").glob("*.yaml")):
                with self.subTest(path=path):
                    config = load_app_config(path)
                    self.assertEqual(config.root, PROJECT_ROOT)
                    self.assertNotIn("${", str(config.dataset_root))
                    self.assertEqual(
                        _resolve_split_manifest_path(config, config.dataset.split_manifest),
                        expected_split,
                    )

    def test_prompt_estimator_configs_select_a_declared_architecture(self):
        for path in sorted((PROJECT_ROOT / "configs" / "prompt_estimator").glob("*.yaml")):
            with self.subTest(path=path):
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                model = build_infrared_prompt_estimator(AutoPromptModelConfig(**payload["model"]))
                self.assertEqual(model.model_name, "FeaturePyramidInfraredPromptEstimator")

    def test_anonymous_release_can_lock_validation_choice_without_git_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"anonymous-checkpoint")
            config = root / "evaluation.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "datasets": {"train": ["irstd_1k"]},
                        "split_policy": {"split_manifest": "splits/paper_split.json"},
                        "evaluation": {"threshold_sweep": [0.25, 0.5]},
                        "metadata": {"stage": "high_resolution_refinement"},
                        "train": {"seed": 42},
                    }
                ),
                encoding="utf-8",
            )
            report = root / "validation.json"
            report.write_text(
                json.dumps(
                    {
                        "role": "validation",
                        "teacher_loaded": False,
                        "cache_loaded": False,
                        "external_prompt_loaded": False,
                        "code": {"revision": "", "dirty": False},
                        "sam2_code": {"revision": "", "dirty": False},
                        "config": artifact_record(config),
                        "checkpoint": artifact_record(checkpoint),
                        "summary": [
                            {"threshold": 0.25, "global_IoU": 0.6, "FApxMP_global": 12.0, "nIoU": 0.5},
                            {"threshold": 0.5, "global_IoU": 0.7, "FApxMP_global": 8.0, "nIoU": 0.6},
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            lock = root / "selection_lock.json"
            self.assertEqual(
                select_operating_point(
                    [
                        "--config",
                        str(config),
                        "--reports",
                        str(report),
                        "--output-lock",
                        str(lock),
                    ]
                ),
                0,
            )
            payload = json.loads(lock.read_text(encoding="utf-8"))
            self.assertTrue(payload["anonymous_release_hash_lock"])
            self.assertEqual(payload["threshold"], 0.5)
            audit = validate_selection_lock(
                lock,
                cfg=yaml.safe_load(config.read_text(encoding="utf-8")),
                checkpoint_path=checkpoint,
                threshold=0.5,
                config_path=config,
            )
            self.assertEqual(audit["lock_type"], "anonymous_release_hash_lock")


if __name__ == "__main__":
    unittest.main()
