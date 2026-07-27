import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from sparksam.config import load_app_config
from sparksam.data import build_dataset_adapter
from sparksam.data import adapters as adapters_module


class GenericMaskAdapterTests(unittest.TestCase):
    def test_generic_binary_mask_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            image = np.zeros((8, 8), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 3:5] = 255
            Image.fromarray(image).save(images / "sample.png")
            Image.fromarray(mask).save(masks / "sample.png")

            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                  "dataset": {"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "binary", "class_map": {}},
                  "runtime": {"artifact_root": "artifacts", "reference_results_root": "reference_results", "output_name": "out", "device": "cpu", "num_workers": 0, "smoke_test": true, "max_samples": 0, "max_images": 0, "save_visuals": false, "seeds": [42]},
                  "evaluation": {"benchmark_version": "v1", "track": "image_prompted_segmentation", "protocol": "mask_supervised", "inference_mode": "box", "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": false}}
                }
                """,
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            adapter = build_dataset_adapter(config)
            loaded = adapter.load(config)
            self.assertEqual(loaded.manifest.sample_count, 1)
            self.assertEqual(loaded.manifest.image_count, 1)
            self.assertEqual(loaded.samples[0].category, "foreground")
            self.assertEqual(loaded.samples[0].frame_id, "sample")
            self.assertEqual(loaded.samples[0].track_id, None)
            self.assertTrue(loaded.samples[0].sample_id.startswith("sample"))
            self.assertIsNotNone(loaded.samples[0].bbox_loose)

    def test_generic_dataset_split_manifest_filters_by_frame_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            for name in ("sample_a", "sample_b"):
                image = np.zeros((8, 8), dtype=np.uint8)
                mask = np.zeros((8, 8), dtype=np.uint8)
                mask[2:6, 3:5] = 255
                Image.fromarray(image).save(images / f"{name}.png")
                Image.fromarray(mask).save(masks / f"{name}.png")
            split_path = root / "split.json"
            split_path.write_text(
                """
                {
                  "datasets": {
                    "generic": {
                      "dataset_id": "generic",
                      "splits": {
                        "train": ["sample_a"],
                        "val": [],
                        "test": ["sample_b"]
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )

            config_path = root / "config.json"
            config_path.write_text(
                f"""
                {{
                  "model": {{"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""}},
                  "dataset": {{"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "binary", "class_map": {{}}, "split_manifest": "{split_path.name}", "split_name": "test"}},
                  "runtime": {{"artifact_root": "artifacts", "reference_results_root": "reference_results", "output_name": "out", "device": "cpu", "num_workers": 0, "smoke_test": true, "max_samples": 0, "max_images": 0, "save_visuals": false, "seeds": [42]}},
                  "evaluation": {{"benchmark_version": "v1", "track": "image_prompted_segmentation", "protocol": "mask_supervised", "inference_mode": "box", "prompt_policy": {{"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": false}}}}
                }}
                """,
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            adapter = build_dataset_adapter(config)
            with patch.object(
                adapters_module,
                "_mask_to_numpy",
                wraps=adapters_module._mask_to_numpy,
            ) as mask_open, patch.object(
                adapters_module,
                "_image_size",
                wraps=adapters_module._image_size,
            ) as image_open:
                loaded = adapter.load(config)

            self.assertEqual([sample.frame_id for sample in loaded.samples], ["sample_b"])
            self.assertEqual(loaded.manifest.sample_count, 1)
            self.assertEqual(loaded.manifest.unfiltered_sample_count, 1)
            self.assertEqual(loaded.manifest.split_name, "test")
            self.assertEqual(loaded.manifest.physical_file_policy, "split_before_decode")
            self.assertEqual(loaded.manifest.discovered_frame_count, 2)
            self.assertEqual(loaded.manifest.requested_frame_count, 1)
            self.assertEqual(loaded.manifest.opened_frame_count, 1)
            self.assertEqual(loaded.manifest.opened_frame_ids, ["sample_b"])
            self.assertEqual(loaded.manifest.opened_image_count, 1)
            self.assertEqual(loaded.manifest.opened_mask_count, 1)
            self.assertTrue(loaded.manifest.opened_frame_ids_sha256)
            self.assertTrue(loaded.manifest.opened_image_paths_sha256)
            self.assertTrue(loaded.manifest.opened_mask_paths_sha256)
            self.assertEqual([call.args[0].stem for call in image_open.call_args_list], ["sample_b"])
            self.assertEqual([call.args[0].stem for call in mask_open.call_args_list], ["sample_b"])
            access = loaded.samples[0].metadata["physical_access"]
            self.assertEqual(access["policy"], "split_before_decode")
            self.assertEqual(access["split_name"], "test")
            self.assertEqual(access["frame_id"], "sample_b")
            self.assertEqual(access["split_manifest_sha256"], loaded.manifest.split_manifest_sha256)

    def test_generic_iter_samples_applies_split_before_opening_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            for name in ("train_frame", "val_frame", "test_frame"):
                image = np.zeros((8, 8), dtype=np.uint8)
                mask = np.zeros((8, 8), dtype=np.uint8)
                mask[2:6, 3:5] = 255
                Image.fromarray(image).save(images / f"{name}.png")
                Image.fromarray(mask).save(masks / f"{name}.png")
            split_path = root / "split.json"
            split_path.write_text(
                """
                {
                  "datasets": {
                    "generic": {
                      "dataset_id": "generic",
                      "splits": {
                        "train": ["train_frame"],
                        "val": ["val_frame"],
                        "test": ["test_frame"]
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                f"""
                {{
                  "model": {{"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""}},
                  "dataset": {{"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "binary", "class_map": {{}}, "split_manifest": "{split_path.name}", "split_name": "val"}},
                  "runtime": {{"artifact_root": "artifacts", "reference_results_root": "reference_results", "output_name": "out", "device": "cpu", "num_workers": 0, "smoke_test": true, "max_samples": 0, "max_images": 0, "save_visuals": false, "seeds": [42]}},
                  "evaluation": {{"benchmark_version": "v1", "track": "image_prompted_segmentation", "protocol": "mask_supervised", "inference_mode": "box", "prompt_policy": {{"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": false}}}}
                }}
                """,
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            adapter = build_dataset_adapter(config)
            with patch.object(
                adapters_module,
                "_mask_to_numpy",
                wraps=adapters_module._mask_to_numpy,
            ) as mask_open, patch.object(
                adapters_module,
                "_image_size",
                wraps=adapters_module._image_size,
            ) as image_open:
                samples = list(adapter.iter_samples(config))

            self.assertEqual([sample.frame_id for sample in samples], ["val_frame"])
            self.assertEqual([call.args[0].stem for call in image_open.call_args_list], ["val_frame"])
            self.assertEqual([call.args[0].stem for call in mask_open.call_args_list], ["val_frame"])
            ledger = adapter.physical_access_ledger()
            self.assertEqual(ledger["discovered_frame_count"], 3)
            self.assertEqual(ledger["opened_frame_ids"], ["val_frame"])

    def test_explicit_generic_adapter_overrides_multimodal_layout_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            (root / "img").mkdir()
            (root / "label").mkdir()
            image = np.zeros((8, 8), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 3:5] = 255
            Image.fromarray(image).save(images / "sample.png")
            Image.fromarray(mask).save(masks / "sample.png")

            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                  "dataset": {"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "binary", "class_map": {}},
                  "runtime": {"artifact_root": "artifacts", "reference_results_root": "reference_results", "output_name": "out", "device": "cpu", "num_workers": 0, "smoke_test": true, "max_samples": 0, "max_images": 0, "save_visuals": false, "seeds": [42]},
                  "evaluation": {"benchmark_version": "v1", "track": "image_prompted_segmentation", "protocol": "mask_supervised", "inference_mode": "box", "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": false}}
                }
                """,
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            adapter = build_dataset_adapter(config)
            loaded = adapter.load(config)

            self.assertEqual(adapter.adapter_name, "generic_image_mask")
            self.assertEqual(loaded.manifest.sample_count, 1)

    def test_generic_mask_skips_image_mask_size_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            image = np.zeros((4, 4), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            Image.fromarray(image).save(images / "sample.png")
            Image.fromarray(mask).save(masks / "sample.png")

            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                  "dataset": {"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "binary", "class_map": {}},
                  "runtime": {"artifact_root": "artifacts", "reference_results_root": "reference_results", "output_name": "out", "device": "cpu", "num_workers": 0, "smoke_test": true, "max_samples": 0, "max_images": 0, "save_visuals": false, "seeds": [42]},
                  "evaluation": {"benchmark_version": "v1", "track": "image_prompted_segmentation", "protocol": "mask_supervised", "inference_mode": "box", "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": false}}
                }
                """,
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = build_dataset_adapter(config).load(config)

            self.assertEqual(loaded.manifest.sample_count, 0)
            self.assertEqual(loaded.manifest.image_count, 0)
            self.assertIn("Skipped 1 image", loaded.manifest.notes)
            self.assertTrue(any("Skipping image/mask size mismatch" in str(item.message) for item in caught))

    def test_generic_mask_can_keep_empty_masks_for_negative_protocol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(images / "background.png")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(masks / "background.png")

            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                  "dataset": {"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "binary", "allow_empty_masks": true, "class_map": {}},
                  "runtime": {"artifact_root": "artifacts", "reference_results_root": "reference_results", "output_name": "out", "device": "cpu", "num_workers": 0, "smoke_test": true, "max_samples": 0, "max_images": 0, "save_visuals": false, "seeds": [42]},
                  "evaluation": {"benchmark_version": "v1", "track": "image_prompted_segmentation", "protocol": "mask_supervised", "inference_mode": "box", "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": false}}
                }
                """,
                encoding="utf-8",
            )
            loaded = build_dataset_adapter(load_app_config(config_path)).load(load_app_config(config_path))

            self.assertEqual(loaded.manifest.sample_count, 1)
            sample = loaded.samples[0]
            self.assertEqual(sample.target_scale, "empty")
            self.assertEqual(sample.category, "background")
            self.assertIsNone(sample.bbox_tight)
            self.assertEqual(float(sample.mask_array.sum()), 0.0)

    def test_generic_instance_masks_keep_image_frame_id_and_unique_sample_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            image = np.zeros((8, 8), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[1:3, 1:3] = 1
            mask[4:6, 4:6] = 2
            Image.fromarray(image).save(images / "frame_0001.png")
            Image.fromarray(mask).save(masks / "frame_0001.png")

            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                  "dataset": {"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "instance_id", "class_map": {"1": "car", "2": "plane"}},
                  "runtime": {"artifact_root": "artifacts", "reference_results_root": "reference_results", "output_name": "out", "device": "cpu", "num_workers": 0, "smoke_test": true, "max_samples": 0, "max_images": 0, "save_visuals": false, "seeds": [42]},
                  "evaluation": {"benchmark_version": "v1", "track": "image_prompted_segmentation", "protocol": "mask_supervised", "inference_mode": "box", "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": false}}
                }
                """,
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            adapter = build_dataset_adapter(config)
            loaded = adapter.load(config)

            self.assertEqual(loaded.manifest.sample_count, 2)
            self.assertEqual(loaded.manifest.image_count, 1)
            self.assertEqual({sample.frame_id for sample in loaded.samples}, {"frame_0001"})
            self.assertEqual(len({sample.sample_id for sample in loaded.samples}), 2)
            self.assertEqual({sample.track_id for sample in loaded.samples}, {"1", "2"})

    def test_generic_mask_matches_pixels_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(images / "Misc_12.png")
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[3:5, 2:4] = 255
            Image.fromarray(mask).save(masks / "Misc_12_pixels0.png")

            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                  "dataset": {"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "binary", "class_map": {}},
                  "runtime": {"artifact_root": "artifacts", "reference_results_root": "reference_results", "output_name": "out", "device": "cpu", "num_workers": 0, "smoke_test": true, "max_samples": 0, "max_images": 0, "save_visuals": false, "seeds": [42]},
                  "evaluation": {"benchmark_version": "v1", "track": "image_prompted_segmentation", "protocol": "mask_supervised", "inference_mode": "box", "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": false}}
                }
                """,
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            adapter = build_dataset_adapter(config)
            loaded = adapter.load(config)

            self.assertEqual(loaded.manifest.sample_count, 1)
            self.assertEqual(loaded.samples[0].frame_id, "Misc_12")
            self.assertEqual(loaded.samples[0].bbox_tight, [2.0, 3.0, 4.0, 5.0])

    def test_binary_mask_prefers_255_when_mixed_positive_values_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(images / "sample.png")
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[6, 6] = 64
            mask[1:3, 2:4] = 255
            Image.fromarray(mask).save(masks / "sample.png")

            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                  "dataset": {"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "binary", "class_map": {}},
                  "runtime": {"artifact_root": "artifacts", "reference_results_root": "reference_results", "output_name": "out", "device": "cpu", "num_workers": 0, "smoke_test": true, "max_samples": 0, "max_images": 0, "save_visuals": false, "seeds": [42]},
                  "evaluation": {"benchmark_version": "v1", "track": "image_prompted_segmentation", "protocol": "mask_supervised", "inference_mode": "box", "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": false}}
                }
                """,
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            adapter = build_dataset_adapter(config)
            loaded = adapter.load(config)

            self.assertEqual(loaded.manifest.sample_count, 1)
            self.assertEqual(loaded.samples[0].bbox_tight, [2.0, 1.0, 4.0, 3.0])
            self.assertEqual(loaded.samples[0].metadata["mask_decode"]["binary_rule"], "value_255_when_mixed_positive_values")


if __name__ == "__main__":
    unittest.main()
