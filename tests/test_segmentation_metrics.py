from __future__ import annotations

import unittest

import numpy as np

from sparksam.evaluation.segmentation_metrics import aggregate_binary_mask_rows, binary_mask_row


class SegmentationMetricsTests(unittest.TestCase):
    def test_global_f1_and_image_mean_dice_are_not_conflated(self) -> None:
        perfect = binary_mask_row(np.asarray([[1.0]], dtype=np.float32), np.asarray([[1.0]], dtype=np.float32), 0.5)
        missed = binary_mask_row(np.asarray([[0.0]], dtype=np.float32), np.asarray([[1.0]], dtype=np.float32), 0.5)
        summary = aggregate_binary_mask_rows([perfect, missed])
        self.assertAlmostEqual(float(summary["global_F1"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(summary["Dice_image_mean"]), 0.5)
        self.assertNotEqual(summary["global_F1"], summary["Dice_image_mean"])

    def test_target_absent_false_positive_rate_is_explicit(self) -> None:
        clean = binary_mask_row(np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32), 0.5)
        false_positive = binary_mask_row(np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32), 0.5)
        summary = aggregate_binary_mask_rows([clean, false_positive])
        self.assertEqual(summary["target_absent_samples"], 2)
        self.assertAlmostEqual(float(summary["target_absent_image_FPR"]), 0.5)


if __name__ == "__main__":
    unittest.main()
