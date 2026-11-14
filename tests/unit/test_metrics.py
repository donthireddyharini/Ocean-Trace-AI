"""
OceanTrace AI — Unit Tests: Evaluation Metrics
Smart India Hackathon 2026 | SIH26143

Tests for ml/evaluation/metrics.py using synthetic ground-truth data
so tests run without any dataset or GPU.
"""

import sys
import os
import unittest

import numpy as np

# Allow running from project root or tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.evaluation.metrics import (
    compute_pixel_accuracy,
    compute_iou_per_class,
    compute_mean_iou,
    compute_precision_recall_f1,
    compute_dice,
    EvaluationReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect(shape=(64, 64), num_classes=5) -> tuple:
    """Perfect prediction == target."""
    target = np.random.randint(0, num_classes, shape, dtype=np.int32)
    pred = target.copy()
    return pred, target


def _all_wrong(shape=(64, 64), num_classes=2) -> tuple:
    """All pixels wrong (class 0 predicted as 1, class 1 as 0)."""
    target = np.zeros(shape, dtype=np.int32)
    pred = np.ones(shape, dtype=np.int32)
    return pred, target


def _checkerboard(shape=(64, 64), num_classes=5) -> tuple:
    """50% correct predictions."""
    target = np.zeros(shape, dtype=np.int32)
    pred = np.zeros(shape, dtype=np.int32)
    # Even rows correct, odd rows wrong (class 1)
    pred[1::2, :] = 1
    return pred, target


# ---------------------------------------------------------------------------
# pixel accuracy
# ---------------------------------------------------------------------------


class TestPixelAccuracy(unittest.TestCase):

    def test_perfect_accuracy(self):
        pred, target = _perfect()
        acc = compute_pixel_accuracy(pred, target)
        self.assertAlmostEqual(acc, 1.0, places=6)

    def test_all_wrong_binary(self):
        pred, target = _all_wrong()
        acc = compute_pixel_accuracy(pred, target)
        self.assertAlmostEqual(acc, 0.0, places=6)

    def test_half_correct(self):
        H, W = 64, 64
        target = np.zeros((H, W), dtype=np.int32)
        pred = np.zeros((H, W), dtype=np.int32)
        # Half the pixels are wrong
        pred[: H // 2, :] = 1
        acc = compute_pixel_accuracy(pred, target)
        self.assertAlmostEqual(acc, 0.5, places=6)

    def test_ignore_index_excluded(self):
        target = np.array([[0, 1, 255], [0, 255, 1]], dtype=np.int32)
        pred   = np.array([[0, 1, 0],   [1, 0,   1]], dtype=np.int32)
        # Valid pixels: (0,0)=correct, (0,1)=correct, (1,0)=wrong, (1,2)=correct → 3/4
        acc = compute_pixel_accuracy(pred, target, ignore_index=255)
        self.assertAlmostEqual(acc, 3 / 4, places=6)

    def test_all_ignored_returns_zero(self):
        target = np.full((10, 10), 255, dtype=np.int32)
        pred = np.zeros((10, 10), dtype=np.int32)
        acc = compute_pixel_accuracy(pred, target, ignore_index=255)
        self.assertEqual(acc, 0.0)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


class TestIoU(unittest.TestCase):

    def test_perfect_iou(self):
        pred, target = _perfect(num_classes=5)
        iou = compute_iou_per_class(pred, target, num_classes=5)
        # Only present classes; perfect overlap → 1.0
        present = [i for i in range(5) if np.isin(i, target)]
        for cls in present:
            self.assertAlmostEqual(iou[cls], 1.0, places=5)

    def test_binary_known_iou(self):
        # 2×2 image, target all 0, pred half 0 half 1
        target = np.array([[0, 0], [0, 0]], dtype=np.int32)
        pred   = np.array([[0, 0], [1, 1]], dtype=np.int32)
        iou = compute_iou_per_class(pred, target, num_classes=2)
        # Class 0: TP=2, FP=0, FN=2 → IoU = 2/(2+0+2) = 0.5
        self.assertAlmostEqual(iou[0], 0.5, places=6)
        # Class 1: TP=0, FP=2, FN=0 → IoU = 0/(0+2+0) = 0.0
        self.assertAlmostEqual(iou[1], 0.0, places=6)

    def test_absent_class_is_nan(self):
        target = np.zeros((10, 10), dtype=np.int32)
        pred = np.zeros((10, 10), dtype=np.int32)
        iou = compute_iou_per_class(pred, target, num_classes=5)
        # Classes 1-4 never appear → NaN
        for cls in range(1, 5):
            self.assertTrue(np.isnan(iou[cls]))

    def test_mean_iou_perfect(self):
        pred, target = _perfect(num_classes=3)
        miou = compute_mean_iou(pred, target, num_classes=3)
        self.assertAlmostEqual(miou, 1.0, places=5)

    def test_mean_iou_all_wrong_binary(self):
        pred, target = _all_wrong()
        miou = compute_mean_iou(pred, target, num_classes=2)
        # Class 0: IoU=0, Class 1: absent (pred=1 but target=0) → 0/(0+N) = 0
        self.assertAlmostEqual(miou, 0.0, places=5)


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------


class TestPrecisionRecallF1(unittest.TestCase):

    def test_perfect_prf(self):
        pred, target = _perfect(num_classes=3)
        for cls in range(3):
            if np.any(target == cls):
                p, r, f1 = compute_precision_recall_f1(pred, target, cls)
                self.assertAlmostEqual(p, 1.0, places=5)
                self.assertAlmostEqual(r, 1.0, places=5)
                self.assertAlmostEqual(f1, 1.0, places=5)

    def test_known_values(self):
        # 10 pixels: 5 oil (1), 5 clean (0)
        target = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
        pred   = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
        # Oil spill (class 1): TP=3, FP=3, FN=2
        p, r, f1 = compute_precision_recall_f1(pred, target, class_idx=1)
        self.assertAlmostEqual(p, 3 / 6, places=5)
        self.assertAlmostEqual(r, 3 / 5, places=5)
        expected_f1 = 2 * (3 / 6) * (3 / 5) / ((3 / 6) + (3 / 5))
        self.assertAlmostEqual(f1, expected_f1, places=5)

    def test_all_predicted_wrong_class(self):
        target = np.ones((20,), dtype=np.int32)
        pred   = np.zeros((20,), dtype=np.int32)
        p, r, f1 = compute_precision_recall_f1(pred, target, class_idx=1)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f1, 0.0)


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------


class TestDice(unittest.TestCase):

    def test_perfect_dice(self):
        pred, target = _perfect(num_classes=3)
        for cls in range(3):
            if np.any(target == cls):
                d = compute_dice(pred, target, class_idx=cls)
                self.assertAlmostEqual(d, 1.0, places=4)

    def test_zero_overlap(self):
        target = np.ones((10, 10), dtype=np.int32)
        pred   = np.zeros((10, 10), dtype=np.int32)
        d = compute_dice(pred, target, class_idx=1)
        # TP=0, pred_sum=0, target_sum=100 → (2*0+ε) / (0+100+ε) ≈ 0
        self.assertLess(d, 1e-3)

    def test_known_dice_value(self):
        # 1D: 4 oil pixels predicted, 4 true oil pixels, 2 overlap
        target = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
        pred   = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int32)
        # intersection=2, sum=4+4=8, dice=(2*2)/(8) = 0.5 (without smooth)
        d = compute_dice(pred, target, class_idx=1, smooth=0.0)
        self.assertAlmostEqual(d, 0.5, places=4)


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------


class TestEvaluationReport(unittest.TestCase):

    def _make_report(self, num_images=4, h=32, w=32, num_classes=5):
        report = EvaluationReport(
            num_classes=num_classes,
            class_names=["Clean Sea", "Oil Spill", "Look-Alike", "Ship", "Land"],
        )
        rng = np.random.default_rng(0)
        for _ in range(num_images):
            target = rng.integers(0, num_classes, (h, w), dtype=int).astype(np.int32)
            pred = rng.integers(0, num_classes, (h, w), dtype=int).astype(np.int32)
            report.update(pred, target)
        return report

    def test_generate_returns_dict(self):
        report = self._make_report()
        results = report.generate()
        self.assertIn("pixel_accuracy", results)
        self.assertIn("mean_iou", results)
        self.assertIn("per_class", results)
        self.assertIn("confusion_matrix", results)

    def test_pixel_accuracy_range(self):
        report = self._make_report()
        results = report.generate()
        acc = results["pixel_accuracy"]
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_mean_iou_range(self):
        report = self._make_report()
        results = report.generate()
        miou = results["mean_iou"]
        self.assertGreaterEqual(miou, 0.0)
        self.assertLessEqual(miou, 1.0)

    def test_perfect_report(self):
        report = EvaluationReport(num_classes=3)
        for _ in range(3):
            pred, target = _perfect(shape=(16, 16), num_classes=3)
            report.update(pred, target)
        results = report.generate()
        self.assertAlmostEqual(results["pixel_accuracy"], 1.0, places=5)
        self.assertAlmostEqual(results["mean_iou"], 1.0, places=5)

    def test_per_class_keys_exist(self):
        class_names = ["Clean Sea", "Oil Spill", "Look-Alike", "Ship", "Land"]
        report = EvaluationReport(num_classes=5, class_names=class_names)
        pred, target = _perfect(shape=(16, 16), num_classes=5)
        report.update(pred, target)
        results = report.generate()
        for name in class_names:
            self.assertIn(name, results["per_class"])

    def test_save_json(self, tmp_path=None):
        import tempfile, json
        report = self._make_report()
        report.generate()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            report.save_json(path)
            with open(path) as fh:
                data = json.load(fh)
            self.assertIn("pixel_accuracy", data)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
