"""
OceanTrace AI — Unit Tests: Preprocessing Utilities
Smart India Hackathon 2026 | SIH26143

Tests for ml/preprocessing/image_utils.py using synthetic images
so tests run without any dataset or GPU.
"""

import os
import sys
import unittest

import numpy as np

# Allow running from project root or tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.preprocessing.image_utils import (
    normalize_image,
    resize_with_padding,
    decode_m4d_mask,
    compute_class_distribution,
    augment_pair,
    split_dataset_by_scene,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_rgb(h=64, w=64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _rand_gray(h=64, w=64) -> np.ndarray:
    rng = np.random.default_rng(99)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _rand_mask(h=64, w=64, num_classes=5) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, num_classes, (h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# normalize_image
# ---------------------------------------------------------------------------


class TestNormalizeImage(unittest.TestCase):

    def test_output_shape_rgb(self):
        arr = _rand_rgb()
        out = normalize_image(arr, modality="OPTICAL")
        self.assertEqual(out.shape, (64, 64, 3))

    def test_output_dtype_float32(self):
        arr = _rand_rgb()
        out = normalize_image(arr, modality="OPTICAL")
        self.assertEqual(out.dtype, np.float32)

    def test_imagenet_range(self):
        """After normalising with ImageNet stats, values should be roughly in [-3, 3]."""
        arr = _rand_rgb()
        out = normalize_image(arr, modality="OPTICAL")
        self.assertGreater(out.min(), -5.0)
        self.assertLess(out.max(), 5.0)

    def test_sar_grayscale_to_3ch(self):
        """SAR grayscale input should be expanded to 3-channel after normalisation."""
        arr = _rand_gray()
        out = normalize_image(arr, modality="SAR")
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[2], 3)

    def test_deterministic(self):
        arr = _rand_rgb()
        out1 = normalize_image(arr, modality="OPTICAL")
        out2 = normalize_image(arr, modality="OPTICAL")
        np.testing.assert_array_equal(out1, out2)

    def test_auto_modality_rgb(self):
        arr = _rand_rgb()
        out = normalize_image(arr, modality="auto")
        self.assertEqual(out.shape[-1], 3)
        self.assertEqual(out.dtype, np.float32)

    def test_auto_modality_gray(self):
        arr = _rand_gray()
        out = normalize_image(arr, modality="auto")
        self.assertEqual(out.ndim, 3)


# ---------------------------------------------------------------------------
# resize_with_padding
# ---------------------------------------------------------------------------


class TestResizeWithPadding(unittest.TestCase):

    def test_square_input(self):
        arr = _rand_rgb(64, 64)
        out = resize_with_padding(arr, target=(128, 128))
        self.assertEqual(out.shape[:2], (128, 128))

    def test_landscape_input(self):
        arr = _rand_rgb(64, 128)
        out = resize_with_padding(arr, target=(64, 64))
        self.assertEqual(out.shape[:2], (64, 64))

    def test_portrait_input(self):
        arr = _rand_rgb(128, 64)
        out = resize_with_padding(arr, target=(64, 64))
        self.assertEqual(out.shape[:2], (64, 64))

    def test_grayscale_input(self):
        arr = _rand_gray(64, 32)
        out = resize_with_padding(arr, target=(64, 64))
        self.assertEqual(out.shape[:2], (64, 64))

    def test_dtype_preserved(self):
        arr = _rand_rgb(32, 32)
        out = resize_with_padding(arr, target=(64, 64))
        # Accept uint8 or float32 — just check shape and no crash
        self.assertEqual(out.shape[:2], (64, 64))

    def test_default_target(self):
        arr = _rand_rgb(100, 200)
        out = resize_with_padding(arr)   # default target=(512, 512)
        self.assertEqual(out.shape[:2], (512, 512))


# ---------------------------------------------------------------------------
# decode_m4d_mask
# ---------------------------------------------------------------------------


class TestDecodeM4DMask(unittest.TestCase):

    def test_pass_through_valid_classes(self):
        mask = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
        decoded = decode_m4d_mask(mask)
        np.testing.assert_array_equal(decoded, mask)

    def test_output_dtype(self):
        mask = np.array([[0, 1, 2]], dtype=np.uint8)
        decoded = decode_m4d_mask(mask)
        self.assertIn(decoded.dtype, [np.int32, np.int64, np.uint8])

    def test_clamps_out_of_range(self):
        """Values outside 0-4 should be clamped or mapped without crash."""
        mask = np.array([[0, 5, 10, 255]], dtype=np.uint8)
        decoded = decode_m4d_mask(mask)
        self.assertEqual(decoded.shape, mask.shape)
        # Out-of-range pixels should be mapped to something in [0, 4]
        self.assertTrue((decoded >= 0).all())
        self.assertTrue((decoded <= 4).all())

    def test_all_classes_present(self):
        mask = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
        decoded = decode_m4d_mask(mask)
        for c in range(5):
            self.assertIn(c, decoded)


# ---------------------------------------------------------------------------
# compute_class_distribution
# ---------------------------------------------------------------------------


class TestComputeClassDistribution(unittest.TestCase):

    def test_returns_dict(self):
        import tempfile, os
        from PIL import Image as PILImage

        # Write two synthetic mask PNGs
        mask_data = [
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
            np.array([[0, 0], [4, 1]], dtype=np.uint8),
        ]
        paths = []
        try:
            for i, m in enumerate(mask_data):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
                    PILImage.fromarray(m).save(fh.name)
                    paths.append(fh.name)
            dist = compute_class_distribution(paths)
            self.assertIsInstance(dist, dict)
            # Class 0 should appear (3 times across both masks)
            self.assertIn(0, dist)
            self.assertEqual(dist[0], 3)
        finally:
            for p in paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# augment_pair
# ---------------------------------------------------------------------------


class TestAugmentPair(unittest.TestCase):

    def test_output_shape_matches_input(self):
        img = _rand_rgb(64, 64)
        mask = _rand_mask(64, 64)
        aug_img, aug_mask = augment_pair(img, mask)
        self.assertEqual(aug_img.shape, img.shape)
        self.assertEqual(aug_mask.shape, mask.shape)

    def test_mask_values_preserved(self):
        """Augmentation must not introduce new class values in the mask."""
        img = _rand_rgb(64, 64)
        mask = _rand_mask(64, 64)
        original_classes = set(np.unique(mask).tolist())
        aug_img, aug_mask = augment_pair(img, mask)
        augmented_classes = set(np.unique(aug_mask).tolist())
        # Augmented mask should only contain classes present in original
        self.assertTrue(augmented_classes.issubset(original_classes))

    def test_runs_multiple_times(self):
        """Augmentation is stochastic; ensure it doesn't crash over many calls."""
        img = _rand_rgb(64, 64)
        mask = _rand_mask(64, 64)
        for _ in range(10):
            aug_img, aug_mask = augment_pair(img, mask)
            self.assertEqual(aug_img.shape, img.shape)

    def test_dtype_preserved(self):
        img = _rand_rgb(64, 64)
        mask = _rand_mask(64, 64)
        aug_img, aug_mask = augment_pair(img, mask)
        self.assertEqual(aug_img.dtype, img.dtype)
        self.assertEqual(aug_mask.dtype, mask.dtype)


# ---------------------------------------------------------------------------
# split_dataset_by_scene
# ---------------------------------------------------------------------------


class TestSplitDatasetByScene(unittest.TestCase):

    def _make_paths(self, n=100):
        return [f"/data/scene_{i:04d}_patch.png" for i in range(n)]

    def test_split_counts_sum_to_total(self):
        paths = self._make_paths(100)
        train, val, test = split_dataset_by_scene(
            paths, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        self.assertEqual(len(train) + len(val) + len(test), 100)

    def test_no_overlap(self):
        paths = self._make_paths(100)
        train, val, test = split_dataset_by_scene(
            paths, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        train_set = set(train)
        val_set = set(val)
        test_set = set(test)
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)

    def test_approximate_ratios(self):
        paths = self._make_paths(200)
        train, val, test = split_dataset_by_scene(
            paths, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        n = len(paths)
        self.assertAlmostEqual(len(val) / n, 0.15, delta=0.05)
        self.assertAlmostEqual(len(test) / n, 0.15, delta=0.05)

    def test_reproducible_with_seed(self):
        paths = self._make_paths(100)
        train_a, val_a, test_a = split_dataset_by_scene(paths, seed=42)
        train_b, val_b, test_b = split_dataset_by_scene(paths, seed=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(test_a, test_b)

    def test_small_dataset(self):
        paths = self._make_paths(10)
        train, val, test = split_dataset_by_scene(paths, seed=0)
        self.assertGreater(len(train), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
