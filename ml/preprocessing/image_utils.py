"""
ml/preprocessing/image_utils.py
=================================
OceanTrace AI — SAR / optical image utility functions.

Functions
---------
load_image(path, modality)
    Load a SAR (grayscale) or RGB image → numpy float32 array.

normalize_image(arr, modality)
    Normalise to [0, 1] or ImageNet stats depending on modality.

resize_with_padding(arr, target)
    Resize + zero-pad to target (H, W) preserving aspect ratio.

decode_m4d_mask(mask_arr)
    Convert M4D raw mask pixel values (0–4) to class index array.

split_dataset_by_scene(paths, val_ratio, test_ratio, seed)
    Scene-aware train/val/test split (groups by filename prefix).

compute_class_distribution(mask_paths)
    Count pixels per class across a list of mask paths.

augment_pair(image, mask)
    Random spatial + photometric augmentation of an image+mask pair.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise ImportError("pip install Pillow")

# ---------------------------------------------------------------------------
# M4D class mapping
# ---------------------------------------------------------------------------
M4D_CLASS_NAMES: Dict[int, str] = {
    0: "Clean Sea",
    1: "Oil Spill",
    2: "Look-Alike",
    3: "Ship",
    4: "Land",
}
NUM_CLASSES = len(M4D_CLASS_NAMES)

# ImageNet normalisation constants (for pre-trained backbone compatibility)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------
def load_image(path: str | Path, modality: str = "auto") -> np.ndarray:
    """
    Load an image from *path* and return a float32 numpy array in [0, 1].

    Parameters
    ----------
    path : str | Path
        File path (PNG, GeoTIFF, etc.).
    modality : {'auto', 'sar', 'rgb'}
        'sar'  → force grayscale load, returned as (H, W, 1)
        'rgb'  → force 3-channel load, returned as (H, W, 3)
        'auto' → detect from image mode; grayscale→(H,W,1), color→(H,W,3)

    Returns
    -------
    np.ndarray of shape (H, W, C), dtype float32, range [0, 1]
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # GeoTIFF via rasterio (optional dependency)
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio  # type: ignore
            with rasterio.open(path) as src:
                arr = src.read().transpose(1, 2, 0).astype(np.float32)
            # Normalise to [0, 1] using per-band min/max
            for c in range(arr.shape[2]):
                band = arr[:, :, c]
                lo, hi = band.min(), band.max()
                arr[:, :, c] = (band - lo) / (hi - lo + 1e-8)
            if modality == "sar" and arr.shape[2] > 1:
                arr = arr[:, :, :1]  # keep only first band
            return arr
        except ImportError:
            pass  # fall through to PIL

    img = Image.open(path)

    if modality == "sar":
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr[:, :, np.newaxis]          # (H, W, 1)

    if modality == "rgb":
        img = img.convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

    # auto
    if img.mode in ("L", "P"):
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr[:, :, np.newaxis]
    else:
        img = img.convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# normalize_image
# ---------------------------------------------------------------------------
def normalize_image(arr: np.ndarray, modality: str = "auto") -> np.ndarray:
    """
    Normalise an image array for model input.

    For SAR (single-channel): stretches to [0, 1] and repeats across 3 channels.
    For RGB / OPTICAL: applies ImageNet mean/std normalisation (channel-wise).

    Parameters
    ----------
    arr : np.ndarray
        Shape (H, W), (H, W, 1), or (H, W, 3).
    modality : str
        'sar', 'rgb', 'optical', or 'auto'.

    Returns
    -------
    np.ndarray float32 with 3 channels.
    """
    arr = arr.astype(np.float32)
    modality_lower = modality.lower()

    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)

    c = arr.shape[-1]
    if modality_lower == "auto":
        modality_lower = "sar" if c == 1 else "rgb"

    if modality_lower in ("sar", "gray", "grayscale"):
        lo, hi = float(arr.min()), float(arr.max())
        norm = (arr - lo) / (hi - lo + 1e-8)
        if norm.shape[-1] == 1:
            norm = np.repeat(norm, 3, axis=-1)
        return norm.astype(np.float32)

    # RGB / OPTICAL — ImageNet normalisation
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.max() > 1.0:
        arr = arr / 255.0
    norm = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return norm.astype(np.float32)


# ---------------------------------------------------------------------------
# resize_with_padding
# ---------------------------------------------------------------------------
def resize_with_padding(
    arr: np.ndarray,
    target: Tuple[int, int] = (512, 512),
    pad_value: float = 0.0,
    is_mask: bool = False,
) -> np.ndarray:
    """
    Resize *arr* to *target* (H, W) with zero-padding to maintain aspect ratio.

    Parameters
    ----------
    arr       : (H, W) or (H, W, C) numpy array.
    target    : (target_H, target_W).
    pad_value : fill value (0 for images, 255 or 0 for masks).
    is_mask   : if True, uses NEAREST interpolation; otherwise BILINEAR.

    Returns
    -------
    np.ndarray of shape (target_H, target_W) or (target_H, target_W, C).
    """
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[:, :, np.newaxis]

    h, w, c = arr.shape
    th, tw = target

    # Scale factor to fit within target while preserving aspect ratio
    scale = min(th / h, tw / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resample = Image.NEAREST if is_mask else Image.BILINEAR
    channels = []
    for ci in range(c):
        pil_ch = Image.fromarray(arr[:, :, ci])
        pil_ch = pil_ch.resize((new_w, new_h), resample)
        channels.append(np.array(pil_ch))
    resized = np.stack(channels, axis=-1)  # (new_h, new_w, c)

    # Pad to target
    out = np.full((th, tw, c), pad_value, dtype=arr.dtype)
    pad_top  = (th - new_h) // 2
    pad_left = (tw - new_w) // 2
    out[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = resized

    return out[:, :, 0] if squeeze else out


# ---------------------------------------------------------------------------
# decode_m4d_mask
# ---------------------------------------------------------------------------
def decode_m4d_mask(mask_arr: np.ndarray) -> np.ndarray:
    """
    Map M4D mask pixel values to class index integers.

    M4D stores masks as 8-bit PNGs with values directly equal to class IDs
    (0–4).  This function validates and returns the mask as int64 (suitable
    for use as CrossEntropy targets in PyTorch).

    Invalid pixel values are mapped to 0 (Clean Sea) with a warning.
    """
    mask = mask_arr.astype(np.int64)
    valid = set(M4D_CLASS_NAMES.keys())
    unique_vals = np.unique(mask)
    invalid = [int(v) for v in unique_vals if int(v) not in valid]
    if invalid:
        import warnings
        warnings.warn(
            f"M4D mask contains unexpected values {invalid}; mapping to 0.",
            UserWarning,
            stacklevel=2,
        )
        for v in invalid:
            mask[mask == v] = 0
    return mask


# ---------------------------------------------------------------------------
# decode_binary_mask
# ---------------------------------------------------------------------------
def decode_binary_mask(mask_arr: np.ndarray, threshold: Optional[int] = None) -> np.ndarray:
    """
    Map binary mask pixel values to binary class integers:
      0 = Background / Sea
      1 = Oil Spill

    Resolves encoding inconsistencies between 0-255 masks (standard 8-bit PNGs)
    and 0-1 masks, filtering out anti-aliasing edge artifacts.

    Parameters
    ----------
    mask_arr : np.ndarray
        Raw mask array (H, W) or (H, W, C).
    threshold : int, optional
        Explicit threshold. If None, dynamically uses mask_max / 2 for [0, 255]
        masks or 0.5 for [0, 1] masks.

    Returns
    -------
    np.ndarray int64 of shape (H, W) with values strictly in {0, 1}.
    """
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    
    mask_max = float(mask_arr.max()) if mask_arr.size > 0 else 0.0
    if mask_max == 0.0:
        return np.zeros(mask_arr.shape, dtype=np.int64)

    if mask_max > 1.0:
        thresh = threshold if threshold is not None else (mask_max / 2.0)
        binary_mask = (mask_arr >= thresh).astype(np.int64)
    else:
        thresh = 0.5 if threshold is None else (threshold / 255.0)
        binary_mask = (mask_arr >= thresh).astype(np.int64)
        
    return binary_mask


# ---------------------------------------------------------------------------
# split_dataset_by_scene
# ---------------------------------------------------------------------------
def split_dataset_by_scene(
    paths: List[str | Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split *paths* into train/val/test ensuring all images from the same
    "scene" end up in the same split.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        return [], [], []

    scenes: Dict[str, List[Path]] = {}
    for p in paths:
        parts = p.stem.split("_")
        if len(parts) >= 2 and parts[0].lower() in ("scene", "img", "sar"):
            scene_key = f"{parts[0]}_{parts[1]}"
        else:
            scene_key = parts[0]
        scenes.setdefault(scene_key, []).append(p)

    rng = random.Random(seed)
    scene_keys = sorted(scenes.keys())
    rng.shuffle(scene_keys)

    n = len(scene_keys)
    if n == 1:
        return sorted(paths), [], []

    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    if n_test + n_val >= n:
        n_test = max(1, n // 3)
        n_val = max(1, n // 3)

    test_scenes = scene_keys[:n_test]
    val_scenes = scene_keys[n_test:n_test + n_val]
    train_scenes = scene_keys[n_test + n_val:]

    def _collect(keys: List[str]) -> List[Path]:
        out = []
        for k in keys:
            out.extend(scenes[k])
        return sorted(out)

    return _collect(train_scenes), _collect(val_scenes), _collect(test_scenes)


# ---------------------------------------------------------------------------
# compute_class_distribution
# ---------------------------------------------------------------------------
def compute_class_distribution(
    mask_paths: List[str | Path],
    dataset: str = "m4d",
) -> Dict[Any, int]:
    """
    Count total pixels per class across *mask_paths*.
    Returns dict mapping class ID (int) -> pixel count.
    """
    counts: Dict[int, int] = {}

    for mp in mask_paths:
        msk = np.array(Image.open(mp))
        for v in np.unique(msk):
            val = int(v)
            counts[val] = counts.get(val, 0) + int(np.sum(msk == v))

    return {k: counts[k] for k in sorted(counts.keys())}


# ---------------------------------------------------------------------------
# augment_pair
# ---------------------------------------------------------------------------
def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random augmentations consistently to an image+mask pair.
    Preserves original image dtype.
    """
    rng = np.random.default_rng(seed)
    orig_dtype = image.dtype

    if np.issubdtype(orig_dtype, np.floating):
        image_work = np.clip(image.astype(np.float32), 0.0, 1.0)
    else:
        image_work = image.astype(np.float32) / 255.0

    # Horizontal flip
    if rng.random() > 0.5:
        image_work = np.fliplr(image_work)
        mask = np.fliplr(mask)

    # Vertical flip
    if rng.random() > 0.5:
        image_work = np.flipud(image_work)
        mask = np.flipud(mask)

    # 90° rotation
    k = int(rng.integers(0, 4))
    if k > 0:
        image_work = np.rot90(image_work, k)
        mask = np.rot90(mask, k)

    # Brightness jitter
    if rng.random() > 0.3:
        factor = float(rng.uniform(0.7, 1.3))
        image_work = np.clip(image_work * factor, 0.0, 1.0)

    # Contrast jitter
    if rng.random() > 0.3:
        mean = image_work.mean()
        factor = float(rng.uniform(0.8, 1.2))
        image_work = np.clip((image_work - mean) * factor + mean, 0.0, 1.0)

    if np.issubdtype(orig_dtype, np.floating):
        res_image = image_work.astype(orig_dtype)
    else:
        res_image = np.clip(image_work * 255.0, 0, 255).astype(orig_dtype)

    return res_image, mask
