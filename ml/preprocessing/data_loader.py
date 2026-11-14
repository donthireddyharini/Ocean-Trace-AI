"""
ml/preprocessing/data_loader.py
=================================
OceanTrace AI — PyTorch Dataset and DataLoader factory for oil-spill segmentation.

Classes
-------
OilSpillDataset
    PyTorch Dataset supporting the M4D Oil Spill format.
    Handles both SAR (grayscale → 3-channel replication) and RGB images.

Functions
---------
create_dataloaders(data_dir, batch_size, num_workers, ...)
    Returns (train_loader, val_loader, test_loader) DataLoader tuple.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise ImportError("pip install torch torchvision")

try:
    from PIL import Image
except ImportError:
    raise ImportError("pip install Pillow")

from ml.preprocessing.image_utils import (
    augment_pair,
    decode_binary_mask,
    decode_m4d_mask,
    load_image,
    normalize_image,
    resize_with_padding,
    split_dataset_by_scene,
)

# ---------------------------------------------------------------------------
# OilSpillDataset
# ---------------------------------------------------------------------------
class OilSpillDataset(Dataset):
    """
    PyTorch Dataset for SAR oil-spill segmentation (M4D format).

    Directory layout expected
    -------------------------
    data_dir/
        images/
            *.png  (SAR or RGB images)
        masks/
            *.png  (segmentation masks with values 0–4)

    Parameters
    ----------
    image_paths : list of Path-like
        Paths to image files.
    mask_paths : list of Path-like
        Corresponding mask file paths (must align 1-to-1 with image_paths).
    modality : {'auto', 'sar', 'rgb'}
        Image loading modality (see image_utils.load_image).
    input_size : (H, W)
        Target spatial size after resize + pad.
    augment : bool
        Whether to apply random augmentations (train only).
    transform : callable, optional
        Additional torchvision-style transform applied to the final
        (image_tensor, mask_tensor) pair.
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        mask_paths: List[Union[str, Path]],
        modality: str = "auto",
        input_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        transform: Optional[Callable] = None,
        is_binary: Optional[bool] = None,
    ) -> None:
        assert len(image_paths) == len(mask_paths), (
            f"Length mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
        )
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths  = [Path(p) for p in mask_paths]
        self.modality    = modality
        self.input_size  = input_size
        self.augment     = augment
        self.transform   = transform
        self.is_binary   = is_binary

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        msk_path = self.mask_paths[idx]

        # Load image → float32 (H, W, C) in [0, 1]
        img = load_image(img_path, modality=self.modality)

        # Convert SAR (H, W, 1) to 3-channel by replication
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)  # (H, W, 3)

        # Load mask
        msk_raw = np.array(Image.open(msk_path))
        if self.is_binary is True:
            msk = decode_binary_mask(msk_raw)
        elif self.is_binary is False:
            msk = decode_m4d_mask(msk_raw)
        else:
            # Auto-detect binary vs M4D
            if msk_raw.max() > 4:
                msk = decode_binary_mask(msk_raw)
            else:
                msk = decode_m4d_mask(msk_raw)

        # Resize + pad
        img = resize_with_padding(img, target=self.input_size, is_mask=False)
        msk = resize_with_padding(msk, target=self.input_size, is_mask=True)

        # Augment
        if self.augment:
            img, msk = augment_pair(img, msk)

        # Normalise image (ImageNet stats for pre-trained backbone)
        img = normalize_image(img, modality="rgb")

        # Convert to tensors
        # img: (H, W, 3) → (3, H, W) float32
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        # msk: (H, W) int64
        msk_tensor = torch.from_numpy(msk.astype(np.int64))

        if self.transform is not None:
            img_tensor, msk_tensor = self.transform(img_tensor, msk_tensor)

        return img_tensor, msk_tensor

    # ------------------------------------------------------------------
    @classmethod
    def from_directory(
        cls,
        data_dir: Union[str, Path],
        modality: str = "auto",
        input_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        transform: Optional[Callable] = None,
        image_subdir: str = "images",
        mask_subdir: str = "masks",
        is_binary: Optional[bool] = None,
    ) -> "OilSpillDataset":
        """
        Build a dataset from a directory with 'images/' and 'masks/' subdirs.
        Images and masks are matched by filename stem.
        """
        data_dir = Path(data_dir)
        img_dir  = data_dir / image_subdir
        msk_dir  = data_dir / mask_subdir

        img_stems = {p.stem: p for p in sorted(img_dir.glob("*.png"))}
        msk_stems = {p.stem: p for p in sorted(msk_dir.glob("*.png"))}

        common = sorted(set(img_stems) & set(msk_stems))
        if not common:
            raise FileNotFoundError(
                f"No matched image–mask pairs in {data_dir}. "
                "Check that images/ and masks/ contain PNG files with matching names."
            )

        image_paths = [img_stems[s] for s in common]
        mask_paths  = [msk_stems[s] for s in common]

        return cls(
            image_paths=image_paths,
            mask_paths=mask_paths,
            modality=modality,
            input_size=input_size,
            augment=augment,
            transform=transform,
            is_binary=is_binary,
        )


# ---------------------------------------------------------------------------
# create_dataloaders
# ---------------------------------------------------------------------------
def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 8,
    num_workers: int = 4,
    modality: str = "auto",
    input_size: Tuple[int, int] = (512, 512),
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    image_subdir: str = "images",
    mask_subdir: str = "masks",
    pin_memory: bool = True,
    persistent_workers: bool = True,
    is_binary: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a dataset directory.

    Parameters
    ----------
    data_dir       : Root dir containing 'images/' and 'masks/' subdirs.
    batch_size     : Samples per batch.
    num_workers    : Worker processes for DataLoader.
    modality       : 'auto' | 'sar' | 'rgb'
    input_size     : (H, W) spatial size for all images.
    val_ratio      : Fraction of data for validation.
    test_ratio     : Fraction of data for test.
    seed           : RNG seed for reproducible splits.
    pin_memory     : Pin memory for CUDA transfer (set False on CPU-only).
    persistent_workers : Keep DataLoader workers alive between epochs.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    img_dir  = data_dir / image_subdir

    all_image_paths = sorted(img_dir.glob("*.png"))
    if not all_image_paths:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    # Scene-aware split
    train_imgs, val_imgs, test_imgs = split_dataset_by_scene(
        all_image_paths, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    def _imgs_to_masks(img_paths: List[Path]) -> List[Path]:
        msk_dir = data_dir / mask_subdir
        return [msk_dir / p.name for p in img_paths]

    train_masks = _imgs_to_masks(train_imgs)
    val_masks   = _imgs_to_masks(val_imgs)
    test_masks  = _imgs_to_masks(test_imgs)

    train_ds = OilSpillDataset(
        train_imgs, train_masks,
        modality=modality, input_size=input_size, augment=True, is_binary=is_binary,
    )
    val_ds = OilSpillDataset(
        val_imgs, val_masks,
        modality=modality, input_size=input_size, augment=False, is_binary=is_binary,
    )
    test_ds = OilSpillDataset(
        test_imgs, test_masks,
        modality=modality, input_size=input_size, augment=False, is_binary=is_binary,
    )

    # Windows: num_workers > 0 requires spawn context — guard against issues
    if os.name == "nt" and num_workers > 0:
        try:
            import multiprocessing
            multiprocessing.set_start_method("spawn", force=False)
        except RuntimeError:
            pass

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(persistent_workers and num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(
        f"DataLoaders created — "
        f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)} samples"
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <data_dir>")
        sys.exit(1)

    train_ldr, val_ldr, test_ldr = create_dataloaders(
        sys.argv[1], batch_size=2, num_workers=0
    )
    imgs, masks = next(iter(train_ldr))
    print(f"Batch image shape : {imgs.shape}")   # (2, 3, 512, 512)
    print(f"Batch mask shape  : {masks.shape}")  # (2, 512, 512)
    print(f"Mask unique values: {masks.unique()}")
