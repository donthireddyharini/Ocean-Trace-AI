"""
OceanTrace AI — SegFormer Inference Wrapper
Smart India Hackathon 2026 | SIH26143

Wraps SegformerForSemanticSegmentation for production inference.
Supports:
  - Loading from fine-tuned .pth checkpoint
  - Loading pretrained encoder only (demo / no-fine-tune mode)
  - Preprocessing SAR and optical images
  - Postprocessing logits to class masks
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports (guarded so the file can be imported without GPU)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    from transformers import SegformerForSemanticSegmentation
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False

# Import InferenceResult from pipeline module
try:
    from ml.inference.inference_pipeline import InferenceResult
except ImportError:
    # Fallback for standalone use
    from dataclasses import dataclass, field

    @dataclass
    class InferenceResult:  # type: ignore[no-redef]
        class_mask: np.ndarray
        class_probabilities: np.ndarray
        spill_detected: bool
        spill_confidence: float
        spill_pixel_fraction: float
        processing_ms: float
        modality: str
        model_version: str
        is_fine_tuned: bool
        warning: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_SIZE = 512
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_OIL_SPILL_IDX = 1

_PRETRAINED_WARNING = (
    "PRETRAINED-ONLY: SegFormer encoder has ImageNet weights but has NOT been "
    "fine-tuned on oil-spill data.  Segmentation output is unreliable."
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SegFormerOilSpill:
    """
    SegFormer-B2 inference wrapper for oil spill semantic segmentation.

    Attributes
    ----------
    CLASS_NAMES : list[str]
        Human-readable class labels (index → name).
    CLASS_COLORS : list[tuple]
        BGR colours for overlay visualisation.
    """

    CLASS_NAMES = ["Background / Sea", "Oil Spill"]
    CLASS_COLORS = [
        (15, 23, 42),    # Background / Sea — dark navy
        (220, 38, 38),   # Oil Spill — red
    ]

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.is_loaded: bool = False
        self.is_fine_tuned: bool = False
        self._model = None
        self.class_names = class_names or list(self.CLASS_NAMES)
        self._device: str = (
            device
            if device
            else ("cuda" if (_TORCH_OK and torch.cuda.is_available()) else "cpu")
        )
        self.preprocess_config: Dict = {
            "image_size": _IMAGE_SIZE,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "SegFormerOilSpill":
        """
        Load a fine-tuned model from a ``.pth`` checkpoint produced by
        the training notebook.

        Parameters
        ----------
        checkpoint_path : str | Path
        device : str, optional

        Returns
        -------
        SegFormerOilSpill
        """
        if not _TORCH_OK or not _TRANSFORMERS_OK:
            raise RuntimeError(
                "PyTorch and 🤗 Transformers are required. "
                "pip install torch transformers"
            )

        checkpoint = torch.load(checkpoint_path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Determine class names and count dynamically from checkpoint
        if "class_names" in checkpoint:
            class_names = checkpoint["class_names"]
        elif "config" in checkpoint and "class_names" in checkpoint["config"]:
            class_names = checkpoint["config"]["class_names"]
        elif "config" in checkpoint and "id2label" in checkpoint["config"]:
            raw_id2label = checkpoint["config"]["id2label"]
            class_names = [raw_id2label[i] if i in raw_id2label else raw_id2label[str(i)] for i in range(len(raw_id2label))]
        else:
            class_names = list(cls.CLASS_NAMES)

        instance = cls(checkpoint_path=checkpoint_path, device=device, class_names=class_names)
        num_classes = len(class_names)
        model_name = checkpoint.get("config", {}).get("pretrained_model_name", checkpoint.get("config", {}).get("model_name", "nvidia/mit-b2"))

        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label={i: n for i, n in enumerate(class_names)},
            label2id={n: i for i, n in enumerate(class_names)},
            ignore_mismatched_sizes=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(instance._device).eval()

        instance._model = model
        instance.is_loaded = True
        instance.is_fine_tuned = True

        # Update preprocess_config from checkpoint if available
        if "preprocessing" in checkpoint:
            instance.preprocess_config.update(checkpoint["preprocessing"])

        logger.info("SegFormer loaded from checkpoint: %s (classes: %s)", checkpoint_path, class_names)
        return instance

    @classmethod
    def from_pretrained_only(
        cls,
        model_name: str = "nvidia/mit-b2",
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ) -> "SegFormerOilSpill":
        """
        Load only the pretrained ImageNet encoder — **no fine-tuning**.

        Useful for demos or when no trained checkpoint is available.
        ``is_fine_tuned`` will be ``False`` and predictions are unreliable.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier (default ``nvidia/mit-b2``).
        device : str, optional
        class_names : list[str], optional

        Returns
        -------
        SegFormerOilSpill
        """
        if not _TORCH_OK or not _TRANSFORMERS_OK:
            raise RuntimeError(
                "PyTorch and 🤗 Transformers are required. "
                "pip install torch transformers"
            )

        resolved_classes = class_names or list(cls.CLASS_NAMES)
        instance = cls(device=device, class_names=resolved_classes)
        num_classes = len(resolved_classes)

        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label={i: n for i, n in enumerate(resolved_classes)},
            label2id={n: i for i, n in enumerate(resolved_classes)},
            ignore_mismatched_sizes=True,
        )
        model.to(instance._device).eval()

        instance._model = model
        instance.is_loaded = True
        instance.is_fine_tuned = False

        logger.warning(_PRETRAINED_WARNING)
        return instance

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def preprocess(
        self,
        image_input,
        modality: str = "auto",
    ) -> "torch.Tensor":
        """
        Preprocess an image for model input.

        Accepts:
          - ``bytes``              raw image bytes
          - ``PIL.Image.Image``
          - ``numpy.ndarray``      (H, W) or (H, W, C) uint8
          - ``str`` / ``Path``     file path

        Parameters
        ----------
        image_input : see above
        modality : ``'auto'``, ``'SAR'``, or ``'OPTICAL'``

        Returns
        -------
        torch.Tensor  shape (1, 3, H, W)
        """
        arr = self._to_numpy(image_input)
        detected = modality if modality != "auto" else self._detect_modality(arr)

        # Resize
        size = self.preprocess_config.get("image_size", _IMAGE_SIZE)
        if _PIL_OK:
            img_pil = PILImage.fromarray(arr).resize((size, size), PILImage.BILINEAR)
            resized = np.array(img_pil).astype(np.float32) / 255.0
        else:
            resized = arr.astype(np.float32) / 255.0

        mean = np.array(self.preprocess_config.get("mean", _MEAN.tolist()), dtype=np.float32)
        std  = np.array(self.preprocess_config.get("std",  _STD.tolist()),  dtype=np.float32)
        resized = (resized - mean) / std  # (H, W, 3)

        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor

    def predict(self, pixel_values_tensor: "torch.Tensor") -> "torch.Tensor":
        """
        Run forward pass and return raw logits.

        Parameters
        ----------
        pixel_values_tensor : torch.Tensor  (B, 3, H, W)

        Returns
        -------
        torch.Tensor  (B, num_classes, H', W')   raw logits (H' may differ from H)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call SegFormerOilSpill.load() first.")
        pixel_values_tensor = pixel_values_tensor.to(self._device)
        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_values_tensor)
        return outputs.logits

    def postprocess(
        self,
        logits: "torch.Tensor",
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert raw logits to a class mask and probability map.

        Parameters
        ----------
        logits : torch.Tensor  (1, C, H', W')
        original_size : (H, W) tuple, optional
            Target size for upsampling. If *None*, uses the logit spatial size.

        Returns
        -------
        class_mask : np.ndarray  (H, W)  int32 class indices
        class_probabilities : np.ndarray  (C, H, W)  float32 probabilities
        """
        if original_size is not None:
            logits = F.interpolate(
                logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (C, H, W)
        class_mask = probs.argmax(axis=0).astype(np.int32)             # (H, W)
        return class_mask, probs.astype(np.float32)

    def run_full_pipeline(
        self,
        image_input,
        modality: str = "auto",
    ) -> "InferenceResult":
        """
        End-to-end: preprocess → predict → postprocess.

        Parameters
        ----------
        image_input : see ``preprocess()``
        modality : ``'auto'``, ``'SAR'``, or ``'OPTICAL'``

        Returns
        -------
        InferenceResult
        """
        import time
        t0 = time.perf_counter()

        arr = self._to_numpy(image_input)
        orig_hw = (arr.shape[0], arr.shape[1])
        detected = modality if modality != "auto" else self._detect_modality(arr)

        tensor = self.preprocess(arr, modality=detected)
        logits = self.predict(tensor)
        class_mask, class_probs = self.postprocess(logits, original_size=orig_hw)

        oil_probs = class_probs[_OIL_SPILL_IDX]
        spill_confidence = float(oil_probs.max())
        spill_pixel_fraction = float((class_mask == _OIL_SPILL_IDX).mean())
        spill_detected = spill_pixel_fraction > 0.001

        processing_ms = (time.perf_counter() - t0) * 1000.0
        return InferenceResult(
            class_mask=class_mask,
            class_probabilities=class_probs,
            spill_detected=spill_detected,
            spill_confidence=spill_confidence,
            spill_pixel_fraction=spill_pixel_fraction,
            processing_ms=processing_ms,
            modality=detected,
            model_version="segformer-b2-oilspill" if self.is_fine_tuned else "mit-b2-pretrained-only",
            is_fine_tuned=self.is_fine_tuned,
            warning="" if self.is_fine_tuned else _PRETRAINED_WARNING,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_checkpoint_available(checkpoint_path: Union[str, Path]) -> bool:
        """Return True if the given checkpoint file exists."""
        return os.path.exists(checkpoint_path)

    @staticmethod
    def _detect_modality(arr: np.ndarray) -> str:
        """Heuristic SAR vs optical detection."""
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            return "SAR"
        if arr.ndim == 3 and arr.shape[2] >= 3:
            ch_std = arr[:, :, :3].std(axis=(0, 1))
            if ch_std.max() < 15:
                return "SAR"
        return "OPTICAL"

    @staticmethod
    def _to_numpy(image_input) -> np.ndarray:
        """Convert any supported input type to (H, W, 3) uint8 numpy array."""
        if isinstance(image_input, np.ndarray):
            arr = image_input
        elif isinstance(image_input, bytes):
            if not _PIL_OK:
                raise RuntimeError("Pillow required. pip install Pillow")
            arr = np.array(PILImage.open(BytesIO(image_input)))
        elif _PIL_OK and isinstance(image_input, PILImage.Image):
            arr = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, (str, Path)):
            if not _PIL_OK:
                raise RuntimeError("Pillow required. pip install Pillow")
            arr = np.array(PILImage.open(image_input))
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr
