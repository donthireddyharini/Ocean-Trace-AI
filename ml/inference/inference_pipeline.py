"""
OceanTrace AI — Inference Pipeline
Smart India Hackathon 2026 | SIH26143

Provides a production-ready inference pipeline for oil spill detection.
Gracefully handles missing fine-tuned weights by falling back to
pretrained-only or demo modes, with clear warnings.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    from transformers import SegformerForSemanticSegmentation
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Full result returned by the inference pipeline."""

    # Pixel-level outputs
    class_mask: np.ndarray          # (H, W)  int class indices
    class_probabilities: np.ndarray # (C, H, W) float probabilities

    # Oil-spill summary
    spill_detected: bool
    spill_confidence: float         # max probability of oil-spill class
    spill_pixel_fraction: float     # fraction of pixels classified as oil spill

    # Metadata
    processing_ms: float
    modality: str                   # 'SAR' or 'OPTICAL'
    model_version: str
    is_fine_tuned: bool             # True = real weights; False = pretrained-only
    warning: str = ""               # Non-empty when in demo / untrained mode


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_CLASS_NAMES = ["Clean Sea", "Oil Spill", "Look-Alike", "Ship", "Land"]
_OIL_SPILL_IDX = 1
_IMAGE_SIZE = 512
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEMO_WARNING = (
    "DEMO MODE: This result is from a randomly-initialised model. "
    "Predictions are NOT reliable. Train the model with the M4D dataset "
    "and load the checkpoint to obtain real detections."
)
_PRETRAINED_WARNING = (
    "PRETRAINED-ONLY MODE: The SegFormer encoder has ImageNet weights but "
    "has NOT been fine-tuned on oil-spill data. Segmentation output is "
    "unreliable. Run the training notebook to produce real weights."
)


class OilSpillInferencePipeline:
    """
    Three-tier graceful degradation::

        1. Fine-tuned checkpoint  → real predictions
        2. Pretrained encoder only → structural output, low confidence
        3. Demo mode (no torch)   → synthetic output with warning

    Parameters
    ----------
    model_path : str or Path, optional
        Path to a `.pth` checkpoint saved by the training notebook.
        When *None* the pipeline tries the default artifact location.
    device : str, optional
        ``'cuda'`` or ``'cpu'``.  Auto-detected when *None*.
    """

    _DEFAULT_CHECKPOINT = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "artifacts", "checkpoints", "best_oilspill_segformer.pth",
    )

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self._status: str = "demo_mode"
        self._model = None
        self._device: Optional[str] = None
        self._model_version: str = "demo-0.0"
        self._is_fine_tuned: bool = False
        self._warning: str = _DEMO_WARNING

        if not _TORCH_OK:
            logger.warning("PyTorch not available. Running in demo mode.")
            return

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- Tier 1: try fine-tuned checkpoint ---
        candidate = Path(model_path) if model_path else Path(self._DEFAULT_CHECKPOINT)
        if candidate.exists():
            try:
                self._load_fine_tuned(candidate)
                return
            except Exception as exc:
                logger.warning("Could not load checkpoint %s: %s", candidate, exc)

        # --- Tier 2: pretrained encoder only ---
        if _TRANSFORMERS_OK:
            try:
                self._load_pretrained_only()
                return
            except Exception as exc:
                logger.warning("Could not load pretrained model: %s", exc)

        # --- Tier 3: demo ---
        logger.warning("No model available. Using demo mode.")

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_fine_tuned(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        num_classes = len(_CLASS_NAMES)
        model_name = checkpoint.get("config", {}).get("model_name", "nvidia/mit-b2")
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label={i: n for i, n in enumerate(_CLASS_NAMES)},
            label2id={n: i for i, n in enumerate(_CLASS_NAMES)},
            ignore_mismatched_sizes=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device).eval()

        self._model = model
        self._status = "fine_tuned"
        self._is_fine_tuned = True
        self._warning = ""
        self._model_version = f"segformer-b2-oilspill-epoch{checkpoint.get('epoch', '?')}"
        logger.info("Fine-tuned checkpoint loaded from %s", checkpoint_path)

    def _load_pretrained_only(self) -> None:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=len(_CLASS_NAMES),
            id2label={i: n for i, n in enumerate(_CLASS_NAMES)},
            label2id={n: i for i, n in enumerate(_CLASS_NAMES)},
            ignore_mismatched_sizes=True,
        )
        model.to(self._device).eval()
        self._model = model
        self._status = "pretrained_only"
        self._is_fine_tuned = False
        self._warning = _PRETRAINED_WARNING
        self._model_version = "mit-b2-pretrained-only"
        logger.warning(_PRETRAINED_WARNING)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True when a model (any tier) is loaded."""
        return self._status != "demo_mode" or not _TORCH_OK

    @property
    def model_status(self) -> str:
        """One of: ``'fine_tuned'``, ``'pretrained_only'``, ``'demo_mode'``."""
        return self._status

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_modality(arr: np.ndarray) -> str:
        """Heuristic: SAR images are single-channel or very low colour variance."""
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            return "SAR"
        if arr.ndim == 3 and arr.shape[2] >= 3:
            ch_std = arr[:, :, :3].std(axis=(0, 1))
            if ch_std.max() < 15:
                return "SAR"
        return "OPTICAL"

    def _to_numpy(self, image_input) -> np.ndarray:
        """Convert any input type to (H, W, 3) uint8 numpy array."""
        if isinstance(image_input, np.ndarray):
            arr = image_input
        elif isinstance(image_input, bytes):
            if not _PIL_OK:
                raise RuntimeError("Pillow required to decode image bytes.")
            arr = np.array(PILImage.open(BytesIO(image_input)))
        elif _PIL_OK and isinstance(image_input, PILImage.Image):
            arr = np.array(image_input)
        elif isinstance(image_input, (str, Path)):
            if not _PIL_OK:
                raise RuntimeError("Pillow required to load image files.")
            arr = np.array(PILImage.open(image_input))
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        # Ensure uint8 HxWxC
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    def _preprocess(self, image_input, modality: str = "auto") -> Tuple["torch.Tensor", str, Tuple[int, int]]:
        """Return (tensor, detected_modality, original_hw)."""
        arr = self._to_numpy(image_input)
        orig_h, orig_w = arr.shape[:2]
        detected = modality if modality != "auto" else self._detect_modality(arr)

        # Resize
        if _PIL_OK:
            img_pil = PILImage.fromarray(arr).resize(
                (_IMAGE_SIZE, _IMAGE_SIZE), PILImage.BILINEAR
            )
            resized = np.array(img_pil).astype(np.float32) / 255.0
        else:
            # Fallback: naive crop/pad
            resized = arr.astype(np.float32) / 255.0

        # Normalise
        resized = (resized - _MEAN) / _STD  # (H, W, 3)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor, detected, (orig_h, orig_w)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, image_input, modality: str = "auto") -> InferenceResult:
        """Run full pipeline on any supported image_input type."""
        t0 = time.perf_counter()

        if self._status == "demo_mode" or self._model is None:
            return self._demo_result(modality, t0)

        tensor, detected, orig_hw = self._preprocess(image_input, modality)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            outputs = self._model(pixel_values=tensor)
            logits = outputs.logits  # (1, C, h', w')

        # Upsample to original size
        logits_up = F.interpolate(
            logits, size=orig_hw, mode="bilinear", align_corners=False
        )
        probs = torch.softmax(logits_up, dim=1).squeeze(0).cpu().numpy()  # (C, H, W)
        class_mask = probs.argmax(axis=0).astype(np.int32)                # (H, W)

        oil_probs = probs[_OIL_SPILL_IDX]
        spill_confidence = float(oil_probs.max())
        spill_pixel_fraction = float((class_mask == _OIL_SPILL_IDX).mean())
        spill_detected = spill_pixel_fraction > 0.001  # >0.1% of pixels

        processing_ms = (time.perf_counter() - t0) * 1000.0
        return InferenceResult(
            class_mask=class_mask,
            class_probabilities=probs,
            spill_detected=spill_detected,
            spill_confidence=spill_confidence,
            spill_pixel_fraction=spill_pixel_fraction,
            processing_ms=processing_ms,
            modality=detected,
            model_version=self._model_version,
            is_fine_tuned=self._is_fine_tuned,
            warning=self._warning,
        )

    def run_from_bytes(self, image_bytes: bytes, modality: str = "auto") -> InferenceResult:
        """Convenience wrapper for byte-encoded images."""
        return self.run(image_bytes, modality=modality)

    # ------------------------------------------------------------------
    # Demo mode
    # ------------------------------------------------------------------

    def _demo_result(self, modality: str, t0: float) -> InferenceResult:
        """Return a clearly-labelled dummy result when no model is loaded."""
        H, W = 512, 512
        num_classes = len(_CLASS_NAMES)
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(num_classes), size=(H, W)).transpose(2, 0, 1)
        class_mask = probs.argmax(axis=0).astype(np.int32)
        processing_ms = (time.perf_counter() - t0) * 1000.0
        return InferenceResult(
            class_mask=class_mask,
            class_probabilities=probs.astype(np.float32),
            spill_detected=False,
            spill_confidence=float(probs[_OIL_SPILL_IDX].max()),
            spill_pixel_fraction=float((class_mask == _OIL_SPILL_IDX).mean()),
            processing_ms=processing_ms,
            modality=modality if modality != "auto" else "UNKNOWN",
            model_version="demo-0.0",
            is_fine_tuned=False,
            warning=_DEMO_WARNING,
        )
