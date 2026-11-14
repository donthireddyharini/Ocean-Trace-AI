"""
OceanTrace AI — Look-Alike Classifier
Smart India Hackathon 2026 | SIH26143

Distinguishes between genuine oil spills and look-alike phenomena
(biogenic films, low-wind areas, algal blooms, etc.) in SAR imagery.

Training requires the GlobalOSD-SAR dataset:
  https://doi.org/10.5281/zenodo.15286918  (CC BY 4.0)

Classes:
  0 = oil_spill  (genuine hydrocarbon release)
  1 = look_alike (natural or non-polluting dark spot)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Heavy optional dependencies
try:
    import torch
    import torch.nn as nn
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
    import torchvision.models as tv_models
    import torchvision.transforms as T
    _TORCHVISION_OK = True
except ImportError:
    _TORCHVISION_OK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES: List[str] = ["Oil Spill", "Look-Alike"]
IMAGE_SIZE: int = 224
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "artifacts", "checkpoints", "look_alike_classifier.pth",
)

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_resnet50(num_classes: int = 2, pretrained: bool = True) -> "nn.Module":
    """ResNet-50 backbone with a replaced classification head."""
    if not _TORCHVISION_OK:
        raise ImportError("torchvision is required. pip install torchvision")
    model = tv_models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> "nn.Module":
    """EfficientNet-B0 backbone with a replaced classification head."""
    if not _TORCHVISION_OK:
        raise ImportError("torchvision is required. pip install torchvision")
    model = tv_models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class LookAlikeClassifier:
    """
    Binary classifier: Oil Spill (0) vs Look-Alike (1).

    Can be initialised without a checkpoint for standalone use; call
    ``is_available()`` first to check whether a trained model exists.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to saved ``.pth`` checkpoint.  Uses the default artifact
        location when *None*.
    backbone : str
        ``'resnet50'`` (default) or ``'efficientnet_b0'``.
    device : str, optional
        ``'cuda'`` or ``'cpu'``.  Auto-detected when *None*.
    threshold : float
        Decision threshold for oil-spill class (default 0.5).
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        backbone: str = "resnet50",
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self._loaded = False
        self._model = None
        self._device: Optional[str] = None
        self._transform = None

        if not _TORCH_OK or not _TORCHVISION_OK:
            logger.warning(
                "PyTorch / torchvision not available. "
                "LookAlikeClassifier running in stub mode."
            )
            return

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._build_transform()

        # Resolve checkpoint
        candidate = Path(checkpoint_path) if checkpoint_path else Path(_DEFAULT_CHECKPOINT)
        if candidate.exists():
            try:
                self._load(candidate, backbone)
            except Exception as exc:
                logger.error("Failed to load look-alike checkpoint: %s", exc)
        else:
            logger.warning(
                "Look-alike checkpoint not found at %s. "
                "Train the model with the GlobalOSD-SAR dataset first.",
                candidate,
            )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_available(checkpoint_path: Optional[Union[str, Path]] = None) -> bool:
        """Return True if a trained checkpoint file exists on disk."""
        path = Path(checkpoint_path) if checkpoint_path else Path(_DEFAULT_CHECKPOINT)
        return path.exists()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_transform(self) -> None:
        if not _TORCHVISION_OK:
            return
        self._transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])

    def _load(self, checkpoint_path: Path, backbone: str) -> None:
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        backbone_name = checkpoint.get("backbone", backbone)

        if backbone_name == "efficientnet_b0":
            model = _build_efficientnet_b0(num_classes=2, pretrained=False)
        else:
            model = _build_resnet50(num_classes=2, pretrained=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device).eval()
        self._model = model
        self._loaded = True
        logger.info("Look-alike classifier loaded from %s", checkpoint_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _preprocess(self, image_input) -> "torch.Tensor":
        """Convert input to a normalised (1, 3, H, W) float32 tensor."""
        if isinstance(image_input, np.ndarray):
            if not _PIL_OK:
                raise RuntimeError("Pillow required for numpy input.")
            img = PILImage.fromarray(image_input.astype(np.uint8))
        elif _PIL_OK and isinstance(image_input, PILImage.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, (str, Path)):
            if not _PIL_OK:
                raise RuntimeError("Pillow required to load images.")
            img = PILImage.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            from io import BytesIO
            if not _PIL_OK:
                raise RuntimeError("Pillow required to decode bytes.")
            img = PILImage.open(BytesIO(image_input)).convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        # Grayscale SAR → 3-channel
        if img.mode != "RGB":
            img = img.convert("RGB")

        tensor = self._transform(img).unsqueeze(0)  # (1, 3, H, W)
        return tensor

    def predict(
        self, image_input, return_probabilities: bool = False
    ) -> Union[int, Tuple[int, np.ndarray]]:
        """
        Classify a single image patch.

        Parameters
        ----------
        image_input : np.ndarray | PIL.Image | str | bytes
        return_probabilities : bool
            If True, also return softmax probabilities.

        Returns
        -------
        int
            Predicted class index (0=Oil Spill, 1=Look-Alike).
        (int, np.ndarray)
            When return_probabilities=True; probs shape (2,).
        """
        if not self._loaded or self._model is None:
            logger.warning("Classifier not loaded; returning stub result.")
            probs = np.array([0.5, 0.5], dtype=np.float32)
            pred = 0
            return (pred, probs) if return_probabilities else pred

        tensor = self._preprocess(image_input).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)  # (1, 2)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (2,)

        pred = int(probs[0] >= self.threshold)  # 0=Oil if above threshold
        # Flip: class 0 is Oil Spill; threshold on oil-spill probability
        pred = 0 if probs[0] >= self.threshold else 1

        return (pred, probs) if return_probabilities else pred

    def predict_batch(
        self, images: List, batch_size: int = 16
    ) -> List[Dict]:
        """
        Classify a list of image inputs.

        Returns
        -------
        List[dict]
            Each dict has keys: 'class_idx', 'class_name', 'probabilities'.
        """
        results = []
        for img in images:
            class_idx, probs = self.predict(img, return_probabilities=True)
            results.append({
                "class_idx": class_idx,
                "class_name": CLASS_NAMES[class_idx],
                "probabilities": {
                    "oil_spill": float(probs[0]),
                    "look_alike": float(probs[1]),
                },
            })
        return results

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded (requires GlobalOSD-SAR training)"
        return f"LookAlikeClassifier(status={status!r}, threshold={self.threshold})"
