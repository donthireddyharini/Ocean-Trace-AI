"""
OceanTrace AI — Segmentation Loss Functions
Smart India Hackathon 2026 | SIH26143

Implements:
  - WeightedCrossEntropyLoss
  - DiceLoss (multi-class)
  - CombinedSegLoss (CE + Dice)
  - FocalLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with per-class weights for class imbalance."""

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight_list = weight
        # Actual tensor assigned in forward (device-agnostic)
        self._weight_tensor: Optional[torch.Tensor] = None

    def _get_weight(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.weight_list is None:
            return None
        if self._weight_tensor is None or self._weight_tensor.device != device:
            self._weight_tensor = torch.tensor(
                self.weight_list, dtype=torch.float32, device=device
            )
        return self._weight_tensor

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) raw class scores
            targets: (B, H, W)   long integer class labels
        Returns:
            scalar loss
        """
        weight = self._get_weight(logits.device)
        return F.cross_entropy(
            logits,
            targets.long(),
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.

    Dice = 1 - (2 * |P ∩ T| + ε) / (|P| + |T| + ε)

    Averaged over present classes (ignores ignore_index pixels).
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        smooth: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W)
            targets: (B, H, W) long
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Build valid-pixel mask
        valid_mask = (targets != self.ignore_index)  # (B, H, W) bool

        # One-hot encode targets; replace ignore pixels with 0
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0
        one_hot = F.one_hot(targets_clamped.long(), self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Zero out ignored pixels in one_hot and probs
        valid = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        one_hot = one_hot * valid
        probs_valid = probs * valid

        # Compute per-class dice
        dims = (0, 2, 3)
        intersection = (probs_valid * one_hot).sum(dim=dims)  # (C,)
        cardinality = probs_valid.sum(dim=dims) + one_hot.sum(dim=dims)  # (C,)

        dice_per_class = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )
        loss_per_class = 1.0 - dice_per_class

        if self.reduction == "mean":
            return loss_per_class.mean()
        elif self.reduction == "sum":
            return loss_per_class.sum()
        else:
            return loss_per_class


# ---------------------------------------------------------------------------


class CombinedSegLoss(nn.Module):
    """
    Combined segmentation loss:  λ_ce * CE  +  λ_dice * Dice

    Default λ_ce=0.5, λ_dice=0.5.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.ce_loss = WeightedCrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index
        )
        self.dice_loss = DiceLoss(
            num_classes=num_classes, ignore_index=ignore_index
        )
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Addresses class imbalance by down-weighting easy examples.
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha_list = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self._alpha_tensor: Optional[torch.Tensor] = None

    def _get_alpha(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.alpha_list is None:
            return None
        if self._alpha_tensor is None or self._alpha_tensor.device != device:
            self._alpha_tensor = torch.tensor(
                self.alpha_list, dtype=torch.float32, device=device
            )
        return self._alpha_tensor

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W)
            targets: (B, H, W) long
        """
        B, C, H, W = logits.shape
        alpha = self._get_alpha(logits.device)

        # Compute log-softmax and softmax
        log_probs = F.log_softmax(logits, dim=1)  # (B, C, H, W)
        probs = torch.exp(log_probs)  # (B, C, H, W)

        # Reshape for gather
        log_probs_flat = log_probs.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
        targets_flat = targets.reshape(-1)  # (N,)

        valid_mask = targets_flat != self.ignore_index
        targets_valid = targets_flat.clone()
        targets_valid[~valid_mask] = 0

        # Gather class-specific values
        log_pt = log_probs_flat.gather(1, targets_valid.unsqueeze(1)).squeeze(1)  # (N,)
        pt = probs_flat.gather(1, targets_valid.unsqueeze(1)).squeeze(1)  # (N,)

        focal_weight = (1.0 - pt) ** self.gamma

        if alpha is not None:
            alpha_t = alpha[targets_valid]
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt
        loss = loss * valid_mask.float()

        if self.reduction == "mean":
            return loss.sum() / valid_mask.float().sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
