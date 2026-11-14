"""
OceanTrace AI — Evaluation Metrics
Smart India Hackathon 2026 | SIH26143

Provides pixel-level segmentation metrics and reporting utilities.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Matplotlib / seaborn — optional for headless environments
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def compute_pixel_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_index: int = 255,
) -> float:
    """
    Fraction of correctly classified pixels (ignoring `ignore_index`).

    Args:
        pred:   (H, W) int array of predicted class indices.
        target: (H, W) int array of ground-truth class indices.
        ignore_index: label value to exclude.

    Returns:
        Pixel accuracy in [0, 1].
    """
    valid = target != ignore_index
    if valid.sum() == 0:
        return 0.0
    return float((pred[valid] == target[valid]).sum()) / float(valid.sum())


def compute_iou_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """
    Per-class Intersection-over-Union.

    Args:
        pred:        (H, W) int32 predicted class indices.
        target:      (H, W) int32 ground-truth class indices.
        num_classes: Number of semantic classes.
        ignore_index: Pixels to skip.

    Returns:
        (num_classes,) float array; NaN for classes absent from both maps.
    """
    valid = target != ignore_index
    pred_v = pred[valid]
    target_v = target[valid]

    iou = np.full(num_classes, np.nan, dtype=np.float64)
    for cls in range(num_classes):
        tp = np.logical_and(pred_v == cls, target_v == cls).sum()
        fp = np.logical_and(pred_v == cls, target_v != cls).sum()
        fn = np.logical_and(pred_v != cls, target_v == cls).sum()
        denom = tp + fp + fn
        if denom > 0:
            iou[cls] = tp / denom
    return iou


def compute_mean_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Mean IoU over classes present in ground truth (NaN classes excluded)."""
    iou = compute_iou_per_class(pred, target, num_classes, ignore_index)
    valid_iou = iou[~np.isnan(iou)]
    if len(valid_iou) == 0:
        return 0.0
    return float(valid_iou.mean())


def compute_precision_recall_f1(
    pred: np.ndarray,
    target: np.ndarray,
    class_idx: int,
    ignore_index: int = 255,
) -> Tuple[float, float, float]:
    """
    Precision, Recall, F1 for a single class (binary view).

    Returns:
        (precision, recall, f1) each in [0, 1].
    """
    valid = target != ignore_index
    pred_v = pred[valid]
    target_v = target[valid]

    tp = np.logical_and(pred_v == class_idx, target_v == class_idx).sum()
    fp = np.logical_and(pred_v == class_idx, target_v != class_idx).sum()
    fn = np.logical_and(pred_v != class_idx, target_v == class_idx).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def compute_dice(
    pred: np.ndarray,
    target: np.ndarray,
    class_idx: int,
    ignore_index: int = 255,
    smooth: float = 1e-6,
) -> float:
    """Dice coefficient for a single class."""
    valid = target != ignore_index
    pred_v = (pred[valid] == class_idx).astype(np.float64)
    target_v = (target[valid] == class_idx).astype(np.float64)
    intersection = (pred_v * target_v).sum()
    return float(
        (2.0 * intersection + smooth) / (pred_v.sum() + target_v.sum() + smooth)
    )


# ---------------------------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Return (num_classes, num_classes) confusion matrix."""
    valid = target != ignore_index
    pred_v = pred[valid].astype(np.int64)
    target_v = target[valid].astype(np.int64)

    # Clamp to valid range to avoid index errors on bad predictions
    pred_v = np.clip(pred_v, 0, num_classes - 1)
    target_v = np.clip(target_v, 0, num_classes - 1)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (target_v, pred_v), 1)
    return cm


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------


class EvaluationReport:
    """
    Accumulates predictions over a full dataset and produces a final report.

    Usage::

        report = EvaluationReport(num_classes=5, class_names=[...])
        for pred, target in zip(preds, targets):
            report.update(pred, target)
        report.generate()
        report.print_summary()
        report.save_json("metrics.json")
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.ignore_index = ignore_index

        self._cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        self._total_pixels = 0
        self._correct_pixels = 0
        self._generated = False
        self.results: Dict = {}

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Add one image pair."""
        self._cm += compute_confusion_matrix(
            pred, target, self.num_classes, self.ignore_index
        )
        valid = target != self.ignore_index
        self._total_pixels += int(valid.sum())
        self._correct_pixels += int((pred[valid] == target[valid]).sum())
        self._generated = False

    def generate(self) -> Dict:
        """Compute all metrics from accumulated state."""
        cm = self._cm.astype(np.float64)
        num_classes = self.num_classes

        # Pixel accuracy
        pixel_acc = (
            self._correct_pixels / self._total_pixels
            if self._total_pixels > 0
            else 0.0
        )

        # Per-class IoU from confusion matrix
        iou_per_class = []
        for c in range(num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = tp + fp + fn
            iou_per_class.append(tp / denom if denom > 0 else float("nan"))

        iou_arr = np.array(iou_per_class)
        valid_iou = iou_arr[~np.isnan(iou_arr)]
        mean_iou = float(valid_iou.mean()) if len(valid_iou) > 0 else 0.0

        # Per-class precision, recall, f1, dice
        per_class = {}
        for c, name in enumerate(self.class_names):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            )
            dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            per_class[name] = {
                "iou": float(iou_arr[c]) if not np.isnan(iou_arr[c]) else None,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "dice": float(dice),
            }

        self.results = {
            "pixel_accuracy": float(pixel_acc),
            "mean_iou": float(mean_iou),
            "per_class": per_class,
            "confusion_matrix": self._cm.tolist(),
        }
        self._generated = True
        return self.results

    def save_json(self, path: Union[str, Path]) -> None:
        """Persist results to JSON file."""
        if not self._generated:
            self.generate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.results, fh, indent=2)

    def print_summary(self) -> None:
        """Print a human-readable metrics table."""
        if not self._generated:
            self.generate()
        r = self.results
        print("\n" + "=" * 60)
        print("OceanTrace AI — Evaluation Report")
        print("=" * 60)
        print(f"  Pixel Accuracy : {r['pixel_accuracy']:.4f}")
        print(f"  Mean IoU       : {r['mean_iou']:.4f}")
        print("-" * 60)
        print(f"  {'Class':<20} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Dice':>8}")
        print("-" * 60)
        for name, m in r["per_class"].items():
            iou_str = f"{m['iou']:.4f}" if m["iou"] is not None else "  N/A "
            print(
                f"  {name:<20} {iou_str:>8} {m['precision']:>8.4f}"
                f" {m['recall']:>8.4f} {m['f1']:>8.4f} {m['dice']:>8.4f}"
            )
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path],
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        cm:          (C, C) confusion matrix (counts).
        class_names: List of class name strings.
        save_path:   File path to save the figure (PNG/PDF).
        normalize:   If True, normalize rows to show recall per class.
        title:       Figure title.
    """
    if not _PLOT_AVAILABLE:
        print("[metrics] matplotlib/seaborn not available; skipping plot.")
        return

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt = ".2f"
    else:
        cm_plot = cm.astype(float)
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[metrics] Confusion matrix saved → {save_path}")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Union[str, Path],
    title: str = "Training Curves",
) -> None:
    """
    Plot training / validation curves.

    Args:
        history:   Dict mapping metric names to lists of values per epoch.
                   Expected keys: 'train_loss', 'val_loss', 'mean_iou', 'oil_spill_iou'.
        save_path: File path to save the figure.
        title:     Figure super-title.
    """
    if not _PLOT_AVAILABLE:
        print("[metrics] matplotlib not available; skipping plot.")
        return

    pairs = [
        ("train_loss", "val_loss", "Loss", True),
        ("mean_iou", "oil_spill_iou", "IoU", False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=15)

    for ax, (k1, k2, ylabel, is_loss) in zip(axes, pairs):
        epochs = range(1, max(len(history.get(k1, [])), len(history.get(k2, []))) + 1)
        if k1 in history:
            ax.plot(epochs, history[k1], label=k1.replace("_", " ").title(), linewidth=2)
        if k2 in history and history[k2]:
            ax.plot(
                range(1, len(history[k2]) + 1),
                history[k2],
                label=k2.replace("_", " ").title(),
                linewidth=2,
                linestyle="--",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[metrics] Training curves saved → {save_path}")
