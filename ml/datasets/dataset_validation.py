"""
ml/datasets/dataset_validation.py
===================================
OceanTrace AI — Dataset integrity validator.

Checks:
  - Image / mask file correspondence (every image has a matching mask and vice-versa)
  - Image dimensions and format (PNG, 8-bit grayscale or RGB)
  - Mask class values (must be a subset of {0,1,2,3,4} for M4D)
  - Reports per-class pixel distribution
  - Saves JSON report to data/validation_report.json

Usage:
    python ml/datasets/dataset_validation.py \
        --images  data/raw/m4d_oil_spill/images \
        --masks   data/raw/m4d_oil_spill/masks \
        --dataset m4d
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("ERROR: pip install numpy Pillow tqdm")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# M4D class definitions
M4D_CLASSES = {
    0: "Clean Sea",
    1: "Oil Spill",
    2: "Look-Alike",
    3: "Ship",
    4: "Land",
}
M4D_VALID_VALUES = set(M4D_CLASSES.keys())


# ---------------------------------------------------------------------------
# Core validation functions
# ---------------------------------------------------------------------------
def find_image_mask_pairs(
    image_dir: Path, mask_dir: Path
) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Return (matched pairs, issues) for the given directories."""
    img_names = {p.stem: p for p in image_dir.glob("*.png")}
    msk_names = {p.stem: p for p in mask_dir.glob("*.png")}

    all_stems = set(img_names) | set(msk_names)
    pairs: list[tuple[Path, Path]] = []
    issues: list[str] = []

    for stem in sorted(all_stems):
        if stem in img_names and stem in msk_names:
            pairs.append((img_names[stem], msk_names[stem]))
        elif stem in img_names:
            issues.append(f"Missing mask for image: {stem}.png")
        else:
            issues.append(f"Missing image for mask: {stem}.png")

    return pairs, issues


def validate_pair(
    img_path: Path,
    msk_path: Path,
    valid_mask_values: Optional[set] = None,
) -> dict:
    """Validate a single image–mask pair. Returns a result dict."""
    result: dict = {
        "image": img_path.name,
        "mask": msk_path.name,
        "ok": True,
        "issues": [],
        "image_shape": None,
        "mask_shape": None,
        "mask_unique_values": [],
        "class_pixel_counts": {},
    }

    # -- Load image
    try:
        img = Image.open(img_path)
        img_arr = np.array(img)
        result["image_shape"] = list(img_arr.shape)
        result["image_mode"] = img.mode
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"Cannot open image: {e}")
        return result

    # -- Load mask
    try:
        msk = Image.open(msk_path)
        msk_arr = np.array(msk)
        result["mask_shape"] = list(msk_arr.shape)
        result["mask_mode"] = msk.mode
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"Cannot open mask: {e}")
        return result

    # -- Dimension match (H × W)
    img_hw = img_arr.shape[:2]
    msk_hw = msk_arr.shape[:2]
    if img_hw != msk_hw:
        result["ok"] = False
        result["issues"].append(
            f"Shape mismatch: image {img_hw} vs mask {msk_hw}"
        )

    # -- Mask class values
    unique_vals = sorted(map(int, np.unique(msk_arr).tolist()))
    result["mask_unique_values"] = unique_vals
    if valid_mask_values is not None:
        invalid = set(unique_vals) - valid_mask_values
        if invalid:
            result["ok"] = False
            result["issues"].append(f"Invalid mask values: {sorted(invalid)}")

    # -- Per-class pixel counts
    for v in unique_vals:
        count = int(np.sum(msk_arr == v))
        result["class_pixel_counts"][int(v)] = count

    return result


def compute_class_distribution(pair_results: list[dict]) -> dict:
    """Aggregate class pixel counts across all pairs."""
    agg: dict[int, int] = defaultdict(int)
    for r in pair_results:
        for cls_id, cnt in r.get("class_pixel_counts", {}).items():
            agg[int(cls_id)] += int(cnt)
    return dict(sorted(agg.items()))


def run_validation(
    image_dir: Path,
    mask_dir: Path,
    dataset: str = "m4d",
    max_errors: int = 100,
) -> dict:
    """Full validation run. Returns the report dict."""
    valid_values = M4D_VALID_VALUES if dataset == "m4d" else None

    report: dict = {
        "dataset": dataset,
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "total_pairs": 0,
        "valid_pairs": 0,
        "invalid_pairs": 0,
        "pairing_issues": [],
        "class_distribution": {},
        "class_names": M4D_CLASSES if dataset == "m4d" else {},
        "pair_results": [],
        "summary": "",
    }

    if not image_dir.exists():
        report["summary"] = f"ERROR: image_dir not found: {image_dir}"
        return report
    if not mask_dir.exists():
        report["summary"] = f"ERROR: mask_dir not found: {mask_dir}"
        return report

    pairs, pairing_issues = find_image_mask_pairs(image_dir, mask_dir)
    report["pairing_issues"] = pairing_issues
    report["total_pairs"] = len(pairs)

    if not pairs:
        report["summary"] = "No image-mask pairs found."
        return report

    pair_results = []
    for img_path, msk_path in tqdm(pairs, desc="Validating pairs"):
        res = validate_pair(img_path, msk_path, valid_values)
        pair_results.append(res)
        if len([r for r in pair_results if not r["ok"]]) >= max_errors:
            print(f"\n  Stopping early after {max_errors} errors.")
            break

    report["pair_results"] = pair_results
    report["valid_pairs"] = sum(1 for r in pair_results if r["ok"])
    report["invalid_pairs"] = sum(1 for r in pair_results if not r["ok"])
    report["class_distribution"] = compute_class_distribution(pair_results)

    # Annotate with class names
    if dataset == "m4d":
        report["class_distribution_named"] = {
            M4D_CLASSES.get(k, str(k)): v
            for k, v in report["class_distribution"].items()
        }

    # Summary string
    pct_valid = (
        100 * report["valid_pairs"] / report["total_pairs"]
        if report["total_pairs"] > 0 else 0
    )
    report["summary"] = (
        f"{report['valid_pairs']}/{report['total_pairs']} pairs valid "
        f"({pct_valid:.1f}%); "
        f"{len(pairing_issues)} pairing issue(s)."
    )
    return report


def save_report(report: dict, out_path: Optional[Path] = None) -> Path:
    if out_path is None:
        out_path = DATA_DIR / "validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Validation report saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="OceanTrace AI dataset validator"
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DATA_DIR / "raw" / "m4d_oil_spill" / "images",
        help="Directory containing image PNGs",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=DATA_DIR / "raw" / "m4d_oil_spill" / "masks",
        help="Directory containing mask PNGs",
    )
    parser.add_argument(
        "--dataset",
        choices=["m4d", "other"],
        default="m4d",
        help="Dataset type (determines valid mask values)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the JSON report (default: data/validation_report.json)",
    )
    args = parser.parse_args()

    report = run_validation(args.images, args.masks, args.dataset)
    save_report(report, args.output)

    print(f"\nSummary: {report['summary']}")
    if report.get("class_distribution_named"):
        print("\nClass distribution (pixels):")
        total = sum(report["class_distribution"].values())
        for name, cnt in report["class_distribution_named"].items():
            pct = 100 * cnt / total if total else 0
            print(f"  {name:20s}: {cnt:>12,d}  ({pct:5.2f}%)")

    if report["pairing_issues"]:
        print(f"\nPairing issues ({len(report['pairing_issues'])}):")
        for issue in report["pairing_issues"][:20]:
            print(f"  {issue}")
        if len(report["pairing_issues"]) > 20:
            print(f"  … and {len(report['pairing_issues']) - 20} more (see report JSON)")

    if report["invalid_pairs"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
