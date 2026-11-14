"""
ml/datasets/dataset_download.py
================================
OceanTrace AI — Dataset acquisition script.

Usage:
    python ml/datasets/dataset_download.py --all
    python ml/datasets/dataset_download.py --dataset ais_brest
    python ml/datasets/dataset_download.py --dataset global_osd_sar
    python ml/datasets/dataset_download.py --dataset hycom_sample
    python ml/datasets/dataset_download.py --info   # print manual-download instructions
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("ERROR: Install requests and tqdm first:  pip install requests tqdm")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INVENTORY_FILE = DATA_DIR / "dataset_inventory.csv"

INVENTORY_FIELDS = [
    "dataset", "local_path", "status", "size_bytes",
    "file_count", "checked_at", "notes",
]


# ---------------------------------------------------------------------------
# Helper: HTTP download with progress bar
# ---------------------------------------------------------------------------
def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        desc=dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))


def _extract_zip(src: Path, dest_dir: Path) -> None:
    print(f"  Extracting {src.name} → {dest_dir}")
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(dest_dir)
    src.unlink()  # remove archive after extraction


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
MANUAL_DATASETS = {
    "m4d_oil_spill": {
        "description": "M4D Oil Spill Detection Dataset (SAR segmentation)",
        "local_path": RAW_DIR / "m4d_oil_spill",
        "steps": [
            "1. Visit https://m4d.iti.gr/oil-spill-detection-dataset/",
            "2. Fill in the academic request form.",
            "3. Wait for approval email (1–3 business days).",
            "4. Download and extract to:  data/raw/m4d_oil_spill/",
            "   Expected structure:",
            "     data/raw/m4d_oil_spill/images/*.png",
            "     data/raw/m4d_oil_spill/masks/*.png",
        ],
    },
    "sentinel1": {
        "description": "Sentinel-1 SAR GRD Imagery (Copernicus Data Space)",
        "local_path": RAW_DIR / "sentinel1",
        "steps": [
            "1. Register at https://dataspace.copernicus.eu/",
            "2. pip install sentinelsat",
            "3. sentinelsat --user U --password P --geometry 'POLYGON((...))' "
            "--producttype GRD --platform S1 --download --path data/raw/sentinel1/",
            "4. Pre-process (calibration → speckle filter → terrain correction) "
            "using ESA SNAP or pyroSAR.",
        ],
    },
    "ais_antwerp": {
        "description": "Antwerp AIS Dataset (Mendeley Data)",
        "local_path": RAW_DIR / "ais_antwerp",
        "steps": [
            "1. Visit https://data.mendeley.com/datasets/smb56stcdx/1",
            "2. Click 'Download All'.",
            "3. Extract CSV files to:  data/raw/ais_antwerp/",
        ],
    },
    "era5": {
        "description": "ERA5 Wind Data (Copernicus CDS)",
        "local_path": RAW_DIR / "era5",
        "steps": [
            "1. Register at https://cds.climate.copernicus.eu/",
            "2. Create ~/.cdsapirc with your API key.",
            "3. pip install cdsapi",
            "4. Run:  python ml/datasets/era5_download.py  (or see DATA_SOURCES.md)",
        ],
    },
}


# ---------------------------------------------------------------------------
# Auto-downloadable datasets
# ---------------------------------------------------------------------------
def download_ais_brest() -> dict:
    """Download Brest AIS dataset from Zenodo record 2563256."""
    dest_dir = RAW_DIR / "ais_brest"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Zenodo API to get file list
    api_url = "https://zenodo.org/api/records/2563256"
    print(f"\n[ais_brest] Fetching Zenodo record metadata …")
    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        record = resp.json()
    except Exception as e:
        return {
            "dataset": "ais_brest",
            "status": f"ERROR: {e}",
            "local_path": str(dest_dir),
            "size_bytes": 0,
            "file_count": 0,
        }

    files = record.get("files", [])
    total_bytes = 0
    downloaded = 0
    for f in files:
        fname = f["key"]
        url = f["links"]["self"]
        dest = dest_dir / fname
        if dest.exists():
            print(f"  [skip] {fname} already exists.")
            total_bytes += dest.stat().st_size
            downloaded += 1
            continue
        print(f"  Downloading {fname} …")
        _download_file(url, dest)
        if fname.endswith(".zip"):
            _extract_zip(dest, dest_dir)
        total_bytes += dest.stat().st_size if dest.exists() else 0
        downloaded += 1

    file_count = len(list(dest_dir.rglob("*.csv")))
    return {
        "dataset": "ais_brest",
        "status": "ok" if downloaded > 0 else "empty",
        "local_path": str(dest_dir),
        "size_bytes": total_bytes,
        "file_count": file_count,
        "notes": f"Zenodo 2563256, {downloaded} archive(s) processed",
    }


def download_global_osd_sar() -> dict:
    """Download GlobalOSD-SAR from Zenodo record 15286918."""
    dest_dir = RAW_DIR / "global_osd_sar"
    dest_dir.mkdir(parents=True, exist_ok=True)

    api_url = "https://zenodo.org/api/records/15286918"
    print(f"\n[global_osd_sar] Fetching Zenodo record metadata …")
    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        record = resp.json()
    except Exception as e:
        return {
            "dataset": "global_osd_sar",
            "status": f"ERROR: {e}",
            "local_path": str(dest_dir),
            "size_bytes": 0,
            "file_count": 0,
        }

    files = record.get("files", [])
    total_bytes = 0
    downloaded = 0
    for f in files:
        fname = f["key"]
        url = f["links"]["self"]
        dest = dest_dir / fname
        if dest.exists():
            print(f"  [skip] {fname} already exists.")
            total_bytes += dest.stat().st_size
            downloaded += 1
            continue
        print(f"  Downloading {fname} …")
        _download_file(url, dest)
        if fname.endswith(".zip"):
            _extract_zip(dest, dest_dir)
        total_bytes += dest.stat().st_size if dest.exists() else 0
        downloaded += 1

    file_count = sum(1 for _ in dest_dir.rglob("*.png")) + sum(
        1 for _ in dest_dir.rglob("*.csv")
    )
    return {
        "dataset": "global_osd_sar",
        "status": "ok" if downloaded > 0 else "empty",
        "local_path": str(dest_dir),
        "size_bytes": total_bytes,
        "file_count": file_count,
        "notes": f"Zenodo 15286918, {downloaded} archive(s) processed",
    }


def download_hycom_sample() -> dict:
    """
    Fetch a small HYCOM ocean-current sample via OPeNDAP / HTTP.
    Uses the publicly accessible GLBv0.08 experimental dataset.
    """
    try:
        import xarray as xr  # type: ignore
    except ImportError:
        return {
            "dataset": "hycom_sample",
            "status": "ERROR: xarray not installed (pip install xarray netCDF4)",
            "local_path": str(RAW_DIR / "hycom"),
            "size_bytes": 0,
            "file_count": 0,
        }

    dest_dir = RAW_DIR / "hycom"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_file = dest_dir / "currents_med_sample.nc"

    if out_file.exists():
        print(f"  [skip] {out_file.name} already exists.")
        return {
            "dataset": "hycom_sample",
            "status": "ok (cached)",
            "local_path": str(dest_dir),
            "size_bytes": out_file.stat().st_size,
            "file_count": 1,
        }

    # Small subsection: Mediterranean, surface layer, single time step
    url = (
        "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z"
        "?water_u[0:1:0][0:1:0][400:1:450][500:1:600]"
        "&water_v[0:1:0][0:1:0][400:1:450][500:1:600]"
        "&lon[500:1:600]&lat[400:1:450]&time[0:1:0]"
    )
    print(f"\n[hycom_sample] Fetching via OPeNDAP …")
    try:
        ds = xr.open_dataset(url)
        ds.to_netcdf(out_file)
        ds.close()
    except Exception as e:
        return {
            "dataset": "hycom_sample",
            "status": f"ERROR: {e}",
            "local_path": str(dest_dir),
            "size_bytes": 0,
            "file_count": 0,
            "notes": "HYCOM OPeNDAP may have downtime; retry later.",
        }

    return {
        "dataset": "hycom_sample",
        "status": "ok",
        "local_path": str(dest_dir),
        "size_bytes": out_file.stat().st_size,
        "file_count": 1,
        "notes": "Mediterranean surface currents, single time step",
    }


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
def _check_local(dataset_key: str, local_path: Path) -> dict:
    """Return inventory row for a local dataset directory."""
    if not local_path.exists():
        return {
            "dataset": dataset_key,
            "status": "missing",
            "local_path": str(local_path),
            "size_bytes": 0,
            "file_count": 0,
        }
    files = list(local_path.rglob("*"))
    size = sum(f.stat().st_size for f in files if f.is_file())
    return {
        "dataset": dataset_key,
        "status": "present",
        "local_path": str(local_path),
        "size_bytes": size,
        "file_count": sum(1 for f in files if f.is_file()),
    }


def write_inventory(rows: list[dict]) -> None:
    INVENTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with open(INVENTORY_FILE, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INVENTORY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row.setdefault("checked_at", now)
            row.setdefault("notes", "")
            writer.writerow(row)
    print(f"\nInventory saved → {INVENTORY_FILE}")


def print_manual_instructions() -> None:
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    for key, info in MANUAL_DATASETS.items():
        status = "PRESENT ✓" if info["local_path"].exists() else "MISSING ✗"
        print(f"\n[{key}] {info['description']}  [{status}]")
        for step in info["steps"]:
            print(f"  {step}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="OceanTrace AI dataset downloader")
    parser.add_argument(
        "--dataset",
        choices=["ais_brest", "global_osd_sar", "hycom_sample", "all"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all auto-downloadable datasets"
    )
    parser.add_argument(
        "--info", action="store_true", help="Print manual download instructions"
    )
    args = parser.parse_args()

    if args.info:
        print_manual_instructions()
        return

    rows: list[dict] = []

    # Always inventory manual datasets
    for key, info in MANUAL_DATASETS.items():
        rows.append(_check_local(key, info["local_path"]))

    download_all = args.all or (args.dataset == "all")

    if download_all or args.dataset == "ais_brest":
        rows.append(download_ais_brest())
    if download_all or args.dataset == "global_osd_sar":
        rows.append(download_global_osd_sar())
    if download_all or args.dataset == "hycom_sample":
        rows.append(download_hycom_sample())

    write_inventory(rows)
    print_manual_instructions()

    print("\nDone.")


if __name__ == "__main__":
    main()
