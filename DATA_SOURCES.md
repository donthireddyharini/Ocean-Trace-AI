# OceanTrace AI — Data Sources & Download Procedures

This document describes how to obtain each dataset used by OceanTrace AI.
See `DATASET_CATALOG.md` for schema details and licensing information.

---

## 1. M4D Oil Spill Detection Dataset (Manual)

**URL:** https://m4d.iti.gr/oil-spill-detection-dataset/

### Steps
1. Visit the URL above in a browser.
2. Fill in the dataset request form (academic affiliation, intended use).
3. Wait for approval email (typically 1–3 business days).
4. Download the provided archive and extract it.
5. Place the extracted contents under `data/raw/m4d_oil_spill/`:
   ```
   data/raw/m4d_oil_spill/
   ├── images/
   │   ├── 00001.png
   │   └── ...
   └── masks/
       ├── 00001.png
       └── ...
   ```

### Expected directory structure after extraction
- `images/` — SAR PNG images (single channel, 8-bit)
- `masks/` — Segmentation masks (same filenames; pixel values 0–4)

---

## 2. GlobalOSD-SAR — Zenodo 15286918 (Auto-download)

**URL:** https://doi.org/10.5281/zenodo.15286918

### Automatic
Run the download script:
```bash
python ml/datasets/dataset_download.py --dataset global_osd_sar
```

### Manual fallback
1. Visit https://zenodo.org/records/15286918
2. Click **Download** on the ZIP archive.
3. Extract to `data/raw/global_osd_sar/`.

---

## 3. Sentinel-1 SAR Imagery (Manual / sentinelsat)

**URL:** https://dataspace.copernicus.eu/

### Steps
1. Register for a free account at https://dataspace.copernicus.eu/
2. Install `sentinelsat`:
   ```bash
   pip install sentinelsat
   ```
3. Use the Copernicus Browser or the CLI to search and download:
   ```bash
   sentinelsat \
     --user YOUR_USER --password YOUR_PASS \
     --geometry "POLYGON((...))" \
     --producttype GRD \
     --platform S1 \
     --start 20230101 --end 20231231 \
     --download --path data/raw/sentinel1/
   ```
4. Pre-process with ESA SNAP or `pyroSAR` (calibration → speckle filter → terrain correction).

### Preprocessing pipeline (recommended)
```
Raw SAFE → Calibration (sigma0) → Lee Speckle Filter → Range-Doppler TC → Export GeoTIFF
```

---

## 4. Brest AIS Dataset — Zenodo 2563256 (Auto-download)

**URL:** https://zenodo.org/records/2563256

### Automatic
```bash
python ml/datasets/dataset_download.py --dataset ais_brest
```

### Manual fallback
1. Visit https://zenodo.org/records/2563256
2. Download the CSV file(s).
3. Place under `data/raw/ais_brest/`.

---

## 5. Antwerp AIS Dataset (Manual)

**URL:** https://data.mendeley.com/datasets/smb56stcdx/1

### Steps
1. Visit the URL above.
2. Click **Download All** (or select individual CSV files).
3. Extract to `data/raw/ais_antwerp/`.

---

## 6. HYCOM Ocean Currents (OPeNDAP / Auto-query)

**URL:** https://tds.hycom.org/thredds/dodsC/

### Python access example
```python
import xarray as xr

url = (
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z"
    "?water_u[0:1:0][0:1:0][0:1:400][0:1:400]"
    "&water_v[0:1:0][0:1:0][0:1:400][0:1:400]"
)
ds = xr.open_dataset(url)
ds.to_netcdf("data/raw/hycom/currents_sample.nc")
```

### Notes
- No registration required.
- Query only the region and time range needed to avoid large downloads.
- Variable list: https://www.hycom.org/dataserver/gofs-3pt1/analysis

---

## 7. ERA5 Wind Data (CDS API)

**URL:** https://cds.climate.copernicus.eu/

### Setup
1. Register at https://cds.climate.copernicus.eu/
2. After login, copy your API key from https://cds.climate.copernicus.eu/user/
3. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_UID:YOUR_API_KEY
   ```
4. Install `cdsapi`:
   ```bash
   pip install cdsapi
   ```

### Download example
```python
import cdsapi

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
        'year': '2023',
        'month': ['01', '02', '03'],
        'day': [f'{d:02d}' for d in range(1, 32)],
        'time': [f'{h:02d}:00' for h in range(24)],
        'area': [60, -10, 30, 40],  # North, West, South, East
        'format': 'netcdf',
    },
    'data/raw/era5/wind_2023_q1.nc'
)
```

---

## Quick-start: download all auto-downloadable datasets

```bash
python ml/datasets/dataset_download.py --all
```

This will:
1. Download Brest AIS (Zenodo 2563256)
2. Download GlobalOSD-SAR (Zenodo 15286918)
3. Query a small HYCOM sample for the Mediterranean
4. Print instructions for the manual datasets
5. Create `data/dataset_inventory.csv`
