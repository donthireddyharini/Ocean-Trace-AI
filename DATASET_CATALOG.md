# OceanTrace AI — Dataset Catalog

> [!IMPORTANT]
> Some datasets require manual registration/download. See `DATA_SOURCES.md` for step-by-step download instructions.

---

## 1. M4D Oil Spill Detection Dataset

| Field | Details |
|---|---|
| **Source** | https://m4d.iti.gr/oil-spill-detection-dataset/ |
| **License** | Academic / Research use only |
| **Format** | SAR PNG images + segmentation mask PNGs |
| **Local path** | `data/raw/m4d_oil_spill/` |

### Classes
| Value | Label | Description |
|---|---|---|
| 0 | Clean Sea | Background (open water, no contamination) |
| 1 | Oil Spill | Confirmed oil spill pixels |
| 2 | Look-Alike | Ambiguous dark patches (bio-film, low wind, etc.) |
| 3 | Ship | Vessel returns |
| 4 | Land | Land pixels (should be excluded from evaluation) |

### Coverage
- Mediterranean Sea
- North Sea
- Various global coastal regions

### Notes
- **Primary training dataset** for semantic segmentation model.
- Manual download required from the M4D website (account/request form).
- Images are single-channel SAR (grayscale), stored as 8-bit PNGs.
- Mask files share the same filename as images, located in a parallel `masks/` directory.
- Approximately 1200 image–mask pairs.

---

## 2. GlobalOSD-SAR (Zenodo 15286918)

| Field | Details |
|---|---|
| **Source** | https://doi.org/10.5281/zenodo.15286918 |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Format** | SAR image patches + classification labels (CSV) |
| **Local path** | `data/raw/global_osd_sar/` |

### Classes
| Label | Description |
|---|---|
| `oil_slick` | Confirmed oil slick patch |
| `look_alike` | Dark formation look-alike (not oil) |

### Notes
- Classification-only dataset — **no pixel-level segmentation masks**.
- Useful for pre-training or fine-tuning a binary classification head.
- Downloaded automatically via Zenodo API (see `ml/datasets/dataset_download.py`).
- Patches are 256×256 or 512×512 SAR image crops.

---

## 3. Sentinel-1 SAR Imagery

| Field | Details |
|---|---|
| **Source** | https://dataspace.copernicus.eu/ |
| **License** | Copernicus free and open data policy |
| **Format** | GeoTIFF / Sentinel SAFE format |
| **Local path** | `data/raw/sentinel1/` |

### Notes
- Requires free registration at Copernicus Data Space.
- Products of interest: GRD (Ground Range Detected), IW (Interferometric Wide) subswath.
- Use `sentinelsat` or the Copernicus Browser to search and download.
- Real satellite imagery — apply preprocessing (calibration, speckle filtering, terrain correction) before model inference.
- VV polarization preferred for oil spill detection.

---

## 4. Brest AIS Dataset

| Field | Details |
|---|---|
| **Source** | https://zenodo.org/records/2563256 |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Format** | CSV |
| **Local path** | `data/raw/ais_brest/` |

### Schema
| Column | Type | Description |
|---|---|---|
| MMSI | int | Maritime Mobile Service Identity |
| BaseDateTime | datetime | UTC timestamp |
| LAT | float | Latitude (decimal degrees) |
| LON | float | Longitude (decimal degrees) |
| SOG | float | Speed over ground (knots) |
| COG | float | Course over ground (degrees) |
| Heading | float | True heading (degrees) |
| VesselName | str | Vessel name |
| VesselType | int | AIS vessel type code |

### Coverage
- Brest, France maritime area
- 2012–2013

### Notes
- Downloaded automatically by `ml/datasets/dataset_download.py`.
- Primary AIS dataset for trajectory anomaly analysis.

---

## 5. Antwerp AIS Dataset

| Field | Details |
|---|---|
| **Source** | https://data.mendeley.com/datasets/smb56stcdx/1 |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Format** | CSV |
| **Local path** | `data/raw/ais_antwerp/` |

### Coverage
- Antwerp port area, Belgium

### Notes
- Supplementary AIS dataset for port/near-shore anomaly detection.
- Manual download from Mendeley Data (no API available).
- Useful for vessel dwell-time and loitering detection near ports.

---

## 6. HYCOM Ocean Currents

| Field | Details |
|---|---|
| **Source** | https://www.hycom.org/ / https://tds.hycom.org/thredds/dodsC/ |
| **License** | Free for research use |
| **Format** | NetCDF via OPeNDAP |
| **Local path** | `data/raw/hycom/` |

### Variables
| Variable | Description |
|---|---|
| `water_u` | Eastward current velocity (m/s) |
| `water_v` | Northward current velocity (m/s) |

### Notes
- Accessed via OPeNDAP (no bulk download needed; query by region/time).
- Used by the `drift_engine` to simulate oil slick drift trajectories.
- Resolution: ~8 km (GLBv0.08) or ~4 km (GLBy0.04).
- Use `xarray` + `netCDF4` to access remotely.

---

## 7. ERA5 Wind Data

| Field | Details |
|---|---|
| **Source** | https://cds.climate.copernicus.eu/ |
| **License** | Copernicus Climate Data Store licence |
| **Format** | GRIB / NetCDF |
| **Local path** | `data/raw/era5/` |

### Variables
| Variable | Description |
|---|---|
| `u10` | 10 m eastward wind component (m/s) |
| `v10` | 10 m northward wind component (m/s) |

### Notes
- Requires free registration at CDS and a valid **CDS API key** in `~/.cdsapirc`.
- Download via `cdsapi` Python library.
- Used alongside HYCOM currents for drift modelling.
- Temporal resolution: hourly; spatial resolution: ~31 km (ERA5) or ~9 km (ERA5-Land).

---

## Summary Table

| # | Dataset | Use | License | Auto-download |
|---|---|---|---|---|
| 1 | M4D Oil Spill | Segmentation training | Academic | ❌ Manual |
| 2 | GlobalOSD-SAR | Classification training | CC BY 4.0 | ✅ Zenodo API |
| 3 | Sentinel-1 SAR | Real inference data | Copernicus Open | ❌ Manual |
| 4 | Brest AIS | Trajectory anomaly | CC BY 4.0 | ✅ Zenodo API |
| 5 | Antwerp AIS | Port anomaly | CC BY 4.0 | ❌ Manual |
| 6 | HYCOM Currents | Drift modelling | Free research | ✅ OPeNDAP |
| 7 | ERA5 Wind | Drift modelling | Copernicus | ✅ CDS API |
