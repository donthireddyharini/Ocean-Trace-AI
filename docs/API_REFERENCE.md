# OceanTrace AI — REST API Reference
**Version:** 0.9.5-production  
**Base URL:** `http://localhost:8000` (Local) / `/api`  
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI) / `http://localhost:8000/redoc`

---

## 1. System & Health

### `GET /api/health`
Checks backend service operational status.
- **Auth:** None
- **Response `200 OK`**:
  ```json
  {
    "status": "operational",
    "service": "OceanTrace AI API",
    "env": "development"
  }
  ```

---

## 2. Authentication & RBAC

### `POST /api/auth/login`
Authenticates user with email and password, returning JWT bearer tokens.
- **Request Body**:
  ```json
  {
    "email": "analyst@oceantrace.ai",
    "password": "Analyst@123"
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1Ni...",
    "refresh_token": "eyJhbGciOiJIUzI1Ni...",
    "token_type": "bearer",
    "expires_in": 1800
  }
  ```

### `POST /api/auth/register`
Registers a new user account with role-based access.
- **Request Body**:
  ```json
  {
    "name": "Commander V. Rao",
    "organization": "Indian Coast Guard",
    "email": "v.rao@coastguard.gov.in",
    "password": "SecurePassword@2026"
  }
  ```

### `GET /api/auth/me`
Retrieves authenticated user profile.
- **Auth:** Bearer Token

---

## 3. Satellite Spill Detection & Characterization

### `POST /api/spill/segformer-detect`
Direct SegFormer Vision AI Detection endpoint for SAR and Optical satellite imagery.
- **Auth:** None (stateless) or Bearer Token
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file`: Satellite image file (PNG, JPG, TIFF)
  - `modality`: `"auto"`, `"SAR"`, or `"OPTICAL"`
- **Response `200 OK`**:
  ```json
  {
    "model": "SegFormer-B2 (Mix Transformer)",
    "encoder": "MiT-B2 Multi-Scale Pyramid",
    "modality": "SAR",
    "confidence": 94.6,
    "area_km2": 14.82,
    "thick_core_km2": 6.22,
    "thin_sheen_km2": 8.6,
    "estimated_volume_m3": 1256.9,
    "estimated_volume_tonnes": 1106.1,
    "length_km": 6.12,
    "width_km": 2.42,
    "aspect_ratio": 2.53,
    "perimeter_km": 17.93,
    "orientation_deg": 118,
    "centroid_px": {"x": 256.0, "y": 256.0},
    "bbox_px": {"x": 64, "y": 64, "w": 384, "h": 384},
    "classes": [
      {"name": "Heavy Emulsion Core", "code": 1, "color": "#dc2626"},
      {"name": "Thin Rainbow Sheen", "code": 2, "color": "#0284c7"},
      {"name": "Look-Alike Boundary", "code": 3, "color": "#d97706"},
      {"name": "Clean Seawater", "code": 0, "color": "#0f172a"}
    ]
  }
  ```

### `POST /api/spill/characterize`
Performs real OpenCV & PIL geometric and morphological analysis on uploaded imagery.
- **Content-Type:** `multipart/form-data`
- **Parameters:** `file`, `spatial_resolution_m` (optional), `center_lat` (optional), `center_lon` (optional)
- **Response `200 OK`**: Returns pixel area, perimeter, shape convexity, compactness, and georeferenced km² if resolution provided.

---

## 4. Ocean Drift & Origin Estimation

### `POST /api/drift/backward`
Stateless backward drift hindcast and forward trajectory prediction.
- **Auth:** Bearer Token
- **Request Body**:
  ```json
  {
    "origin_lat": 18.922,
    "origin_lon": 72.834,
    "area_km2": 15.4,
    "wind_dir_deg": 245.0,
    "wind_speed_kt": 14.5,
    "current_dir_deg": 190.0,
    "current_speed_kt": 0.8
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "probable_origin_lat": 18.7841,
    "probable_origin_lon": 72.6953,
    "estimated_age_h": 28.5,
    "origin_window": "T-34h to T-22h",
    "spill_polygon": [[18.92, 72.83], ...],
    "forecast_polygon": [[19.05, 72.95], ...],
    "compute_ms": 1.42
  }
  ```

---

## 5. Environmental Oceanographic Data

### `GET /api/environment/{lat}/{lon}`
Retrieves ocean current velocity and surface wind vectors from real Open-Meteo Marine APIs.
- **Response `200 OK`**:
  ```json
  {
    "lat": 18.92,
    "lon": 72.83,
    "current_speed_kt": 0.65,
    "current_dir_deg": 195.0,
    "wind_speed_kt": 12.3,
    "wind_dir_deg": 240.0,
    "current_source": "Open-Meteo Marine API (real)",
    "wind_source": "Open-Meteo API (real)",
    "data_coverage": "full"
  }
  ```

---

## 6. AIS Vessel Correlation & Attribution

### `POST /api/ais/analyze`
Vectorised single-pass numpy attribution engine evaluating suspect vessels.
- **Auth:** Bearer Token
- **Request Body**:
  ```json
  {
    "origin_lat": 18.784,
    "origin_lon": 72.695,
    "spill_time_h": -28.5,
    "drift_bearing_deg": 45.0,
    "vessels": [
      {
        "mmsi": "636019842",
        "name": "PACIFIC PIONEER",
        "type": "Crude Oil Tanker",
        "flag": "Liberia",
        "lat": 18.791,
        "lon": 72.688,
        "heading_deg": 48.0,
        "timestamp_offset_h": -28.0
      }
    ]
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "funnel": [48, 14, 6, 3, 1],
    "vessels": [
      {
        "mmsi": "636019842",
        "name": "PACIFIC PIONEER",
        "type": "Crude Oil Tanker",
        "flag": "Liberia",
        "distance_km": 1.08,
        "distance_nmi": 0.58,
        "spatial": 25,
        "temporal": 24,
        "trajectory": 20,
        "drift_compat": 18,
        "behavior": 8,
        "total": 95,
        "status": "HIGH INVESTIGATION PRIORITY"
      }
    ],
    "compute_ms": 3.42
  }
  ```

---

## 7. Reports & Audit Trail

### `POST /api/reports/generate`
Generates a structured investigation report JSON and printable dossier.

### `GET /api/audit-logs`
Administrative audit trail of actions taken in the system.
