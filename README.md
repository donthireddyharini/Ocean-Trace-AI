# OceanTrace AI — Research-Grade Maritime Satellite Oil Spill Detection & AIS Attribution Platform

**Smart India Hackathon 2026 — Problem Statement SIH26143**  
*Organization: NTRO · Category: Software · Theme: Space Technology*  
*Status: Research-Grade Implementation with Full AI/ML Pipeline, Particle Drift Hindcast, and FastAPI Backend*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%200.115-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20%2B%20SegFormer--B2-EE4C2C.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-58%20Passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> **OceanTrace AI** is an end-to-end maritime intelligence platform that detects oil spills from Sentinel-1 SAR and optical satellite imagery, reconstructs the probable spill origin using ocean currents and wind hindcasts, and correlates historical AIS trajectories using a vectorised attribution engine to identify suspect vessels with transparent, explainable evidence.

---

## 🌐 Live System URLs

| Component | Target URL | Description |
|---|---|---|
| **Web Dashboard** | `[(https://ocean-trace-ai-eight.vercel.app/)` | Full Tactical Cyber-Maritime C4ISR Command Center |
| **Interactive Alternate** | `http://localhost:8080/oceantrace-ai.html` | Direct standalone frontend |
| **FastAPI REST API** | `http://localhost:8000` | Operational Python REST & ML backend |
| **Swagger UI Docs** | `http://localhost:8000/docs` | Interactive API documentation |
| **ReDoc Reference** | `http://localhost:8000/redoc` | OpenAPI 3.1 specification viewer |
| **Cloud Production** | `https://smart-india-hackthn.vercel.app/` | Vercel production deployment |

---

## 🏗️ System Architecture

```
                                  OCEANTRACE AI SYSTEM ARCHITECTURE
                                  
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                           FRONTEND DASHBOARD (Leaflet + JS)                            │
  │   - Tactical C4ISR Map    - 48h AIS Time-Scrubber   - Satellite Scene Studio           │
  │   - Bonn Volumetric Intel - Multi-factor Scoring    - Voice & Text Copilot             │
  └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                              │ HTTP / JSON REST APIs
                                              ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                            FASTAPI BACKEND SERVICE (:8000)                             │
  │  ├── Auth & RBAC (JWT, bcrypt, Admin / Analyst / Viewer)                               │
  │  ├── /api/spill/segformer-detect  ──► SegFormer Vision Transformer Inference          │
  │  ├── /api/spill/characterize      ──► OpenCV Geometric & Morphological Analysis        │
  │  ├── /api/drift/backward          ──► Particle-based Ocean Drift Hindcast              │
  │  ├── /api/environment/{lat}/{lon} ──► Live Marine Current & Wind Fetch (Open-Meteo)    │
  │  └── /api/ais/analyze             ──► Vectorised NumPy Spatiotemporal Scoring Engine   │
  └───────────────────────┬────────────────────────────────────────┬───────────────────────┘
                          │                                        │
                          ▼                                        ▼
  ┌──────────────────────────────────────┐     ┌───────────────────────────────────────────┐
  │         AI / ML PIPELINE             │     │           DATASET CATALOG                 │
  │ - SegFormer-B2 (nvidia/mit-b2)       │     │ - M4D Marine Oil Spill Dataset (SAR)      │
  │ - Combined CE + Soft Dice Loss       │     │ - GlobalOSD-SAR (Zenodo 15286918)         │
  │ - Look-Alike False-Positive Filter   │     │ - Brest Historical AIS (Zenodo 2563256)   │
  │ - Reproducible Kaggle/Colab Notebook │     │ - HYCOM Currents & ERA5 Wind Fields       │
  └──────────────────────────────────────┘     └───────────────────────────────────────────┘
```

---

## 🔬 Core Capabilities

### 1. Vision Transformer Satellite Spill Detection (SegFormer-B2)
- Architecture: Hierarchical Mix Transformer (`nvidia/mit-b2`) encoder with All-MLP lightweight decoder.
- Multi-class output: **Clean Sea (0)**, **Heavy Emulsion Core (1)**, **Thin Rainbow Sheen (2)**, **Look-Alike Boundary (3)**, **Ship Scatterer (4)**.
- Volumetric estimation using **Bonn Agreement** oil appearance codes (200 µm emulsion vs 1.5 µm iridescent sheen).
- Dual-modality support: Automatic detection of **Sentinel-1 SAR** (grayscale backscatter dampening) vs **Optical Satellite** (RGB multi-spectral variance).

### 2. OpenCV Real Spill Geometry Characterization
- Computes exact pixel area, perimeter, centroid coordinates, and bounding box.
- Shape descriptors: Aspect ratio, compactness ($4\pi A / P^2$), convexity ($A / A_{hull}$), and orientation angle.
- Calculates true geographic area ($\text{km}^2$) and real lat/lon centroid when spatial resolution ($m/\text{pixel}$) is supplied.

### 3. Particle-Based Hydrodynamic Drift Hindcast
- 3% wind drift factor combined with vector ocean surface currents.
- Reverse trajectory simulation (hindcasting) from detection timestamp back to estimated release window.
- Forward trajectory forecasts at **+6h**, **+12h**, **+24h**, and **+48h**.
- Multi-band probabilistic uncertainty contours (68%, 90%, 95% confidence intervals).

### 4. Vectorised AIS Vessel Attribution Engine
- Evaluates suspect vessels in a single-pass NumPy batch scoring pipeline:
  $$\text{Score} = \text{Spatial}(25) + \text{Temporal}(25) + \text{Trajectory}(20) + \text{Drift}(20) + \text{Behavior}(10)$$
- Detects AIS transmission gaps ("dark periods"), speed-drops, and trajectory alignment with drift bearing.
- Complies with SIH legal requirements: Labels candidate vessels as *High Investigation Priority*, never declaring guilt based on algorithmic scoring alone.

---

## 🚀 Quick Start Guide

### 1. Run the Backend
```powershell
cd oceantrace-backend\backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000 --reload
```
Swagger UI will be live at `http://localhost:8000/docs`.

### 2. Run the Frontend
```powershell
# From project root
python -m http.server 8080
```
Open `http://localhost:8080` in Chrome, Edge, or Firefox.

### 3. Automated Test Verification
Run the complete unit and integration test suite:
```powershell
$env:PYTHONPATH=".;oceantrace-backend\backend"
python -m pytest tests/ -v
```
**Results:** `58 passed` across unit and integration tests!

---

## 📁 Repository Structure

```
OceanTrace-AI/
├── notebooks/
│   └── oil_spill_segformer_training.ipynb   # Executable Kaggle/Colab GPU notebook
├── ml/
│   ├── datasets/                            # Dataset download, validation & AIS processor
│   ├── preprocessing/                       # Normalization, resizing, augmentation, scene split
│   ├── segmentation/                        # SegFormer PyTorch model wrapper & loss functions
│   ├── classification/                      # Look-alike CNN classifier
│   ├── evaluation/                          # mIoU, Dice, Precision, Recall, Confusion Matrix
│   └── inference/                           # Inference pipeline & particle drift simulator
├── oceantrace-backend/
│   ├── start_backend.bat                    # One-click Windows starter
│   ├── start_backend.sh                     # Linux/macOS starter
│   └── backend/
│       ├── app/
│       │   ├── routers/                     # auth, spill, drift, ais, environment, characterize
│       │   ├── services/                    # spill_characterization, environmental_data, etc.
│       │   └── main.py                      # FastAPI app entry point
├── tests/
│   ├── unit/                                # 52 passing unit tests
│   └── integration/                         # 6 passing end-to-end integration tests
├── data/
│   ├── samples/                             # Realistic AIS CSV sample dataset
│   └── dataset_inventory.csv                # Automated inventory tracking
├── docs/
│   ├── ARCHITECTURE.md                      # System architecture specification
│   ├── API_REFERENCE.md                     # Full REST API endpoint documentation
│   ├── TRAINING_GUIDE.md                    # Kaggle & Colab GPU training instructions
│   ├── DEPLOYMENT.md                        # Production hosting guide
│   └── FRONTEND_BACKEND_INTEGRATION.md      # Dual-mode client integration guide
├── reports/
│   └── INVESTIGATION_REPORT_TEMPLATE.md     # Official intelligence dossier template
├── DATASET_CATALOG.md                       # Comprehensive dataset catalog
├── DATA_SOURCES.md                          # Data access procedures
├── MODEL_CARD.md                            # SegFormer-B2 model card & ethical boundaries
├── requirements_ml.txt                      # Complete ML stack requirements
└── README.md                                # Project documentation
```

---

## ⚖️ Scientific & Legal Disclaimer

*OceanTrace AI provides analytical and investigative correlation based on Synthetic Aperture Radar (SAR) observations, hydrodynamic drift modeling, and historical AIS trajectories. Spatiotemporal correlation does not establish legal responsibility or environmental liability. All findings must be independently validated by competent maritime authorities before enforcement action.*
