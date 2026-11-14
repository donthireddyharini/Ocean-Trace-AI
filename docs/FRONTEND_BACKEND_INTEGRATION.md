# OceanTrace AI — Frontend to Backend Integration Architecture

The OceanTrace AI dashboard is engineered with a modular dual-execution pattern:
1. **Live Backend Mode (Connected)**: Connects to the FastAPI REST API at `http://localhost:8000`. Requests for AIS analysis, drift hindcasts, spill segmentation, and user sessions route through real server-side endpoints.
2. **Local Resilient Fallback Mode (Autonomous)**: If the backend is temporarily offline or unreachable, the dashboard automatically falls back to in-browser deterministic simulation so investigators and hackathon evaluators are never blocked by network latency.

---

## 1. Connecting the Dashboard to Live Backend

1. Start both servers:
   - Backend on `http://localhost:8000`
   - Frontend on `http://localhost:8080`
2. Open `http://localhost:8080` in your web browser.
3. Click **Settings** in the top navigation bar.
4. Scroll to **Live Backend Integration**:
   - **Backend Base URL:** `http://localhost:8000`
   - **Access Token:** Obtain from `POST http://localhost:8000/api/auth/login` (or click "Generate Local Test Token").
5. Click **Save Settings**.
6. When running AIS analysis or spill detection, you will see the **`LIVE BACKEND · X.XX ms`** verification badge confirming server-side vectorised processing!

---

## 2. API Mapping & Endpoints

| Dashboard Action | Client-Side Trigger | FastAPI Server Endpoint |
|---|---|---|
| Ingest & Detect Spill | "Run Detection" | `POST /api/spill/segformer-detect` |
| Image Analysis | Geometric Calculation | `POST /api/spill/characterize` |
| Hindcast Drift | "Trace Origin" | `POST /api/drift/backward` |
| Environmental Fetch | Auto-coordinates | `GET /api/environment/{lat}/{lon}` |
| AIS Vessel Attribution | "Analyze AIS & Rank" | `POST /api/ais/analyze` |
| Dossier Generation | "Generate Report" | `POST /api/reports/generate` |
