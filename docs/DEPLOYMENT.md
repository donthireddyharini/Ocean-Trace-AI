# OceanTrace AI — Deployment & Operation Guide

## 1. Local Full-Stack Execution

### Step 1: Start the Backend (FastAPI)
The backend manages JWT auth, vectorised AIS correlation, drift simulation, environmental data, and SegFormer inference.

On Windows PowerShell:
```powershell
cd "c:\Users\donth_\Downloads\Smart-India-Hackthn-main\Smart-India-Hackthn-main\oceantrace-backend\backend"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000 --reload
```
Interactive API docs are immediately accessible at:
👉 `http://localhost:8000/docs`

### Step 2: Start the Frontend (Web Dashboard)
Open a separate PowerShell terminal:
```powershell
cd "c:\Users\donth_\Downloads\Smart-India-Hackthn-main\Smart-India-Hackthn-main"
python -m http.server 8080
```
Open your browser to:
👉 `http://localhost:8080` (or `http://localhost:8080/oceantrace-ai.html`)

---

## 2. Cloud Deployment

### Frontend: Vercel (Current Production Host)
The frontend is already configured for zero-configuration static deployment via `vercel.json`:
- Deployed at: `https://smart-india-hackthn.vercel.app/`
- Automatic rewrites route `/` and `/oceantrace-ai.html` to `index.html`.

### Backend: Render / Railway / AWS ECS
Deploy the `oceantrace-backend/backend` folder as a Python web service:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables:**
  - `ENV`: `production`
  - `JWT_SECRET`: Generate a 64-character secret
  - `CORS_ORIGINS`: `https://smart-india-hackthn.vercel.app,http://localhost:8080`
  - `DATABASE_URL`: `sqlite:///./oceantrace.db` or PostgreSQL URI
