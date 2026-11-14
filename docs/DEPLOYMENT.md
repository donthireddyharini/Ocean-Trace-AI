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

---

## 3. Production Model Hosting & Vercel Serverless Integration

Due to AWS Lambda / Vercel Serverless 250MB uncompressed bundle limits, large PyTorch models (~800MB–1GB+) cannot be packaged directly into a single serverless zip. OceanTrace AI provides three production-grade model hosting architectures:

### Architecture A: Dedicated Inference Microservice (Recommended)
Deploy a lightweight GPU or CPU container on **Hugging Face Spaces** (Free Docker/FastAPI) or **Render**:
1. Host the PyTorch `SegFormer-B2` forward pass in a dedicated endpoint.
2. In Vercel Project Settings -> **Environment Variables**, set:
   ```ini
   MODEL_INFERENCE_URL=https://<your-space-name>.hf.space/predict
   ```
3. The Vercel serverless API will securely proxy satellite images to your persistent microservice, compute inference in real-time, and return structured segmentation results without any bundle size restrictions.

### Architecture B: Persistent Object Storage / Hugging Face Hub Download
Store `best_oilspill_segformer.pth` on Hugging Face Hub (Private/Public) or S3/GCS:
1. In Vercel Environment Variables, set:
   ```ini
   HF_MODEL_REPO=your-username/oceantrace-segformer-b2
   HF_TOKEN=hf_...  # optional if repo is private
   # Or direct signed HTTPS download URL:
   CHECKPOINT_URL=https://your-bucket.s3.amazonaws.com/best_oilspill_segformer.pth
   ```
2. On initial invocation, the backend downloads the checkpoint into `/tmp/best_oilspill_segformer.pth` and caches it across warm invocations.

### Architecture C: Local / Dedicated Server Mount
For on-premises, Docker, or VM deployments (AWS EC2, GCP Compute Engine, Render):
1. Place `best_oilspill_segformer.pth` directly in `oceantrace_artifacts/best_oilspill_segformer.pth`.
2. The detector automatically discovers local files on startup and activates local PyTorch inference.
