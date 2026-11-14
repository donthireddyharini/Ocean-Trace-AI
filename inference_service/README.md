---
title: OceanTrace AI SegFormer Inference Microservice
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# OceanTrace AI — SegFormer-B2 Inference Microservice
**Smart India Hackathon 2026 | Problem Statement SIH26143**

Dedicated cloud inference container for **SegFormer-B2** (`nvidia/mit-b2`) marine oil spill semantic segmentation.

## Quick Start on Hugging Face Spaces
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Choose **Docker** as the SDK.
3. Push or upload the files from this `inference_service/` folder to the repository.
4. Set Space variables (Settings -> Variables and secrets):
   - `MODEL_CHECKPOINT_PATH` (or upload `best_oilspill_segformer.pth`)
5. Once running, copy your Space direct URL (e.g., `https://yourusername-oceantrace-model.hf.space`).
6. In your Vercel Project Settings -> **Environment Variables**, set:
   ```ini
   MODEL_INFERENCE_URL=https://yourusername-oceantrace-model.hf.space/predict
   ```

## API Endpoints
- `GET /health` — Microservice health status.
- `GET /model/status` — Current SegFormer-B2 checkpoint loading state.
- `POST /predict` — High-speed segmentation forward pass for SAR and Optical imagery.
