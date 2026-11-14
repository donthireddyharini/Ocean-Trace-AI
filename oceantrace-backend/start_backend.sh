#!/usr/bin/env bash
echo "==================================================="
echo "    OceanTrace AI - Maritime Intelligence Backend"
echo "==================================================="
cd "$(dirname "$0")/backend"
[ ! -f .env ] && cp .env.example .env
python3 -m pip install -r requirements.txt -q
echo "Starting FastAPI on http://localhost:8000 ..."
echo "API Docs: http://localhost:8000/docs"
python3 -m uvicorn app.main:app --port 8000 --reload --host 0.0.0.0
