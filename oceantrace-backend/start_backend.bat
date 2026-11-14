@echo off
echo ===================================================
echo     OceanTrace AI - Maritime Intelligence Backend
echo ===================================================
cd %~dp0backend
if not exist .env (
    echo Creating default .env file...
    copy .env.example .env
)
echo Checking dependencies...
python -m pip install -r requirements.txt -q
echo Starting FastAPI application server on http://localhost:8000 ...
echo Swagger UI Documentation: http://localhost:8000/docs
python -m uvicorn app.main:app --port 8000 --reload
pause
