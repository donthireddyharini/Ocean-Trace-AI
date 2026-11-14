import sys
import os
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "oceantrace-backend" / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app, _seed_demo_users
from app.database import init_db

# Ensure database is initialized in serverless environment
try:
    init_db()
    _seed_demo_users()
except Exception as e:
    print(f"Serverless init warning: {e}")
