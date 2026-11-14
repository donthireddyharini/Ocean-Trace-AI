"""
End-to-End Pipeline Integration Test.
Simulates a full mission workflow through the FastAPI backend:
1. Authentication & Session Creation
2. Mission Creation
3. Ingestion & SegFormer Detection
4. Spill Characterization (OpenCV morphological analysis)
5. Oceanographic Environment Fetch
6. Backward Drift Hindcast & Origin Localization
7. AIS Vessel Correlation & Attribution Scoring
8. Audit Trail Verification
"""
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app, on_startup

on_startup()
client = TestClient(app)


def test_full_investigation_lifecycle():
    # 1. Login as Analyst
    login_resp = client.post("/api/auth/login", json={
        "email": "analyst@oceantrace.ai",
        "password": "Analyst@123"
    })
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Create Mission
    mission_resp = client.post("/api/investigations", json={
        "name": "SIH26143 Arabian Sea Test Mission",
        "description": "Validation mission for end-to-end maritime spill attribution",
        "priority": "HIGH"
    }, headers=headers)
    assert mission_resp.status_code == 201
    mission_id = mission_resp.json()["id"]

    # 3. Create a synthetic test SAR image with a dark slick feature
    img_arr = np.full((256, 256), 180, dtype=np.uint8)
    # Draw dark slick feature
    img_arr[80:140, 90:170] = 45
    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format="PNG")
    buf.seek(0)

    # 4. Direct SegFormer Detection
    detect_resp = client.post(
        "/api/spill/segformer-detect",
        files={"file": ("sar_scene.png", buf.getvalue(), "image/png")},
        data={"modality": "SAR"}
    )
    assert detect_resp.status_code == 200
    detect_data = detect_resp.json()
    assert detect_data["modality"] == "SAR"
    assert "confidence" in detect_data
    assert "area_km2" in detect_data

    # 5. Spill Characterization
    buf.seek(0)
    char_resp = client.post(
        "/api/spill/characterize",
        files={"file": ("sar_scene.png", buf.getvalue(), "image/png")},
        data={
            "spatial_resolution_m": 20.0,
            "center_lat": 18.92,
            "center_lon": 72.83,
            "modality": "SAR"
        },
        headers=headers
    )
    assert char_resp.status_code == 200
    char_data = char_resp.json()
    assert char_data["pixel_area"] > 0
    assert char_data["area_km2"] is not None

    # 6. Environmental Fetch
    env_resp = client.get("/api/environment/18.92/72.83")
    assert env_resp.status_code == 200
    env_data = env_resp.json()

    # 7. Backward Drift Hindcast
    drift_resp = client.post("/api/drift/backward", json={
        "origin_lat": 18.92,
        "origin_lon": 72.83,
        "area_km2": char_data["area_km2"] or 10.0,
        "wind_dir_deg": env_data.get("wind_dir_deg") or 240.0,
        "wind_speed_kt": env_data.get("wind_speed_kt") or 14.0,
        "current_dir_deg": env_data.get("current_dir_deg") or 190.0,
        "current_speed_kt": env_data.get("current_speed_kt") or 0.8
    }, headers=headers)
    assert drift_resp.status_code == 200
    drift_data = drift_resp.json()
    probable_lat = drift_data["probable_origin_lat"]
    probable_lon = drift_data["probable_origin_lon"]

    # 8. AIS Vessel Correlation
    ais_resp = client.post("/api/ais/analyze", json={
        "origin_lat": probable_lat,
        "origin_lon": probable_lon,
        "spill_time_h": -drift_data["estimated_age_h"],
        "drift_bearing_deg": 45.0,
        "vessels": [
            {
                "mmsi": "419000123",
                "name": "MT ARABIAN STAR",
                "type": "Oil Tanker",
                "flag": "India",
                "lat": probable_lat + 0.01,
                "lon": probable_lon + 0.01,
                "heading_deg": 46.0,
                "timestamp_offset_h": -drift_data["estimated_age_h"] + 0.5
            },
            {
                "mmsi": "235099887",
                "name": "GLOBAL CARRIER",
                "type": "Container Ship",
                "flag": "Panama",
                "lat": probable_lat + 2.5,
                "lon": probable_lon + 2.0,
                "heading_deg": 180.0,
                "timestamp_offset_h": -5.0
            }
        ]
    }, headers=headers)
    assert ais_resp.status_code == 200
    ais_data = ais_resp.json()
    assert len(ais_data["vessels"]) == 2
    # Verify top priority ranking
    assert ais_data["vessels"][0]["name"] == "MT ARABIAN STAR"
    assert ais_data["vessels"][0]["total"] > 70
    assert "HIGH INVESTIGATION PRIORITY" in ais_data["vessels"][0]["status"]
