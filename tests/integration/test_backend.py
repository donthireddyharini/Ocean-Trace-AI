"""
Integration tests for OceanTrace AI FastAPI backend.
Tests health, auth, environment, drift, and AIS scoring endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app, on_startup

# Ensure database tables and seed users are initialized
on_startup()

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "OceanTrace AI" in data["service"]


def test_auth_demo_login():
    response = client.post("/api/auth/login", json={
        "email": "analyst@oceantrace.ai",
        "password": "Analyst@123"
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_environment_endpoint():
    response = client.get("/api/environment/18.92/72.83")
    assert response.status_code == 200
    data = response.json()
    assert "wind_speed_kt" in data
    assert "current_speed_kt" in data
    assert data["data_coverage"] in ["full", "partial", "demo_fallback"]


def test_drift_backward_stateless():
    # Login to get token
    login_resp = client.post("/api/auth/login", json={
        "email": "analyst@oceantrace.ai",
        "password": "Analyst@123"
    })
    token = login_resp.json()["access_token"]

    payload = {
        "origin_lat": 18.922,
        "origin_lon": 72.834,
        "area_km2": 12.5,
        "wind_dir_deg": 240.0,
        "wind_speed_kt": 15.0,
        "current_dir_deg": 180.0,
        "current_speed_kt": 0.9
    }
    response = client.post(
        "/api/drift/backward",
        json=payload,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "probable_origin_lat" in data
    assert "probable_origin_lon" in data
    assert len(data["spill_polygon"]) > 0


def test_ais_analyze_vectorised():
    login_resp = client.post("/api/auth/login", json={
        "email": "analyst@oceantrace.ai",
        "password": "Analyst@123"
    })
    token = login_resp.json()["access_token"]

    payload = {
        "origin_lat": 18.78,
        "origin_lon": 72.69,
        "spill_time_h": -24.0,
        "drift_bearing_deg": 45.0,
        "vessels": [
            {
                "mmsi": "123456789",
                "name": "TEST TANKER",
                "type": "Tanker",
                "flag": "Panama",
                "lat": 18.79,
                "lon": 72.70,
                "heading_deg": 48.0,
                "timestamp_offset_h": -23.5
            },
            {
                "mmsi": "987654321",
                "name": "FAR CARGO",
                "type": "Cargo",
                "flag": "Liberia",
                "lat": 16.0,
                "lon": 70.0,
                "heading_deg": 180.0,
                "timestamp_offset_h": -2.0
            }
        ]
    }
    response = client.post(
        "/api/ais/analyze",
        json=payload,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["vessels"]) == 2
    assert data["vessels"][0]["total"] >= data["vessels"][1]["total"]
    assert data["compute_ms"] >= 0
