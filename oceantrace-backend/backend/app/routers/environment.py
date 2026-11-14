"""
Environment router — exposes real environmental data (wind, ocean currents)
from Open-Meteo and HYCOM via the EnvironmentalDataService.
"""
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.services.environmental_data import EnvironmentalDataService

router = APIRouter(prefix="/api/environment", tags=["environment"])

_svc = EnvironmentalDataService()


@router.get("/health")
def environment_health():
    """
    Quick reachability check for the upstream environmental data APIs.
    Returns individual status for Open-Meteo Wind and Open-Meteo Marine.
    """
    import requests

    results = {}

    # Open-Meteo wind API (use a fixed test coordinate)
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 15.0, "longitude": 40.0,
                "hourly": "windspeed_10m",
                "forecast_days": 1, "timezone": "UTC"
            },
            timeout=6
        )
        results["openmeteo_wind"] = "reachable" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as e:
        results["openmeteo_wind"] = f"unreachable ({str(e)[:40]})"

    # Open-Meteo Marine API
    try:
        r = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": 15.0, "longitude": 40.0,
                "hourly": "ocean_current_velocity",
                "forecast_days": 1, "timezone": "UTC"
            },
            timeout=6
        )
        results["openmeteo_marine"] = "reachable" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as e:
        results["openmeteo_marine"] = f"unreachable ({str(e)[:40]})"

    return {"status": "ok", "apis": results}


@router.get("/{lat}/{lon}")
def get_environmental_conditions(lat: float, lon: float, timestamp: str = None):
    """
    Return real wind and ocean current conditions for a given location.
    Falls back to physics-based estimates when APIs are unreachable,
    with data_coverage='demo_fallback' and explicit warnings.
    """
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=422, detail="lat must be in [-90, 90]")
    if not (-180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="lon must be in [-180, 180]")

    conditions = _svc.get_conditions(lat=lat, lon=lon, timestamp=timestamp)
    return JSONResponse(content=asdict(conditions))
