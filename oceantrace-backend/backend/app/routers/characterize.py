"""
Spill characterization router.
POST /api/spill/characterize — uploads a satellite image and returns
pixel-level and (optionally) geographic oil spill geometry.
Requires ANALYST or ADMIN role.
"""
import dataclasses

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional

from app.security import require_role
from app.services.spill_characterization import SpillCharacterizer

router = APIRouter(prefix="/api/spill", tags=["spill"])

_characterizer = SpillCharacterizer()

# Maximum upload size: 50 MB
_MAX_SIZE_BYTES = 50 * 1024 * 1024


@router.post("/characterize")
async def characterize_spill(
    file: UploadFile = File(..., description="Satellite image (PNG, JPEG, GeoTIFF preview)"),
    spatial_resolution_m: Optional[float] = Form(
        None,
        description="Meters per pixel — enables geographic area/perimeter output"
    ),
    center_lat: Optional[float] = Form(
        None,
        description="Image centre latitude (decimal degrees)"
    ),
    center_lon: Optional[float] = Form(
        None,
        description="Image centre longitude (decimal degrees)"
    ),
    modality: str = Form(
        "auto",
        description="Image modality: 'sar', 'optical', or 'auto'"
    ),
    _user=Depends(require_role("ANALYST", "ADMIN")),
):
    """
    Analyse an uploaded satellite image for oil spill extent and shape.

    Returns SpillGeometry as JSON:
    - Pixel-level measurements always present.
    - Geographic measurements (area_km2, perimeter_km, centroid lat/lon)
      only when spatial_resolution_m is provided.
    - detection_note explains data availability.
    """
    if file.content_type not in {
        "image/png", "image/jpeg", "image/tiff", "image/geotiff",
        "application/octet-stream",
    }:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}. Use PNG, JPEG, or TIFF."
        )

    image_bytes = await file.read()
    if len(image_bytes) > _MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large: {len(image_bytes) // 1024} KB. Maximum is 50 MB."
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        geometry = _characterizer.analyze_from_bytes(
            image_bytes=image_bytes,
            modality=modality,
            spatial_resolution_m=spatial_resolution_m,
            center_lat=center_lat,
            center_lon=center_lon,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Image analysis failed: {str(exc)[:200]}"
        )

    # Convert dataclass to plain dict (handles Optional fields cleanly)
    result = dataclasses.asdict(geometry)
    return JSONResponse(content=result)
