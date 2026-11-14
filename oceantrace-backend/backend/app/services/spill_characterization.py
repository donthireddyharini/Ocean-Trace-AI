"""
Real image analysis pipeline for oil spill characterization.
Uses OpenCV and PIL for pixel-level analysis.
NOTE: Geographic area calculations require real GeoTIFF metadata.
When georeferencing is unavailable, pixel-based measurements are returned
and geographic measurements are marked as unavailable.
"""
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from PIL import Image
import io
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class SpillGeometry:
    # Pixel measurements (always available)
    pixel_area: int
    pixel_perimeter: float
    centroid_px: Tuple[float, float]
    bbox_px: Tuple[int, int, int, int]  # x, y, w, h

    # Shape descriptors
    aspect_ratio: float
    orientation_deg: float
    compactness: float
    convexity: float

    # Geographic measurements (only if georeferencing available)
    area_km2: Optional[float] = None
    perimeter_km: Optional[float] = None
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None

    # Metadata
    image_width_px: int = 0
    image_height_px: int = 0
    spatial_resolution_m: Optional[float] = None  # meters per pixel
    georef_available: bool = False

    # Analysis metadata
    confidence: float = 0.0
    detection_note: str = ""


class SpillCharacterizer:
    def __init__(self):
        self.min_spill_area_px = 100  # Minimum pixels to count as spill

    def analyze_from_bytes(self, image_bytes: bytes,
                           modality: str = 'auto',
                           spatial_resolution_m: Optional[float] = None,
                           center_lat: Optional[float] = None,
                           center_lon: Optional[float] = None) -> SpillGeometry:
        """
        Analyze an uploaded satellite image for oil spill characteristics.
        Returns real measurements based on actual pixel analysis.
        Geographic measurements returned only when spatial_resolution_m provided.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
        img_array = np.array(image)

        # Detect oil spill pixels using adaptive thresholding
        spill_mask = self._detect_spill_pixels(img_array, modality)

        if spill_mask.sum() < self.min_spill_area_px:
            # Return minimal result - no significant spill detected
            h, w = img_array.shape
            return SpillGeometry(
                pixel_area=int(spill_mask.sum()),
                pixel_perimeter=0.0,
                centroid_px=(w / 2, h / 2),
                bbox_px=(0, 0, w, h),
                aspect_ratio=1.0,
                orientation_deg=0.0,
                compactness=0.0,
                convexity=0.0,
                image_width_px=w,
                image_height_px=h,
                confidence=0.1,
                detection_note="No significant oil spill signature detected in this image"
            )

        return self._compute_geometry(spill_mask, img_array.shape,
                                      spatial_resolution_m, center_lat, center_lon)

    def _detect_spill_pixels(self, gray_array: np.ndarray, modality: str) -> np.ndarray:
        """
        SAR: Oil spill = dark pixels (dampened backscatter below median)
        Optical: Oil spill = dark brownish region below median luminance
        """
        if cv2 is not None:
            # Adaptive thresholding based on image statistics
            blur = cv2.GaussianBlur(gray_array, (5, 5), 0)
            median_val = np.median(blur)

            # SAR: backscatter dampening detection
            threshold = max(median_val * 0.75, median_val - 25.0)
            dark_mask = (blur < threshold).astype(np.uint8)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            return cleaned.astype(bool)
        else:
            median_val = np.median(gray_array)
            threshold = max(median_val * 0.75, median_val - 25.0)
            return (gray_array < threshold)

    def _compute_geometry(self, mask: np.ndarray, shape: tuple,
                          spatial_res: Optional[float],
                          center_lat: Optional[float],
                          center_lon: Optional[float]) -> SpillGeometry:
        """Compute all geometry metrics from binary mask."""
        h, w = shape
        mask_uint8 = mask.astype(np.uint8)

        # Pixel area
        pixel_area = int(mask.sum())

        if cv2 is not None:
            # Contours for perimeter and shape
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                main_contour = None
                perimeter_px = 0.0
            else:
                main_contour = max(contours, key=cv2.contourArea)
                perimeter_px = float(cv2.arcLength(main_contour, True))

            # Centroid using moments
            M = cv2.moments(mask_uint8)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = w / 2, h / 2

            # Bounding box
            if main_contour is not None:
                bx, by, bw, bh = cv2.boundingRect(main_contour)
            else:
                bx, by, bw, bh = 0, 0, w, h
        else:
            # Pure numpy fallback
            y_idxs, x_idxs = np.where(mask)
            if len(x_idxs) > 0:
                cx, cy = float(np.mean(x_idxs)), float(np.mean(y_idxs))
                bx, by = int(np.min(x_idxs)), int(np.min(y_idxs))
                bw, bh = int(np.max(x_idxs) - bx + 1), int(np.max(y_idxs) - by + 1)
                perimeter_px = float(2 * (bw + bh))
            else:
                cx, cy = w / 2, h / 2
                bx, by, bw, bh = 0, 0, w, h
                perimeter_px = 0.0
            main_contour = None

        # Shape descriptors
        aspect_ratio = float(bw / max(bh, 1))
        hull = cv2.convexHull(main_contour) if main_contour is not None else None
        hull_area = float(cv2.contourArea(hull)) if hull is not None else float(pixel_area)
        convexity = float(pixel_area / max(hull_area, 1))
        compactness = float(4 * np.pi * pixel_area / max(perimeter_px ** 2, 1))

        # Orientation via PCA or fitEllipse
        orientation_deg = 0.0
        if main_contour is not None and len(main_contour) >= 5:
            try:
                (_, _), (_, _), angle = cv2.fitEllipse(main_contour)
                orientation_deg = float(angle)
            except Exception:
                pass

        # Geographic measurements
        area_km2 = None
        perimeter_km = None
        centroid_lat = None
        centroid_lon = None
        georef_available = False

        if spatial_res is not None:
            # spatial_res in meters/pixel
            area_m2 = pixel_area * (spatial_res ** 2)
            area_km2 = round(area_m2 / 1e6, 4)
            perimeter_km = round(perimeter_px * spatial_res / 1000, 3)
            georef_available = True

            if center_lat is not None and center_lon is not None:
                # Approximate centroid lat/lon from pixel offset
                dx_px = cx - w / 2
                dy_px = cy - h / 2
                # Approx: 1 degree lat ≈ 111 km
                deg_per_px = spatial_res / 111000
                centroid_lat = round(center_lat - dy_px * deg_per_px, 6)
                centroid_lon = round(center_lon + dx_px * deg_per_px /
                                     max(np.cos(np.radians(center_lat)), 0.01), 6)

        confidence = min(0.95, 0.5 + (pixel_area / (h * w)) * 3.0)

        return SpillGeometry(
            pixel_area=pixel_area,
            pixel_perimeter=round(perimeter_px, 2),
            centroid_px=(round(cx, 1), round(cy, 1)),
            bbox_px=(bx, by, bw, bh),
            aspect_ratio=round(aspect_ratio, 3),
            orientation_deg=round(orientation_deg, 1),
            compactness=round(compactness, 4),
            convexity=round(convexity, 4),
            area_km2=area_km2,
            perimeter_km=perimeter_km,
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            image_width_px=w,
            image_height_px=h,
            spatial_resolution_m=spatial_res,
            georef_available=georef_available,
            confidence=round(confidence, 3),
            detection_note="" if area_km2 else (
                "Geographic area unavailable: no spatial resolution metadata. "
                "Provide spatial_resolution_m parameter."
            )
        )
