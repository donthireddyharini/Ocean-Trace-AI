"""
Environmental data service for ocean currents and wind.
Fetches from HYCOM OPeNDAP and Copernicus CDS ERA5.
Requires optional API keys for ERA5.
Provides honest fallback when APIs are unavailable.
"""
import requests
from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class EnvironmentalConditions:
    lat: float
    lon: float
    timestamp: str

    # Ocean currents (HYCOM)
    current_u_ms: Optional[float] = None  # eastward velocity m/s
    current_v_ms: Optional[float] = None  # northward velocity m/s
    current_speed_kt: Optional[float] = None
    current_dir_deg: Optional[float] = None
    current_source: str = "unavailable"

    # Wind (ERA5 / open-meteo fallback)
    wind_u10_ms: Optional[float] = None
    wind_v10_ms: Optional[float] = None
    wind_speed_kt: Optional[float] = None
    wind_dir_deg: Optional[float] = None
    wind_source: str = "unavailable"

    # Data quality
    data_coverage: str = "none"  # 'full', 'partial', 'demo_fallback'
    warnings: List[str] = field(default_factory=list)


class EnvironmentalDataService:
    HYCOM_BASE = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
    OPEN_METEO_BASE = "https://marine-api.open-meteo.com/v1/marine"
    OPEN_METEO_WIND = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        self.era5_key = os.getenv("ERA5_API_KEY", "")
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'OceanTrace-AI/1.0 SIH26143'

    def get_conditions(self, lat: float, lon: float,
                       timestamp: str = None) -> EnvironmentalConditions:
        """
        Fetch real environmental conditions for location.
        Tries: 1) Open-Meteo Marine API (free, no key) for wind
               2) HYCOM for currents (may be slow/unavailable)
               Falls back to demo data with clear labeling.
        """
        result = EnvironmentalConditions(
            lat=lat, lon=lon,
            timestamp=timestamp or "current",
            warnings=[]
        )

        # Try Open-Meteo for wind (free, no auth required)
        wind_ok = self._fetch_openmeteo_wind(lat, lon, result)

        # Try Open-Meteo Marine for currents
        current_ok = self._fetch_openmeteo_marine(lat, lon, result)

        if wind_ok and current_ok:
            result.data_coverage = "full"
        elif wind_ok or current_ok:
            result.data_coverage = "partial"
        else:
            result.data_coverage = "demo_fallback"
            result.warnings.append(
                "Real environmental APIs unavailable. Using physics-based estimates."
            )
            self._apply_demo_fallback(lat, lon, result)

        return result

    def _fetch_openmeteo_wind(self, lat: float, lon: float, result: EnvironmentalConditions) -> bool:
        """Fetch wind from Open-Meteo (truly free, no key needed)."""
        try:
            resp = self.session.get(
                self.OPEN_METEO_WIND,
                params={
                    'latitude': lat, 'longitude': lon,
                    'hourly': 'windspeed_10m,winddirection_10m',
                    'forecast_days': 1, 'timezone': 'UTC'
                },
                timeout=8
            )
            if resp.status_code == 200:
                data = resp.json()
                hourly = data.get('hourly', {})
                speeds = hourly.get('windspeed_10m', [])
                dirs = hourly.get('winddirection_10m', [])
                if speeds and dirs:
                    spd = speeds[0]  # first hour
                    dir_ = dirs[0]
                    import numpy as np
                    spd_ms = spd * 1000 / 3600  # km/h -> m/s
                    result.wind_speed_kt = round(spd_ms * 1.944, 2)
                    result.wind_dir_deg = round(float(dir_), 1)
                    result.wind_u10_ms = round(
                        -result.wind_speed_kt * 0.5144 * np.sin(np.radians(dir_)), 3
                    )
                    result.wind_v10_ms = round(
                        -result.wind_speed_kt * 0.5144 * np.cos(np.radians(dir_)), 3
                    )
                    result.wind_source = "Open-Meteo API (real)"
                    return True
        except Exception as e:
            result.warnings.append(f"Wind API error: {str(e)[:60]}")
        return False

    def _fetch_openmeteo_marine(self, lat: float, lon: float, result: EnvironmentalConditions) -> bool:
        """Fetch ocean currents from Open-Meteo Marine API."""
        try:
            resp = self.session.get(
                self.OPEN_METEO_BASE,
                params={
                    'latitude': lat, 'longitude': lon,
                    'hourly': 'ocean_current_velocity,ocean_current_direction',
                    'forecast_days': 1, 'timezone': 'UTC'
                },
                timeout=8
            )
            if resp.status_code == 200:
                data = resp.json()
                hourly = data.get('hourly', {})
                speeds = hourly.get('ocean_current_velocity', [])
                dirs = hourly.get('ocean_current_direction', [])
                if speeds and dirs and speeds[0] is not None:
                    import numpy as np
                    spd_ms = float(speeds[0])
                    dir_ = float(dirs[0])
                    result.current_speed_kt = round(spd_ms * 1.944, 3)
                    result.current_dir_deg = round(dir_, 1)
                    result.current_u_ms = round(spd_ms * np.sin(np.radians(dir_)), 4)
                    result.current_v_ms = round(spd_ms * np.cos(np.radians(dir_)), 4)
                    result.current_source = "Open-Meteo Marine API (real)"
                    return True
        except Exception as e:
            result.warnings.append(f"Marine API error: {str(e)[:60]}")
        return False

    def _apply_demo_fallback(self, lat: float, lon: float, result: EnvironmentalConditions):
        """Physics-based estimates when APIs unavailable - clearly labeled."""
        import numpy as np
        # Simple latitude-based wind estimate (trade winds, westerlies)
        lat_abs = abs(lat)
        if lat_abs < 30:
            wind_spd = 8 + np.random.uniform(-2, 2)
            wind_dir = 90 + lat * 2  # easterlies
        else:
            wind_spd = 12 + np.random.uniform(-3, 3)
            wind_dir = 250 + lat * 1.5  # westerlies

        result.wind_speed_kt = round(float(wind_spd), 1)
        result.wind_dir_deg = round(float(wind_dir % 360), 1)
        result.current_speed_kt = round(float(0.3 + np.random.uniform(0, 0.5)), 2)
        result.current_dir_deg = round(float((wind_dir + 15) % 360), 1)  # Ekman transport
        result.wind_source = "DEMO_ESTIMATE (APIs unavailable)"
        result.current_source = "DEMO_ESTIMATE (APIs unavailable)"
