"""
AIS Data Processor for OceanTrace AI.
Handles loading, cleaning, and feature extraction from real AIS CSV data.
Supports Brest AIS (Zenodo 2563256) format and generic NMEA-derived CSVs.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


class AISDataProcessor:
    """Process and analyze AIS vessel trajectory data."""

    # Realistic vessel speed bounds (knots)
    MAX_VESSEL_SPEED_KT = 50.0
    MIN_VALID_LAT = -90.0
    MAX_VALID_LAT = 90.0

    # Brest AIS canonical column names (Zenodo 2563256)
    BREST_COLUMNS = [
        "MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG",
        "Heading", "VesselName", "IMO", "CallSign", "VesselType",
        "Status", "Length", "Width", "Draft",
    ]

    # Minimum columns we need for any analysis
    REQUIRED_COLUMNS = {"MMSI", "LAT", "LON", "SOG", "BaseDateTime"}

    @classmethod
    def load_brest_format(cls, csv_path: str) -> pd.DataFrame:
        """
        Load Brest AIS dataset format (Zenodo 2563256).
        Columns: MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading,
                 VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"AIS data file not found: {csv_path}")

        df = pd.read_csv(
            path,
            dtype={"MMSI": str, "VesselName": str, "IMO": str,
                   "CallSign": str, "VesselType": str, "Status": str},
            low_memory=False,
        )

        # Rename columns to match Brest format if they differ (case-insensitive)
        df.columns = [c.strip() for c in df.columns]
        col_map = {c.upper(): c for c in df.columns}

        # Parse timestamps
        for ts_col in ("BaseDateTime", "BASEDATETIME", "timestamp", "TIMESTAMP"):
            if ts_col in df.columns or ts_col.upper() in col_map:
                actual = col_map.get(ts_col.upper(), ts_col)
                df[actual] = pd.to_datetime(df[actual], infer_datetime_format=True,
                                            errors="coerce")
                if actual != "BaseDateTime":
                    df.rename(columns={actual: "BaseDateTime"}, inplace=True)
                break

        # Ensure numeric columns
        for num_col in ("LAT", "LON", "SOG", "COG", "Heading"):
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        return df

    @classmethod
    def load_generic_csv(cls, csv_path: str) -> pd.DataFrame:
        """Auto-detect and load generic AIS CSV format."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"AIS data file not found: {csv_path}")

        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        # Build a normalisation map: common aliases -> canonical name
        alias_map = {
            # MMSI
            "mmsi": "MMSI", "vessel_id": "MMSI", "shipid": "MMSI",
            # Latitude
            "lat": "LAT", "latitude": "LAT", "y": "LAT",
            # Longitude
            "lon": "LON", "lng": "LON", "longitude": "LON", "x": "LON",
            # Speed over ground
            "sog": "SOG", "speed": "SOG", "speed_over_ground": "SOG",
            # Course over ground
            "cog": "COG", "course": "COG", "course_over_ground": "COG",
            # Heading
            "heading": "Heading", "true_heading": "Heading",
            # Timestamp
            "basedatetime": "BaseDateTime", "timestamp": "BaseDateTime",
            "datetime": "BaseDateTime", "time": "BaseDateTime",
            # Names / types
            "vesselname": "VesselName", "name": "VesselName", "shipname": "VesselName",
            "vesseltype": "VesselType", "ship_type": "VesselType",
        }

        rename = {c: alias_map[c.lower()] for c in df.columns if c.lower() in alias_map}
        df.rename(columns=rename, inplace=True)

        missing = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        # Parse types
        df["MMSI"] = df["MMSI"].astype(str)
        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"],
                                            infer_datetime_format=True, errors="coerce")
        for num_col in ("LAT", "LON", "SOG", "COG", "Heading"):
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        return df

    @classmethod
    def clean_data(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Remove invalid observations.
        Returns: (cleaned_df, quality_report)
        quality_report: {total, removed_invalid_coords, removed_speed_outliers,
                         removed_duplicates, final_count}
        """
        total = len(df)
        report: dict = {"total": total}

        # 1. Invalid coordinates
        lat_ok = df["LAT"].between(cls.MIN_VALID_LAT, cls.MAX_VALID_LAT)
        lon_ok = df["LON"].between(-180.0, 180.0)
        coord_mask = lat_ok & lon_ok & df["LAT"].notna() & df["LON"].notna()
        removed_coords = int((~coord_mask).sum())
        df = df[coord_mask].copy()
        report["removed_invalid_coords"] = removed_coords

        # 2. Speed outliers
        if "SOG" in df.columns:
            speed_mask = df["SOG"].between(0.0, cls.MAX_VESSEL_SPEED_KT)
            removed_speed = int((~speed_mask).sum())
            df = df[speed_mask].copy()
        else:
            removed_speed = 0
        report["removed_speed_outliers"] = removed_speed

        # 3. Duplicate rows (same MMSI + timestamp + position)
        dup_cols = ["MMSI", "BaseDateTime", "LAT", "LON"]
        dup_cols = [c for c in dup_cols if c in df.columns]
        before_dup = len(df)
        df.drop_duplicates(subset=dup_cols, inplace=True)
        removed_dups = before_dup - len(df)
        report["removed_duplicates"] = int(removed_dups)

        report["final_count"] = len(df)
        return df.reset_index(drop=True), report

    @classmethod
    def filter_spatiotemporal(
        cls,
        df: pd.DataFrame,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """Filter to geographic region and time window."""
        # Spatial filter
        dist = df.apply(
            lambda row: cls.haversine_km(center_lat, center_lon, row["LAT"], row["LON"]),
            axis=1,
        )
        spatial_mask = dist <= radius_km

        # Temporal filter
        if "BaseDateTime" in df.columns:
            ts = df["BaseDateTime"]
            time_mask = (ts >= pd.Timestamp(start_dt)) & (ts <= pd.Timestamp(end_dt))
        else:
            time_mask = pd.Series([True] * len(df), index=df.index)

        return df[spatial_mask & time_mask].copy().reset_index(drop=True)

    @classmethod
    def compute_vessel_tracks(cls, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group by MMSI and sort by timestamp to get individual vessel tracks."""
        tracks: Dict[str, pd.DataFrame] = {}
        for mmsi, group in df.groupby("MMSI"):
            if "BaseDateTime" in group.columns:
                group = group.sort_values("BaseDateTime")
            tracks[str(mmsi)] = group.reset_index(drop=True)
        return tracks

    @classmethod
    def detect_ais_gaps(
        cls,
        track_df: pd.DataFrame,
        gap_threshold_minutes: int = 60,
    ) -> List[dict]:
        """
        Detect AIS transmission gaps (potential dark periods).
        A gap doesn't prove wrongdoing - must be noted as 'observed gap', not 'dark activity'.
        Returns list of {start, end, duration_minutes, lat_before, lon_before}
        """
        if "BaseDateTime" not in track_df.columns or len(track_df) < 2:
            return []

        sorted_df = track_df.sort_values("BaseDateTime").reset_index(drop=True)
        gaps: List[dict] = []
        threshold = timedelta(minutes=gap_threshold_minutes)

        for i in range(1, len(sorted_df)):
            t_prev = sorted_df.loc[i - 1, "BaseDateTime"]
            t_curr = sorted_df.loc[i, "BaseDateTime"]
            if pd.isna(t_prev) or pd.isna(t_curr):
                continue
            duration = t_curr - t_prev
            if duration > threshold:
                gaps.append(
                    {
                        "start": t_prev.isoformat() if hasattr(t_prev, "isoformat") else str(t_prev),
                        "end": t_curr.isoformat() if hasattr(t_curr, "isoformat") else str(t_curr),
                        "duration_minutes": round(duration.total_seconds() / 60, 1),
                        "lat_before": float(sorted_df.loc[i - 1, "LAT"]),
                        "lon_before": float(sorted_df.loc[i - 1, "LON"]),
                        "note": "Observed AIS transmission gap. Does not confirm illicit activity.",
                    }
                )
        return gaps

    @classmethod
    def compute_behavioral_features(cls, track_df: pd.DataFrame) -> dict:
        """
        Compute track-level behavioral features:
        - speed_changes: list of significant speed change events (>5kt)
        - course_changes: list of significant course changes (>30deg)
        - loitering: detected slow-moving periods (SOG < 1kt for >30 min)
        - ais_gaps: detected transmission gaps
        - track_length_km: total track distance
        - time_in_area: hours vessel was in search area
        """
        if len(track_df) < 2:
            return {
                "speed_changes": [],
                "course_changes": [],
                "loitering": [],
                "ais_gaps": [],
                "track_length_km": 0.0,
                "time_in_area": 0.0,
            }

        sorted_df = track_df.sort_values("BaseDateTime").reset_index(drop=True) \
            if "BaseDateTime" in track_df.columns else track_df.reset_index(drop=True)

        # Speed changes
        speed_changes: List[dict] = []
        if "SOG" in sorted_df.columns:
            sog = sorted_df["SOG"].fillna(0)
            for i in range(1, len(sog)):
                delta = abs(float(sog.iloc[i]) - float(sog.iloc[i - 1]))
                if delta > 5.0:
                    speed_changes.append(
                        {
                            "index": i,
                            "delta_kt": round(delta, 2),
                            "from_kt": round(float(sog.iloc[i - 1]), 2),
                            "to_kt": round(float(sog.iloc[i]), 2),
                        }
                    )

        # Course changes
        course_changes: List[dict] = []
        if "COG" in sorted_df.columns:
            cog = sorted_df["COG"].fillna(0)
            for i in range(1, len(cog)):
                delta = abs(float(cog.iloc[i]) - float(cog.iloc[i - 1]))
                # Normalise across 0/360 boundary
                delta = min(delta, 360.0 - delta)
                if delta > 30.0:
                    course_changes.append(
                        {
                            "index": i,
                            "delta_deg": round(delta, 1),
                            "from_deg": round(float(cog.iloc[i - 1]), 1),
                            "to_deg": round(float(cog.iloc[i]), 1),
                        }
                    )

        # Loitering: continuous slow movement
        loitering: List[dict] = []
        if "SOG" in sorted_df.columns and "BaseDateTime" in sorted_df.columns:
            slow_mask = sorted_df["SOG"].fillna(0) < 1.0
            in_loiter = False
            loiter_start_idx = 0
            for i, slow in enumerate(slow_mask):
                if slow and not in_loiter:
                    in_loiter = True
                    loiter_start_idx = i
                elif not slow and in_loiter:
                    in_loiter = False
                    t0 = sorted_df.loc[loiter_start_idx, "BaseDateTime"]
                    t1 = sorted_df.loc[i - 1, "BaseDateTime"]
                    if pd.notna(t0) and pd.notna(t1):
                        dur_min = (t1 - t0).total_seconds() / 60
                        if dur_min >= 30:
                            loitering.append(
                                {
                                    "start": str(t0),
                                    "end": str(t1),
                                    "duration_minutes": round(dur_min, 1),
                                    "lat": float(sorted_df.loc[loiter_start_idx, "LAT"]),
                                    "lon": float(sorted_df.loc[loiter_start_idx, "LON"]),
                                }
                            )

        # AIS gaps
        ais_gaps = cls.detect_ais_gaps(sorted_df, gap_threshold_minutes=60)

        # Track length
        track_length_km = 0.0
        for i in range(1, len(sorted_df)):
            track_length_km += cls.haversine_km(
                float(sorted_df.loc[i - 1, "LAT"]), float(sorted_df.loc[i - 1, "LON"]),
                float(sorted_df.loc[i, "LAT"]), float(sorted_df.loc[i, "LON"]),
            )

        # Time in area (hours between first and last fix)
        time_in_area = 0.0
        if "BaseDateTime" in sorted_df.columns:
            t_first = sorted_df["BaseDateTime"].iloc[0]
            t_last = sorted_df["BaseDateTime"].iloc[-1]
            if pd.notna(t_first) and pd.notna(t_last):
                time_in_area = round((t_last - t_first).total_seconds() / 3600, 2)

        return {
            "speed_changes": speed_changes,
            "course_changes": course_changes,
            "loitering": loitering,
            "ais_gaps": ais_gaps,
            "track_length_km": round(track_length_km, 3),
            "time_in_area": time_in_area,
        }

    @classmethod
    def haversine_km(cls, lat1, lon1, lat2, lon2) -> float:
        """Calculate distance between two points in km."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
