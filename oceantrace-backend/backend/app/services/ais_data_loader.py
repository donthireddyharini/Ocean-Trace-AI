"""
AIS Data Loader - loads and processes real AIS CSV data.
Supports Brest AIS (Zenodo 2563256) and standard NMEA-derived CSV formats.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timezone

from app.schemas import VesselRecord


class AISDataLoader:
    """
    Load, normalise, and filter AIS CSV records.

    Supported formats
    -----------------
    brest   : Zenodo 2563256 — columns MMSI, BaseDateTime, LAT, LON, SOG, COG,
              Heading, VesselName, VesselType
    standard: Generic AIS CSV — lower-case or varied column names are mapped
              automatically via STANDARD_ALIASES.
    auto    : (default) Try brest first, then standard.
    """

    # Brest AIS canonical column names
    BREST_COLUMNS = {
        'mmsi': 'MMSI',
        'lat': 'LAT',
        'lon': 'LON',
        'sog': 'SOG',
        'cog': 'COG',
        'timestamp': 'BaseDateTime',
        'vessel_name': 'VesselName',
        'vessel_type': 'VesselType',
    }

    # Aliases for generic/standard CSVs (key = canonical, values = accepted names)
    STANDARD_ALIASES: dict = {
        'MMSI': ['mmsi', 'MMSI', 'vessel_id'],
        'LAT': ['lat', 'LAT', 'latitude', 'Latitude'],
        'LON': ['lon', 'LON', 'longitude', 'Longitude'],
        'SOG': ['sog', 'SOG', 'speed', 'Speed'],
        'COG': ['cog', 'COG', 'course', 'Course'],
        'BaseDateTime': ['BaseDateTime', 'timestamp', 'Timestamp', 'datetime', 'DateTime', 'time'],
        'VesselName': ['VesselName', 'vessel_name', 'name', 'Name', 'shipname'],
        'VesselType': ['VesselType', 'vessel_type', 'type', 'Type', 'ship_type'],
    }

    # ------------------------------------------------------------------ load
    @classmethod
    def load_csv(cls, csv_path: str, format: str = 'auto') -> pd.DataFrame:
        """
        Load an AIS CSV file.

        Parameters
        ----------
        csv_path : path to the CSV file
        format   : 'brest', 'standard', or 'auto'

        Returns
        -------
        Normalised DataFrame with canonical column names, parsed timestamps,
        valid lat/lon, and rows with critical missing values removed.
        """
        df = pd.read_csv(csv_path, low_memory=False)

        if format == 'auto':
            if 'BaseDateTime' in df.columns and 'LAT' in df.columns:
                format = 'brest'
            else:
                format = 'standard'

        if format == 'brest':
            df = cls._normalise_brest(df)
        else:
            df = cls._normalise_standard(df)

        df = cls._clean(df)
        return df

    @classmethod
    def _normalise_brest(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Brest format is already canonical — just ensure expected cols exist."""
        expected = set(cls.BREST_COLUMNS.values())
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Brest AIS CSV missing columns: {missing}")
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], utc=True, errors='coerce')
        return df

    @classmethod
    def _normalise_standard(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Map aliased column names to canonical names."""
        rename_map: dict = {}
        for canonical, aliases in cls.STANDARD_ALIASES.items():
            for alias in aliases:
                if alias in df.columns and canonical not in df.columns:
                    rename_map[alias] = canonical
                    break
        df = df.rename(columns=rename_map)
        if 'BaseDateTime' in df.columns:
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], utc=True, errors='coerce')
        return df

    @classmethod
    def _clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid or missing critical fields."""
        for col in ('MMSI', 'LAT', 'LON'):
            if col in df.columns:
                df = df.dropna(subset=[col])

        if 'LAT' in df.columns:
            df = df[(df['LAT'] >= -90) & (df['LAT'] <= 90)]
        if 'LON' in df.columns:
            df = df[(df['LON'] >= -180) & (df['LON'] <= 180)]
        if 'SOG' in df.columns:
            df['SOG'] = pd.to_numeric(df['SOG'], errors='coerce').fillna(0.0)
        if 'COG' in df.columns:
            df['COG'] = pd.to_numeric(df['COG'], errors='coerce').fillna(0.0)

        df = df.reset_index(drop=True)
        return df

    # --------------------------------------------------------- spatial filter
    @classmethod
    def filter_by_region_and_time(
        cls,
        df: pd.DataFrame,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame:
        """
        Filter AIS records to a geographic region (great-circle distance)
        and an inclusive time window.

        Parameters
        ----------
        df          : normalised AIS DataFrame (from load_csv)
        center_lat  : region centre latitude
        center_lon  : region centre longitude
        radius_km   : search radius in kilometres
        start_time  : ISO-8601 string, e.g. '2024-01-01T00:00:00Z'
        end_time    : ISO-8601 string
        """
        if df.empty:
            return df

        # Spatial filter — vectorised haversine
        lat_r = np.radians(df['LAT'].values)
        lon_r = np.radians(df['LON'].values)
        clat_r = np.radians(center_lat)
        clon_r = np.radians(center_lon)

        dlat = lat_r - clat_r
        dlon = lon_r - clon_r
        a = np.sin(dlat / 2) ** 2 + np.cos(clat_r) * np.cos(lat_r) * np.sin(dlon / 2) ** 2
        dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        spatial_mask = dist_km <= radius_km

        # Temporal filter
        if 'BaseDateTime' in df.columns:
            t0 = pd.Timestamp(start_time, tz='UTC')
            t1 = pd.Timestamp(end_time, tz='UTC')
            ts = df['BaseDateTime']
            # coerce non-tz-aware to UTC
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize('UTC')
            time_mask = (ts >= t0) & (ts <= t1)
        else:
            time_mask = pd.Series([True] * len(df), index=df.index)

        return df[spatial_mask & time_mask].reset_index(drop=True)

    # -------------------------------------------------- conversion to schema
    @classmethod
    def to_vessel_records(cls, df: pd.DataFrame, reference_time: str) -> List[VesselRecord]:
        """
        Convert a filtered AIS DataFrame to a list of VesselRecord objects
        suitable for the AISAnalyzer scorer.

        One VesselRecord per MMSI (uses the most recent position per vessel).

        Parameters
        ----------
        df             : filtered AIS DataFrame
        reference_time : ISO-8601 string — the detection/spill time
        """
        if df.empty:
            return []

        ref_ts = pd.Timestamp(reference_time, tz='UTC')
        records: List[VesselRecord] = []

        # Group by MMSI and pick the latest position
        if 'BaseDateTime' in df.columns:
            ts_col = df['BaseDateTime']
            if ts_col.dt.tz is None:
                ts_col = ts_col.dt.tz_localize('UTC')
            df = df.copy()
            df['_ts'] = ts_col
            latest = df.sort_values('_ts').groupby('MMSI', as_index=False).last()
        else:
            latest = df.groupby('MMSI', as_index=False).last()

        for _, row in latest.iterrows():
            mmsi = str(row.get('MMSI', 'UNKNOWN'))
            name = str(row.get('VesselName', mmsi)) if pd.notna(row.get('VesselName')) else mmsi
            vtype = str(row.get('VesselType', 'Unknown')) if pd.notna(row.get('VesselType')) else 'Unknown'

            # Timestamp offset in hours relative to detection
            offset_h = 0.0
            if '_ts' in row.index and pd.notna(row['_ts']):
                delta = row['_ts'] - ref_ts
                offset_h = delta.total_seconds() / 3600.0

            records.append(VesselRecord(
                mmsi=mmsi,
                name=name,
                type=vtype,
                flag='Unknown',
                lat=float(row.get('LAT', 0.0)),
                lon=float(row.get('LON', 0.0)),
                heading_deg=float(row.get('COG', 0.0)),
                timestamp_offset_h=round(offset_h, 2),
            ))

        return records

    # ------------------------------------------------- track statistics
    @classmethod
    def compute_track_statistics(cls, df: pd.DataFrame) -> dict:
        """
        Compute per-vessel track statistics:
        - mean / max speed (SOG)
        - mean / max heading change between consecutive points
        - AIS gap count (gaps > 30 minutes in sorted timestamps)

        Returns
        -------
        dict keyed by MMSI string with statistics sub-dict.
        """
        if df.empty:
            return {}

        stats: dict = {}

        for mmsi, group in df.groupby('MMSI'):
            group = group.copy()

            # Sort by time if available
            if 'BaseDateTime' in group.columns:
                ts = group['BaseDateTime']
                if ts.dt.tz is None:
                    ts = ts.dt.tz_localize('UTC')
                group['_ts'] = ts
                group = group.sort_values('_ts')

            sog = pd.to_numeric(group.get('SOG', pd.Series(dtype=float)), errors='coerce').dropna()
            cog = pd.to_numeric(group.get('COG', pd.Series(dtype=float)), errors='coerce').dropna()

            # Heading changes
            heading_changes: List[float] = []
            cog_vals = cog.values
            for i in range(1, len(cog_vals)):
                delta = abs(cog_vals[i] - cog_vals[i - 1])
                if delta > 180:
                    delta = 360 - delta
                heading_changes.append(delta)

            # AIS gaps > 30 min
            gap_count = 0
            if '_ts' in group.columns:
                ts_sorted = group['_ts'].dropna().reset_index(drop=True)
                for i in range(1, len(ts_sorted)):
                    gap_min = (ts_sorted[i] - ts_sorted[i - 1]).total_seconds() / 60
                    if gap_min > 30:
                        gap_count += 1

            stats[str(mmsi)] = {
                'point_count': len(group),
                'mean_sog_kt': round(float(sog.mean()), 2) if len(sog) else None,
                'max_sog_kt': round(float(sog.max()), 2) if len(sog) else None,
                'mean_heading_change_deg': round(float(np.mean(heading_changes)), 2) if heading_changes else None,
                'max_heading_change_deg': round(float(np.max(heading_changes)), 2) if heading_changes else None,
                'ais_gap_count': gap_count,
            }

        return stats
