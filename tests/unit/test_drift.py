"""
Unit tests for ParticleDriftSimulator.
"""
import pytest
from ml.inference.drift_simulation import ParticleDriftSimulator


def test_drift_initialization_zero_forcing():
    sim = ParticleDriftSimulator(wind_dir_deg=0, wind_speed_kt=0, current_dir_deg=0, current_speed_kt=0)
    assert sim.net_speed_ms == 0.0
    assert sim.net_speed_kt == 0.0


def test_drift_origin_estimation():
    # 20 kt wind coming from/blowing toward 90 deg, 1 kt current at 90 deg
    sim = ParticleDriftSimulator(wind_dir_deg=90, wind_speed_kt=20, current_dir_deg=90, current_speed_kt=1.0)
    result = sim.estimate_origin(observed_lat=15.0, observed_lon=40.0, area_km2=10.0)

    assert result.probable_origin_lat is not None
    assert result.probable_origin_lon is not None
    assert result.estimated_spill_age_hours >= 12.0
    assert len(result.backward_trajectory) > 10
    assert len(result.forward_trajectory) == 49
    assert result.forward_6h is not None
    assert result.forward_12h is not None
    assert result.forward_24h is not None
    assert result.forward_48h is not None
    assert len(result.origin_probability_bands) == 3


def test_particle_advance():
    sim = ParticleDriftSimulator(wind_dir_deg=0, wind_speed_kt=10, current_dir_deg=0, current_speed_kt=0.5)
    lat2, lon2 = sim._advance_particle(0.0, 0.0, speed_km_h=10.0, bearing_deg=0.0, dt_hours=1.0)
    # Moving due North from Equator increases latitude
    assert lat2 > 0.0
    assert abs(lon2) < 0.001
