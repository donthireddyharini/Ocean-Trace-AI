# MARITIME FORENSIC INTELLIGENCE REPORT
**CONFIDENTIAL // LAW ENFORCEMENT & MARITIME ADMINISTRATION USE ONLY**  
**Investigation Reference:** {investigation_id}  
**Classification:** RESTRICTED  
**Generated On:** {timestamp_utc} UTC  
**Issuing Authority:** National Maritime Domain Awareness (NMDA) / NTRO / Coast Guard Intelligence Cell

---

### MANDATORY REGULATORY & SCIENTIFIC DISCLAIMER
> *This document provides investigative correlation and analytical intelligence derived from Synthetic Aperture Radar (SAR) satellite imagery, oceanographic hydrodynamic models, and historical Automatic Identification System (AIS) trajectories. Spatiotemporal proximity and algorithmic priority rankings do NOT constitute definitive legal proof of liability or intentional discharge. All findings require physical ground-truthing, sensor calibration, and legal verification prior to enforcement action.*

---

## 1. Executive Summary
- **Spill Detection Timestamp:** {detection_timestamp}
- **Estimated Incident Region:** {coordinates_lat_lon}
- **Estimated Slick Area:** {area_km2} km² ({thick_core_km2} km² heavy emulsion core)
- **Estimated Discharge Volume (Bonn Agreement):** ~{volume_tonnes} Metric Tonnes
- **Top Associated Suspect Vessel:** {top_vessel_name} (MMSI: {top_vessel_mmsi})
- **Attribution Confidence Level:** {attribution_confidence_pct}% ({priority_status})

---

## 2. Satellite Earth Observation & Spill Segmentation
- **Platform:** Sentinel-1 C-SAR / EO Constellation
- **Acquisition Mode:** Interferometric Wide Swath (IW), VV Polarization
- **AI Detection Model:** SegFormer-B2 (Hierarchical Mix Transformer MiT-B2)
- **Confidence Score:** {model_confidence}%
- **False-Positive Look-Alike Screening:** {look_alike_rejection_score}% rejection probability against biogenic slicks and low-wind calm zones

---

## 3. Hydrodynamic Drift Hindcast & Origin Reconstruction
- **Surface Wind Forcing:** {wind_speed_kt} knots @ {wind_dir_deg}° (Source: ERA5 / Open-Meteo)
- **Surface Ocean Current:** {current_speed_kt} knots @ {current_dir_deg}° (Source: HYCOM / Copernicus Marine)
- **Estimated Elapsed Discharge Age:** {elapsed_age_hours} hours
- **Probable Origin Geographic Bounding Box:**
  - Latitude: {origin_lat} ± {origin_uncertainty_km} km
  - Longitude: {origin_lon} ± {origin_uncertainty_km} km
- **Temporal Release Window:** {release_window_utc}

---

## 4. AIS Vessel Attribution & Trajectory Analysis
Total Vessels Screened: **{total_vessels_screened}**  
Candidates Passing Spatial Radius: **{vessels_spatial}**  
Candidates Passing Temporal Window: **{vessels_temporal}**  
Candidates Demonstrating Trajectory Alignment: **{vessels_trajectory}**

### Ranked Investigation Priority Roster

| Rank | MMSI | Vessel Name | Type | Flag | Distance from Origin | Spatial (25) | Temporal (25) | Traj (20) | Drift (20) | Behav (10) | Total Score | Priority Status |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | {mmsi_1} | {name_1} | {type_1} | {flag_1} | {dist_1} km | {s_1} | {t_1} | {tr_1} | {d_1} | {b_1} | **{tot_1}** | {stat_1} |
| 2 | {mmsi_2} | {name_2} | {type_2} | {flag_2} | {dist_2} km | {s_2} | {t_2} | {tr_2} | {d_2} | {b_2} | **{tot_2}** | {stat_2} |

---

## 5. Chain of Custody & Algorithmic Audit Trail
- **Analysis Execution Hash:** `{sha256_audit_hash}`
- **Responsible Analyst:** {analyst_name} ({analyst_org})
- **System Environment:** OceanTrace AI Engine v0.9.5-production (SIH26143)
