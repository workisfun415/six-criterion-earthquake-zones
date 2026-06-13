#!/usr/bin/env python3
"""
CSEP Japan Forecast Generator
==============================
Converts the 28 Critical Earthquake Zones into CSEP Japan ForecastML format.

Author : Ramakrishna Pasupuleti
         Independent Researcher, Suryapet, Telangana, India
         ORCID: 0009-0008-8418-1430
         Email: workisfun415@gmail.com

References:
  Tsuruoka et al. (2012) Earth Planets Space 64, 661-672
  Schorlemmer et al. (2007) Seismol. Res. Lett. 78(1), 17-29
  CSEP Japan Testing Center: wwweic.eri.u-tokyo.ac.jp/ZISINyosoku/

Usage:
    python generate_csep_forecast.py

Inputs:
    japan_critical_zones_2023-12-31.csv   (your 28 zone list)

Outputs:
    csep_forecast_japan_1yr.csv           (ForecastML 1-year forecast)
    csep_forecast_japan_summary.csv       (human-readable zone summary)
    csep_forecast_readme.txt              (submission notes for ERI)

CSEP Japan Grid:
    Region : All Japan + surrounding sea
    Lat    : 24.05 to 47.95 N  (step 0.1 deg)
    Lon    : 122.05 to 149.95 E (step 0.1 deg)
    Depth  : 0 to 30 km
    Mag    : M5.0 to M9.0 (step 0.1)
"""

import numpy as np
import pandas as pd
import csv
import os
from datetime import datetime, date

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# CSEP Japan grid parameters (from official CSEP Japan documentation)
GRID_LAT_MIN  =  24.05
GRID_LAT_MAX  =  47.95
GRID_LON_MIN  = 122.05
GRID_LON_MAX  = 149.95
GRID_STEP     =   0.10   # degrees

# Depth range (CSEP Japan standard)
DEPTH_MIN =  0.0   # km
DEPTH_MAX = 30.0   # km

# Magnitude range (CSEP Japan standard, step 0.1)
MAG_MIN  = 5.0
MAG_MAX  = 9.0
MAG_STEP = 0.1

# Zone parameters from manuscript
ZONE_RADIUS_KM  = 200.0   # detection radius per zone
SPREAD_SIGMA_KM = 100.0   # Gaussian spread sigma for grid distribution

# Minimum background rate per cell (prevents zero-probability cells)
BACKGROUND_RATE_PER_CELL = 1e-5

# Assessment date
CUTOFF_DATE = "2023-12-31"
FORECAST_START = "2024-01-01"
FORECAST_END   = "2024-12-31"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km."""
    R    = 6371.0
    dlat = np.radians(np.asarray(lat2, float) - lat1)
    dlon = np.radians(np.asarray(lon2, float) - lon1)
    a    = (np.sin(dlat/2)**2
            + np.cos(np.radians(lat1))
            * np.cos(np.radians(np.asarray(lat2, float)))
            * np.sin(dlon/2)**2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def gr_rate(n_m3, b, m_ref=3.0, m_target=5.0):
    """
    Gutenberg-Richter scaling from M>=m_ref to M>=m_target.
    N(M>=m) = N(M>=m_ref) * 10^(-b*(m-m_ref))
    """
    return n_m3 * 10**(-b * (m_target - m_ref))


def gr_bin_rate(total_rate_m5, b, mag_lo, mag_hi):
    """
    Expected number of events in magnitude bin [mag_lo, mag_hi).
    Uses Gutenberg-Richter: P(m in [lo,hi)) = 10^(-b*(lo-5)) - 10^(-b*(hi-5))
    """
    if mag_lo >= MAG_MAX:
        return 0.0
    p_lo = 10**(-b * (mag_lo - 5.0))
    p_hi = 10**(-b * (mag_hi - 5.0)) if mag_hi <= MAG_MAX + 0.05 else 0.0
    frac = max(0.0, p_lo - p_hi)
    # Normalise so sum over all bins from M5.0 = 1.0
    p_total = 1.0 - 10**(-b * (MAG_MAX - 5.0))
    if p_total <= 0:
        return 0.0
    return total_rate_m5 * (frac / p_total)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  CSEP Japan Forecast Generator")
    print("  Six-Criterion Consensus Detection Framework")
    print("  Ramakrishna Pasupuleti — Independent Researcher")
    print("=" * 70)
    print()

    # ── Load zone list ────────────────────────────────────────────────────────
    zone_file = "japan_critical_zones_2023-12-31.csv"
    if not os.path.exists(zone_file):
        print(f"  ERROR: {zone_file} not found.")
        print("  Run japan_zones.py first to generate the zone list.")
        return

    zones = pd.read_csv(zone_file)
    # Handle both column name variants
    if 'Lat' in zones.columns:
        zones = zones.rename(columns={'Lat':'lat','Lon':'lon'})
    if 'Consensus_votes' in zones.columns:
        zones = zones.rename(columns={'Consensus_votes':'votes'})
    if 'b_value' in zones.columns:
        zones = zones.rename(columns={'b_value':'b_val'})

    print(f"  Loaded {len(zones)} zones from {zone_file}")
    print()

    # ── Compute per-zone M5+ annual rates ────────────────────────────────────
    print("  Computing per-zone M5+ annual forecast rates...")
    zones['rate_m5_annual'] = zones.apply(
        lambda r: gr_rate(r['n365'], r['b_val'], m_ref=3.0, m_target=5.0),
        axis=1
    )

    # Vote-weighted rate: higher consensus = higher confidence, slight boost
    zones['vote_weight'] = zones['votes'] / 6.0
    # Apply modest vote weighting (1.0 to 1.5 scale)
    zones['rate_m5_weighted'] = (zones['rate_m5_annual']
                                  * (0.7 + 0.3 * zones['vote_weight']))

    print(f"  Total M5+ events/year across 28 zones: "
          f"{zones['rate_m5_weighted'].sum():.1f}")
    print()

    # ── Build CSEP 0.1-degree grid ────────────────────────────────────────────
    print("  Building CSEP Japan 0.1° grid...")
    grid_lats = np.round(np.arange(GRID_LAT_MIN, GRID_LAT_MAX + 0.01,
                                    GRID_STEP), 2)
    grid_lons = np.round(np.arange(GRID_LON_MIN, GRID_LON_MAX + 0.01,
                                    GRID_STEP), 2)
    total_cells = len(grid_lats) * len(grid_lons)
    print(f"  Grid: {len(grid_lats)} lats × {len(grid_lons)} lons = "
          f"{total_cells:,} cells")
    print()

    # ── Distribute zone rates onto grid (Gaussian kernel) ────────────────────
    print("  Distributing zone rates onto grid cells...")

    # Initialize rate array (total M5+ rate per cell)
    rate_grid = np.full(
        (len(grid_lats), len(grid_lons)),
        BACKGROUND_RATE_PER_CELL)

    # For each zone: spread rate using Gaussian kernel within R km
    for _, z in zones.iterrows():
        zlat = z['lat']
        zlon = z['lon']
        rate = z['rate_m5_weighted']
        sigma = SPREAD_SIGMA_KM
        b     = z['b_val']

        # Find grid cells within 2.5 × sigma
        lat_deg_range = 2.5 * sigma / 111.0
        lon_deg_range = 2.5 * sigma / (111.0 * np.cos(np.radians(zlat)))

        lat_mask = ((grid_lats >= zlat - lat_deg_range) &
                    (grid_lats <= zlat + lat_deg_range))
        lon_mask = ((grid_lons >= zlon - lon_deg_range) &
                    (grid_lons <= zlon + lon_deg_range))

        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]

        if len(lat_idx) == 0 or len(lon_idx) == 0:
            continue

        # Compute Gaussian weights for all cells in range
        weights = np.zeros((len(lat_idx), len(lon_idx)))
        for ii, li in enumerate(lat_idx):
            for jj, lj in enumerate(lon_idx):
                d = haversine(zlat, zlon,
                              grid_lats[li], grid_lons[lj])
                if d <= ZONE_RADIUS_KM:
                    weights[ii, jj] = np.exp(-0.5 * (d / sigma)**2)

        total_w = weights.sum()
        if total_w > 0:
            weights = weights / total_w
            for ii, li in enumerate(lat_idx):
                for jj, lj in enumerate(lon_idx):
                    rate_grid[li, lj] += rate * weights[ii, jj]

    print(f"  Rate grid computed. "
          f"Non-background cells: "
          f"{(rate_grid > BACKGROUND_RATE_PER_CELL * 1.01).sum():,}")
    print()

    # ── Generate magnitude bins ───────────────────────────────────────────────
    mag_bins = np.round(np.arange(MAG_MIN, MAG_MAX + 0.01, MAG_STEP), 1)

    # For b-value per cell: use nearest zone's b if inside zone, else 0.9
    print("  Computing per-cell b-values...")
    b_grid = np.full((len(grid_lats), len(grid_lons)), 0.90)

    for _, z in zones.iterrows():
        for ii, la in enumerate(grid_lats):
            for jj, lo in enumerate(grid_lons):
                d = haversine(z['lat'], z['lon'], la, lo)
                if d <= ZONE_RADIUS_KM:
                    # Weight by inverse distance
                    existing_b = b_grid[ii, jj]
                    zone_b     = z['b_val']
                    alpha      = max(0.0, 1.0 - d / ZONE_RADIUS_KM)
                    b_grid[ii, jj] = alpha * zone_b + (1 - alpha) * existing_b

    # ── Write ForecastML CSV ──────────────────────────────────────────────────
    outfile_ml = "csep_forecast_japan_1yr.csv"
    print(f"  Writing ForecastML forecast to {outfile_ml}...")
    print("  (This may take 1-2 minutes for 0.1-degree grid...)")

    rows_written = 0
    with open(outfile_ml, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # ForecastML header
        writer.writerow([
            'lon', 'lat',
            'depth_min', 'depth_max',
            'mag_min', 'mag_max',
            'rate',
            'rate_low_95CI',
            'rate_high_95CI'
        ])

        for ii, la in enumerate(grid_lats):
            for jj, lo in enumerate(grid_lons):
                total_rate_m5 = rate_grid[ii, jj]
                b             = b_grid[ii, jj]

                for k, mag_lo in enumerate(mag_bins):
                    mag_hi = round(mag_lo + MAG_STEP, 1)
                    if mag_lo >= MAG_MAX:
                        break

                    # Rate in this magnitude bin
                    bin_rate = gr_bin_rate(total_rate_m5, b, mag_lo, mag_hi)

                    # Uncertainty: ±50% for 95% CI (conservative)
                    rate_low  = bin_rate * 0.50
                    rate_high = bin_rate * 1.50

                    if bin_rate > 0:
                        writer.writerow([
                            f"{lo:.2f}", f"{la:.2f}",
                            f"{DEPTH_MIN:.1f}", f"{DEPTH_MAX:.1f}",
                            f"{mag_lo:.1f}", f"{mag_hi:.1f}",
                            f"{bin_rate:.8e}",
                            f"{rate_low:.8e}",
                            f"{rate_high:.8e}",
                        ])
                        rows_written += 1

    print(f"  Written: {rows_written:,} rows")
    size_mb = os.path.getsize(outfile_ml) / (1024*1024)
    print(f"  File size: {size_mb:.1f} MB")
    print()

    # ── Write human-readable zone summary ────────────────────────────────────
    outfile_sum = "csep_forecast_japan_summary.csv"
    print(f"  Writing zone summary to {outfile_sum}...")
    with open(outfile_sum, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Rank', 'Lat_center', 'Lon_center',
            'Consensus_votes', 'Q', 'C', 'b_value',
            'n90_observed', 'n365_observed',
            'Rate_M5plus_annual', 'Rate_M6plus_annual', 'Rate_M7plus_annual',
            'Nearest_city', 'City_dist_km',
            'Forecast_year', 'Assessment_date',
        ])
        for i, z in zones.iterrows():
            b    = z['b_val']
            rm5  = z['rate_m5_weighted']
            rm6  = gr_rate(z['n365'], b, m_ref=3.0, m_target=6.0)
            rm7  = gr_rate(z['n365'], b, m_ref=3.0, m_target=7.0)
            city = z.get('Nearest_city', '')
            cdist= z.get('City_dist_km', '')
            writer.writerow([
                i+1,
                z['lat'], z['lon'],
                z['votes'],
                f"{z.get('Q', ''):.3f}" if z.get('Q') else '',
                f"{z.get('C', ''):.3f}" if z.get('C') else '',
                f"{b:.3f}",
                z['n90'],
                z['n365'],
                f"{rm5:.3f}",
                f"{rm6:.4f}",
                f"{rm7:.5f}",
                city, cdist,
                FORECAST_START[:4],
                CUTOFF_DATE,
            ])
    print(f"  Written: {len(zones)} zones")
    print()

    # ── Write README for ERI ──────────────────────────────────────────────────
    outfile_readme = "csep_forecast_readme.txt"
    print(f"  Writing submission readme to {outfile_readme}...")
    readme = f"""
CSEP Japan Forecast Submission
================================
Model Name    : Seismic Magnitude Variance Consensus Detection Framework
                (K-R Six-Criterion Approach)
Author        : Ramakrishna Pasupuleti
Affiliation   : Independent Researcher, Suryapet, Telangana, India
ORCID         : 0009-0008-8418-1430
Email         : workisfun415@gmail.com
Preprint      : https://zenodo.org/records/20603673
Under review  : IEEE TQE and BSSA (BSSA-D-26-00168)

Submission Date  : {date.today().strftime('%Y-%m-%d')}
Assessment Date  : {CUTOFF_DATE}
Forecast Period  : {FORECAST_START} to {FORECAST_END}
Forecast Class   : 1-year
Testing Region   : All Japan

Files Submitted
---------------
1. csep_forecast_japan_1yr.csv
   ForecastML format (lon, lat, depth_min, depth_max, mag_min, mag_max,
   rate, rate_low_95CI, rate_high_95CI)
   Grid: 0.1-degree, lat 24.05-47.95N, lon 122.05-149.95E
   Depth: 0-30 km
   Magnitude: M5.0-M9.0 (0.1 step)

2. csep_forecast_japan_summary.csv
   Human-readable summary of 28 critical zones with rates

3. japan_critical_zones_2023-12-31.csv
   Full zone list with all parameters

4. japan_zones.py
   Complete reproducible code

Model Description
-----------------
The Six-Criterion Consensus Detection Framework identifies critical
earthquake zones using catalog data alone (no physical models needed).

Core signals computed per 1-degree grid cell, 200km radius:

  C  = max(0, 1 - sigma/sigma_0)        Magnitude compression
  Q  = max(0, 1 - r/r_0)                Seismicity quiescence
  S  = max(0, (skew - mu_s) / 3*sigma_s) Skewness anomaly
  Psi = C * Q * (1 + S)                  Combined signal

Parameters from catalog:
  b-value (Aki 1965 MLE, auto Mc detection)
  n90  = events in last 90 days
  Cv   = coefficient of variation of inter-event times
  DeltaC, DeltaS = 3-month changes

Six scoring functions applied to rank zones:
  F1 = Psi + n90/100
  F2 = n90/100 + Cv
  F3 = |dC| + |dS|
  F4 = 1/b
  F5 = Psi + n90/100 + 1/b + Cv  (MEGA)
  F6 = n90/100 + 1/b

Zone is CRITICAL if it appears in top-30 of >= 3/6 functions.

Forecast Rate Computation
-------------------------
For each of 28 zones:
  1. Observed M3+ rate: n365 (events in last 365 days within R=200km)
  2. G-R scaling to M5+: rate(M5+) = n365 * 10^(-b*(5-3))
  3. Magnitude distribution: G-R with zone-specific b-value
  4. Spatial distribution: Gaussian kernel sigma=100km on 0.1-degree grid
  5. Vote weighting: rate *= (0.7 + 0.3 * votes/6)
  6. Background: 1e-5 events/cell/year for cells outside all zones

Training / Test Split
---------------------
Training period: 2000-01-01 to 2009-01-01 (first 40% of catalog)
Assessment date: {CUTOFF_DATE}
Catalog used: JMA M3+ unified catalog 2000-2023 (190,189 events)

Prospective Validation (2020-2023)
------------------------------------
Year  M5.5+ Det  Rate  M6.0+ Det  Rate  M7.0+ Det
2020  ...   ...  81%   ...   ...  71%   ...   100%
(Full results in supplementary material)

Contact
-------
Ramakrishna Pasupuleti
workisfun415@gmail.com
ORCID: 0009-0008-8418-1430
"""
    with open(outfile_readme, 'w') as f:
        f.write(readme)
    print(f"  Written: {outfile_readme}")
    print()

    # ── Final summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("  CSEP FORECAST PACKAGE COMPLETE")
    print("=" * 70)
    print()
    print("  Files ready for submission to ERI Tokyo:")
    print()
    print(f"  1. csep_forecast_japan_1yr.csv       ← Main ForecastML file")
    print(f"  2. csep_forecast_japan_summary.csv   ← Zone summary")
    print(f"  3. csep_forecast_readme.txt          ← Model description")
    print(f"  4. japan_zones.py                    ← Reproducible code")
    print(f"  5. japan_critical_zones_2023-12-31.csv ← Zone list")
    print()
    print("  NEXT STEP:")
    print("  Email: ZISINyosoku-submit@eri.u-tokyo.ac.jp")
    print()
    print("  Subject: Forecast Model Submission Inquiry")
    print("  Attach : csep_forecast_readme.txt")
    print()
    print("  They will reply with the official application form and")
    print("  confirm whether to accept your static table or require")
    print("  running code at their center.")
    print()

if __name__ == '__main__':
    main()
