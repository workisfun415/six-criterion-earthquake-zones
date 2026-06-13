#!/usr/bin/env python3
"""
Japan Critical Earthquake Zones — Six-Criterion Consensus Framework
=====================================================================
Author : Ramakrishna Pasupuleti
         Independent Researcher, Suryapet, Telangana, India
         ORCID: 0009-0008-8418-1430

Usage:
    # Latest assessment
    py japan_zones.py --catalog jma_M3plus_2000_2023.csv

    # Specific date (paper's Dec 31 2023)
    py japan_zones.py --catalog jma_M3plus_2000_2023.csv --cutoff 2023-12-31

    # Year-by-year test 2020-2023 (reproduces paper Table 2)
    py japan_zones.py --catalog jma_M3plus_2000_2023.csv --yearly

    # Both at once
    py japan_zones.py --catalog jma_M3plus_2000_2023.csv --cutoff 2023-12-31 --yearly

Requirements:
    pip install numpy pandas
"""

import numpy as np
import pandas as pd
import csv
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# CONFIGURATION — exact BSSA manuscript parameters
# ============================================================
LAT_MIN, LAT_MAX = 30, 46
LON_MIN, LON_MAX = 128, 148
GRID_STEP   = 1.0       # 1-degree grid
RADIUS_KM   = 200       # search radius per cell
W           = 20        # rolling window (events)
MIN_MAG     = 3.0       # minimum magnitude
TRAIN_END   = '2009-01-01'  # first 40% of 2000-2023
Q_THRESH    = 0.90      # quiescence filter
TOP_N       = 30        # top-N per scoring function
MIN_VOTES   = 3         # consensus threshold
DETECT_R    = 200       # detection radius for validation (km)

CITIES = [
    ('Tokyo',    35.68, 139.69),
    ('Osaka',    34.69, 135.50),
    ('Sendai',   38.27, 140.87),
    ('Sapporo',  43.06, 141.35),
    ('Fukuoka',  33.59, 130.40),
    ('Nagoya',   35.18, 136.91),
    ('Noto',     37.49, 137.27),
    ('Hokkaido', 42.50, 143.00),
    ('Sanriku',  39.50, 143.00),
    ('Ibaraki',  36.30, 140.50),
    ('Aomori',   40.80, 140.70),
    ('Fukushima',37.80, 141.50),
    ('Miyagi',   38.30, 141.00),
    ('Niigata',  37.91, 139.02),
    ('Suruga',   34.80, 138.50),
    ('Hyuga',    31.70, 131.50),
    ('Kushiro',  42.97, 144.38),
]


# ============================================================
# UTILITIES
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(np.asarray(lat2, float) - lat1)
    dlon = np.radians(np.asarray(lon2, float) - lon1)
    a = (np.sin(dlat/2)**2
         + np.cos(np.radians(lat1))
         * np.cos(np.radians(np.asarray(lat2, float)))
         * np.sin(dlon/2)**2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def nearest_city(lat, lon):
    best = min(CITIES, key=lambda c: haversine(lat, lon, c[1], c[2]))
    return best[0], round(haversine(lat, lon, best[1], best[2]), 1)


def estimate_Mc(mags, min_mag=3.0):
    """Maximum curvature Mc estimate (Wiemer & Wyss 2000)."""
    mags = np.asarray(mags)
    mags = mags[mags >= min_mag]
    if len(mags) < 50:
        return min_mag
    bins   = np.arange(min_mag, float(np.max(mags)) + 0.1, 0.1)
    counts, edges = np.histogram(mags, bins=bins)
    return float(edges[np.argmax(counts)]) if len(counts) > 0 else min_mag


def b_value_aki(mags, Mc):
    """Aki (1965) MLE b-value."""
    mags = np.asarray(mags)
    mags = mags[mags >= Mc]
    if len(mags) < 20:
        return 1.0
    mean_m = float(np.mean(mags))
    if mean_m <= Mc:
        return 1.0
    return float(np.log10(np.e) / (mean_m - Mc))


# ============================================================
# STEP 1: LOAD CATALOG
# ============================================================
def load_catalog(filepath):
    print(f"\n  Loading: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)

    # Normalise column names — handles both lat/lon and latitude/longitude
    df.columns = [c.strip().lower() for c in df.columns]
    col_map = {}
    for c in df.columns:
        if c in ('lat',):          col_map[c] = 'lat'
        if c == 'latitude':        col_map[c] = 'lat'
        if c in ('lon','long'):    col_map[c] = 'lon'
        if c == 'longitude':       col_map[c] = 'lon'
        if c in ('mag','magnitude'): col_map[c] = 'mag'
    df = df.rename(columns=col_map)

    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    for col in ['lat','lon','mag']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['time','lat','lon','mag'])
    df = df[df['mag'] >= MIN_MAG]
    df = df[(df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) &
            (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)]
    df = df.sort_values('time').reset_index(drop=True)

    print(f"  Events : {len(df):,}")
    print(f"  Period : {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"  Mag    : M{df['mag'].min():.1f} – M{df['mag'].max():.1f}")

    # Compute global Mc for b-value fallback
    global_Mc = estimate_Mc(df['mag'].values, MIN_MAG)
    print(f"  Global Mc (completeness): M{global_Mc:.1f}")
    return df, global_Mc


# ============================================================
# STEP 2: BUILD GRID + TRAINING BASELINES
# ============================================================
def build_grid(df):
    print(f"\n  Building 1°×1° grid (R={RADIUS_KM}km, W={W}, train<{TRAIN_END})...")
    t  = df['time'].values.astype('datetime64[s]')
    la = df['lat'].values
    lo = df['lon'].values
    m  = df['mag'].values
    train_end = np.datetime64(TRAIN_END)

    grid        = {}
    total_cells = 0

    for glat in np.arange(LAT_MIN, LAT_MAX, GRID_STEP):
        for glon in np.arange(LON_MIN, LON_MAX, GRID_STEP):
            total_cells += 1

            # Bounding-box pre-filter
            box = ((la > glat-3) & (la < glat+3) &
                   (lo > glon-3) & (lo < glon+3))
            idx = np.where(box)[0]
            if len(idx) < W + 30:
                continue

            # Exact radius
            ds  = haversine(glat, glon, la[idx], lo[idx])
            idx = idx[ds <= RADIUS_KM]
            if len(idx) < W + 30:
                continue

            # Training subset
            tr = idx[t[idx] < train_end]
            if len(tr) < W + 10:
                continue

            lm_tr = m[tr]
            lt_tr = t[tr]

            # σ₀: mean rolling std over W events
            rs = [np.std(lm_tr[max(0, k-W+1):k+1])
                  for k in range(W-1, len(lm_tr))]
            if len(rs) < 5:
                continue
            s0 = float(np.mean(rs))
            if s0 <= 0:
                continue

            # r₀: mean event rate (events/day)
            rates = []
            for k in range(min(len(tr)-W, len(rs))):
                dt = float((lt_tr[min(k+W-1, len(lt_tr)-1)] - lt_tr[k])
                           / np.timedelta64(1, 'D'))
                if dt > 0:
                    rates.append(W / dt)
            r0 = float(np.mean(rates)) if rates else 1.0

            # Skewness baseline
            sks_list = [float(pd.Series(lm_tr[max(0,k):k+W]).skew())
                        for k in range(min(len(tr)-W, len(rs)))]
            sk0  = float(np.mean(sks_list)) if sks_list else 0.0
            sks_ = float(max(np.std(sks_list)*3, 0.5)) if sks_list else 0.5

            grid[(glat, glon)] = {
                'idx': idx, 's0': s0, 'r0': r0,
                'sk0': sk0, 'sks': sks_
            }

    print(f"  Grid cells with baselines: {len(grid)} / {total_cells}")
    return grid, t, la, lo, m


# ============================================================
# STEPS 3-4: COMPUTE FEATURES AT CUTOFF DATE
# ============================================================
def compute_features(grid, t, la, lo, m, cutoff, global_Mc):
    cut64  = np.datetime64(cutoff)
    cut_3m = cut64 - np.timedelta64(90,  'D')
    cut_1y = cut64 - np.timedelta64(365, 'D')
    cells  = []

    for (glat, glon), gc in grid.items():
        idx = gc['idx']
        s0  = gc['s0']
        r0  = gc['r0']

        mask = t[idx] < cut64
        cloc = idx[mask]
        if len(cloc) < W + 5:
            continue

        lm = m[cloc]
        lt = t[cloc]
        nl = len(cloc)

        # C — compression
        sig = float(np.std(lm[-W:]))
        C   = max(0.0, 1.0 - sig / s0)

        # Q — quiescence
        dt_last = float((lt[-1] - lt[max(0, nl-W)])
                        / np.timedelta64(1, 'D'))
        rn = W / dt_last if dt_last > 0 else 0.0
        Q  = max(0.0, 1.0 - rn / r0) if r0 > 0 else 0.0

        if Q < Q_THRESH:
            continue

        # S — skewness anomaly
        S   = max(0.0, (float(pd.Series(lm[-W:]).skew()) - gc['sk0'])
                  / gc['sks'])
        Psi = C * Q * (1.0 + S)

        # ΔC, ΔS — 3-month change
        m3loc = idx[t[idx] < cut_3m]
        C3 = C; S3 = 0.0
        if len(m3loc) >= W + 5:
            lm3 = m[m3loc]
            C3  = max(0.0, 1.0 - float(np.std(lm3[-W:])) / s0)
            S3  = max(0.0, (float(pd.Series(lm3[-W:]).skew()) - gc['sk0'])
                      / gc['sks'])
        dC = C3 - C
        dS = S  - S3

        # b-value — auto Mc
        yr_loc   = idx[(t[idx] >= cut_1y) & (t[idx] < cut64)]
        yr_m     = m[yr_loc]
        Mc_local = estimate_Mc(yr_m, MIN_MAG) if len(yr_m) >= 50 else global_Mc
        b_val    = b_value_aki(yr_m, Mc_local) if len(yr_m) >= 20 else 1.0

        # n90, n365, Cv
        n90  = int(np.sum((t[idx] >= cut_3m) & (t[idx] < cut64)))
        n365 = int(len(yr_loc))
        yr_t = t[yr_loc]
        iet  = np.diff(yr_t) / np.timedelta64(1, 'D')
        iet  = iet[iet > 0]
        Cv   = (float(np.std(iet) / np.mean(iet))
                if len(iet) > 5 and float(np.mean(iet)) > 0 else 1.0)

        max_m1y  = float(np.max(yr_m)) if len(yr_m) > 0 else MIN_MAG
        city, cd = nearest_city(glat, glon)

        cells.append({
            'lat': glat, 'lon': glon,
            'Q': round(Q,4),   'C': round(C,4),
            'S': round(S,4),   'Psi': round(Psi,4),
            'dC': round(dC,4), 'dS': round(dS,4),
            'b_val': round(b_val,3),
            'n90': n90, 'n365': n365,
            'Cv': round(Cv,3),
            'max_m1y': round(max_m1y,1),
            'city': city, 'city_d': cd,
        })

    return cells


# ============================================================
# STEPS 5-6: SIX SCORING FUNCTIONS + CONSENSUS
# ============================================================
def consensus_zones(cells, min_votes=MIN_VOTES, top_n=TOP_N):
    def inv_b(c): return 1.0 / max(c['b_val'], 0.3)

    scoring = [
        ('F1:Psi+n90',   lambda c: c['Psi'] + c['n90']/100),
        ('F2:n90+Cv',    lambda c: c['n90']/100 + c['Cv']),
        ('F3:|dC|+|dS|', lambda c: abs(c['dC']) + abs(c['dS'])),
        ('F4:1/b',       inv_b),
        ('F5:MEGA',      lambda c: c['Psi'] + c['n90']/100 + inv_b(c) + c['Cv']),
        ('F6:n90+1/b',   lambda c: c['n90']/100 + inv_b(c)),
    ]

    votes  = {}
    detail = {}
    for fname, fn in scoring:
        ranked = sorted(cells, key=lambda c: -fn(c))[:top_n]
        for c in ranked:
            key = (c['lat'], c['lon'])
            votes[key]  = votes.get(key, 0) + 1
            detail.setdefault(key, []).append(fname)

    result = []
    for c in cells:
        key = (c['lat'], c['lon'])
        v   = votes.get(key, 0)
        if v >= min_votes:
            c['votes']    = v
            c['selected'] = ' | '.join(detail.get(key, []))
            result.append(c)

    result.sort(key=lambda x: (-x['votes'], -(x['Psi'] + x['n90']/100)))
    return result


# ============================================================
# PRINT TABLE
# ============================================================
def print_zones(zones, cutoff_str, total_cells):
    print()
    print("=" * 125)
    print(f"  JAPAN CRITICAL ZONES  |  Assessment: {cutoff_str[:10]}")
    print(f"  {len(zones)} consensus zones  |  "
          f"{len(zones)/total_cells*100:.0f}% of grid  |  "
          f">=3/6 functions agree")
    print("=" * 125)
    print(f"  {'#':>3}  {'Lat':>6}  {'Lon':>7}  {'V':>2}  "
          f"{'Q':>6}  {'C':>5}  {'S':>5}  {'Psi':>6}  "
          f"{'b':>5}  {'n90':>5}  {'Cv':>5}  {'MaxM':>5}  "
          f"{'City':<14}  {'Dist':>5}")
    print("  " + "─" * 110)
    for i, z in enumerate(zones):
        print(f"  {i+1:>3}  {z['lat']:>5.1f}N  {z['lon']:>6.1f}E  "
              f"{z['votes']:>2}  {z['Q']:>6.3f}  {z['C']:>5.3f}  "
              f"{z['S']:>5.3f}  {z['Psi']:>6.3f}  {z['b_val']:>5.3f}  "
              f"{z['n90']:>5}  {z['Cv']:>5.3f}  {z['max_m1y']:>5.1f}  "
              f"{z['city']:<14}  {z['city_d']:>4.0f}km")
    print()


# ============================================================
# SAVE CSV
# ============================================================
def save_zones(zones, outpath, cutoff_str):
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Rank','Lat','Lon','Consensus_votes',
                    'Q','C','S','Psi','Delta_C','Delta_S',
                    'b_value','Cv','n90','n365','Max_mag_1yr',
                    'Nearest_city','City_dist_km',
                    'Region','Assessment_date','Selected_by'])
        for i, z in enumerate(zones):
            w.writerow([i+1, z['lat'], z['lon'], z['votes'],
                        z['Q'], z['C'], z['S'], z['Psi'],
                        z['dC'], z['dS'],
                        z['b_val'], z['Cv'],
                        z['n90'], z['n365'], z['max_m1y'],
                        z['city'], z['city_d'],
                        'Japan', cutoff_str[:10], z['selected']])
    print(f"  Saved: {outpath}  ({len(zones)} zones)")


# ============================================================
# YEARLY ROLLING TEST (reproduces paper Table 2)
# ============================================================
def yearly_test(grid, t, la, lo, m, global_Mc, total_cells):
    print()
    print("=" * 90)
    print("  YEARLY ROLLING DETECTION TEST  (reproduces paper Table 2)")
    print("=" * 90)
    print(f"\n  {'Year':>6}  {'Zones':>6}  "
          f"{'M5.5+':>6}  {'Det':>4}  {'Rate':>6}  "
          f"{'M6.0+':>6}  {'Det':>4}  {'Rate':>6}  "
          f"{'M7.0+':>5}  {'Det':>3}")
    print("  " + "-" * 75)

    grand = {'n55':0,'d55':0,'n60':0,'d60':0,'n70':0,'d70':0}

    for yr in [2020, 2021, 2022, 2023]:
        cutoff    = f'{yr}-01-01'
        cut64     = np.datetime64(cutoff)
        win_end   = np.datetime64(f'{yr+1}-01-01')

        # Get consensus zones at Jan 1 of this year
        cells = compute_features(grid, t, la, lo, m, cut64, global_Mc)
        zones = consensus_zones(cells)

        # Get observed mainshocks in next 12 months
        obs_mask = ((t >= cut64) & (t < win_end) & (m >= 5.5))
        obs_idx  = np.where(obs_mask)[0]

        # Simple declustering: skip if larger event within 5 days + 150km
        mainshocks = []
        for i in obs_idx:
            prior = np.where(
                (t < t[i]) &
                (t > t[i] - np.timedelta64(5,'D')) &
                (m >= 5.0))[0]
            is_main = True
            for pi in prior:
                if haversine(la[i], lo[i], la[pi], lo[pi]) < 150:
                    is_main = False
                    break
            if is_main:
                mainshocks.append(i)

        # Check detection
        det55 = det60 = det70 = 0
        n55 = n60 = n70 = 0

        for i in mainshocks:
            ev_m = m[i]
            if ev_m >= 5.5: n55 += 1
            if ev_m >= 6.0: n60 += 1
            if ev_m >= 7.0: n70 += 1

            detected = any(
                haversine(la[i], lo[i], z['lat'], z['lon']) <= DETECT_R
                for z in zones)
            if detected:
                if ev_m >= 5.5: det55 += 1
                if ev_m >= 6.0: det60 += 1
                if ev_m >= 7.0: det70 += 1

        grand['n55']+=n55; grand['d55']+=det55
        grand['n60']+=n60; grand['d60']+=det60
        grand['n70']+=n70; grand['d70']+=det70

        r55 = f"{100*det55//n55}%" if n55 > 0 else "--"
        r60 = f"{100*det60//n60}%" if n60 > 0 else "--"

        print(f"  {yr:>6}  {len(zones):>6}  "
              f"{n55:>6}  {det55:>4}  {r55:>6}  "
              f"{n60:>6}  {det60:>4}  {r60:>6}  "
              f"{n70:>5}  {det70:>3}")

    # Totals
    print("  " + "-" * 75)
    tr55 = f"{100*grand['d55']//grand['n55']}%" if grand['n55']>0 else "--"
    tr60 = f"{100*grand['d60']//grand['n60']}%" if grand['n60']>0 else "--"
    tr70 = f"{100*grand['d70']//grand['n70']}%" if grand['n70']>0 else "--"
    print(f"  {'TOTAL':>6}  {'':>6}  "
          f"{grand['n55']:>6}  {grand['d55']:>4}  {tr55:>6}  "
          f"{grand['n60']:>6}  {grand['d60']:>4}  {tr60:>6}  "
          f"{grand['n70']:>5}  {grand['d70']:>3}")
    print()
    print(f"  Paper target: 81% M>=5.5  |  71% M>=6.0  |  100% M>=7.0")
    print()


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Japan Critical Zones — Six-Criterion Consensus Framework')
    parser.add_argument('--catalog', required=True,
                        help='Path to catalog CSV  (jma_M3plus_2000_2023.csv)')
    parser.add_argument('--cutoff',  default='2023-12-31',
                        help='Assessment date  (default: 2023-12-31)')
    parser.add_argument('--yearly',  action='store_true',
                        help='Run year-by-year test 2020-2023')
    parser.add_argument('--min-votes', type=int, default=MIN_VOTES,
                        help='Minimum consensus votes (default 3)')
    parser.add_argument('--output',  default=None,
                        help='Output CSV filename (auto-named if not given)')
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Japan Critical Earthquake Zones Detection Framework     ║")
    print("║  Ramakrishna Pasupuleti — Independent Researcher         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Load
    print()
    print("Step 1 — Loading catalog...")
    df, global_Mc = load_catalog(args.catalog)

    # Build grid
    print()
    print("Step 2 — Building grid and training baselines...")
    grid, t, la, lo, m = build_grid(df)
    total_cells = len(grid)

    # Cutoff
    cutoff_str = args.cutoff + 'T00:00:00'
    cutoff     = np.datetime64(cutoff_str)
    print(f"\n  Assessment date: {args.cutoff}")

    # Yearly test
    if args.yearly:
        print()
        print("Step 3a — Running yearly detection test 2020–2023...")
        yearly_test(grid, t, la, lo, m, global_Mc, total_cells)

    # Single assessment
    print("Step 3b — Computing C, Q, S, Ψ and parameters...")
    cells = compute_features(grid, t, la, lo, m, cutoff, global_Mc)
    print(f"  Q > {Q_THRESH} zones: {len(cells)}")

    print()
    print("Step 4 — Applying six scoring functions and consensus...")
    zones = consensus_zones(cells, min_votes=args.min_votes)

    if not zones:
        print(f"  No zones with >= {args.min_votes} votes.")
        sys.exit(1)

    # Print
    print_zones(zones, cutoff_str, total_cells)

    # Save
    outpath = args.output or f"japan_critical_zones_{args.cutoff}.csv"
    save_zones(zones, outpath, cutoff_str)

    print()
    print("Done.")
    print()


if __name__ == '__main__':
    main()