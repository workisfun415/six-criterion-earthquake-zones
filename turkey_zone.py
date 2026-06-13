#!/usr/bin/env python3
"""
Reproduce Turkey Critical Zones — BSSA Manuscript Exact Method
================================================================
Author: Ramakrishna Pasupuleti
Usage:  python turkey_zones.py
        (data_final.xlsx must be in same folder)
"""
import numpy as np
import pandas as pd
import csv, os
np.random.seed(42)

# ============================================================
# CONFIGURATION — EXACT SAME AS BSSA MANUSCRIPT
# ============================================================
GRID_STEP = 0.5       # degrees
RADIUS_KM = 50        # km
W = 20                 # sliding window (events)
MIN_MAG = 3.0
TRAIN_END = '2010-01-01'
Q_THRESHOLD = 0.9
TOP_N = 30             # top-N per scoring function
MIN_VOTES = 3          # consensus threshold
DETECT_RADIUS = 100    # km for validation

CITIES = [
    ('Istanbul',41.01,28.98),('Ankara',39.93,32.85),('Izmir',38.42,27.14),
    ('Gaziantep',37.06,37.38),('Malatya',38.35,38.32),('Van',38.49,43.38),
    ('Elazig',38.67,39.22),('Erzurum',39.90,41.27),('Denizli',37.77,29.09),
    ('Mugla',37.21,28.36),('K.Maras',37.58,36.93),('Bingol',38.88,40.50),
    ('Duzce',40.84,31.16),('Manisa',38.61,27.43),('Konya',37.87,32.48),
    ('Erzincan',39.75,39.49),('Sivas',39.75,37.02),('Hatay',36.20,36.15),
]

# ============================================================
# FUNCTIONS
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def nearest_city(lat, lon):
    best = min(CITIES, key=lambda c: haversine(lat, lon, c[1], c[2]))
    return best[0], haversine(lat, lon, best[1], best[2])

# ============================================================
# STEP 1: LOAD AFAD CATALOG
# ============================================================
print("="*70)
print("  TURKEY CRITICAL ZONES — BSSA Exact Reproduction")
print("="*70)

filename = 'data_final.xlsx'
if not os.path.exists(filename):
    print(f"  ERROR: {filename} not found in current folder!")
    print(f"  Put data_final.xlsx in: {os.getcwd()}")
    exit(1)

print(f"\n  Loading {filename}...")
afad = pd.read_excel(filename)
afad['time'] = pd.to_datetime(afad['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
afad = afad.rename(columns={'Latitude': 'lat', 'Longitude': 'lon', 'Magnitude': 'mag'})
afad = afad[['time', 'lat', 'lon', 'mag']].dropna()
afad = afad[afad['mag'] >= MIN_MAG]
afad = afad[(afad['lat'] >= 36) & (afad['lat'] <= 42.5) &
            (afad['lon'] >= 26) & (afad['lon'] <= 45)]
afad = afad.sort_values('time').reset_index(drop=True)

t = afad['time'].values
la = afad['lat'].values
lo = afad['lon'].values
m = afad['mag'].values

print(f"  Events: {len(afad):,}")
print(f"  Period: {afad['time'].min().date()} to {afad['time'].max().date()}")

# ============================================================
# STEP 2: BUILD GRID WITH TRAINING BASELINES
# ============================================================
print(f"\n  Building grid (0.5°, R=50km, W=20)...")
train_end = np.datetime64(TRAIN_END)

grid = {}
for glat in np.arange(36, 42.5, GRID_STEP):
    for glon in np.arange(26, 45, GRID_STEP):
        box = (la > glat-1) & (la < glat+1) & (lo > glon-1) & (lo < glon+1)
        loc = np.where(box)[0]
        if len(loc) < W + 30:
            continue
        ds = np.array([haversine(glat, glon, la[j], lo[j]) for j in loc])
        loc = loc[ds <= RADIUS_KM]
        if len(loc) < W + 30:
            continue

        tr = loc[t[loc] < train_end]
        if len(tr) < W + 10:
            continue
        lm_tr = m[tr]
        lt_tr = t[tr]

        # sigma_0
        rs = [np.std(lm_tr[max(0, k-W+1):k+1]) for k in range(W-1, len(lm_tr))]
        if len(rs) < 5:
            continue
        s0 = np.mean(rs)
        if s0 <= 0:
            continue

        # r_0
        rates = []
        for k in range(min(len(tr)-W, len(rs))):
            dt = (lt_tr[min(k+W-1, len(lt_tr)-1)] - lt_tr[k]) / np.timedelta64(1, 'D')
            if dt > 0:
                rates.append(W / dt)
        r0 = np.mean(rates) if rates else 1

        # skewness baseline
        sks_list = [float(pd.Series(lm_tr[max(0,k):k+W]).skew())
                    for k in range(min(len(tr)-W, len(rs)))]
        sk0 = np.mean(sks_list) if sks_list else 0
        sks = max(np.std(sks_list) * 3, 0.5)

        grid[(glat, glon)] = {'loc': loc, 's0': s0, 'r0': r0, 'sk0': sk0, 'sks': sks}

total = len(grid)
print(f"  Grid cells: {total}")

# ============================================================
# FUNCTION: COMPUTE FEATURES + CONSENSUS AT A CUTOFF DATE
# ============================================================
def get_consensus_zones(cutoff_date):
    """Compute 6-criterion consensus zones at a given cutoff date."""
    cutoff = np.datetime64(cutoff_date)
    cut_3m = cutoff - np.timedelta64(90, 'D')
    cut_1y = cutoff - np.timedelta64(365, 'D')

    cells = []
    for (glat, glon), gc in grid.items():
        loc = gc['loc']; s0 = gc['s0']; r0 = gc['r0']
        mask = t[loc] < cutoff
        cloc = loc[mask]
        if len(cloc) < W + 5:
            continue
        lm = m[cloc]; lt = t[cloc]; nl = len(cloc)

        # C, Q, S
        sig = np.std(lm[-W:])
        C = max(0, 1 - sig / s0)
        dt_last = (lt[-1] - lt[max(0, nl-W)]) / np.timedelta64(1, 'D')
        rn = W / dt_last if dt_last > 0 else 0
        Q = max(0, 1 - rn / r0) if r0 > 0 else 0
        if Q < Q_THRESHOLD:
            continue
        S = max(0, (float(pd.Series(lm[-W:]).skew()) - gc['sk0']) / gc['sks'])
        Psi = C * Q * (1 + S)

        # Delta C, Delta S
        m3 = t[loc] < cut_3m; c3 = loc[m3]
        C3, S3 = C, 0
        if len(c3) >= W + 5:
            C3 = max(0, 1 - np.std(m[c3][-W:]) / s0)
            S3 = max(0, (float(pd.Series(m[c3][-W:]).skew()) - gc['sk0']) / gc['sks'])
        Cd = C3 - C; Si = S - S3

        # b-value
        yr = loc[(t[loc] >= cut_1y) & (t[loc] < cutoff)]
        yr_m = m[yr]
        b_val = 1.0
        if len(yr_m) >= 20:
            mm = np.mean(yr_m[yr_m >= MIN_MAG])
            if mm > MIN_MAG:
                b_val = np.log10(np.e) / (mm - MIN_MAG)

        # n90, Cv
        n90 = int(np.sum((t[loc] >= cut_3m) & (t[loc] < cutoff)))
        n365 = len(yr)
        yr_t = t[yr]; iet = np.diff(yr_t) / np.timedelta64(1, 'D')
        iet = iet[iet > 0]
        Cv = np.std(iet) / np.mean(iet) if len(iet) > 5 and np.mean(iet) > 0 else 1.0
        max_mag = np.max(yr_m) if len(yr_m) > 0 else MIN_MAG

        city_name, city_dist = nearest_city(glat, glon)
        cells.append({
            'lat': glat, 'lon': glon, 'Q': Q, 'C': C, 'S': S, 'Psi': Psi,
            'Cd': Cd, 'Si': Si, 'n90': n90, 'n365': n365,
            'b_val': b_val, 'Cv': Cv, 'max_mag': max_mag,
            'city': city_name, 'city_d': city_dist
        })

    # 6 scoring functions
    sfuncs = [
        ('F1: Psi+n90',   lambda c: c['Psi'] + c['n90']/100),
        ('F2: n90+Cv',    lambda c: c['n90']/100 + c['Cv']),
        ('F3: |dC|+|dS|', lambda c: abs(c['Cd']) + abs(c['Si'])),
        ('F4: 1/b',       lambda c: 1/max(c['b_val'], 0.3)),
        ('F5: MEGA',      lambda c: c['Psi'] + c['n90']/100 + 1/max(c['b_val'],0.3) + c['Cv']),
        ('F6: n90+1/b',   lambda c: c['n90']/100 + 1/max(c['b_val'], 0.3)),
    ]

    votes = {}; vote_detail = {}
    for sname, sf in sfuncs:
        ranked = sorted(cells, key=lambda c: -sf(c))[:TOP_N]
        for c in ranked:
            k = (c['lat'], c['lon'])
            votes[k] = votes.get(k, 0) + 1
            if k not in vote_detail: vote_detail[k] = []
            vote_detail[k].append(sname)

    result = []
    for c in cells:
        k = (c['lat'], c['lon'])
        v = votes.get(k, 0)
        if v >= MIN_VOTES:
            c['votes'] = v
            c['selected_by'] = vote_detail.get(k, [])
            result.append(c)

    result.sort(key=lambda x: (-x['votes'], -x['Psi'] - x['n90']/100))
    return result, len(cells)

# ============================================================
# LATEST ZONE LIST
# ============================================================
latest_date = afad['time'].max().strftime('%Y-%m-%d')
print(f"\n{'='*70}")
print(f"  Computing zones at: {latest_date}")
print(f"{'='*70}")

zones, n_q9 = get_consensus_zones(latest_date)
print(f"  Q > 0.9 zones: {n_q9}")
print(f"  Consensus >= 3: {len(zones)} zones ({len(zones)/total*100:.0f}% of grid)")

# Print table
print(f"\n  {'#':>3} {'Lat':>5} {'Lon':>5} {'V':>2} {'Q':>6} {'C':>5} {'S':>5} "
      f"{'Psi':>6} {'b':>5} {'n90':>4} {'Cv':>5} {'Mmax':>5} {'City':<12} {'km':>4}")
print(f"  {'-'*95}")
for i, z in enumerate(zones):
    print(f"  {i+1:>3} {z['lat']:>4.1f}N {z['lon']:>4.1f}E {z['votes']:>2} "
          f"{z['Q']:>6.3f} {z['C']:>5.2f} {z['S']:>5.2f} {z['Psi']:>6.3f} "
          f"{z['b_val']:>5.2f} {z['n90']:>4} {z['Cv']:>5.2f} {z['max_mag']:>5.1f} "
          f"{z['city']:<12} {z['city_d']:>3.0f}")

# Save CSV
outfile = f"turkey_critical_zones_{latest_date}.csv"
with open(outfile, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Rank','Lat','Lon','Votes','Q','C','S','Psi','Delta_C','Delta_S',
                'b_value','Cv','n90','n365','Max_mag','City','City_km','Selected_by'])
    for i, z in enumerate(zones):
        w.writerow([i+1, z['lat'], z['lon'], z['votes'],
                    f"{z['Q']:.3f}", f"{z['C']:.3f}", f"{z['S']:.3f}", f"{z['Psi']:.3f}",
                    f"{z['Cd']:+.3f}", f"{z['Si']:+.3f}", f"{z['b_val']:.3f}", f"{z['Cv']:.3f}",
                    z['n90'], z['n365'], f"{z['max_mag']:.1f}",
                    z['city'], f"{z['city_d']:.0f}",
                    ' | '.join(z['selected_by'])])
print(f"\n  Saved: {outfile}")

# ============================================================
# YEAR-BY-YEAR VALIDATION (2020-2026)
# ============================================================
print(f"\n{'='*70}")
print(f"  YEARLY ROLLING VALIDATION (2020-2026)")
print(f"{'='*70}")

print(f"\n  {'Year':>6} {'Zones':>6} {'M>=5.5':>7} {'Det':>4} {'Rate':>6} "
      f"{'M>=6.0':>7} {'Det':>4} {'Rate':>6} {'Key Events'}")
print(f"  {'-'*85}")

grand = {'eq55':0, 'det55':0, 'eq60':0, 'det60':0}

for year in range(2020, 2027):
    cutoff = np.datetime64(f'{year}-01-01')
    window_end = min(cutoff + np.timedelta64(365, 'D'), t[-1] + np.timedelta64(1, 'D'))

    # Get zones at this cutoff
    yr_zones, _ = get_consensus_zones(f'{year}-01-01')

    # Get M>=5.5 mainshocks in next 12 months
    fx = (t >= cutoff) & (t < window_end) & (m >= 5.5)
    fi = np.where(fx)[0]
    eqs = []
    for i in fi:
        prior = np.where((t < t[i]) & (t > t[i] - np.timedelta64(5, 'D')) & (m >= 5.0))[0]
        ok = True
        for pi in prior:
            if haversine(la[i], lo[i], la[pi], lo[pi]) < 100:
                ok = False; break
        if ok:
            cn, cd = nearest_city(la[i], lo[i])
            eqs.append((la[i], lo[i], m[i], pd.Timestamp(t[i]).strftime('%Y-%m-%d'), cn))

    eqs55 = eqs
    eqs60 = [e for e in eqs if e[2] >= 6.0]

    # Check detection
    det55 = []; det60 = []; key_events = []
    for elat, elon, emag, edate, ecity in eqs55:
        found = False
        for z in yr_zones:
            if haversine(elat, elon, z['lat'], z['lon']) <= DETECT_RADIUS:
                found = True
                d = haversine(elat, elon, z['lat'], z['lon'])
                if emag >= 6.0:
                    key_events.append(f"{ecity} M{emag:.1f} {d:.0f}km")
                break
        if found:
            det55.append(emag)
            if emag >= 6.0: det60.append(emag)

    grand['eq55'] += len(eqs55); grand['det55'] += len(det55)
    grand['eq60'] += len(eqs60); grand['det60'] += len(det60)

    r55 = f"{len(det55)/len(eqs55)*100:.0f}%" if eqs55 else "--"
    r60 = f"{len(det60)/len(eqs60)*100:.0f}%" if eqs60 else "--"
    keys = ', '.join(key_events[:2]) if key_events else ""

    print(f"  {year:>6} {len(yr_zones):>6} {len(eqs55):>7} {len(det55):>4} {r55:>6} "
          f"{len(eqs60):>7} {len(det60):>4} {r60:>6} {keys}")

# Totals
print(f"  {'-'*85}")
tr55 = f"{grand['det55']/grand['eq55']*100:.0f}%" if grand['eq55'] else "--"
tr60 = f"{grand['det60']/grand['eq60']*100:.0f}%" if grand['eq60'] else "--"
print(f"  {'TOTAL':>6} {'':>6} {grand['eq55']:>7} {grand['det55']:>4} {tr55:>6} "
      f"{grand['eq60']:>7} {grand['det60']:>4} {tr60:>6}")

# Missed M>=6.0
print(f"\n  MISSED M>=6.0 EVENTS:")
for year in range(2020, 2027):
    cutoff = np.datetime64(f'{year}-01-01')
    window_end = min(cutoff + np.timedelta64(365, 'D'), t[-1] + np.timedelta64(1, 'D'))
    yr_zones, _ = get_consensus_zones(f'{year}-01-01')

    fx = (t >= cutoff) & (t < window_end) & (m >= 6.0)
    fi = np.where(fx)[0]
    for i in fi:
        prior = np.where((t < t[i]) & (t > t[i] - np.timedelta64(5, 'D')) & (m >= 5.0))[0]
        ok = True
        for pi in prior:
            if haversine(la[i], lo[i], la[pi], lo[pi]) < 100: ok = False; break
        if not ok: continue

        best_d = min(haversine(la[i], lo[i], z['lat'], z['lon']) for z in yr_zones) if yr_zones else 999
        if best_d > DETECT_RADIUS:
            cn, _ = nearest_city(la[i], lo[i])
            print(f"    X M{m[i]:.1f} {pd.Timestamp(t[i]).strftime('%Y-%m-%d')} "
                  f"({la[i]:.1f}N,{lo[i]:.1f}E) {cn} nearest={best_d:.0f}km "
                  f"{'<-- OFFSHORE' if best_d > 200 else ''}")

print(f"\n{'='*70}")
print(f"  DONE. Zones saved to: {outfile}")
print(f"  Onshore M>=6.0 detection: check above")
print(f"{'='*70}")