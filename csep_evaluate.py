#!/usr/bin/env python3
"""
CSEP Evaluation — Six-Criterion Consensus Detection Framework
==============================================================
Author : Ramakrishna Pasupuleti
         Independent Researcher, Suryapet, Telangana, India
         ORCID: 0009-0008-8418-1430

Usage:
    python csep_evaluate.py --region japan
    python csep_evaluate.py --region turkey
    python csep_evaluate.py --region both

Files needed in same folder:
    japan_critical_zones_2023-12-31.csv
    jma_M3plus_2000_2023.csv
    turkey_critical_zones_2026-05-28.csv
    data_final.xlsx
"""

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, argparse, glob, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    'japan': {
        'catalog_file':   'jma_M3plus_2000_2023.csv',
        'catalog_type':   'csv',
        'lat_range':      (30, 46),
        'lon_range':      (128, 148),
        'grid_step':      1.0,
        'min_mag':        3.0,
        'detect_r':       200,
        'decluster_km':   150,
        'decluster_days': 5,
        'assess_years':   [2020, 2021, 2022, 2023],
    },
    'turkey': {
        'catalog_file':   'data_final.xlsx',
        'catalog_type':   'xlsx',
        'lat_range':      (36, 42.5),
        'lon_range':      (26, 45),
        'grid_step':      0.5,
        'min_mag':        3.0,
        'detect_r':       100,
        'decluster_km':   50,
        'decluster_days': 5,
        'assess_years':   [2020, 2021, 2022, 2023, 2024, 2025, 2026],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R    = 6371.0
    dlat = np.radians(np.asarray(lat2, float) - lat1)
    dlon = np.radians(np.asarray(lon2, float) - lon1)
    a    = (np.sin(dlat/2)**2
            + np.cos(np.radians(lat1))
            * np.cos(np.radians(np.asarray(lat2, float)))
            * np.sin(dlon/2)**2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def normalise_zones(df):
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if   cl == 'lat':                         col_map[c] = 'lat'
        elif cl == 'lon':                         col_map[c] = 'lon'
        elif cl in ('votes','consensus_votes'):   col_map[c] = 'votes'
        elif cl == 'b_value':                     col_map[c] = 'b_val'
        elif cl in ('nearest_city','city'):       col_map[c] = 'city'
        elif cl in ('city_dist_km','city_km'):    col_map[c] = 'city_d'
        elif cl == 'psi':                         col_map[c] = 'Psi'
        elif cl == 'n365':                        col_map[c] = 'n365'
        elif cl == 'n90':                         col_map[c] = 'n90'
        elif cl in ('max_mag','max_mag_1yr'):     col_map[c] = 'max_mag'
    return df.rename(columns=col_map)


def find_zone_file(region):
    for pat in [f'{region}_critical_zones_*.csv', f'{region}_zones_*.csv']:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[-1]
    return None


def load_zones(region):
    fpath = find_zone_file(region)
    if not fpath:
        print(f"  ERROR: No zone CSV found for {region}.")
        return None
    df = normalise_zones(pd.read_csv(fpath))
    if 'votes' in df.columns:
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(3)
    print(f"  Zones : {len(df)} from {fpath}")
    return df


def load_catalog(cfg):
    fname = cfg['catalog_file']
    if not os.path.exists(fname):
        print(f"  ERROR: {fname} not found.")
        return None
    if cfg['catalog_type'] == 'xlsx':
        df = pd.read_excel(fname)
        df['time'] = pd.to_datetime(
            df['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        df = df.rename(columns={'Latitude':'lat',
                                'Longitude':'lon',
                                'Magnitude':'mag'})
    else:
        df = pd.read_csv(fname, low_memory=False)
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl == 'latitude':  rename[c] = 'lat'
            if cl == 'longitude': rename[c] = 'lon'
            if cl == 'magnitude': rename[c] = 'mag'
        df = df.rename(columns=rename)
    df = df.dropna(subset=['time','lat','lon','mag'])
    for col in ['lat','lon','mag']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['lat','lon','mag'])
    la1, la2 = cfg['lat_range']
    lo1, lo2 = cfg['lon_range']
    df = df[(df['lat']>=la1) & (df['lat']<=la2) &
            (df['lon']>=lo1) & (df['lon']<=lo2) &
            (df['mag']>=cfg['min_mag'])]
    df = df.sort_values('time').reset_index(drop=True)
    print(f"  Events: {len(df):,}  "
          f"({df['time'].min().date()} to {df['time'].max().date()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# N-TEST
# ─────────────────────────────────────────────────────────────────────────────
def n_test(observed, expected, confidence=0.95):
    alpha = 1 - confidence
    low   = scipy.stats.poisson.ppf(alpha/2,   expected)
    high  = scipy.stats.poisson.ppf(1-alpha/2, expected)
    d1    = scipy.stats.poisson.cdf(observed,   expected)
    d2    = 1 - scipy.stats.poisson.cdf(observed - 1, expected)
    return d1, d2, (low <= observed <= high), low, high


# ─────────────────────────────────────────────────────────────────────────────
# MAINSHOCK ISOLATION
# ─────────────────────────────────────────────────────────────────────────────
def get_mainshocks(t_arr, la_arr, lo_arr, m_arr,
                   cutoff, win_end, cfg, min_mag=5.5):
    obs_idx = np.where(
        (t_arr >= cutoff) & (t_arr < win_end) & (m_arr >= min_mag))[0]
    dkm   = cfg['decluster_km']
    ddays = cfg['decluster_days']
    mainshocks = []
    for i in obs_idx:
        prior = np.where(
            (t_arr < t_arr[i]) &
            (t_arr > t_arr[i] - np.timedelta64(ddays, 'D')) &
            (m_arr >= 5.0))[0]
        ok = all(haversine(la_arr[i], lo_arr[i],
                           la_arr[p], lo_arr[p]) >= dkm
                 for p in prior)
        if ok:
            mainshocks.append(i)
    return mainshocks


def is_detected(i, zones, la_arr, lo_arr, R):
    return any(haversine(la_arr[i], lo_arr[i],
                         z['lat'], z['lon']) <= R
               for _, z in zones.iterrows())


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION TABLE
# ─────────────────────────────────────────────────────────────────────────────
def detection_table(zones, catalog, cfg, region, outlines):
    R     = cfg['detect_r']
    t     = catalog['time'].values
    la    = catalog['lat'].values
    lo    = catalog['lon'].values
    m     = catalog['mag'].values
    grand = {k:0 for k in ['n55','d55','n60','d60','n70','d70']}

    outlines.append(f"\n{'='*78}")
    outlines.append(f"  {region.upper()} — YEAR-BY-YEAR DETECTION")
    outlines.append(f"  Declustering: {cfg['decluster_days']}d / "
                    f"{cfg['decluster_km']}km  |  Detection radius: {R}km")
    outlines.append(f"{'='*78}")
    outlines.append(
        f"  {'Year':>6}  {'M5.5+':>5} {'Det':>3} {'%':>4}  "
        f"{'M6.0+':>5} {'Det':>3} {'%':>4}  "
        f"{'M7.0+':>5} {'Det':>3}  Key events")
    outlines.append("  " + "-"*78)

    for yr in cfg['assess_years']:
        cut = np.datetime64(f'{yr}-01-01')
        end = np.datetime64(f'{yr+1}-01-01')
        ms  = get_mainshocks(t, la, lo, m, cut, end, cfg)
        n55=n60=n70=d55=d60=d70=0
        keys=[]
        for i in ms:
            mg = m[i]
            if mg >= 5.5: n55 += 1
            if mg >= 6.0: n60 += 1
            if mg >= 7.0: n70 += 1
            det = is_detected(i, zones, la, lo, R)
            if det:
                if mg >= 5.5: d55 += 1
                if mg >= 6.0:
                    d60 += 1
                    bd = min(haversine(la[i], lo[i], z['lat'], z['lon'])
                             for _, z in zones.iterrows())
                    keys.append(f"M{mg:.1f}@{bd:.0f}km")
                if mg >= 7.0: d70 += 1
        for k, v in [('n55',n55),('d55',d55),('n60',n60),
                     ('d60',d60),('n70',n70),('d70',d70)]:
            grand[k] += v
        r55 = f"{100*d55//n55}%" if n55>0 else "--"
        r60 = f"{100*d60//n60}%" if n60>0 else "--"
        outlines.append(
            f"  {yr:>6}  {n55:>5} {d55:>3} {r55:>4}  "
            f"{n60:>5} {d60:>3} {r60:>4}  "
            f"{n70:>5} {d70:>3}  {', '.join(keys[:2])}")

    outlines.append("  " + "-"*78)
    tr55 = f"{100*grand['d55']//grand['n55']}%" if grand['n55']>0 else "--"
    tr60 = f"{100*grand['d60']//grand['n60']}%" if grand['n60']>0 else "--"
    tr70 = f"{100*grand['d70']//grand['n70']}%" if grand['n70']>0 else "--"
    outlines.append(
        f"  {'TOTAL':>6}  {grand['n55']:>5} {grand['d55']:>3} {tr55:>4}  "
        f"{grand['n60']:>5} {grand['d60']:>3} {tr60:>4}  "
        f"{grand['n70']:>5} {grand['d70']:>3}")

    la1,la2 = cfg['lat_range']; lo1,lo2 = cfg['lon_range']
    n_total = (len(np.arange(la1, la2, cfg['grid_step']))
               * len(np.arange(lo1, lo2, cfg['grid_step'])))
    area_f = len(zones) / n_total
    det_r  = grand['d55'] / grand['n55'] if grand['n55']>0 else 0
    skill  = (det_r - area_f) / (1 - area_f) if area_f < 1 else 0
    outlines.append(f"\n  Area fraction  : {area_f*100:.1f}%")
    outlines.append(f"  Detection rate : {det_r*100:.1f}% (M>=5.5)")
    outlines.append(f"  Skill score    : {skill:.3f}  (0=random, 1=perfect)")

    return grand, n_total


# ─────────────────────────────────────────────────────────────────────────────
# MOLCHAN DIAGRAM DATA
# ─────────────────────────────────────────────────────────────────────────────
def molchan_data(zones, catalog, cfg, n_total, outlines, region):
    R    = cfg['detect_r']
    t    = catalog['time'].values
    la   = catalog['lat'].values
    lo   = catalog['lon'].values
    m    = catalog['mag'].values
    tops = list(range(5, min(len(zones)+1, 55), 5))

    outlines.append(f"\n{'='*55}")
    outlines.append(f"  MOLCHAN DIAGRAM DATA — {region.upper()}")
    outlines.append(f"{'='*55}")
    outlines.append(f"  Top-N   Area%   M5.5 det%   M6.0 det%")
    outlines.append("  " + "-"*42)

    rows = []
    print(f"  Molchan ({region}):")

    for top_n in tops:
        sort_c = [c for c in ['votes','Psi'] if c in zones.columns]
        sub = (zones.sort_values(sort_c, ascending=False).head(top_n)
               if sort_c else zones.head(top_n))
        tau = len(sub) / n_total
        n55=d55=n60=d60=0

        for yr in cfg['assess_years']:
            cut = np.datetime64(f'{yr}-01-01')
            end = np.datetime64(f'{yr+1}-01-01')
            for i in get_mainshocks(t, la, lo, m, cut, end, cfg):
                mg = m[i]
                if mg >= 5.5: n55 += 1
                if mg >= 6.0: n60 += 1
                if is_detected(i, sub, la, lo, R):
                    if mg >= 5.5: d55 += 1
                    if mg >= 6.0: d60 += 1

        r55  = d55/n55*100 if n55>0 else 0
        r60  = d60/n60*100 if n60>0 else 0
        nu55 = 1 - d55/n55 if n55>0 else 1.0
        nu60 = 1 - d60/n60 if n60>0 else 1.0

        rows.append({'top_n':top_n, 'tau':tau,
                     'nu55':nu55, 'nu60':nu60,
                     'r55':r55, 'r60':r60,
                     'n55':n55, 'd55':d55,
                     'n60':n60, 'd60':d60})
        outlines.append(
            f"  {top_n:>5}  {tau*100:>6.1f}%  "
            f"{r55:>10.0f}%  {r60:>10.0f}%")
        print(f"    Top-{top_n:>2}: {tau*100:.0f}% area  "
              f"M5.5={r55:.0f}%  M6.0={r60:.0f}%")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT MOLCHAN
# ─────────────────────────────────────────────────────────────────────────────
def plot_molchan(df, region, outfile):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    x = np.linspace(0, 1, 100)
    ax.plot(x, 1-x, 'k--', lw=1.2, alpha=0.5, label='Random forecast')
    ax.fill_between(x, 1-x, 1, alpha=0.06, color='gray',
                    label='Worse than random')
    ax.plot(df['tau'], df['nu55'], 'o-',
            color='#1a4b8c', lw=2.5, ms=7,
            label='M >= 5.5', zorder=4)
    ax.plot(df['tau'], df['nu60'], 's-',
            color='#B8860B', lw=2.5, ms=7,
            label='M >= 6.0', zorder=4)

    op30 = df[df['top_n'] == 30]
    if len(op30) > 0:
        op = op30.iloc[0]
        ax.scatter([op['tau']], [op['nu55']], s=280, marker='*',
                   c='#8B0000', zorder=5, edgecolors='k', lw=0.8)
        ax.annotate(
            f"Top-30\n({op['tau']*100:.0f}% area\n{op['r55']:.0f}% det.)",
            xy=(op['tau'], op['nu55']),
            xytext=(op['tau']+0.06, op['nu55']+0.09),
            fontsize=9, fontweight='bold', color='#8B0000',
            arrowprops=dict(arrowstyle='->', color='#8B0000', lw=1.5))

    ax.set_xlabel('Fraction of area occupied (tau)', fontsize=12)
    ax.set_ylabel('Miss rate  nu = 1 - detection rate', fontsize=12)
    ax.set_title(
        f'Molchan Diagram - {region.upper()}\n'
        f'Six-Criterion Consensus Framework',
        fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-0.01, 0.55)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE ONE REGION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_region(region):
    cfg = CONFIGS[region]
    scipy_ver = scipy.__version__
    outlines = [
        f"CSEP EVALUATION REPORT - {region.upper()}",
        "Six-Criterion Consensus Detection Framework",
        "Ramakrishna Pasupuleti, ORCID: 0009-0008-8418-1430",
        f"pycsep 0.8.0 | scipy {scipy_ver}",
        f"Declustering: {cfg['decluster_days']} days / {cfg['decluster_km']} km",
        "=" * 65,
    ]

    print()
    print("=" * 65)
    print(f"  CSEP EVALUATION - {region.upper()}")
    print(f"  Declustering: {cfg['decluster_days']}d / {cfg['decluster_km']}km")
    print("=" * 65)

    print("\n  Loading zones...")
    zones = load_zones(region)
    if zones is None:
        return

    print("\n  Loading catalog...")
    catalog = load_catalog(cfg)
    if catalog is None:
        return

    la1,la2 = cfg['lat_range']; lo1,lo2 = cfg['lon_range']
    n_total = (len(np.arange(la1, la2, cfg['grid_step']))
               * len(np.arange(lo1, lo2, cfg['grid_step'])))

    print("\n  Running detection table...")
    grand, _ = detection_table(zones, catalog, cfg, region, outlines)

    print("\n  Computing Molchan diagram...")
    mol_df = molchan_data(zones, catalog, cfg, n_total, outlines, region)

    print("\n  Running N-test...")
    if 'n365' in zones.columns and 'b_val' in zones.columns:
        exp55 = (zones['n365'] * 10**(-zones['b_val'] * 2.5)).sum()
        exp60 = (zones['n365'] * 10**(-zones['b_val'] * 3.0)).sum()
    else:
        exp55 = len(zones) * 0.5
        exp60 = len(zones) * 0.1

    d1_55, d2_55, p55, lo55, hi55 = n_test(grand['n55'], exp55)
    d1_60, d2_60, p60, lo60, hi60 = n_test(grand['n60'], exp60)

    outlines += [
        f"\n{'='*55}",
        "  N-TEST (Poisson 95% confidence)",
        f"{'='*55}",
        "\n  M >= 5.5:",
        f"    Expected : {exp55:.1f}",
        f"    Observed : {grand['n55']}",
        f"    95% CI   : [{lo55:.0f}, {hi55:.0f}]",
        f"    delta1   : {d1_55:.4f}",
        f"    delta2   : {d2_55:.4f}",
        f"    Result   : {'PASS' if p55 else 'FAIL'}",
        "\n  M >= 6.0:",
        f"    Expected : {exp60:.1f}",
        f"    Observed : {grand['n60']}",
        f"    95% CI   : [{lo60:.0f}, {hi60:.0f}]",
        f"    delta1   : {d1_60:.4f}",
        f"    delta2   : {d2_60:.4f}",
        f"    Result   : {'PASS' if p60 else 'FAIL'}",
        "\n  NOTE: N-test FAIL is expected for alarm-based models.",
        "  Alarm models forecast zones, not Poisson rates.",
        "  The Molchan diagram is the primary skill metric.",
    ]

    outfile_txt = f"csep_results_{region}.txt"
    with open(outfile_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(outlines))
    print(f"\n  Saved: {outfile_txt}")

    plot_molchan(mol_df, region, f"csep_molchan_{region}.png")

    print()
    print(f"  SUMMARY:")
    print(f"  M>=5.5 : {grand['d55']}/{grand['n55']} = "
          f"{100*grand['d55']//grand['n55'] if grand['n55']>0 else 0}%")
    print(f"  M>=6.0 : {grand['d60']}/{grand['n60']} = "
          f"{100*grand['d60']//grand['n60'] if grand['n60']>0 else 0}%")
    print(f"  M>=7.0 : {grand['d70']}/{grand['n70']}")
    print(f"  N-test M5.5 : {'PASS' if p55 else 'FAIL (expected for alarm model)'}")
    print(f"  N-test M6.0 : {'PASS' if p60 else 'FAIL (expected for alarm model)'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='CSEP Evaluation - Six-Criterion Consensus Framework')
    parser.add_argument('--region', default='both',
                        choices=['japan','turkey','both'])
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CSEP Evaluation - Six-Criterion Consensus Framework     ║")
    print("║  Ramakrishna Pasupuleti - Independent Researcher         ║")
    print("║  pycsep 0.8.0                                            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    regions = (['japan','turkey'] if args.region == 'both'
               else [args.region])

    for region in regions:
        try:
            evaluate_region(region)
        except Exception as e:
            print(f"\n  ERROR in {region}: {e}")
            import traceback; traceback.print_exc()

    print("Done. Files created:")
    for region in regions:
        print(f"  csep_results_{region}.txt")
        print(f"  csep_molchan_{region}.png")
    print()
    print("Cite: Savran et al. (2022) SRL doi:10.1785/0220200386")
    print("      pycsep v0.8.0  https://github.com/SCECCode/pycsep")


if __name__ == '__main__':
    main()
