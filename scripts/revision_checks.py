"""Reviewer-response diagnostics computed from the saved final outputs (no reprocessing).

Writes tables to outputs/tables/revision_checks/ and the full Table S2 to outputs/tables/.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/tables/revision_checks'
OUT.mkdir(parents=True, exist_ok=True)
lines = []


def log(s=''):
    print(s)
    lines.append(s)


g = gpd.read_file(ROOT / 'data/processed/risk_index/cgsri_grid10km.gpkg')
N = len(g)
COMP = {'GWH': 'gw_hazard', 'SH': 'sub_hazard', 'CS': 'climate_stress', 'E': 'exposure', 'V': 'vulnerability'}
X = g[list(COMP.values())].astype(float).copy()
X.columns = list(COMP)


def mm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def top_q(s, n=91):
    r = s.rank(method='first')
    return set(r[r > len(s) - n].index)


def quintile(s):
    return pd.qcut(s.rank(method='first'), 5, labels=['Very low', 'Low', 'Moderate', 'High', 'Very high'])


# ── 1. UGPI built-up growth cap sensitivity ─────────────────────────────────
log('== 1. UGPI built-up growth-rate cap sensitivity ==')
bgr = g['builtup_growth_pct']
assert np.allclose(g['n_BUGR'], mm(bgr).clip(0, 1), atol=1e-6), 'n_BUGR is not uncapped min-max'
log('Implemented n_BUGR = uncapped min-max of builtup_growth_pct (verified).')
u0 = (g['n_BD'] + g['n_PD'] + g['n_BUGR']) / 3
assert np.allclose(u0, g['ugpi_urban_only'])
rows = []
for p in [90, 95, 99]:
    cap = bgr.quantile(p / 100)
    uc = (g['n_BD'] + g['n_PD'] + mm(bgr.clip(upper=cap))) / 3
    rho = spearmanr(u0, uc)[0]
    q75 = len(set(u0[u0 >= u0.quantile(.75)].index) & set(uc[uc >= uc.quantile(.75)].index))
    q90 = len(set(u0[u0 >= u0.quantile(.90)].index) & set(uc[uc >= uc.quantile(.90)].index))
    rows.append(dict(cap_percentile=p, cap_value_pct=round(cap, 1), spearman_vs_uncapped=round(rho, 3),
                     q75_cells_shared=q75, q75_cells_total=int((u0 >= u0.quantile(.75)).sum()),
                     q90_cells_shared=q90, q90_cells_total=int((u0 >= u0.quantile(.90)).sum())))
    log(f'  P{p} cap ({cap:.0f}%): rho={rho:.3f}  Q75 shared {q75}  Q90 shared {q90}')
pd.DataFrame(rows).to_csv(OUT / 'ugpi_cap_sensitivity.csv', index=False)

# ── 2. UGPI completeness and the GRACE-coverage split ──────────────────────
log('\n== 2. UGPI completeness / GRACE-coverage split ==')
hg = g['drought_freq'].notna()
log(f'  3-indicator UGPI missing values: {int(g["ugpi_urban_only"].isna().sum())} of {N}')
log(f'  6-indicator (original) UGPI missing: {int(g["ugpi_equal"].isna().sum())}  '
    f'(= cells without nearest GRACE drought value: {int((~hg).sum())})')
log(f'  GWSA pixel trend range in covered cells: {g.loc[hg,"gwsa_slope"].min():.2f} to {g.loc[hg,"gwsa_slope"].max():.2f} mm/yr')
log(f'  GWH: zero in {int((g.gw_hazard[hg] == 0).sum())} covered cells; '
    f'non-covered cells use 0.5 x normalised UGPI (range {g.gw_hazard[~hg].min():.3f}-{g.gw_hazard[~hg].max():.3f})')
base5 = X.mean(axis=1)
assert np.allclose(base5, g['cgsri'])
VH5 = top_q(base5)
alt = X.copy()
alt['GWH'] = 0.0
cs_chirps = mm(g['chirps_below_normal_pct'])
alt['CS'] = cs_chirps
s_alt = alt.mean(axis=1)
log(f'  Harmonised alternative (GWH=0 everywhere, CS=CHIRPS-only everywhere): '
    f'rho={spearmanr(base5, s_alt)[0]:.3f}, VH retained {len(top_q(s_alt) & VH5)}/91')
sub = base5[hg]
vh_sub = top_q(sub, n=int(round(len(sub) / 5)))
log(f'  Excluding the {int((~hg).sum())} non-covered cells: VH (n={len(vh_sub)}) of 330-cell ranking '
    f'shared with full-sample VH: {len(vh_sub & VH5)}; full-sample VH cells inside 330: {len(VH5 & set(sub.index))}')
pd.DataFrame([dict(test='GWH=0 & CS=CHIRPS-only for all cells', spearman=round(spearmanr(base5, s_alt)[0], 3),
                   vh_retained=len(top_q(s_alt) & VH5)),
              dict(test='exclude 123 non-GRACE cells (330-cell quintiles)', spearman=np.nan,
                   vh_retained=len(vh_sub & VH5))]).to_csv(OUT / 'grace_coverage_sensitivity.csv', index=False)

# ── 3. Weighting schemes, E+V, 4-component, component importance ─────────
log('\n== 3. Weighting schemes ==')
SCH = {
    'equal_5comp': dict(GWH=.2, SH=.2, CS=.2, E=.2, V=.2),
    'primary_4comp_noSH': dict(GWH=.25, CS=.25, E=.25, V=.25),
    'hazard_dominant': dict(GWH=.25, SH=.25, CS=1 / 6, E=1 / 6, V=1 / 6),
    'exposure_dominant': dict(GWH=1 / 6, SH=1 / 6, CS=1 / 6, E=.25, V=.25),
    'SH_sceptic': dict(GWH=.2375, SH=.05, CS=.2375, E=.2375, V=.2375),
    'GWH_removed': dict(SH=.25, CS=.25, E=.25, V=.25),
}
s2 = g[['cell_id', 'centroid_lat', 'centroid_lon', 'population_2020']].copy()
summ = []
for k, w in SCH.items():
    s = sum(X[c] * w.get(c, 0) for c in COMP)
    s2[f'cgsri_{k}'] = s.round(4)
    s2[f'class_{k}'] = quintile(s).astype(str)
    vh = top_q(s)
    summ.append(dict(scheme=k, spearman_vs_equal=round(spearmanr(s, base5)[0], 3), vh_retained=len(vh & VH5),
                     vh_pop_millions=round(g.loc[list(vh), 'population_2020'].sum() / 1e6, 2)))
    log(f'  {k:20s} rho={summ[-1]["spearman_vs_equal"]:.3f} VH retained {summ[-1]["vh_retained"]}/91 '
        f'pop {summ[-1]["vh_pop_millions"]}M')
pd.DataFrame(summ).to_csv(OUT / 'weighting_schemes.csv', index=False)
s2.to_csv(ROOT / 'outputs/tables/supplementary_table_S2_sensitivity_analysis.csv', index=False)
ev = X.E + X.V
s4 = sum(X[c] * .25 for c in ['GWH', 'CS', 'E', 'V'])
log(f'  E+V top quintile shared with 5-comp VH: {len(top_q(ev) & VH5)}; with 4-comp VH: {len(top_q(ev) & top_q(s4))}')
log(f'  pearson(E,UGPI)={np.corrcoef(X.E, g.ugpi_urban_only)[0, 1]:.2f}  pearson(E,V)={np.corrcoef(X.E, X.V)[0, 1]:.2f}')
log(f'  rho(E,V)={spearmanr(X.E, X.V)[0]:.2f}  rho(n_BD,n_PD)={spearmanr(g.n_BD, g.n_PD)[0]:.2f} '
    f'pearson(n_BD,n_PD)={np.corrcoef(g.n_BD, g.n_PD)[0, 1]:.2f}  rho(E,UGPI)={spearmanr(X.E, g.ugpi_urban_only)[0]:.2f}')
only56 = list(VH5 - top_q(ev))
log(f'  56 CGSRI-only VH cells: mean SH {X.SH[only56].mean():.2f} vs all {X.SH.mean():.2f}; '
    f'mean elev {g.elev_m[only56].mean():.1f} m vs all {g.elev_m.mean():.1f} m')
log(f'  Total pop High+VeryHigh (5comp): {g.loc[quintile(base5).isin(["High","Very high"]),"population_2020"].sum()/1e6:.2f}M; '
    f'VH 5comp {g.loc[list(VH5),"population_2020"].sum()/1e6:.3f}M; VH 4comp {g.loc[list(top_q(s4)),"population_2020"].sum()/1e6:.3f}M')

# ── 4. Q90 co-location cells ────────────────────────────────────────────────
log('\n== 4. Q90 co-location cells ==')
u, sh = g['ugpi_urban_only'], g['sub_hazard']
hh = g[(u >= u.quantile(.9)) & (sh >= sh.quantile(.9))]
log(hh[['cell_id', 'centroid_lat', 'centroid_lon', 'insar_n_valid']].round(3).to_string(index=False))
hh[['cell_id', 'centroid_lat', 'centroid_lon', 'insar_n_valid', 'ugpi_urban_only', 'sub_hazard']].to_csv(
    OUT / 'q90_colocation_cells.csv', index=False)
a = int(((u >= u.quantile(.75)) & (sh >= sh.quantile(.75))).sum())
nu, ns = int((u >= u.quantile(.75)).sum()), int((sh >= sh.quantile(.75)).sum())
log(f'  Q75 HH={a}, margins {nu}x{ns}, expected under independence = {nu * ns / N:.1f}')
nu9, ns9 = int((u >= u.quantile(.9)).sum()), int((sh >= sh.quantile(.9)).sum())
log(f'  Q90 HH={len(hh)}, margins {nu9}x{ns9}, expected = {nu9 * ns9 / N:.1f}')

# ── 5. Drought episodes ─────────────────────────────────────────────────────
log('\n== 5. Drought episodes ==')
d = pd.read_csv(ROOT / 'data/processed/groundwater_anomalies/gwsa_by_region_monthly.csv', parse_dates=['time'])
d = d[d.time <= '2023-12-31'].set_index('time')
z = d['sgwa'].dropna()
cal = pd.date_range(z.index.min(), '2023-12-01', freq='MS')
log(f'  valid months {len(z)} of {len(cal)} calendar months')
below = z < -0.5
ep, cur = [], []
for t, b in below.items():
    if b:
        cur.append(t)
    elif cur:
        ep.append(cur)
        cur = []
if cur:
    ep.append(cur)
erows = []
for e in ep:
    seg = z[e[0]:e[-1]]
    calm = (e[-1].year - e[0].year) * 12 + e[-1].month - e[0].month + 1
    cls = pd.cut(seg, [-np.inf, -2, -1.5, -1, -0.5], labels=['extreme', 'severe', 'moderate', 'mild']).value_counts()
    erows.append(dict(start=e[0].strftime('%Y-%m'), end=e[-1].strftime('%Y-%m'), calendar_months=calm,
                      observed_months=len(seg), missing_months=calm - len(seg), min_sgwa=round(seg.min(), 2),
                      min_month=seg.idxmin().strftime('%Y-%m'), **{f'n_{k}': int(v) for k, v in cls.items()}))
E = pd.DataFrame(erows)
E.to_csv(OUT / 'drought_episodes.csv', index=False)
log(E[E.observed_months >= 2].to_string(index=False))

# ── 6. GWSA-rainfall lag correlation ───────────────────────────────────────
log('\n== 6. GWSA vs CHIRPS rainfall anomaly (same Ghana box) ==')
r = pd.read_csv(ROOT / 'data/processed/climate_indices/rainfall_anomaly_monthly.csv', parse_dates=['date']).set_index('date')
ra = r['precip_anom_mm']
ra.index = ra.index.to_period('M')
ra = ra[~ra.index.duplicated()].asfreq('M')
gw = d['gwsa_mm'].dropna()
gw.index = gw.index.to_period('M')
lrows = []
for lag in range(0, 25):
    x = ra.shift(lag).reindex(gw.index)
    m = x.notna()
    rr, pp = pearsonr(gw[m], x[m])
    lrows.append(dict(lag_months=lag, r=round(rr, 3), p=pp, n=int(m.sum())))
cum = ra.rolling(12).sum()
for lag in [0, 3, 6]:
    x = cum.shift(lag).reindex(gw.index)
    m = x.notna()
    rr, pp = pearsonr(gw[m], x[m])
    lrows.append(dict(lag_months=f'cum12_lag{lag}', r=round(rr, 3), p=pp, n=int(m.sum())))
L = pd.DataFrame(lrows)
L.to_csv(OUT / 'gwsa_rainfall_lag_correlation.csv', index=False)
log(L.to_string(index=False))

# ── 7. Lake Volta order of magnitude ───────────────────────────────────────
log('\n== 7. Domain area for reservoir-equivalent estimate ==')
ds = xr.open_dataset(ROOT / 'data/raw/grace/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04CRI.nc')
ds = ds.assign_coords(lon=((ds.lon + 180) % 360 - 180)).sortby('lon')
lm = ds['land_mask'].sel(lat=slice(4.0, 11.5), lon=slice(-3.5, 1.5))
lat = np.deg2rad(lm.lat)
cell_km2 = (111.32 * 0.5) ** 2 * np.cos(lat)
area = float((lm * cell_km2).sum())
log(f'  Land-masked GRACE area in 4.0-11.5N, 3.5W-1.5E: {area:,.0f} km2 ({int(lm.sum())} 0.5deg cells)')
log(f'  1 m level change over 8,500 km2 of lake = {8500 / area * 1000:.0f} mm EWH over this domain')

(OUT / 'summary.txt').write_text('\n'.join(lines))

# ── 8. SBAS network structure ───────────────────────────────────────────────
import h5py
log('\n== 8. SBAS network ==')
with h5py.File(ROOT / 'data/processed/subsidence/mintpy/inputs/ifgramStack.h5', 'r') as f:
    dates = [[str(x.decode() if isinstance(x, bytes) else x) for x in row] for row in f["date"][:]]
    bperp = f['bperp'][:]
    keep = f['dropIfgram'][:]
pairs = [tuple(p) for p in dates]
nodes = sorted({d for p in pairs for d in p})
parent = {n: n for n in nodes}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for a, b in pairs:
    parent[find(a)] = find(b)
comps = {}
for n in nodes:
    comps.setdefault(find(n), []).append(n)
sizes = sorted((len(v) for v in comps.values()), reverse=True)
edges = {frozenset(p) for p in pairs}
adj = {n: set() for n in nodes}
for a, b in pairs:
    adj[a].add(b)
    adj[b].add(a)
tri = sum(1 for a, b in pairs for c in adj[a] & adj[b]) // 3
dt = [(pd.Timestamp(b) - pd.Timestamp(a)).days for a, b in pairs]
log(f'  pairs {len(pairs)} (kept {int(keep.sum())}), dates {len(nodes)}, temporal components {len(comps)}, '
    f'largest {sizes[0]} dates, singletons {sum(s == 1 for s in sizes)}, closed triangles {tri}')
log(f'  temporal baselines: {pd.Series(dt).value_counts().to_dict()}; |Bperp|>500: '
    f'{[(pairs[i], round(float(bperp[i]), 1), bool(keep[i])) for i in np.where(np.abs(bperp) > 500)[0]]}')
comp_rows = sorted(([min(v), max(v), len(v)] for v in comps.values()))
pd.DataFrame(comp_rows, columns=['first_date', 'last_date', 'n_dates']).to_csv(OUT / 'sbas_temporal_components.csv', index=False)
pd.DataFrame({'date1': [p[0] for p in pairs], 'date2': [p[1] for p in pairs], 'dt_days': dt,
              'bperp_m': np.round(bperp, 1), 'used': keep}).to_csv(OUT / 'sbas_pair_list.csv', index=False)

# ── 9. GRACE missing months ─────────────────────────────────────────────────
log('\n== 9. Missing GRACE months (Apr 2002 - Dec 2023) ==')
obs = set(d['gwsa_mm'].dropna().index.to_period('M'))
calm = pd.period_range('2002-04', '2023-12', freq='M')
miss = [str(p) for p in calm if p not in obs]
log(f'  calendar {len(calm)}, observed {len(obs & set(calm))}, missing {len(miss)}: {miss}')

# ── 10. InSAR coverage by built-up fraction; northern cells ────────────────
log('\n== 10. InSAR coverage and northern cells ==')
cov = g[g.insar_n_valid > 0].copy()
cov['bu_class'] = pd.qcut(cov['builtup_density_2020'], 3, labels=['low', 'mid', 'high'])
log(cov.groupby('bu_class', observed=True)['insar_n_valid'].agg(['count', 'median', 'min', 'max']).to_string())
log(f'  max built-up fraction any cell: {g.builtup_density_2020.max():.3f}')
north = g.centroid_lat >= 7.0
log(f'  grid cells >=7.0N: {int(north.sum())}; of these with >=10 InSAR px: {int((north & (g.insar_n_valid >= 10)).sum())}')
log(f'  VH 5comp cells >=6.3N: {sum(g.loc[list(VH5), "centroid_lat"] >= 6.3)}; '
    f'VH 4comp cells >=6.3N: {sum(g.loc[list(top_q(s4)), "centroid_lat"] >= 6.3)}')
log(f'  VH 5comp cells without GRACE coverage: {len(VH5 & set(g.index[~hg]))}; '
    f'VH 4comp: {len(top_q(s4) & set(g.index[~hg]))}')
alt4 = (0 + mm(g['chirps_below_normal_pct']) + X.E + X.V) / 4
log(f'  harmonised 4comp (GWH=0, CS CHIRPS-only): rho={spearmanr(s4, alt4)[0]:.3f}, VH retained {len(top_q(alt4) & top_q(s4))}/91')

# ── 11. Two-point planar ramp sensitivity ──────────────────────────────────
log('\n== 11. Ramp sensitivity ==')
REF = (5.55, -0.42)
KM = 110.6


def dxy(lat, lon):
    return (lon - REF[1]) * KM * np.cos(np.deg2rad(REF[0])), (lat - REF[0]) * KM


kx, ky = dxy(5.60, -0.19)            # Kotoka: observed LOS +9.0, assumed stable
ux, uy = dxy(5.75, -0.31)            # KUTU
ax_, ay_ = dxy(5.56, -0.20)          # ACRA (anchor of the GNSS comparison)
kutu_los = -7.4 * np.cos(np.deg2rad(38))
A = np.array([[kx, ky], [ux - ax_, uy - ay_]])
a, b = np.linalg.solve(A, [9.0, kutu_los])
log(f'  plane: {a:.3f} mm/yr/km east, {b:.3f} mm/yr/km north (|grad| {np.hypot(a, b):.2f})')
has = g.insar_n_valid >= 10
cx, cy = dxy(g.centroid_lat, g.centroid_lon)
v0 = g.insar_vel_mmyr
v1 = v0 - (a * cx + b * cy)


def haz(v):
    vc = v[has].clip(-60, 20)
    return 1 - (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)


h0, h1 = haz(v0), haz(v1)
log(f'  66-cell InSAR_haz rank correlation original vs ramp-corrected: rho={spearmanr(h0, h1)[0]:.3f}')
log(f'  distance-velocity rho (66 cells): {spearmanr(np.hypot(cx[has], cy[has]), v0[has])[0]:.2f}; '
    f'east-offset vs velocity rho: {spearmanr(cx[has], v0[has])[0]:.2f}')
elev_inv = 1 - mm(g.elev_m.clip(0, 200))
proxy = (elev_inv + mm(g.builtup_density_2020) + g.gw_hazard) / 3
w = np.clip(g.insar_n_valid / 500, 0, 1)
w = np.where(g.centroid_lat < 7.0, np.minimum(w, 0.7), w)
w = np.where(has, w, 0)


def sh(hz):
    full = proxy.copy()
    full[has] = hz
    return (1 - w) * proxy + w * full


log(f'  SH reproduction max abs error: {np.abs(sh(h0) - g.sub_hazard).max():.2e}')
X1 = X.copy()
X1['SH'] = sh(h1)
s5r = X1.mean(axis=1)
log(f'  5comp CGSRI with ramp-corrected SH: rho={spearmanr(base5, s5r)[0]:.3f}, VH retained {len(top_q(s5r) & VH5)}/91')
u_, shr = g['ugpi_urban_only'], X1['SH']
hh_r = set(g.index[(u_ >= u_.quantile(.9)) & (shr >= shr.quantile(.9))])
log(f'  Q90 HH under ramp-corrected SH: {len(hh_r)} cells; shared with original 8: {len(hh_r & set(hh.index))}')
(OUT / 'summary.txt').write_text('\n'.join(lines))
