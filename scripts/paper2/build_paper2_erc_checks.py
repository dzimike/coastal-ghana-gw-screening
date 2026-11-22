"""Paper 2 (ERC version): robustness checks added to build_uhgpi_v2.py. Run that script first.

Does not change the index. Adds:
 1. Day and night dry-season LST from the raw MOD11A2 granules (fast pixel-to-cell aggregation),
    validated against the daytime series used in the index.
 2. Night-time LST: trends, spatial pattern and urban contrast at equal distance from the sea.
 3. Sensitivity of the daytime trend to Terra's orbit drift (2001-2019 against 2001-2023).
 4. Sensitivity of the warming-urban growth association to how growth is measured, to the
    extrapolated 2025 GHSL epoch, to the climatic gradient and to the Tarkwa-Prestea mining area.
 5. Cloud sampling: trend in the number of valid composites against the LST trend.
 6. Regional groundwater context from a 9-member GRACE x GLDAS ensemble, with and without Lake Volta.

Outputs: data/processed/lst/erc/ and outputs/tables/paper2_erc/ (build_log.txt holds every number).
"""
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pymannkendall as mk
import xarray as xr
from pyhdf.SD import SD, SDC
from pyproj import Transformer
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/modis_lst'
V2 = ROOT / 'data/processed/lst/v2'
P3 = ROOT / 'data/processed/paper3'
OUT = ROOT / 'data/processed/lst/erc'
TAB = ROOT / 'outputs/tables/paper2_erc'
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
GRID = ROOT / 'data/processed/risk_index/cgsri_grid10km_v2.gpkg'
YEARS = (2001, 2023)
MIN_COMPOSITES, MIN_SEASONS = 3, 18
N_SURR = 9999
TARKWA = (-1.995, 5.301)
MINING_RADIUS_KM = 30
SINU = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs'
PIX = 926.625433
rng = np.random.default_rng(20261121)
lines = []


def log(s=''):
    print(s, flush=True)
    lines.append(str(s))


# ── 1. day and night LST per cell, Dec-Feb composites ───────────────────────
grid = gpd.read_file(GRID)[['cell_id', 'col_id', 'row_id', 'geometry']]
cache = OUT / 'lst_djf_day_night_by_cell.parquet'


def tile_index(h):
    """Row/col of pixels in a tile that fall in a grid cell, and the cell index."""
    x0 = (h - 18) * PIX * 1200
    y0 = (9 - 8) * PIX * 1200
    cols, rows = np.meshgrid(np.arange(1200), np.arange(1200))
    x, y = x0 + (cols + .5) * PIX, y0 - (rows + .5) * PIX
    e, n = Transformer.from_crs(SINU, grid.crs, always_xy=True).transform(x.ravel(), y.ravel())
    pts = gpd.GeoDataFrame({'i': np.arange(e.size)}, geometry=gpd.points_from_xy(e, n), crs=grid.crs)
    xmin, ymin, xmax, ymax = grid.total_bounds
    pts = pts[(e >= xmin) & (e <= xmax) & (n >= ymin) & (n <= ymax)]
    j = gpd.sjoin(pts, grid[['geometry']], predicate='within', how='inner')
    j = j[~j.index.duplicated()]
    return j['i'].values, j['index_right'].values


if cache.exists():
    long = pd.read_parquet(cache)
else:
    idx = {h: tile_index(h) for h in (17, 18)}
    rows = []
    files = sorted(RAW.glob('MOD11A2.A*.hdf'))
    dates = {}
    for f in files:
        m = re.search(r'A(\d{7})\.h(\d\d)v08', f.name)
        d = pd.to_datetime(m.group(1), format='%Y%j')
        if d.month in (12, 1, 2):
            dates.setdefault(d, {})[int(m.group(2))] = f
    for k, (d, tiles) in enumerate(sorted(dates.items())):
        sums = {b: np.zeros(len(grid)) for b in ('Day', 'Night')}
        cnts = {b: np.zeros(len(grid)) for b in ('Day', 'Night')}
        for h, f in tiles.items():
            hdf = SD(str(f), SDC.READ)
            for b in ('Day', 'Night'):
                v = hdf.select(f'LST_{b}_1km')[:].astype(float).ravel()
                q = hdf.select(f'QC_{b}')[:].astype(np.uint8).ravel()
                pi, ci = idx[h]
                ok = (v[pi] != 0) & ((q[pi] & 3) == 0)
                np.add.at(sums[b], ci[ok], v[pi][ok] * 0.02 - 273.15)
                np.add.at(cnts[b], ci[ok], 1)
            hdf.end()
        for b in ('Day', 'Night'):
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = sums[b] / cnts[b]
            rows.append(pd.DataFrame({'date': d, 'band': b, 'cell_id': grid.cell_id.values, 'lst': mean,
                                      'npix': cnts[b].astype(int)}))
        if k % 100 == 0:
            print('composite', k, d.date(), flush=True)
    long = pd.concat(rows, ignore_index=True)
    long.to_parquet(cache)

long['ds'] = np.where(long.date.dt.month == 12, long.date.dt.year + 1, long.date.dt.year)
long = long[long.ds.between(*YEARS) & long.lst.notna()]


def seasons(band):
    w = long[long.band == band].pivot_table(index='ds', columns='cell_id', values='lst', aggfunc=['mean', 'count'])
    return w['mean'].where(w['count'] >= MIN_COMPOSITES), w['count']


day, nday = seasons('Day')
night, nnight = seasons('Night')
a = pd.read_csv(V2 / 'uhgpi_grid10km_v2.csv')
a = a[a.uhgpi.notna()].reset_index(drop=True)
v = gpd.read_file(GRID).drop(columns='geometry').set_index('cell_id')
for c in ('builtup_m2_2000', 'builtup_m2_2020', 'builtup_m2_2025', 'builtup_growth_pct'):
    a[c] = a.cell_id.map(v[c])
ref = pd.read_csv(V2 / 'lst_dry_season_2001_2023.csv', index_col=0)[a.cell_id]
cmn = day.reindex(columns=a.cell_id)
x, y = ref.values.ravel(), cmn.values.ravel()
ok = np.isfinite(x) & np.isfinite(y)
log('1. PIPELINE CHECK (fast day aggregation vs series used in the index)')
log(f'cell-seasons compared {ok.sum()}; r {np.corrcoef(x[ok], y[ok])[0, 1]:.4f}; mean diff {np.mean(y[ok] - x[ok]):+.3f} C; '
    f'RMSD {np.sqrt(np.mean((y[ok] - x[ok]) ** 2)):.3f} C')


# ── helpers ─────────────────────────────────────────────────────────────────
def build_null(df):
    c, r = df.col_id.values, df.row_id.values
    W = ((np.abs(c[:, None] - c[None, :]) <= 1) & (np.abs(r[:, None] - r[None, :]) <= 1)).astype(float)
    np.fill_diagonal(W, 0)
    n = len(df)
    H = np.eye(n) - 1.0 / n
    vals, vecs = np.linalg.eigh(H @ W @ H)
    return vecs[:, np.abs(vals) > 1e-8]


def msr(x, y, V, method='spearman', n=N_SURR):
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = (lambda u: stats.spearmanr(u, y)[0]) if method == 'spearman' else (lambda u: np.corrcoef(u, y)[0, 1])
    obs = f(x)
    z = (x - x.mean()) / x.std()
    coef = V.T @ z
    sur = (rng.choice([-1.0, 1.0], size=(n, coef.size)) * coef) @ V.T
    null = np.array([f(s) for s in sur])
    return obs, (1 + (np.abs(null) >= abs(obs)).sum()) / (n + 1)


def partial(df, x, y, ctrl):
    R = df[[x, y] + ctrl].rank()
    X = np.column_stack([np.ones(len(R)), R[ctrl].values])
    res = lambda s: s - X @ np.linalg.lstsq(X, s, rcond=None)[0]
    return res(R[x].values), res(R[y].values)


FALLBACK = {'n': 0}


def hr_test(values):
    """Hamed-Rao slope and p; the original Mann-Kendall p is used if the corrected variance is negative."""
    with np.errstate(invalid='ignore'):
        h = mk.hamed_rao_modification_test(values)
    if np.isfinite(h.p):
        return h.slope, h.p
    FALLBACK['n'] += 1
    return h.slope, mk.original_test(values).p


def trends(s, ids, min_seasons=MIN_SEASONS):
    out = {}
    for c in ids:
        ts = s[c].dropna()
        if len(ts) >= min_seasons:
            out[c] = hr_test(ts.values)
    return pd.DataFrame.from_dict(out, orient='index', columns=['slope', 'p'])


V = build_null(a)
assoc = []

# ── 2. night-time LST ───────────────────────────────────────────────────────
log('\n2. NIGHT-TIME LST')
nt = trends(night, a.cell_id)
a['night_slope'], a['night_p'] = a.cell_id.map(nt.slope), a.cell_id.map(nt.p)
a['night_recent'] = a.cell_id.map(night.loc[2019:2023].mean())
a['night_valid'] = a.cell_id.map(nnight.reindex(columns=a.cell_id).mean())
nn = a.dropna(subset=['night_slope', 'night_recent']).reset_index(drop=True)
nn['night_anom'] = nn.night_recent - nn.night_recent.mean()
nn['night_fdr'] = multipletests(nn.night_p, method='fdr_bh')[0]
s = nn.night_slope
log(f'cells with >= {MIN_SEASONS} valid night seasons: {len(nn)} of {len(a)}; mean valid composites per season '
    f'{nn.night_valid.mean():.1f}; recent night LST {nn.night_recent.min():.1f}-{nn.night_recent.max():.1f} C')
log(f'night Sen slope median {s.median():+.4f} C/yr (IQR {s.quantile(.25):+.4f} to {s.quantile(.75):+.4f}); positive '
    f'{int((s > 0).sum())}/{len(s)} ({100 * (s > 0).mean():.0f}%); p<0.05 {int((nn.night_p < .05).sum())} '
    f'({int(((nn.night_p < .05) & (s > 0)).sum())} positive); FDR {int(nn.night_fdr.sum())} '
    f'({int((nn.night_fdr & (s > 0)).sum())} positive)')
dom_n = night[nn.cell_id].mean(axis=1)
hs, hp = hr_test(dom_n.values)
log(f'cells where Hamed-Rao variance was negative (original MK p used): {FALLBACK["n"]}')
log(f'domain-mean night LST {dom_n.mean():.2f} C; Sen {hs:+.4f} C/yr; Hamed-Rao p {hp:.3g}')
Vn = build_null(nn)
for lab, xv, yv in [('night trend vs built-up growth', 'night_slope', 'n_BUGR'),
                    ('night trend vs UGPI', 'night_slope', 'ugpi_urban_only'),
                    ('night anomaly vs UGPI', 'night_anom', 'ugpi_urban_only'),
                    ('night anomaly vs distance to sea', 'night_anom', 'dist_sea_km'),
                    ('night anomaly vs rainfall', 'night_anom', 'chirps_mean_rain_mm'),
                    ('night anomaly vs elevation', 'night_anom', 'elev_m'),
                    ('night anomaly vs day anomaly', 'night_anom', 'lst_anom')]:
    rho, pm = msr(nn[xv], nn[yv], Vn)
    assoc.append(dict(test=lab, n=len(nn), rho=round(rho, 3), p_msr=pm))
    log(f'{lab:40s} rho {rho:+.3f}  p_MSR {pm:.4f}')
e1, e2 = partial(nn, 'night_anom', 'ugpi_urban_only', ['dist_sea_km', 'chirps_mean_rain_mm', 'elev_m'])
rho, pm = msr(e1, e2, Vn, 'pearson')
assoc.append(dict(test='night anomaly vs UGPI | distance, rainfall, elevation', n=len(nn), rho=round(rho, 3), p_msr=pm))
log(f'partial night anomaly vs UGPI | distance, rainfall, elevation: {rho:+.3f}  p_MSR {pm:.4f}')
top = nn.ugpi_urban_only >= a.ugpi_urban_only.quantile(.8)
bands = pd.cut(nn.dist_sea_km, [-1, 10, 20, 30, 40, 50], labels=['0-10', '10-20', '20-30', '30-40', '40-50'])
wb = nn.groupby([bands, top], observed=True).night_anom.agg(['mean', 'size']).unstack()
wb.columns = ['other_mean', 'top_mean', 'other_n', 'top_n']
wb['diff'] = wb.top_mean - wb.other_mean
log('night anomaly, top-quintile UGPI vs other cells, by distance band:\n' + wb.round(2).to_string())
wb.to_csv(TAB / 'night_ugpi_contrast_within_bands.csv')
nn[['cell_id', 'night_slope', 'night_p', 'night_fdr', 'night_recent', 'night_anom', 'night_valid']].to_csv(
    OUT / 'night_lst_by_cell.csv', index=False)
dom_n.rename('domain_night_lst').to_csv(OUT / 'domain_night_lst.csv')

# ── 3. Terra orbit drift ────────────────────────────────────────────────────
log('\n3. TERRA ORBIT DRIFT: DAYTIME TRENDS BY END YEAR')
season = pd.read_csv(V2 / 'lst_dry_season_2001_2023.csv', index_col=0)
drift = []
for band, ser in (('day', season), ('night', night)):
    for end in (2019, 2020, 2021, 2022, 2023):
        t = trends(ser.loc[:end], a.cell_id, min_seasons=int(round(MIN_SEASONS * (end - 2000) / 23)))
        j = a.set_index('cell_id').join(t, how='inner')
        sig = multipletests(j.p, method='fdr_bh')[0]
        dm = ser.loc[:end, a.cell_id].mean(axis=1)
        ds_, dp = hr_test(dm.values)
        drift.append(dict(band=band, end=end, cells=len(j), median_slope=round(j.slope.median(), 4),
                          pct_positive=round(100 * (j.slope > 0).mean(), 1), fdr_cells=int(sig.sum()),
                          rho_vs_growth=round(stats.spearmanr(j.slope, j.n_BUGR)[0], 3),
                          domain_slope=round(ds_, 4), domain_p=round(dp, 4)))
dd = pd.DataFrame(drift)
log(dd.to_string(index=False))
dd.to_csv(TAB / 'trend_end_year_sensitivity.csv', index=False)

# independent air-temperature comparison (TerraClimate Tmax/Tmin at the cell centroids, Dec-Feb)
log('\nTerraClimate air temperature, Dec-Feb mean over the analysis cells')
lat_da = xr.DataArray(a.centroid_lat.values, dims='c')
lon_da = xr.DataArray(a.centroid_lon.values, dims='c')
tc = {}
for var in ('tmax', 'tmin'):
    parts = []
    for y in range(2000, 2024):
        f = ROOT / f'data/raw/terraclimate/TerraClimate_{var}_{y}.nc'
        if f.exists():
            with xr.open_dataset(f) as d:
                da = d[var].sel(lat=lat_da, lon=lon_da, method='nearest').load()
            parts.append(da.mean('c').to_series())
    if parts:
        m = pd.concat(parts).sort_index()
        m = m[m.index.month.isin([12, 1, 2])]
        lab = np.where(m.index.month == 12, m.index.year + 1, m.index.year)
        cnt = m.groupby(lab).count()
        tc[var] = m.groupby(lab).mean()[cnt == 3].loc[2001:2023]
tcr = []
for var, ser in (('tmax', season), ('tmin', night)):
    if var not in tc:
        log(f'{var}: files not available')
        continue
    lsts = ser[a.cell_id].mean(axis=1)
    j = pd.concat([tc[var].rename('air'), lsts.rename('lst')], axis=1).dropna()
    for end in (2019, 2023):
        k = j.loc[:end]
        sa, pa = hr_test(k.air.values)
        sl, pl = hr_test(k.lst.values)
        t_ = k.index.values.astype(float)
        dr = np.corrcoef(k.air - np.polyval(np.polyfit(t_, k.air, 1), t_), k.lst - np.polyval(np.polyfit(t_, k.lst, 1), t_))[0, 1]
        tcr.append(dict(variable=var, lst='day' if var == 'tmax' else 'night', end=end, n=len(k), air_slope=round(sa, 4), air_p=round(pa, 4),
                        lst_slope=round(sl, 4), lst_p=round(pl, 4), r=round(np.corrcoef(k.air, k.lst)[0, 1], 3), r_detrended=round(dr, 3)))
tcd = pd.DataFrame(tcr)
log(tcd.to_string(index=False))
tcd.to_csv(TAB / 'terraclimate_comparison.csv', index=False)
t19 = trends(season.loc[:2019], a.cell_id, min_seasons=int(round(MIN_SEASONS * 19 / 23))).reindex(a.cell_id)
sub = a.assign(slope19=t19.slope.values).dropna(subset=['slope19']).reset_index(drop=True)
rho, pm = msr(sub.slope19, sub.n_BUGR, build_null(sub))
assoc.append(dict(test='trend 2001-2019 vs built-up growth', n=len(sub), rho=round(rho, 3), p_msr=pm))
log(f'trend 2001-2019 vs built-up growth: rho {rho:+.3f}, p_MSR {pm:.4f}, n {len(sub)}')

# ── 4. warming vs urban growth ──────────────────────────────────────────────
log('\n4. WARMING AND BUILT-UP GROWTH: SENSITIVITY')
a['growth_pct_2000_2020'] = np.where(a.builtup_m2_2000 > 0, 100 * (a.builtup_m2_2020 / a.builtup_m2_2000 - 1), np.nan)
a['abs_change_2000_2020_pp'] = 100 * (a.builtup_m2_2020 - a.builtup_m2_2000) / 1e8
a['abs_change_2000_2025_pp'] = 100 * (a.builtup_m2_2025 - a.builtup_m2_2000) / 1e8
a['density_2000'] = 100 * a.builtup_m2_2000 / 1e8
log(f'cells with zero built-up in 2000: {int((a.builtup_m2_2000 == 0).sum())}; median 2000 built-up fraction '
    f'{a.density_2000.median():.2f}%; rho(relative growth, 2000 built-up fraction) '
    f'{stats.spearmanr(a.n_BUGR, a.density_2000)[0]:+.3f}; rho(relative growth, absolute change) '
    f'{stats.spearmanr(a.n_BUGR, a.abs_change_2000_2025_pp)[0]:+.3f}')
log(f'domain-level zero-baseline cells set by growth formula: n_BUGR values at those cells '
    f'{a.loc[a.builtup_m2_2000 == 0, "n_BUGR"].describe()[["count", "min", "max"]].round(3).to_dict()}')
lat, lon = a.centroid_lat.values, a.centroid_lon.values
dkm = np.hypot((lon - TARKWA[0]) * 111.32 * np.cos(np.radians(5.3)), (lat - TARKWA[1]) * 110.6)
a['mining_zone'] = dkm <= MINING_RADIUS_KM
log(f'cells within {MINING_RADIUS_KM} km of Tarkwa: {int(a.mining_zone.sum())}; median trend there '
    f'{a.loc[a.mining_zone, "sen_slope"].median():+.4f} vs {a.loc[~a.mining_zone, "sen_slope"].median():+.4f} elsewhere; FDR-significant there '
    f'{int(a.loc[a.mining_zone, "fdr_sig"].astype(str).str.lower().eq("true").sum())}')
CTRL = ['dist_sea_km', 'chirps_mean_rain_mm', 'elev_m']
tests = [('relative growth 2000-2025 (baseline)', a, 'n_BUGR'),
         ('relative growth 2000-2020 (no extrapolated epoch)', a.dropna(subset=['growth_pct_2000_2020']), 'growth_pct_2000_2020'),
         ('absolute built-up change 2000-2020', a, 'abs_change_2000_2020_pp'),
         ('absolute built-up change 2000-2025', a, 'abs_change_2000_2025_pp'),
         ('relative growth, Tarkwa-Prestea excluded', a[~a.mining_zone], 'n_BUGR'),
         ('absolute change 2000-2020, Tarkwa-Prestea excluded', a[~a.mining_zone], 'abs_change_2000_2020_pp')]
out4 = []
for lab, df, col in tests:
    df = df.reset_index(drop=True)
    Vs = build_null(df)
    rho, pm = msr(df.sen_slope, df[col], Vs)
    e1, e2 = partial(df, 'sen_slope', col, CTRL)
    rp, pp = msr(e1, e2, Vs, 'pearson')
    e1, e2 = partial(df, 'sen_slope', col, CTRL + ['density_2000'])
    rb, pb = msr(e1, e2, Vs, 'pearson')
    out4.append(dict(measure=lab, n=len(df), rho=round(rho, 3), p_msr=round(pm, 4), partial_climate=round(rp, 3), p_partial_climate=round(pp, 4),
                     partial_climate_baseline=round(rb, 3), p_partial_baseline=round(pb, 4)))
o4 = pd.DataFrame(out4)
log(o4.to_string(index=False))
o4.to_csv(TAB / 'warming_growth_sensitivity.csv', index=False)

# ── 5. cloud sampling ───────────────────────────────────────────────────────
log('\n5. CLOUD SAMPLING')
nv = nday.reindex(columns=a.cell_id).reindex(range(YEARS[0], YEARS[1] + 1)).fillna(0)
tt = nv.index.values.astype(float)
a['valid_trend'] = a.cell_id.map({c: np.polyfit(tt, nv[c].values, 1)[0] for c in a.cell_id})
log(f'OLS trend in valid composites per season: median {a.valid_trend.median():+.3f} per yr; IQR {a.valid_trend.quantile(.25):+.3f} to {a.valid_trend.quantile(.75):+.3f}')
dom_valid = nv.mean(axis=1)
lr = stats.linregress(tt, dom_valid.values)
log(f'domain mean valid composites per season {dom_valid.mean():.2f}; OLS {lr.slope:+.3f}/yr; p {lr.pvalue:.3f}; '
    f'r(domain LST, domain valid composites) {np.corrcoef(season[a.cell_id].mean(axis=1), dom_valid)[0, 1]:+.3f}')
rho, pm = msr(a.sen_slope, a.valid_trend, V)
assoc.append(dict(test='LST trend vs trend in valid composites', n=len(a), rho=round(rho, 3), p_msr=pm))
log(f'LST trend vs trend in valid composites: rho {rho:+.3f}, p_MSR {pm:.4f}')
e1, e2 = partial(a, 'sen_slope', 'n_BUGR', ['valid_trend'] + CTRL)
rho, pm = msr(e1, e2, V, 'pearson')
assoc.append(dict(test='LST trend vs relative growth | sampling trend, distance, rainfall, elevation', n=len(a), rho=round(rho, 3), p_msr=pm))
log(f'LST trend vs relative growth | sampling trend + climate covariates: {rho:+.3f}, p_MSR {pm:.4f}')

# ── 6. regional groundwater context (ensemble) ──────────────────────────────
log('\n6. REGIONAL GROUNDWATER CONTEXT')
raw = pd.read_csv(P3 / 'gwsa_members_uncorrected.csv', index_col=0, parse_dates=True)
cor = pd.read_csv(P3 / 'gwsa_members_lake_corrected.csv', index_col=0, parse_dates=True)
cor = cor[[c for c in cor.columns if c.endswith('A6000') or c.endswith('A8502')]]
dom = season[a.cell_id].mean(axis=1)
lst = dom - dom.mean()


def z_score(s):
    return (s - s.mean()) / s.std()


def annual_z(s):
    z = z_score(s)
    n = z.groupby(z.index.year).count()
    return z.groupby(z.index.year).mean()[n >= 6]


def djf_z(s):
    z = z_score(s)
    z = z[z.index.month.isin([12, 1, 2])]
    lab = np.where(z.index.month == 12, z.index.year + 1, z.index.year)
    n = z.groupby(lab).count()
    return z.groupby(lab).mean()[n >= 2]


def detrend(s):
    t = s.index.values.astype(float)
    return s - np.polyval(np.polyfit(t, s.values, 1), t)


def lag1(s):
    f = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
    return f.corr(f.shift(1))


def corr_eff(x, y):
    r = np.corrcoef(x, y)[0, 1]
    r1, r2 = lag1(x), lag1(y)
    ne = len(x) * (1 - r1 * r2) / (1 + r1 * r2)
    ne = float(np.clip(ne, 3, len(x)))
    t = r * np.sqrt((ne - 2) / (1 - r * r))
    return r, ne, 2 * stats.t.sf(abs(t), ne - 2)


def deficit(g, l):
    d = g < -0.5
    k = int(d.sum())
    if k == 0 or k == len(g):
        return k, np.nan, np.nan, np.nan, ''
    obs = l[d].mean() - l[~d].mean()
    v = l.values
    sel = np.array([rng.permutation(len(v))[:k] for _ in range(9999)])
    m1 = v[sel].mean(axis=1)
    m0 = (v.sum() - m1 * k) / (len(v) - k)
    return k, l[d].mean(), l[~d].mean(), (1 + ((m1 - m0) >= obs).sum()) / 10000, ','.join(map(str, g.index[d]))


def member_stats(series, member, setname):
    out = []
    for pairing, fn in (('calendar year', annual_z), ('Dec-Feb', djf_z)):
        j = pd.concat([fn(series.dropna()).rename('g'), lst.rename('l')], axis=1).dropna().loc[2002:2023]
        for form in ('raw', 'detrended'):
            g, l = (j.g, j.l) if form == 'raw' else (z_score(detrend(j.g)), detrend(j.l))
            r, ne, p = corr_eff(g, l)
            k, ld, lo, pp, yrs = deficit(g, l)
            out.append(dict(member=member, set=setname, pairing=pairing, form=form, n=len(g), r=r, n_eff=ne, p=p,
                            deficit_years=k, lst_deficit=ld, lst_other=lo, diff=ld - lo, p_hotter=pp, years=yrs))
    return out


rows = []
for c in raw.columns:
    rows += member_stats(raw[c], c, 'uncorrected')
for c in cor.columns:
    rows += member_stats(cor[c], c, 'lake-corrected')
rows += member_stats(raw.mean(axis=1), 'ensemble mean', 'uncorrected')
rows += member_stats(cor.mean(axis=1), 'ensemble mean', 'lake-corrected')
tg = pd.DataFrame(rows)
tg['q'] = np.nan
tg['q_hotter'] = np.nan
mem = tg.member != 'ensemble mean'
tg.loc[mem, 'q'] = multipletests(tg.loc[mem, 'p'], method='fdr_bh')[1]
hot = mem & tg.p_hotter.notna()
tg.loc[hot, 'q_hotter'] = multipletests(tg.loc[hot, 'p_hotter'], method='fdr_bh')[1]
tg.round(4).to_csv(TAB / 'temporal_gwsa_lst_ensemble.csv', index=False)
log(f'tests across members: {int(mem.sum())}; q<0.05 for correlations: {int((tg.q < .05).sum())}; q<0.05 for deficit-years-hotter: {int((tg.q_hotter < .05).sum())}')
for (st, pa, fo), m in tg[mem].groupby(['set', 'pairing', 'form']):
    e = tg[(~mem) & (tg.set == st) & (tg.pairing == pa) & (tg.form == fo)].iloc[0]
    log(f'{st} | {pa} | {fo}: n {e.n}; ensemble r {e.r:+.2f} (n_eff {e.n_eff:.1f}, p {e.p:.2f}); members r {m.r.min():+.2f} to {m.r.max():+.2f}; '
        f'p<0.05 {int((m.p < .05).sum())}/{len(m)}, q<0.05 {int((m.q < .05).sum())}; deficit years {e.years}; LST {e.lst_deficit:+.2f} vs {e.lst_other:+.2f} C; '
        f'p(hotter) {e.p_hotter:.2f}; members p(hotter)<0.05 {int((m.p_hotter < .05).sum())}, q<0.05 {int((m.q_hotter < .05).sum())}; diff range {m["diff"].min():+.2f} to {m["diff"].max():+.2f}')
frame = lambda d, f: pd.DataFrame({c: f(d[c].dropna()) for c in d})
for lab, f in (('annual', annual_z), ('djf', djf_z)):
    u, k = frame(raw, f), frame(cor, f)
    o = pd.DataFrame({'uncorrected': f(raw.mean(axis=1).dropna()), 'lake_corrected': f(cor.mean(axis=1).dropna()),
                      'unc_min': u.min(axis=1), 'unc_max': u.max(axis=1), 'cor_min': k.min(axis=1), 'cor_max': k.max(axis=1)})
    o['lst_anom'] = lst
    o.loc[2002:2023].round(3).to_csv(TAB / f'{lab}_sgwa_ensemble.csv')
    log(f'{lab} years: ' + ','.join(map(str, o.loc[2002:2023].dropna(subset=['uncorrected']).index)))

pd.DataFrame(assoc).to_csv(TAB / 'associations_additional.csv', index=False)
a.to_csv(OUT / 'analysis_cells_erc.csv', index=False)
(TAB / 'build_log.txt').write_text('\n'.join(lines) + '\n')
print('done')
