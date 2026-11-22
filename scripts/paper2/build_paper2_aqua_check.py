"""Paper 2 (ERC version): test the Terra LST results against Aqua MODIS (MYD11A2 v6.1).

Run download_modis_aqua.py and build_paper2_erc_checks.py first. Both sensors are aggregated with the
same pixel-to-cell method and compared over their common dry seasons, 2003-2023.
Outputs: data/processed/lst/erc/lst_djf_aqua_by_cell.parquet and outputs/tables/paper2_erc/aqua_*.
"""
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pymannkendall as mk
from pyhdf.SD import SD, SDC
from pyproj import Transformer
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/modis_lst_aqua'
E = ROOT / 'data/processed/lst/erc'
TAB = ROOT / 'outputs/tables/paper2_erc'
GRID = ROOT / 'data/processed/risk_index/cgsri_grid10km_v2.gpkg'
YEARS = (2003, 2023)
MIN_COMPOSITES = 3
MIN_SEASONS = int(round(18 * 21 / 23))
N_SURR = 9999
SINU = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs'
PIX = 926.625433
CTRL = ['dist_sea_km', 'chirps_mean_rain_mm', 'elev_m']
rng = np.random.default_rng(20261122)
lines = []


def log(s=''):
    print(s, flush=True)
    lines.append(str(s))


grid = gpd.read_file(GRID)[['cell_id', 'geometry']]
cache = E / 'lst_djf_aqua_by_cell.parquet'


def tile_index(h):
    x0, y0 = (h - 18) * PIX * 1200, PIX * 1200
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
    aq = pd.read_parquet(cache)
else:
    idx = {h: tile_index(h) for h in (17, 18)}
    dates = {}
    for f in sorted(RAW.glob('MYD11A2.A*.hdf')):
        m = re.search(r'A(\d{7})\.h(\d\d)v08', f.name)
        d = pd.to_datetime(m.group(1), format='%Y%j')
        if d.month in (12, 1, 2):
            dates.setdefault(d, {})[int(m.group(2))] = f
    rows = []
    for d, tiles in sorted(dates.items()):
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
            rows.append(pd.DataFrame({'date': d, 'band': b, 'cell_id': grid.cell_id.values, 'lst': mean, 'npix': cnts[b].astype(int)}))
    aq = pd.concat(rows, ignore_index=True)
    aq.to_parquet(cache)

te = pd.read_parquet(E / 'lst_djf_day_night_by_cell.parquet')


def seasons(long, band):
    x = long[(long.band == band) & long.lst.notna()].copy()
    x['ds'] = np.where(x.date.dt.month == 12, x.date.dt.year + 1, x.date.dt.year)
    x = x[x.ds.between(*YEARS)]
    w = x.pivot_table(index='ds', columns='cell_id', values='lst', aggfunc=['mean', 'count'])
    return w['mean'].where(w['count'] >= MIN_COMPOSITES), w['count']


def hr(v):
    with np.errstate(invalid='ignore'):
        h = mk.hamed_rao_modification_test(v)
    return h.slope, (h.p if np.isfinite(h.p) else mk.original_test(v).p)


def trends(s, ids, min_seasons):
    out = {c: hr(s[c].dropna().values) for c in ids if s[c].notna().sum() >= min_seasons}
    return pd.DataFrame.from_dict(out, orient='index', columns=['slope', 'p'])


def build_null(df):
    c, r = df.col_id.values, df.row_id.values
    W = ((np.abs(c[:, None] - c[None, :]) <= 1) & (np.abs(r[:, None] - r[None, :]) <= 1)).astype(float)
    np.fill_diagonal(W, 0)
    H = np.eye(len(df)) - 1.0 / len(df)
    vals, vecs = np.linalg.eigh(H @ W @ H)
    return vecs[:, np.abs(vals) > 1e-8]


def msr(x, y, V, method='spearman'):
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = (lambda u: stats.spearmanr(u, y)[0]) if method == 'spearman' else (lambda u: np.corrcoef(u, y)[0, 1])
    z = (x - x.mean()) / x.std()
    coef = V.T @ z
    sur = (rng.choice([-1.0, 1.0], size=(N_SURR, coef.size)) * coef) @ V.T
    null = np.array([f(s) for s in sur])
    obs = f(x)
    return obs, (1 + (np.abs(null) >= abs(obs)).sum()) / (N_SURR + 1)


def partial(df, x, y, ctrl):
    R = df[[x, y] + ctrl].rank()
    X = np.column_stack([np.ones(len(R)), R[ctrl].values])
    res = lambda s: s - X @ np.linalg.lstsq(X, s, rcond=None)[0]
    return res(R[x].values), res(R[y].values)


def detrend(s):
    t = s.index.values.astype(float)
    return s - np.polyval(np.polyfit(t, s.values, 1), t)


a = pd.read_csv(E / 'analysis_cells_erc.csv')
S = {('Terra', b): seasons(te, b) for b in ('Day', 'Night')}
S.update({('Aqua', b): seasons(aq, b) for b in ('Day', 'Night')})
log(f'Aqua files {len(list(RAW.glob("MYD11A2.A*.hdf")))}; composites {aq.date.nunique()}; seasons {YEARS[0]}-{YEARS[1]}; minimum valid seasons per cell {MIN_SEASONS}')
for b in ('Day', 'Night'):
    log(f'{b}: mean valid composites per cell-season, Terra {S[("Terra", b)][1].reindex(columns=a.cell_id).mean().mean():.1f}, '
        f'Aqua {S[("Aqua", b)][1].reindex(columns=a.cell_id).mean().mean():.1f}')

# trends by sensor and final season
rows = []
SL = {}
for (sensor, band), (s, _) in S.items():
    for end in (2019, 2021, 2023):
        ms = int(round(MIN_SEASONS * (end - 2002) / 21))
        sub = s.loc[:end]
        t = trends(sub, a.cell_id, ms)
        if end == 2023:
            SL[(sensor, band)] = t
        dm = sub[t.index].mean(axis=1)
        ds, dp = hr(dm.values)
        rows.append(dict(sensor=sensor, band=band.lower(), end=end, cells=len(t), median_slope=round(t.slope.median(), 4),
                         pct_positive=round(100 * (t.slope > 0).mean(), 1), fdr_cells=int(multipletests(t.p, method='fdr_bh')[0].sum()),
                         domain_slope=round(ds, 4), domain_p=round(dp, 4)))
tr = pd.DataFrame(rows).sort_values(['band', 'sensor', 'end'])
log('\nTRENDS, COMMON PERIOD STARTING 2003\n' + tr.to_string(index=False))
tr.to_csv(TAB / 'aqua_trend_comparison.csv', index=False)

# agreement between sensors
log('\nAGREEMENT BETWEEN SENSORS')
ag = []
for b in ('Day', 'Night'):
    ta, aa = S[('Terra', b)][0], S[('Aqua', b)][0]
    ids = SL[('Terra', b)].index.intersection(SL[('Aqua', b)].index)
    rs = stats.spearmanr(SL[('Terra', b)].loc[ids, 'slope'], SL[('Aqua', b)].loc[ids, 'slope'])[0]
    ra = stats.spearmanr(ta.loc[2019:2023, ids].mean(), aa.loc[2019:2023, ids].mean())[0]
    dt, da = ta[ids].mean(axis=1), aa[ids].mean(axis=1)
    diff = dt - da
    h = hr(diff.values)
    pre, post = diff.loc[:2019], diff.loc[2021:]
    wt = stats.mannwhitneyu(post, pre, alternative='two-sided')
    ag.append(dict(band=b.lower(), cells=len(ids), rho_cell_slopes=round(rs, 3), rho_recent_lst=round(ra, 3),
                   r_domain=round(np.corrcoef(dt, da)[0, 1], 3), r_domain_detrended=round(np.corrcoef(detrend(dt), detrend(da))[0, 1], 3),
                   terra_minus_aqua_mean=round(diff.mean(), 3), diff_slope=round(h[0], 4), diff_slope_p=round(h[1], 4),
                   diff_2003_2019=round(pre.mean(), 3), diff_2021_2023=round(post.mean(), 3), shift=round(post.mean() - pre.mean(), 3),
                   shift_p=round(wt.pvalue, 4)))
agd = pd.DataFrame(ag)
log(agd.T.to_string(header=False))
agd.to_csv(TAB / 'aqua_sensor_agreement.csv', index=False)

# warming and built-up growth, Aqua daytime and night-time
log('\nWARMING AND BUILT-UP GROWTH')
grow = []
for sensor in ('Terra', 'Aqua'):
    for band in ('Day', 'Night'):
        j = a.merge(SL[(sensor, band)].rename_axis('cell_id').reset_index(), on='cell_id').reset_index(drop=True)
        V = build_null(j)
        for lab, sub in (('all cells', j), ('Tarkwa-Prestea excluded', j[~j.mining_zone].reset_index(drop=True))):
            Vs = V if len(sub) == len(j) else build_null(sub)
            for name, col in (('relative growth', 'n_BUGR'), ('absolute change 2000-2020', 'abs_change_2000_2020_pp'), ('UGPI', 'ugpi_urban_only')):
                rho, p = msr(sub.slope, sub[col], Vs)
                e1, e2 = partial(sub, 'slope', col, CTRL)
                rp, pp = msr(e1, e2, Vs, 'pearson')
                grow.append(dict(sensor=sensor, band=band.lower(), cells=lab, n=len(sub), measure=name, rho=round(rho, 3), p_msr=round(p, 4),
                                 partial_climate=round(rp, 3), p_partial=round(pp, 4)))
gd = pd.DataFrame(grow)
allp = np.r_[gd.p_msr.values, gd.p_partial.values]
q = multipletests(allp, method='fdr_bh')[1]
gd['q_msr'], gd['q_partial'] = np.round(q[:len(gd)], 4), np.round(q[len(gd):], 4)
log(f'Benjamini-Hochberg control over all {len(allp)} tests; q < 0.05: {int((q < .05).sum())}')
log(gd.to_string(index=False))
gd.to_csv(TAB / 'aqua_warming_growth.csv', index=False)

# urban contrast at equal distance from the sea, 2019-2023
log('\nURBAN CONTRAST (top-quintile UGPI minus other cells), 2019-2023')
top = a.ugpi_urban_only >= a.ugpi_urban_only.quantile(.8)
bands = pd.cut(a.dist_sea_km, [-1, 10, 20, 30, 40, 50], labels=['0-10', '10-20', '20-30', '30-40', '40-50'])
ct = {}
for (sensor, band), (s, _) in S.items():
    v = a.cell_id.map(s.loc[2019:2023].mean())
    v = v - v.mean()
    g = v.groupby([bands, top], observed=True).mean().unstack()
    ct[f'{sensor} {band.lower()}'] = g[True] - g[False]
ct = pd.DataFrame(ct).round(2)
log(ct.to_string())
ct.to_csv(TAB / 'aqua_urban_contrast.csv')

dom = pd.DataFrame({f'{sen.lower()}_{b.lower()}': s[a.cell_id].mean(axis=1) for (sen, b), (s, _) in S.items()})
dom.round(3).to_csv(TAB / 'aqua_domain_series.csv')
(TAB / 'aqua_build_log.txt').write_text('\n'.join(lines) + '\n')
print('done')
