"""Paper 2 (v2): dry-season LST trends and the Urban Heat-Groundwater Pressure Index (UHGPI).

Replaces build_compound_heat_gw_index.py / finalize_paper2_after_lst.py (v1 CUHGSI). Changes:
- Domain: cells whose centroid lies within 50 km of the sea (Natural Earth 10 m coastline). The
  shared 453-cell grid was buffered from the whole national boundary, so 151 cells lie > 50 km
  from the sea along the Ivorian and Togolese borders; the full grid is kept as a sensitivity case.
- Seasons: dry season = Dec-Feb, labelled by the Jan-Feb year, 2001-2023 only (the 2000 season
  has Feb only; 2024 lies outside the GRACE window). A season value is the mean of the cell's
  8-day composites and requires >= 3 valid composites. Trends need >= 18 valid seasons.
- Trend test: Hamed-Rao modified Mann-Kendall with Sen's slope; Benjamini-Hochberg FDR across cells.
- Components (Paper 1 v2 layers, no imputation): LST trend, recent LST anomaly (2019-2023),
  UGPI (3 indicators: built-up fraction, population density, built-up growth), climate stress
  (CHIRPS below-normal frequency). Groundwater storage (GRACE) has no spatial information at 10 km
  and is used only as temporal context, not as an index component.
- Cells without a valid LST trend are excluded, not filled.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats
from shapely.geometry import box
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / 'data/processed/risk_index/cgsri_grid10km_v2.gpkg'
LST8 = ROOT / 'data/processed/lst/lst_8day_by_cell.csv'
GWSA = ROOT / 'data/processed/groundwater_anomalies/gwsa_by_region_monthly.csv'
LAKE = ROOT / 'outputs/tables/revision_checks/lake_volta_monthly.csv'
BOUNDARY = ROOT / 'data/raw/boundaries/ghana_boundary.gpkg'
NE_COAST = Path.home() / '.local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_coastline.shp'
OUT = ROOT / 'data/processed/lst/v2'
TAB = ROOT / 'outputs/tables/paper2_v2'
CLASSES = ['Very low', 'Low', 'Moderate', 'High', 'Very high']
COMP = ['n_LST_trend', 'n_LST_anom', 'n_UGPI', 'n_CS']
YEARS = (2001, 2023)
ANOM_YEARS = (2019, 2023)
MIN_COMPOSITES, MIN_SEASONS, COAST_KM = 3, 18, 50.0
N_MC, N_SURR = 10000, 9999
rng = np.random.default_rng(20261118)
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
log_lines = []


def log(s=''):
    print(s)
    log_lines.append(str(s))


def mm(s):
    return (s - s.min()) / (s.max() - s.min())


def quintiles(s):
    return pd.Series(pd.qcut(s.rank(method='first'), 5, labels=CLASSES).astype(str), index=s.index)


# ── grid, distance to sea, land fraction ────────────────────────────────────
g = gpd.read_file(V2)
coast = gpd.read_file(NE_COAST).clip(box(-3.4, 4.4, 1.4, 6.4)).to_crs(g.crs).union_all()
g['dist_sea_km'] = g.geometry.centroid.distance(coast) / 1000
land = gpd.read_file(BOUNDARY).to_crs(g.crs).union_all()
g['land_frac'] = g.geometry.intersection(land).area / g.geometry.area
g['coastal'] = g['dist_sea_km'] <= COAST_KM
log(f'grid cells {len(g)}; centroid <= {COAST_KM:.0f} km from sea: {g.coastal.sum()}; '
    f'> {COAST_KM:.0f} km: {(~g.coastal).sum()} (max {g.dist_sea_km.max():.0f} km, '
    f'{g.loc[~g.coastal, "population_2020"].sum() / 1e6:.2f} M people)')

# ── dry-season LST per cell ─────────────────────────────────────────────────
d8 = pd.read_csv(LST8, parse_dates=['date'])
cells = [c for c in d8.columns if c != 'date']
d8 = d8[d8.date.dt.month.isin([12, 1, 2])].copy()
d8['ds'] = np.where(d8.date.dt.month == 12, d8.date.dt.year + 1, d8.date.dt.year)
d8 = d8[d8.ds.between(*YEARS)]
nvalid = d8.groupby('ds')[cells].count()
season = d8.groupby('ds')[cells].mean().where(nvalid >= MIN_COMPOSITES)
season.to_csv(OUT / 'lst_dry_season_2001_2023.csv')
nseas = season.notna().sum()
log(f'dry seasons {YEARS[0]}-{YEARS[1]}: {len(season)}; composites per season '
    f'{int(d8.groupby("ds").date.nunique().min())}-{int(d8.groupby("ds").date.nunique().max())}; '
    f'cell-seasons dropped for < {MIN_COMPOSITES} composites: {int(((nvalid < MIN_COMPOSITES) & (nvalid > 0)).sum().sum())}')

rows = []
for c in cells:
    ts = season[c].dropna()
    r = dict(cell_id=c, n_seasons=len(ts), mean_valid_composites=float(nvalid[c].mean()))
    if len(ts) >= MIN_SEASONS:
        hr = mk.hamed_rao_modification_test(ts.values)
        r.update(sen_slope=hr.slope, hr_z=hr.z, hr_p=hr.p)
    rows.append(r)
tr = pd.DataFrame(rows)
g = g.merge(tr, on='cell_id', how='left')
anom_mean = season.loc[ANOM_YEARS[0]:ANOM_YEARS[1]].mean()
g['lst_recent'] = g.cell_id.map(anom_mean)

# ── analysis set ────────────────────────────────────────────────────────────
def analysis_set(mask, label):
    a = g[mask & g.sen_slope.notna() & g.lst_recent.notna()].copy()
    a['lst_anom'] = a.lst_recent - a.lst_recent.mean()
    a['fdr_sig'] = multipletests(a.hr_p, method='fdr_bh')[0]
    a['n_LST_trend'] = mm(a.sen_slope)
    a['n_LST_anom'] = mm(a.lst_anom)
    a['n_UGPI'] = mm(a.ugpi_urban_only)
    a['n_CS'] = mm(a.chirps_below_normal_pct)
    a['uhgpi'] = a[COMP].mean(axis=1)
    a['uhgpi_class'] = quintiles(a.uhgpi)
    a['domain'] = label
    return a


a = analysis_set(g.coastal, 'coastal50')
full = analysis_set(pd.Series(True, index=g.index), 'fullgrid')
log(f'coastal cells with valid trend and anomaly: {len(a)} of {g.coastal.sum()}; '
    f'excluded (< {MIN_SEASONS} seasons): {g.coastal.sum() - len(a)}')

# ── LST trend results ───────────────────────────────────────────────────────
s = a.sen_slope
p10, p90 = np.percentile(s, [10, 90])
log('\nLST TRENDS (coastal set)')
log(f'median Sen slope {s.median():+.4f} C/yr (IQR {s.quantile(.25):+.4f} to {s.quantile(.75):+.4f}); '
    f'p10 {p10:+.4f}, p90 {p90:+.4f}; positive {int((s > 0).sum())}/{len(s)} ({100 * (s > 0).mean():.0f}%)')
log(f'Hamed-Rao p < 0.05: {int((a.hr_p < .05).sum())} ({100 * (a.hr_p < .05).mean():.0f}%), of which positive '
    f'{int(((a.hr_p < .05) & (s > 0)).sum())}; BH-FDR q < 0.05: {int(a.fdr_sig.sum())} '
    f'({int((a.fdr_sig & (s > 0)).sum())} positive)')
fs = full.sen_slope
log(f'full grid: n {len(full)}, median {fs.median():+.4f}, positive {100 * (fs > 0).mean():.0f}%, '
    f'p<0.05 {int((full.hr_p < .05).sum())}, FDR {int(full.fdr_sig.sum())}')

# domain-mean dry-season series and its trend
dom = season[a.cell_id].mean(axis=1)
hr = mk.hamed_rao_modification_test(dom.values)
log(f'domain-mean dry-season LST {dom.mean():.2f} C; Sen {hr.slope:+.4f} C/yr, Hamed-Rao p {hr.p:.3g}')
log(f'cell recent ({ANOM_YEARS[0]}-{ANOM_YEARS[1]}) LST: mean {a.lst_recent.mean():.1f}, '
    f'range {a.lst_recent.min():.1f}-{a.lst_recent.max():.1f} C')

# ── spatial null (Moran spectral randomisation) on the analysis set ────────
def msr_basis(df):
    c, r = df.col_id.values, df.row_id.values
    W = ((np.abs(c[:, None] - c[None, :]) <= 1) & (np.abs(r[:, None] - r[None, :]) <= 1)).astype(float)
    np.fill_diagonal(W, 0)
    n = len(df)
    H = np.eye(n) - 1.0 / n
    vals, vecs = np.linalg.eigh(H @ W @ H)
    return W, vecs[:, np.abs(vals) > 1e-8]


W, V = msr_basis(a)


def morans_i(x):
    z = x - x.mean()
    return len(z) / W.sum() * (z @ W @ z) / (z @ z)


def surrogates(x, n=N_SURR):
    z = (x - x.mean()) / x.std()
    coef = V.T @ z
    return (rng.choice([-1.0, 1.0], size=(n, coef.size)) * coef) @ V.T


def msr_corr(x, y, method='spearman'):
    f = (lambda u, v: stats.spearmanr(u, v)[0]) if method == 'spearman' else (lambda u, v: np.corrcoef(u, v)[0, 1])
    obs = f(x, y)
    null = np.array([f(sx, y) for sx in surrogates(np.asarray(x, float))])
    return obs, (1 + (np.abs(null) >= abs(obs)).sum()) / (len(null) + 1)


# ── associations: LST vs urban metrics, coastal-cool-island controls ───────
log('\nASSOCIATIONS (Spearman rho; p from naive test and from Moran spectral randomisation)')
assoc = []
pairs = [('sen_slope', 'n_BUGR', 'LST trend vs built-up growth'),
         ('sen_slope', 'ugpi_urban_only', 'LST trend vs UGPI'),
         ('sen_slope', 'dist_sea_km', 'LST trend vs distance to sea'),
         ('lst_anom', 'ugpi_urban_only', 'LST anomaly vs UGPI'),
         ('lst_anom', 'dist_sea_km', 'LST anomaly vs distance to sea'),
         ('lst_anom', 'land_frac', 'LST anomaly vs land fraction'),
         ('lst_anom', 'mean_valid_composites', 'LST anomaly vs valid composites per season'),
         ('lst_anom', 'chirps_mean_rain_mm', 'LST anomaly vs mean annual rainfall'),
         ('lst_anom', 'elev_m', 'LST anomaly vs elevation'),
         ('ugpi_urban_only', 'dist_sea_km', 'UGPI vs distance to sea')]
for x, y, lab in pairs:
    rho, p_naive = stats.spearmanr(a[x], a[y])
    _, p_msr = msr_corr(a[x].values, a[y].values)
    assoc.append(dict(pair=lab, rho=round(rho, 3), p_naive=p_naive, p_msr=p_msr))
    log(f'{lab:45s} rho {rho:+.3f}  p {p_naive:.3g}  p_MSR {p_msr:.3g}')
# partial: LST anomaly vs UGPI controlling for distance to sea, rainfall and elevation (rank-based)
R = a[['lst_anom', 'ugpi_urban_only', 'dist_sea_km', 'chirps_mean_rain_mm', 'elev_m']].rank()
X = np.column_stack([np.ones(len(R)), R[['dist_sea_km', 'chirps_mean_rain_mm', 'elev_m']].values])
res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
e1, e2 = res(R.lst_anom.values), res(R.ugpi_urban_only.values)
pr = np.corrcoef(e1, e2)[0, 1]
_, p_pr = msr_corr(e1, e2, method='pearson')
assoc.append(dict(pair='LST anomaly vs UGPI | distance, rainfall, elevation (partial)', rho=round(pr, 3),
                  p_naive=np.nan, p_msr=p_pr))
log(f'partial rho (LST anomaly, UGPI | distance to sea, rainfall, elevation) {pr:+.3f}  p_MSR {p_pr:.3g}')
# coastal band means
bands = pd.cut(a.dist_sea_km, [-1, 10, 20, 30, 40, 50])
bt = a.groupby(bands, observed=True).agg(n=('cell_id', 'size'), lst_anom=('lst_anom', 'mean'),
                                         sen_slope=('sen_slope', 'median'), ugpi=('ugpi_urban_only', 'mean'))
log(bt.round(3).to_string())
# within-band urban contrast: high-UGPI (top quintile) minus others, by distance band
a['ugpi_top'] = a.ugpi_urban_only >= a.ugpi_urban_only.quantile(0.8)
wb = a.groupby([bands, 'ugpi_top'], observed=True).lst_anom.mean().unstack()
wb['diff_top_minus_rest'] = wb[True] - wb[False]
wb = wb.join(a.groupby([bands, 'ugpi_top'], observed=True).size().unstack().add_prefix('n_'))
log('LST anomaly, top-quintile UGPI vs rest, within distance bands:\n' + wb.round(2).to_string())
pd.DataFrame(assoc).to_csv(TAB / 'associations.csv', index=False)
bt.to_csv(TAB / 'distance_bands.csv')
wb.to_csv(TAB / 'ugpi_contrast_within_bands.csv')
mor = {k: round(morans_i(a[k].values), 3) for k in ['sen_slope', 'lst_anom', 'ugpi_urban_only', 'chirps_below_normal_pct', 'uhgpi']}
log(f"Moran's I: {mor}")

# ── index sensitivity ───────────────────────────────────────────────────────
log('\nUHGPI SENSITIVITY')
base = a.uhgpi
top = base.rank(method='first') > len(a) * 0.8
sens = []


def compare(label, alt):
    rho = stats.spearmanr(base, alt)[0]
    alt_cls = quintiles(alt)
    stab = (alt_cls == a.uhgpi_class).mean()
    alt_top = alt.rank(method='first') > len(a) * 0.8
    sens.append(dict(scheme=label, spearman=round(rho, 3), tier_agreement=round(stab, 3),
                     top_quintile_retained=int((top & alt_top).sum()), top_quintile_n=int(top.sum())))


W_SCHEMES = {'heat-weighted (0.35,0.35,0.15,0.15)': [.35, .35, .15, .15],
             'pressure-weighted (0.15,0.15,0.35,0.35)': [.15, .15, .35, .35]}
for lab, w in W_SCHEMES.items():
    compare(lab, a[COMP] @ np.array(w))
for c in COMP:
    compare(f'drop {c}', a[[k for k in COMP if k != c]].mean(axis=1))
compare('rank (percentile) normalisation', a[['sen_slope', 'lst_anom', 'ugpi_urban_only', 'chirps_below_normal_pct']].rank(pct=True).mean(axis=1))
compare('UGPI log1p-scaled', a[['n_LST_trend', 'n_LST_anom', 'n_CS']].assign(u=mm(np.log1p(a.ugpi_urban_only * 100))).mean(axis=1))
# residual LST anomaly: remove the climatic gradient (rainfall, elevation, distance to sea) by OLS
Xr = np.column_stack([np.ones(len(a)), a[['chirps_mean_rain_mm', 'elev_m', 'dist_sea_km']].values])
a['lst_anom_resid'] = a.lst_anom.values - Xr @ np.linalg.lstsq(Xr, a.lst_anom.values, rcond=None)[0]
r2 = 1 - a.lst_anom_resid.var() / a.lst_anom.var()
log(f'LST anomaly explained by rainfall, elevation and distance to sea: R2 = {r2:.2f}')
compare('residual LST anomaly (climate gradient removed)', a[['n_LST_trend', 'n_UGPI', 'n_CS']].assign(r=mm(a.lst_anom_resid)).mean(axis=1))
wmc = rng.dirichlet(np.ones(4), N_MC)
mc = a[COMP].values @ wmc.T
mc_rho = np.array([stats.spearmanr(base, mc[:, k])[0] for k in range(0, N_MC, 10)])
mc_top = (pd.DataFrame(mc).rank(method='first') > len(a) * 0.8).mean(axis=1).values
a['p_top_quintile_mc'] = mc_top
sens.append(dict(scheme=f'Monte Carlo Dirichlet(1,1,1,1), {N_MC} draws', spearman=round(float(np.median(mc_rho)), 3),
                 tier_agreement=np.nan, top_quintile_retained=int((top & (mc_top >= 0.5)).sum()),
                 top_quintile_n=int(top.sum())))
sens_df = pd.DataFrame(sens)
log(sens_df.to_string(index=False))
log(f'MC Spearman 5th percentile {np.percentile(mc_rho, 5):.3f}; cells in top quintile in >= 90% of draws: '
    f'{int((mc_top >= 0.9).sum())}; base top-quintile cells with P >= 0.5: {int((top & (mc_top >= 0.5)).sum())}/{int(top.sum())}')
imp = pd.DataFrame([dict(component=c, spearman_with_index=round(stats.spearmanr(a[c], base)[0], 3),
                         sd=round(a[c].std(), 3), mean=round(a[c].mean(), 3)) for c in COMP])
log(imp.to_string(index=False))
sens_df.to_csv(TAB / 'uhgpi_sensitivity.csv', index=False)
imp.to_csv(TAB / 'uhgpi_component_importance.csv', index=False)
cc = a[COMP].corr(method='spearman').round(3)
cc.to_csv(TAB / 'component_correlations.csv')
log('component Spearman correlations:\n' + cc.to_string())

# full-grid comparison on shared cells
sh = a[['cell_id', 'uhgpi']].merge(full[['cell_id', 'uhgpi', 'uhgpi_class']], on='cell_id', suffixes=('', '_full'))
log(f'full-grid index vs coastal index on {len(sh)} shared cells: Spearman {stats.spearmanr(sh.uhgpi, sh.uhgpi_full)[0]:.3f}')
log(f'full grid: top-quintile cells > 50 km from sea: '
    f'{int(((full.uhgpi_class == "Very high") & (full.dist_sea_km > COAST_KM)).sum())} of {int((full.uhgpi_class == "Very high").sum())}')

# ── bivariate co-location: LST trend x UGPI (tertiles) ─────────────────────
log('\nCO-LOCATION OF WARMING TREND AND URBAN PRESSURE')
t1 = pd.qcut(a.sen_slope.rank(method='first'), 3, labels=[1, 2, 3]).astype(int)
t2 = pd.qcut(a.ugpi_urban_only.rank(method='first'), 3, labels=[1, 2, 3]).astype(int)
a['bivar_trend'], a['bivar_ugpi'] = t1, t2
ct = pd.crosstab(t1, t2)
log('rows: LST-trend tertile; cols: UGPI tertile\n' + ct.to_string())
coloc = []
for q in (2 / 3, 0.8, 0.9):
    hx = a.sen_slope >= a.sen_slope.quantile(q)
    hy = a.ugpi_urban_only >= a.ugpi_urban_only.quantile(q)
    obs = int((hx & hy).sum())
    exp = hx.sum() * hy.sum() / len(a)
    orr, pf = stats.fisher_exact(pd.crosstab(hx, hy).values, alternative='two-sided')
    surr = surrogates(a.sen_slope.values)
    cnt = np.array([int(((sx >= np.quantile(sx, q)) & hy.values).sum()) for sx in surr])
    p_lo = (1 + (cnt <= obs).sum()) / (N_SURR + 1)
    p_hi = (1 + (cnt >= obs).sum()) / (N_SURR + 1)
    coloc.append(dict(threshold=f'top {round((1 - q) * 100)}%', observed=obs, expected=round(exp, 1),
                      odds_ratio=round(orr, 2), fisher_p=round(pf, 4), msr_p_fewer=round(p_lo, 4),
                      msr_p_more=round(p_hi, 4), null_mean=round(cnt.mean(), 1)))
co = pd.DataFrame(coloc)
log(co.to_string(index=False))
co.to_csv(TAB / 'colocation_trend_ugpi.csv', index=False)

# ── exposure ────────────────────────────────────────────────────────────────
log('\nEXPOSURE (WorldPop 2020)')
popt = a.groupby('uhgpi_class').population_2020.sum().reindex(CLASSES) / 1e6
log(f'coastal analysis set population {a.population_2020.sum() / 1e6:.2f} M; by class (M): {popt.round(2).to_dict()}')
hh = (a.bivar_trend == 3) & (a.bivar_ugpi == 3)
log(f'top-tertile warming x top-tertile UGPI: {int(hh.sum())} cells, {a.loc[hh, "population_2020"].sum() / 1e6:.2f} M people')
log(f'top-quintile UHGPI cells: mean distance to sea {a.loc[a.uhgpi_class == "Very high", "dist_sea_km"].mean():.0f} km; '
    f'lon range {a.loc[a.uhgpi_class == "Very high", "centroid_lon"].min():.2f} to {a.loc[a.uhgpi_class == "Very high", "centroid_lon"].max():.2f}')

dist = gpd.read_file(ROOT / 'data/raw/boundaries/gadm41_GHA.gpkg', layer='ADM_ADM_2').to_crs(g.crs)[['NAME_1', 'NAME_2', 'geometry']]
cent = gpd.GeoDataFrame(a[['cell_id']], geometry=g.set_index('cell_id').loc[a.cell_id].geometry.centroid.values, crs=g.crs)
nm = gpd.sjoin_nearest(cent, dist, how='left').drop_duplicates('cell_id').set_index('cell_id')
a['region'] = a.cell_id.map(nm.NAME_1)
a['district'] = a.cell_id.map(nm.NAME_2)
vh = a[a.uhgpi_class == 'Very high']
log('top-quintile UHGPI cells by region: ' + vh.region.value_counts().to_string().replace('\n', '; '))
log('top-quintile UHGPI cells by district: ' + vh.district.value_counts().head(12).to_string().replace('\n', '; '))
sig = a[a.fdr_sig]
log('FDR-significant warming cells by region: ' + sig.region.value_counts().to_string().replace('\n', '; '))
log(f'FDR-significant warming cells: median slope {sig.sen_slope.median():+.4f}; mean distance to sea {sig.dist_sea_km.mean():.0f} km')
hhc = a[(a.bivar_trend == 3) & (a.bivar_ugpi == 3)]
log('top-tertile warming x UGPI cells by district: ' + hhc.district.value_counts().head(12).to_string().replace('\n', '; '))

# ── temporal context: GWSA vs dry-season LST ───────────────────────────────
log('\nTEMPORAL CONTEXT')
gw = pd.read_csv(GWSA, parse_dates=['time']).set_index('time').loc['2002-04':'2023-12'].dropna(subset=['gwsa_mm'])
t = (gw.index - gw.index[0]).days.values / 365.2425
v = gw.gwsa_mm.values
slope = np.median([(v[j] - v[i]) / (t[j] - t[i]) for i in range(len(v)) for j in range(i + 1, len(v))])
log(f'GWSA: {len(gw)} months; time-based Sen slope {slope:+.2f} mm/yr')
gw['ds'] = np.where(gw.index.month == 12, gw.index.year + 1, gw.index.year)
det = v - np.polyval(np.polyfit(t, v, 1), t)
gw['sgwa_dt'] = (det - det.mean()) / det.std()
lst_anom_series = dom - dom.mean()
temporal = []
for lab, ser in [('calendar-year SGWA', gw.groupby(gw.index.year).sgwa.mean()),
                 ('calendar-year detrended SGWA', gw.groupby(gw.index.year).sgwa_dt.mean()),
                 ('DJF SGWA', gw[gw.index.month.isin([12, 1, 2])].groupby('ds').sgwa.mean()),
                 ('DJF detrended SGWA', gw[gw.index.month.isin([12, 1, 2])].groupby('ds').sgwa_dt.mean())]:
    j = pd.concat([ser.rename('gw'), lst_anom_series.rename('lst')], axis=1).dropna()
    r, p = stats.pearsonr(j.gw, j.lst)
    rho, ps = stats.spearmanr(j.gw, j.lst)
    temporal.append(dict(series=lab, n=len(j), pearson_r=round(r, 3), p=round(p, 3), spearman=round(rho, 3), p_s=round(ps, 3)))
lake = pd.read_csv(LAKE, parse_dates=['time']).set_index('time')
lake['corr'] = lake.gwsa_mm - lake.lake_ewh_mm_fullsupply
lc = lake['corr']
lake['z'] = (lc - lc.mean()) / lc.std()
ser = lake.groupby(lake.index.year).z.mean()[lake.groupby(lake.index.year).z.count() >= 6]
j = pd.concat([ser.rename('gw'), lst_anom_series.rename('lst')], axis=1).dropna()
r, p = stats.pearsonr(j.gw, j.lst)
rho, ps = stats.spearmanr(j.gw, j.lst)
temporal.append(dict(series='calendar-year SGWA, Lake Volta removed (2002-2018)', n=len(j), pearson_r=round(r, 3),
                     p=round(p, 3), spearman=round(rho, 3), p_s=round(ps, 3)))
tdf = pd.DataFrame(temporal)
log(tdf.to_string(index=False))
ann = gw.groupby(gw.index.year).sgwa.mean()
dry_years = ann[ann < -0.5].index
la = lst_anom_series.reindex(range(2002, 2024))
in_d = la[la.index.isin(dry_years)]
out_d = la[~la.index.isin(dry_years)]
perm = np.array([rng.permutation(la.values)[:len(in_d)].mean() for _ in range(9999)])
p_perm = (1 + (perm >= in_d.mean()).sum()) / 10000
log(f'groundwater-deficit years (annual SGWA < -0.5): {list(dry_years)}; mean dry-season LST anomaly '
    f'{in_d.mean():+.2f} C vs {out_d.mean():+.2f} C in other years; one-sided permutation p (deficit years hotter) {p_perm:.3f}')
tdf.to_csv(TAB / 'temporal_gwsa_lst.csv', index=False)
pd.DataFrame({'lst_anom_domain': lst_anom_series, 'sgwa_calendar': ann}).to_csv(TAB / 'annual_series.csv')

# ── save ────────────────────────────────────────────────────────────────────
keep = ['cell_id', 'col_id', 'row_id', 'centroid_lon', 'centroid_lat', 'dist_sea_km', 'land_frac', 'n_seasons',
        'mean_valid_composites', 'sen_slope', 'hr_z', 'hr_p', 'fdr_sig', 'lst_recent', 'lst_anom', 'n_BUGR',
        'ugpi_urban_only', 'chirps_below_normal_pct', 'chirps_mean_rain_mm', 'elev_m', 'population_2020'] + COMP + [
        'uhgpi', 'uhgpi_class', 'p_top_quintile_mc', 'bivar_trend', 'bivar_ugpi', 'lst_anom_resid', 'region', 'district']
out = g[['cell_id', 'geometry', 'coastal']].merge(a[keep], on='cell_id', how='left')
out.to_file(OUT / 'uhgpi_grid10km_v2.gpkg', driver='GPKG')
out.drop(columns='geometry').to_csv(OUT / 'uhgpi_grid10km_v2.csv', index=False)
full[['cell_id', 'dist_sea_km', 'sen_slope', 'hr_p', 'fdr_sig', 'lst_anom', 'uhgpi', 'uhgpi_class']].to_csv(
    OUT / 'uhgpi_fullgrid_sensitivity.csv', index=False)
season[a.cell_id].mean(axis=1).rename('domain_lst').to_csv(OUT / 'domain_dry_season_lst.csv')
(TAB / 'build_log.txt').write_text('\n'.join(log_lines) + '\n')
log(f'\nwrote {OUT} and {TAB}')
