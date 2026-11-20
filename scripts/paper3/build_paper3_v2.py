"""Paper 3 (v2): separating Lake Volta storage from GRACE-derived groundwater storage over southern Ghana.

Domain: the 123 JPL-land 0.5-degree cells in 4.0-11.5N, 3.5W-1.5E (the regional domain of Paper 1).
Period: April 2002 - December 2025. Anomalies are relative to the 2004-2009 calendar-month climatology.

Ensemble
  GRACE/GRACE-FO mascons: JPL RL06.3 (CRI, scaled), CSR RL06.3, GSFC RL06v2.0
  Land surface models:    GLDAS-2.1 Noah, CLSM and VIC (1 degree), soil moisture + canopy water
  -> 9 GWSA members, GWSA = TWSA - SMSA - CWSA
  Lake Volta storage:     G-REALM 10-day levels (target 000393), dS = A * dh, A = 6,000 or 8,502 km2
  -> 18 lake-corrected members.  Bui reservoir (impounded from June 2011) is a separate sensitivity case.
Anchor check: JPL x GLDAS Noah 0.25 deg for April 2002 - December 2023 must reproduce Paper 1 (12.04 mm/yr).

Every statistic quoted in the manuscript is written to outputs/tables/paper3_v2/build_log.txt.
"""
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
import xarray as xr
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw'
INT = ROOT / 'data/interim/paper3'
OUTD = ROOT / 'data/processed/paper3'
TAB = ROOT / 'outputs/tables/paper3_v2'
for d in (INT, OUTD, TAB):
    d.mkdir(parents=True, exist_ok=True)

LAT, LON = (4.0, 11.5), (-3.5, 1.5)
BASE = ('2004-01-01', '2009-12-31')
START, END = '2002-04-01', '2025-12-01'
A_LAKE = {'A6000': 6000.0, 'A8502': 8502.0}      # km2: reduced area and full-supply area (Lake Volta)
BUI_KM3 = {'dead_4.85': 4.85, 'gross_12.57': 12.57}  # Bui: gross 12.57 km3, active 7.72 km3
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.append(s)


def month_index(times):
    return pd.DatetimeIndex(times).to_period('M').to_timestamp()


def anomaly(s):
    b = s[BASE[0]:BASE[1]]
    clim = b.groupby(b.index.month).mean()
    return s - s.index.month.map(clim).values


# ------------------------------------------------------------------ domain mask (JPL land cells)
jpl = xr.open_dataset(RAW / 'grace/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04CRI.nc')
jpl = jpl.assign_coords(lon=((jpl.lon + 180) % 360 - 180)).sortby('lon')
box = dict(lat=slice(*LAT), lon=slice(*LON))
MASK = jpl.land_mask.sel(**box) == 1                       # 0.5 deg, 15 x 10
A_DOMAIN = float((MASK * (111.32 * 0.5) ** 2 * np.cos(np.deg2rad(MASK.lat))).sum())
log(f'Domain: {int(MASK.sum())} JPL land cells of 0.5 deg, area {A_DOMAIN:,.0f} km2')


def to_mask_grid(da):
    """Map any regular lat/lon grid onto the 0.5 deg mask cells (cell-mean of finer grids, parent of coarser)."""
    da = da.sel(lat=slice(LAT[0] - 1, LAT[1] + 1), lon=slice(LON[0] - 1, LON[1] + 1))
    res = float(abs(da.lat[1] - da.lat[0]))
    if res < 0.5 - 1e-6:
        da = da.coarsen(lat=int(round(0.5 / res)), lon=int(round(0.5 / res)), boundary='trim').mean()
    return da.reindex(lat=MASK.lat, lon=MASK.lon, method='nearest', tolerance=max(res, 0.5) / 2 + 1e-6)


def domain_mean(da):
    g = to_mask_grid(da).where(MASK)
    s = g.mean(['lat', 'lon'], skipna=True).to_series()
    s.index = month_index(s.index)
    return s[~s.index.duplicated(keep='first')]


# ------------------------------------------------------------------ GRACE products
def grace_jpl():
    da = (jpl.lwe_thickness * jpl.scale_factor).sel(**box) * 10.0
    s = da.where(MASK).mean(['lat', 'lon']).to_series()
    s.index = month_index(s.index)
    return s[~s.index.duplicated(keep='first')]


def grace_csr():
    ds = xr.open_dataset(RAW / 'grace_ensemble/csr/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc',
                         decode_times=False)
    t = pd.Timestamp('2002-01-01') + pd.to_timedelta(ds.time.values.astype(float), unit='D')
    ds = ds.assign_coords(time=t, lon=((ds.lon + 180) % 360 - 180)).sortby('lon')
    return domain_mean(ds.lwe_thickness * 10.0)


def grace_gsfc():
    f = next((RAW / 'grace_ensemble/gsfc').glob('GSFC*HALFDEGREE.nc'))
    ds = xr.open_dataset(f)
    var = next(v for v in ds.data_vars if re.search(r'lwe|cmwe|tws|mass', v, re.I) and ds[v].ndim == 3)
    ren = {c: n for c, n in [('latitude', 'lat'), ('longitude', 'lon')] if c in ds.dims or c in ds.coords}
    ds = ds.rename(ren)
    if float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360 - 180))
    ds = ds.sortby('lon').sortby('lat')
    units = str(ds[var].attrs.get('units', 'cm')).lower()
    scale = 10.0 if 'cm' in units else 1.0
    return domain_mean(ds[var] * scale)


# ------------------------------------------------------------------ GLDAS
LSM = {
    'Noah': ('gldas_1deg/noah', ['SoilMoi0_10cm_inst', 'SoilMoi10_40cm_inst', 'SoilMoi40_100cm_inst',
                                 'SoilMoi100_200cm_inst']),
    'CLSM': ('gldas_1deg/clsm', ['SoilMoist_P_inst']),
    'VIC': ('gldas_1deg/vic', ['SoilMoi0_30cm_inst', 'SoilMoi_depth2_inst', 'SoilMoi_depth3_inst']),
}


def gldas(name, sub, sm_vars, extra=()):
    cache = INT / f'gldas_{name.lower()}_domain.csv'
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    files = sorted((RAW / sub).glob('*.nc4'))
    rows = {}
    for f in files:
        ds = xr.open_dataset(f)
        sm = sum(ds[v] for v in sm_vars)
        r = {'sm': domain_mean(sm).iloc[0], 'cw': domain_mean(ds['CanopInt_inst']).iloc[0]}
        for v in extra:
            r[v] = domain_mean(ds[v]).iloc[0]
        rows[month_index(ds.time.values)[0]] = r
        ds.close()
    df = pd.DataFrame(rows).T.sort_index()
    df.to_csv(cache)
    return df


# ------------------------------------------------------------------ Lake Volta and Bui
def lake_volta_levels():
    rows = []
    for ln in open(RAW / 'lake_volta/gwm/lake000393.10d.2.txt'):
        p = ln.split()
        if (len(p) >= 16 and re.fullmatch(r'[A-Z][A-Z0-9/]+', p[0]) and re.fullmatch(r'\d{8}', p[2])
                and p[2] != '99999999' and float(p[5]) < 900):
            rows.append((pd.to_datetime(p[2]), p[0], float(p[5]), float(p[6])))
    d = pd.DataFrame(rows, columns=['date', 'mission', 'h', 'err']).set_index('date')
    d.to_csv(OUTD / 'lake_volta_grealm_10day.csv')
    m = d.h.resample('MS').mean()
    return d, m


def bui_storage(km3):
    """Linear filling from 8 June 2011 to the end of 2013, constant afterwards (scenario)."""
    idx = pd.date_range('2002-01-01', END, freq='MS')
    f = np.clip((idx - pd.Timestamp('2011-06-01')).days / (pd.Timestamp('2014-01-01') - pd.Timestamp('2011-06-01')).days, 0, 1)
    return pd.Series(km3 * 1e9 * f / (A_DOMAIN * 1e6) * 1000, index=idx)    # mm EWH


# ------------------------------------------------------------------ statistics
def dec_year(idx):
    return idx.year + (idx.dayofyear - 0.5) / 365.25


def trend(s):
    s = s.dropna()
    sl = stats.theilslopes(s.values, dec_year(s.index))[0]
    r = mk.hamed_rao_modification_test(s.values)
    return sl, r.p


def neff(x, y):
    r1 = pd.Series(x).autocorr(1)
    r2 = pd.Series(y).autocorr(1)
    return len(x) * (1 - r1 * r2) / (1 + r1 * r2)


def corr_eff(a, b):
    m = a.notna() & b.notna()
    x, y = a[m].values, b[m].values
    r = np.corrcoef(x, y)[0, 1]
    ne = max(neff(x, y), 3.0)
    t = r * np.sqrt((ne - 2) / (1 - r * r))
    return r, 2 * stats.t.sf(abs(t), ne - 2), ne, int(m.sum())


def detrend(s):
    s = s.dropna()
    x = dec_year(s.index)
    return s - np.polyval(np.polyfit(x, s.values, 1), x)


def zscore(s):
    return (s - s.mean()) / s.std()


def episodes(z, thr=-0.5, gap=2, min_len=3):
    """Runs of z < thr on the monthly calendar; runs separated by <= gap months are merged; >= min_len observed months."""
    full = z.reindex(pd.date_range(z.index.min(), z.index.max(), freq='MS'))
    dry = (full < thr).values
    runs, i = [], 0
    while i < len(dry):
        if dry[i]:
            j = i
            while j + 1 < len(dry) and dry[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    out = []
    for a, b in merged:
        seg = full.iloc[a:b + 1]
        if (seg < thr).sum() >= min_len:
            out.append(dict(start=seg.index[0].strftime('%Y-%m'), end=seg.index[-1].strftime('%Y-%m'),
                            months=b - a + 1, dry_obs=int((seg < thr).sum()), mean=round(seg.mean(), 2),
                            min=round(seg.min(), 2), min_month=seg.idxmin().strftime('%Y-%m'),
                            left_censored=a == 0))
    return out


def spi(p, k):
    """Standardised Precipitation Index on k-month sums (gamma fit per calendar month, 1981-2025)."""
    acc = p.rolling(k).sum()
    out = pd.Series(np.nan, index=p.index)
    for m in range(1, 13):
        x = acc[acc.index.month == m].dropna()
        q = (x == 0).mean()
        a, loc, sc = stats.gamma.fit(x[x > 0], floc=0)
        cdf = q + (1 - q) * stats.gamma.cdf(x, a, loc=loc, scale=sc)
        out.loc[x.index] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    return out


# ------------------------------------------------------------------ climate drivers (domain box means)
def chirps_domain():
    cache = INT / 'chirps_domain_monthly.csv'
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    ds = xr.open_dataset(RAW / 'chirps/chirps-v2.0.monthly.nc', chunks={'time': 60})
    v = next(k for k in ds.data_vars if 'precip' in k.lower())
    rn = {c: n for c, n in [('latitude', 'lat'), ('longitude', 'lon')] if c in ds.coords}
    da = ds.rename(rn)[v].sel(lat=slice(*LAT), lon=slice(*LON), time=slice('1981-01-01', END))
    s = domain_mean(da.compute())
    s.name = 'precip_mm'
    s.to_csv(cache)
    return s


def pdsi_domain():
    cache = INT / 'pdsi_domain_monthly.csv'
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    parts = []
    for f in sorted((RAW / 'terraclimate').glob('TerraClimate_PDSI_*.nc')):
        ds = xr.open_dataset(f)
        parts.append(domain_mean(ds['PDSI'].sortby('lat')))
    s = pd.concat(parts).sort_index()
    s.name = 'pdsi'
    s.to_csv(cache)
    return s


# ================================================================== build
log('=' * 80)
d10, lvl = lake_volta_levels()
log(f'G-REALM Volta: {len(d10)} valid 10-day heights {d10.index.min():%Y-%m-%d} to {d10.index.max():%Y-%m-%d};',
    'missions', d10.mission.value_counts().to_dict())
v2 = xr.open_dataset(RAW / 'lake_volta/Africa_Volta_WaterHeightV2_393.nc')
h2 = v2.surface_water_height.to_series()
h2 = h2[(v2.outlier_flag.to_series() != 1) & h2.notna() & (h2 < 900)].resample('MS').mean()
j = pd.concat([lvl, h2], axis=1, keys=['op', 'v2']).dropna()
log(f'Operational vs G-REALM V2 monthly levels: n={len(j)}, r={j.corr().iloc[0, 1]:.4f}, '
    f'mean diff {(j.op - j.v2).mean():+.3f} m, SD diff {(j.op - j.v2).std():.3f} m')
cal = pd.date_range('2002-01-01', END, freq='MS')
lvl_c = lvl.reindex(cal)
nmiss = int(lvl_c[START:END].isna().sum())
lvl_c = lvl_c.interpolate(limit=3, limit_area='inside')
log(f'Lake months missing in {START[:7]}..{END[:7]}: {nmiss}, filled by linear interpolation (<= 3 months)')
dh = anomaly(lvl_c)
log(f'Lake level anomaly range {dh.min():.2f} to {dh.max():+.2f} m; annual means:',
    dh.groupby(dh.index.year).mean().round(2).to_dict())
lake = {k: dh * a / A_DOMAIN * 1000 for k, a in A_LAKE.items()}      # mm EWH over the domain
log('1 m of lake level =', {k: round(a / A_DOMAIN * 1000, 1) for k, a in A_LAKE.items()}, 'mm EWH over the domain')

tws = {'JPL': grace_jpl(), 'CSR': grace_csr(), 'GSFC': grace_gsfc()}
for k, s in tws.items():
    s = s[START:END].dropna()
    log(f'TWSA {k}: {len(s)} months {s.index.min():%Y-%m}..{s.index.max():%Y-%m}')
months = sorted(set.intersection(*[set(s[START:END].dropna().index) for s in tws.values()]))
months = pd.DatetimeIndex(months)
log(f'Months with all three GRACE solutions: {len(months)}')
twsa = pd.DataFrame({k: anomaly(s.reindex(cal.union(s.index)))for k, s in tws.items()}).reindex(months)

lsm = {k: gldas(k, *v) for k, v in LSM.items()}
clsm_full = gldas('CLSM_full', 'gldas_1deg/clsm', ['SoilMoist_P_inst'], extra=('TWS_inst',))
smsa = pd.DataFrame({k: anomaly(v.sm.reindex(cal)) for k, v in lsm.items()}).reindex(months)
cwsa = pd.DataFrame({k: anomaly(v.cw.reindex(cal)) for k, v in lsm.items()}).reindex(months)
log('SMSA SD by model (mm):', smsa.std().round(1).to_dict(), ' r(Noah,CLSM,VIC):',
    smsa.corr().round(2).values[np.triu_indices(3, 1)].tolist())

gw = pd.DataFrame({f'{g}-{m}': twsa[g] - smsa[m] - cwsa[m] for g, m in product(tws, LSM)})
lk = pd.DataFrame(lake).reindex(months)
gwc = pd.DataFrame({f'{c}|{a}': gw[c] - lk[a] for c, a in product(gw.columns, A_LAKE)})
gw.to_csv(OUTD / 'gwsa_members_uncorrected.csv')
gwc.to_csv(OUTD / 'gwsa_members_lake_corrected.csv')
lk.to_csv(OUTD / 'lake_volta_storage_ewh.csv')
ens = pd.DataFrame({'gwsa_raw_mean': gw.mean(1), 'gwsa_raw_min': gw.min(1), 'gwsa_raw_max': gw.max(1),
                    'gwsa_corr_mean': gwc.mean(1), 'gwsa_corr_min': gwc.min(1), 'gwsa_corr_max': gwc.max(1),
                    'twsa_mean': twsa.mean(1), 'smsa_mean': smsa.mean(1), 'lake_A6000': lk.A6000,
                    'lake_A8502': lk.A8502})
ens.to_csv(OUTD / 'gwsa_ensemble_summary.csv')

# --- anchor: reproduce Paper 1 with JPL x Noah 0.25
n025 = sorted((RAW / 'gldas').glob('*.nc4'))
anc_cache = INT / 'gldas_noah025_domain.csv'
if anc_cache.exists():
    n25 = pd.read_csv(anc_cache, index_col=0, parse_dates=True)
else:
    rows = {}
    for f in n025:
        ds = xr.open_dataset(f)
        sm = sum(ds[v] for v in LSM['Noah'][1])
        rows[month_index(ds.time.values)[0]] = {'sm': domain_mean(sm).iloc[0], 'cw': domain_mean(ds.CanopInt_inst).iloc[0]}
    n25 = pd.DataFrame(rows).T.sort_index()
    n25.to_csv(anc_cache)
p1 = pd.read_csv(ROOT / 'data/interim/time_series/gwsa_monthly.csv', parse_dates=['time']).dropna(subset=['gwsa_mm'])
p1 = p1.set_index(month_index(p1.time)).gwsa_mm
jt = anomaly(tws['JPL'].reindex(cal.union(tws['JPL'].index)))
anc = (jt - anomaly(n25.sm.reindex(cal)) - anomaly(n25.cw.reindex(cal))).reindex(p1.index)
log(f'Anchor JPL x Noah0.25 vs Paper 1 series: max abs diff {np.nanmax(np.abs(anc - p1)):.2f} mm '
    f'(note: Paper 1 averaged GLDAS over its own valid cells), trend {trend(p1)[0]:.2f} (Paper 1 file) / {trend(anc)[0]:.2f} mm/yr')
n1 = anomaly(lsm['Noah'].sm.reindex(cal)).reindex(p1.index)
log(f'Noah 1 deg vs 0.25 deg SMSA (2002-2023): r={np.corrcoef(n1.dropna(), anomaly(n25.sm.reindex(cal)).reindex(n1.dropna().index))[0, 1]:.3f}')

# ------------------------------------------------------------------ 1. how much of GWSA is lake storage?
log('\n--- Lake contribution (per member; median [min-max] across the 9 GWSA members x 2 lake areas)')
rows = []
for c, a in product(gw.columns, A_LAKE):
    m = gw[c].notna() & lk[a].notna()
    x, l = gw[c][m], lk[a][m]
    r, p, ne, n = corr_eff(x, l)
    rows.append(dict(member=c, area=a, r_lake_gwsa=r, p_eff=p, neff=ne, var_explained=1 - (x - l).var() / x.var(),
                     sd_raw=x.std(), sd_corr=(x - l).std(), r_lake_twsa=np.corrcoef(twsa[c.split('-')[0]][m], l)[0, 1]))
lc = pd.DataFrame(rows)
lc.to_csv(TAB / 'lake_contribution_by_member.csv', index=False)
for col in ['r_lake_gwsa', 'p_eff', 'var_explained', 'sd_raw', 'sd_corr', 'r_lake_twsa']:
    log(f'{col}: median {lc[col].median():.3f} [{lc[col].min():.3f}-{lc[col].max():.3f}]')
log(f'Lake storage SD: A6000 {lk.A6000.std():.1f} mm, A8502 {lk.A8502.std():.1f} mm; range A8502 '
    f'{lk.A8502.min():.0f} to {lk.A8502.max():+.0f} mm')
# empirical recovery: regression slope of GWSA (and TWSA) on the A8502 lake term; 1 = full recovery of lake mass
rows = []
for c in gw.columns:
    m = gw[c].notna() & lk.A8502.notna()
    for nm, y in [('GWSA', gw[c][m]), ('GWSA_detr', detrend(gw[c][m])),
                  ('TWSA', twsa[c.split('-')[0]][m])]:
        x = lk.A8502[m] if nm != 'GWSA_detr' else detrend(lk.A8502[m])
        rows.append(dict(member=c, y=nm, beta=np.polyfit(x.values, y.values, 1)[0]))
bt = pd.DataFrame(rows)
bt.to_csv(TAB / 'lake_regression_slopes.csv', index=False)
for nm, g in bt.groupby('y', sort=False):
    log(f'Regression slope of {nm} on lake term (A8502): median {g.beta.median():.2f} [{g.beta.min():.2f}-{g.beta.max():.2f}]')

# ------------------------------------------------------------------ 2. trends
log('\n--- Trends (Sen slope on a time axis, mm/yr; Hamed-Rao p)')
periods = {'2002-04..2025-12': (START, END), '2002-04..2023-12': (START, '2023-12-01'),
           '2010-01..2025-12': ('2010-01-01', END)}
rows = []
for pname, (a, b) in periods.items():
    for c in gw.columns:
        sl, p = trend(gw[c][a:b])
        rows.append(dict(period=pname, member=c, lake='none', slope=sl, p=p))
        for ar in A_LAKE:
            sl, p = trend(gwc[f'{c}|{ar}'][a:b])
            rows.append(dict(period=pname, member=c, lake=ar, slope=sl, p=p))
        for bk, km3 in BUI_KM3.items():
            sl, p = trend((gwc[f'{c}|A8502'] - bui_storage(km3).reindex(months))[a:b])
            rows.append(dict(period=pname, member=c, lake=f'A8502+Bui_{bk}', slope=sl, p=p))
        sl, p = trend((gw[c] - 0.5 * lk.A8502)[a:b])
        rows.append(dict(period=pname, member=c, lake='A8502_half_recovery', slope=sl, p=p))
tr = pd.DataFrame(rows)
tr.to_csv(TAB / 'trends_by_member.csv', index=False)
for (pname, lake_), g in tr.groupby(['period', 'lake'], sort=False):
    log(f'{pname:17s} {lake_:22s} slope median {g.slope.median():6.2f} [{g.slope.min():6.2f}..{g.slope.max():6.2f}]'
        f'  p<0.05 in {int((g.p < 0.05).sum())}/{len(g)}  (p median {g.p.median():.3f})')
log(f'Bui storage at full filling: dead {BUI_KM3["dead_4.85"] * 1e9 / (A_DOMAIN * 1e6) * 1000:.1f} mm, '
    f'gross {BUI_KM3["gross_12.57"] * 1e9 / (A_DOMAIN * 1e6) * 1000:.1f} mm EWH')
for g in tws:
    for k, (a, b) in periods.items():
        log(f'TWSA {g} {k}: {trend(twsa[g][a:b])[0]:.2f} mm/yr; lake A8502 {trend(lk.A8502.reindex(months)[a:b])[0]:.2f} mm/yr')

# ------------------------------------------------------------------ 3. drought index and episodes
log('\n--- Standardised GWSA (GDSI) and episodes, ensemble-mean series')
series = {'raw': gw.mean(1), 'corrected': gwc.mean(1)}
zs = {}
for k, s in series.items():
    zs[f'{k}_z'] = zscore(s)
    zs[f'{k}_zdetr'] = zscore(detrend(s))
Z = pd.DataFrame(zs)
Z.to_csv(OUTD / 'gdsi_ensemble_mean.csv')
ep_rows = []
for k in Z:
    z = Z[k].dropna()
    log(f'{k}: months < -0.5 {int((z < -0.5).sum())}/{len(z)}, < -1.0 {int((z < -1).sum())}, < -1.5 {int((z < -1.5).sum())}, '
        f'< -2.0 {int((z < -2).sum())}; min {z.min():.2f} ({z.idxmin():%Y-%m})')
    for e in episodes(z):
        ep_rows.append(dict(series=k, **e))
        log('   episode', e)
pd.DataFrame(ep_rows).to_csv(TAB / 'episodes_ensemble_mean.csv', index=False)

log('\nMean GDSI in 2015-01..2016-12 by member (raw vs lake-corrected, not detrended / detrended)')
rows = []
for c in gw.columns:
    for ar in [None] + list(A_LAKE) + ['half']:
        s = gw[c] if ar is None else (gw[c] - 0.5 * lk.A8502 if ar == 'half' else gwc[f'{c}|{ar}'])
        rows.append(dict(member=c, lake=ar or 'none', z=zscore(s)['2015':'2016'].mean(),
                         zdetr=zscore(detrend(s))['2015':'2016'].mean(),
                         dry_months=int((zscore(s)['2015':'2016'] < -0.5).sum())))
e2 = pd.DataFrame(rows)
e2.to_csv(TAB / 'episode2015_16_by_member.csv', index=False)
for lake_, g in e2.groupby('lake', sort=False):
    log(f'  {lake_:6s} z median {g.z.median():+.2f} [{g.z.min():+.2f},{g.z.max():+.2f}]  detrended median '
        f'{g.zdetr.median():+.2f} [{g.zdetr.min():+.2f},{g.zdetr.max():+.2f}]  months<-0.5 median {g.dry_months.median():.0f}')

# robustness of candidate episodes across members and lake scenarios
WINDOWS = {'2002-04..2003-12': ('2002-04', '2003-12'), '2004-05..2005-09': ('2004-05', '2005-09'),
           '2006-05..2007-09': ('2006-05', '2007-09'), '2015-01..2016-12': ('2015-01', '2016-12'),
           '2020-06..2022-06': ('2020-06', '2022-06'), '2025-01..2025-12': ('2025-01', '2025-12')}
SCEN = {'none': lambda c: gw[c], 'A6000': lambda c: gwc[f'{c}|A6000'], 'A8502': lambda c: gwc[f'{c}|A8502'],
        'half': lambda c: gw[c] - 0.5 * lk.A8502}
rows = []
for wname, (a, b) in WINDOWS.items():
    for sc, fn in SCEN.items():
        for c in gw.columns:
            s = fn(c)
            rows.append(dict(window=wname, lake=sc, member=c, z=zscore(s)[a:b].mean(), zdetr=zscore(detrend(s))[a:b].mean()))
wr = pd.DataFrame(rows)
wr.to_csv(TAB / 'episode_windows_by_member.csv', index=False)
log('\nEpisode windows: mean z (not detrended | detrended), median [min, max] over 9 members; n members with mean < -0.5')
for (wname, sc), g in wr.groupby(['window', 'lake'], sort=False):
    log(f'  {wname} {sc:6s} z {g.z.median():+.2f} [{g.z.min():+.2f},{g.z.max():+.2f}] n<-0.5={int((g.z < -0.5).sum())} | '
        f'detr {g.zdetr.median():+.2f} [{g.zdetr.min():+.2f},{g.zdetr.max():+.2f}] n<-0.5={int((g.zdetr < -0.5).sum())}')

# member agreement: share of members in drought (z < -0.5) per month
agree = pd.DataFrame({'raw': (gw.apply(zscore) < -0.5).mean(1), 'corrected': (gwc.apply(zscore) < -0.5).mean(1),
                      'raw_detr': (gw.apply(lambda s: zscore(detrend(s))) < -0.5).mean(1),
                      'corrected_detr': (gwc.apply(lambda s: zscore(detrend(s))) < -0.5).mean(1)})
agree.to_csv(OUTD / 'drought_member_agreement.csv')

# ------------------------------------------------------------------ 4. climate linkage
log('\n--- Climate linkage (ensemble-mean series; p from effective sample size)')
pr = chirps_domain()
pdsi = pdsi_domain()
pr.index = month_index(pr.index)
pdsi.index = month_index(pdsi.index)
drivers = {'PDSI': pdsi}
for k in (1, 3, 6, 12, 24):
    drivers[f'SPI-{k}'] = spi(pr['1981-01-01':END], k)
pd.DataFrame(drivers).to_csv(OUTD / 'climate_drivers_domain.csv')
rows = []
for target, s in [('GWSA raw', gw.mean(1)), ('GWSA lake-corrected', gwc.mean(1)), ('Lake storage', lk.A8502),
                  ('TWSA', twsa.mean(1))]:
    for dname, dv in drivers.items():
        for lag in range(0, 7):
            x = s.reindex(cal)
            y = dv.reindex(cal).shift(lag)
            r, p, ne, n = corr_eff(x, y)
            rd, pd_, ned, _ = corr_eff(detrend(x.dropna()).reindex(cal), detrend(y.dropna()).reindex(cal))
            rows.append(dict(target=target, driver=dname, lag=lag, n=n, r=r, p_eff=p, neff=ne, r_detr=rd, p_detr=pd_))
for sc in ['A6000', 'A8502', 'half']:
    s = pd.concat([SCEN[sc](c) for c in gw.columns], axis=1).mean(1)
    for dname, dv in drivers.items():
        x, y = s.reindex(cal), dv.reindex(cal)
        r, p, ne, n = corr_eff(x, y)
        rd, pd_, ned, _ = corr_eff(detrend(x.dropna()).reindex(cal), detrend(y.dropna()).reindex(cal))
        rows.append(dict(target=f'GWSA lake {sc}', driver=dname, lag=0, n=n, r=r, p_eff=p, neff=ne, r_detr=rd, p_detr=pd_))
cl = pd.DataFrame(rows)
cl.to_csv(TAB / 'climate_linkage.csv', index=False)
for (t, dname), g in cl.groupby(['target', 'driver'], sort=False):
    b = g.loc[g.r.abs().idxmax()]
    l0 = g[g.lag == 0].iloc[0]
    log(f'{t:20s} {dname:7s} lag0 r={l0.r:+.2f} (p_eff {l0.p_eff:.3f}, detr {l0.r_detr:+.2f} p {l0.p_detr:.3f}) | '
        f'max |r| lag {int(b.lag)} r={b.r:+.2f} p_eff {b.p_eff:.3f}; r range over lags {g.r.min():+.2f}..{g.r.max():+.2f}')

# ------------------------------------------------------------------ 5. CLSM check and Noah/VIC-only sensitivity
# GLDAS-2.1 CLSM has no separate groundwater store: TWS_inst = profile soil moisture + canopy (+ SWE)
d = (clsm_full.TWS_inst - clsm_full.sm - clsm_full.cw).abs().max()
log(f'\nCLSM TWS_inst - profile SM - canopy: max abs {d:.3f} mm (profile SM is the whole modelled subsurface column)')
nv = [c for c in gw.columns if not c.endswith('CLSM')]
for per, (a, b) in periods.items():
    g = tr[(tr.period == per) & tr.member.isin(nv)]
    log(f'Noah/VIC only {per}: ' + '; '.join(f'{k} {v.slope.median():.2f} [{v.slope.min():.2f}..{v.slope.max():.2f}] '
                                            f'sig {int((v.p < 0.05).sum())}/{len(v)}'
                                            for k, v in g.groupby('lake', sort=False)))
g = lc[lc.member.isin(nv)]
log(f'Noah/VIC only: var explained by lake median {g.var_explained.median():.2f} [{g.var_explained.min():.2f}-{g.var_explained.max():.2f}]')
g = wr[wr.member.isin(nv) & (wr.window == '2015-01..2016-12')]
log('Noah/VIC only 2015-16 mean z: ' + '; '.join(f'{k} {v.z.median():+.2f} [{v.z.min():+.2f},{v.z.max():+.2f}]'
                                                  for k, v in g.groupby('lake', sort=False)))

# ------------------------------------------------------------------ 6. variance decomposition of the trend
t_full = tr[(tr.period == '2002-04..2025-12') & tr.lake.isin(list(A_LAKE))].copy()
t_full['grace'] = t_full.member.str.split('-').str[0]
t_full['lsm'] = t_full.member.str.split('-').str[1]
log('\nTrend spread (2002-2025, lake-corrected) by factor: mean slope per level')
for f in ['grace', 'lsm', 'lake']:
    log(f'  {f}:', t_full.groupby(f).slope.mean().round(2).to_dict())

(TAB / 'build_log.txt').write_text('\n'.join(LOG) + '\n')
print('\nwrote', TAB / 'build_log.txt')
