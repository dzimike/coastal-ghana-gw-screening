"""Paper 3: checks added for Journal of Hydrology: Regional Studies. Run after build_paper3_v2.py.

1. What the averaging domain contains (share of each country).
2. Size of the lake-correction uncertainty relative to the groundwater signal that remains.
3. How representative 2002-2025 is of the longer rainfall (1981-2025) and lake-level (1992-2026) records.
Outputs: outputs/tables/paper3_v2/domain_composition.csv, long_term_context.csv, jhrs_checks_log.txt
"""
from pathlib import Path

import cartopy.io.shapereader as shp
import geopandas as gpd
import numpy as np
import pandas as pd
import pymannkendall as mk
import xarray as xr
from scipy import stats
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
PRO = ROOT / 'data/processed/paper3'
TAB = ROOT / 'outputs/tables/paper3_v2'
INT = ROOT / 'data/interim/paper3'
lines = []


def log(s=''):
    print(s, flush=True)
    lines.append(str(s))


# 1. domain composition
jpl = xr.open_dataset(ROOT / 'data/raw/grace/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04CRI.nc')
jpl = jpl.assign_coords(lon=((jpl.lon + 180) % 360 - 180)).sortby('lon')
m = jpl.land_mask.sel(lat=slice(4.0, 11.5), lon=slice(-3.5, 1.5)) == 1
cells = [box(float(x) - .25, float(y) - .25, float(x) + .25, float(y) + .25)
         for y in m.lat.values for x in m.lon.values if bool(m.sel(lat=y, lon=x))]
dom = gpd.GeoDataFrame(geometry=cells, crs=4326).to_crs(6933)
area = dom.area.sum() / 1e6
dom_u = dom.union_all()
world = gpd.read_file(shp.natural_earth(resolution='10m', category='cultural', name='admin_0_countries'))[['ADMIN', 'geometry']].to_crs(6933)
rows = [dict(country=r.ADMIN, km2_in_domain=r.geometry.intersection(dom_u).area / 1e6, country_km2=r.geometry.area / 1e6)
        for r in world[world.intersects(dom_u)].itertuples()]
comp = pd.DataFrame(rows).sort_values('km2_in_domain', ascending=False)
comp['share_of_domain_pct'] = 100 * comp.km2_in_domain / area
comp['share_of_country_pct'] = 100 * comp.km2_in_domain / comp.country_km2
comp.round(1).to_csv(TAB / 'domain_composition.csv', index=False)
log(f'1. DOMAIN: {len(cells)} cells, equal-area {area:,.0f} km2')
log(comp.round(1).to_string(index=False))
log(f'not assigned to a country (coastal cell fractions): {100 - comp.share_of_domain_pct.sum():.1f}%')

# 2. correction uncertainty
R = pd.read_csv(PRO / 'gwsa_members_uncorrected.csv', index_col=0, parse_dates=True)
C = pd.read_csv(PRO / 'gwsa_members_lake_corrected.csv', index_col=0, parse_dates=True)
L = pd.read_csv(PRO / 'lake_volta_storage_ewh.csv', index_col=0, parse_dates=True).loc[R.index]
t = (R.index - R.index[0]).days.values / 365.2425
det = lambda s: s - np.polyval(np.polyfit(t, s.values, 1), t)
A6 = C[[c for c in C if c.endswith('A6000')]]
A8 = C[[c for c in C if c.endswith('A8502')]]
d = L.A8502 - L.A6000
log('\n2. CORRECTION UNCERTAINTY (mm)')
log(f'lake-area bound difference: median |d| {d.abs().median():.1f}; detrended SD {det(d).std():.1f}')
log(f'detrended SD of ensemble-mean corrected GWSA: A6000 {det(A6.mean(axis=1)).std():.1f}, A8502 {det(A8.mean(axis=1)).std():.1f}; '
    f'uncorrected {det(R.mean(axis=1)).std():.1f}')
log(f'member half-range of uncorrected GWSA: median {((R.max(axis=1) - R.min(axis=1)) / 2).median():.1f}')
ann = lambda s: s.groupby(s.index.year).mean()
log(f'2012 to 2016 fall in annual means: GWSA {ann(R.mean(axis=1))[2012] - ann(R.mean(axis=1))[2016]:.2f}; lake A8502 '
    f'{ann(L.A8502)[2012] - ann(L.A8502)[2016]:.2f}; lake A6000 {ann(L.A6000)[2012] - ann(L.A6000)[2016]:.2f}')

# 3. long-term context
p = pd.read_csv(INT / 'chirps_domain_monthly.csv', index_col=0, parse_dates=True).iloc[:, 0]
yr = p.groupby(p.index.year).sum()
yr = yr[p.groupby(p.index.year).count() == 12].loc[1981:2025]
pre, stu = yr.loc[1981:2001], yr.loc[2002:2025]
mw = stats.mannwhitneyu(stu, pre, alternative='two-sided')
h = mk.hamed_rao_modification_test(yr.values)
log('\n3. LONG-TERM CONTEXT')
log(f'CHIRPS annual rainfall 1981-2025: mean {yr.mean():.0f} mm, SD {yr.std():.0f}, range {yr.min():.0f} ({yr.idxmin()}) to {yr.max():.0f} ({yr.idxmax()})')
log(f'1981-2001 mean {pre.mean():.0f}; 2002-2025 mean {stu.mean():.0f} ({100 * (stu.mean() / pre.mean() - 1):+.1f}%); Mann-Whitney p {mw.pvalue:.4f}; '
    f'Sen {h.slope:+.2f} mm/yr, Hamed-Rao p {h.p:.4f}')
log(f'study-period range covers {100 * ((yr >= stu.min()) & (yr <= stu.max())).mean():.0f}% of the 45 years; driest {stu.idxmin()} percentile in 1981-2025 '
    f'{100 * (yr <= stu.min()).mean():.0f}; wettest {stu.idxmax()} percentile {100 * (yr <= stu.max()).mean():.0f}')
for a, b in ((2002, 2007), (2008, 2013), (2014, 2017), (2015, 2016), (2018, 2025)):
    s = yr.loc[a:b]
    log(f'  {a}-{b}: mean {s.mean():.0f} mm ({100 * (s.mean() / yr.mean() - 1):+.1f}% of the 1981-2025 mean)')
g = pd.read_csv(PRO / 'lake_volta_grealm_10day.csv', parse_dates=['date']).set_index('date').h
mo = g.groupby(g.index.to_period('M')).mean()
mo.index = mo.index.to_timestamp()
la = mo.groupby(mo.index.year).mean()
la = la[mo.groupby(mo.index.year).count() >= 9]
log(f'G-REALM annual mean level, relative to reference: 1993-2001 mean {la.loc[1993:2001].mean():+.2f} m; 2002-2007 {la.loc[2002:2007].mean():+.2f}; '
    f'2008-2013 {la.loc[2008:2013].mean():+.2f}; 2014-2018 {la.loc[2014:2018].mean():+.2f}; 2019-2025 {la.loc[2019:2025].mean():+.2f}')
log(f'range of annual levels 1993-2001: {la.loc[1993:2001].min():+.2f} to {la.loc[1993:2001].max():+.2f}; 2002-2025: {la.loc[2002:2025].min():+.2f} to {la.loc[2002:2025].max():+.2f}')
pd.DataFrame({'annual_rainfall_mm': yr, 'lake_level_m': la}).round(2).to_csv(TAB / 'long_term_context.csv')
(TAB / 'jhrs_checks_log.txt').write_text('\n'.join(lines) + '\n')
