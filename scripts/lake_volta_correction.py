"""Remove Lake Volta storage from the regional GWSA (April 2002 - March 2019) and compare results.

Lake storage anomaly (EWH over the GRACE averaging domain) = A_lake * dh / A_domain, with dh the
G-REALM V2 monthly mean level relative to its 2004-2009 calendar-month climatology. A constant
full-supply area is used (upper bound; the area shrinks at low levels), with a lower-area sensitivity case.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/tables/revision_checks'
A_DOMAIN = 376_774.0          # km2, land-masked GRACE cells in 4-11.5N, 3.5W-1.5E (revision_checks.py)
A_LAKE = {'full_supply_8502': 8502.0, 'reduced_6000': 6000.0}

ds = xr.open_dataset(ROOT / 'data/raw/lake_volta/Africa_Volta_WaterHeightV2_393.nc')
h = ds['surface_water_height'].to_series()
flag = ds['outlier_flag'].to_series()
h = h[(flag != 1) & h.notna() & (h < 900)]
lvl = h.resample('MS').mean()

g = pd.read_csv(ROOT / 'data/processed/groundwater_anomalies/gwsa_by_region_monthly.csv', parse_dates=['time'])
g['time'] = g.time.dt.to_period('M').dt.to_timestamp()
g = g.set_index('time')['gwsa_mm'].dropna()
g = g['2002-04':'2019-03']
lvl = lvl.reindex(g.index)
base = lvl['2004':'2009']
clim = base.groupby(base.index.month).mean()
dh = lvl - lvl.index.month.map(clim).values

lines, rows = [], []


def log(s):
    print(s)
    lines.append(s)


log(f'Lake level months matched to GWSA: {int(dh.notna().sum())} of {len(g)}; '
    f'level range {lvl.min():.2f}-{lvl.max():.2f} m; dh range {dh.min():.2f} to {dh.max():.2f} m')


def sen_mk(s):
    s = s.dropna()
    t = (s.index - s.index[0]).days.values / 365.2425
    v = s.values
    sl = np.median([(v[j] - v[i]) / (t[j] - t[i]) for i in range(len(v)) for j in range(i + 1, len(v))])
    return sl, mk.hamed_rao_modification_test(v).p


def episodes(s):
    z = (s - s.mean()) / s.std()
    out, cur = [], []
    for t, b in (z < -0.5).items():
        if b:
            cur.append(t)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    res = []
    for e in out:
        if len(e) < 3:
            continue
        seg = z[e[0]:e[-1]]
        res.append(f"{e[0]:%Y-%m}–{e[-1]:%Y-%m} ({len(seg)} obs; min {seg.min():.2f}; extreme {(seg < -2).sum()})")
    return res


m = dh.notna()
for label, a in [('uncorrected', None)] + list(A_LAKE.items()):
    s = g[m].copy() if a is None else (g - dh * a * 1000.0 / A_DOMAIN)[m]
    rsa = None if a is None else dh[m] * a * 1000.0 / A_DOMAIN
    sl, p = sen_mk(s)
    sl8, p8 = sen_mk(s['2008':])
    eps = episodes(s)
    log(f'{label:18s} n={len(s)}  Sen {sl:+.2f} mm/yr (p={p:.1e})  2008-2019 {sl8:+.2f} (p={p8:.2f})'
        + ('' if rsa is None else f'  lake term SD {rsa.std():.1f} mm, range {rsa.min():.0f} to {rsa.max():.0f}, '
                                  f'r(lake, GWSA)={np.corrcoef(rsa, g[m])[0, 1]:.2f}'))
    for e in eps:
        log(f'    episode {e}')
    rows.append(dict(series=label, n=len(s), sen_mm_yr=round(sl, 2), p=p, sen_2008_2019=round(sl8, 2), p_2008=p8,
                     sd_mm=round(s.std(), 1), episodes='; '.join(eps)))
pd.DataFrame(rows).to_csv(OUT / 'lake_volta_correction.csv', index=False)
pd.DataFrame({'gwsa_mm': g, 'lake_dh_m': dh,
              'lake_ewh_mm_fullsupply': dh * A_LAKE['full_supply_8502'] * 1000 / A_DOMAIN}).to_csv(
    OUT / 'lake_volta_monthly.csv')
(OUT / 'lake_volta_summary.txt').write_text('\n'.join(lines))
