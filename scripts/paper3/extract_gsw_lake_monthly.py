"""Monthly water area of Lake Volta from JRC Global Surface Water v1.4 (Pekel et al., 2016).

Reads only the Lake Volta window of the openly published 30 m tiles (no Earth Engine account needed).
1. Lake mask: the connected water body with occurrence > 0, inside a 0.05 degree buffer of the Natural Earth lake
   and north of the Akosombo Dam.
2. For each month, water and valid (observed) area are tallied per occurrence value (1-100). Unobserved pixels are
   filled with an isotonic estimate of the probability of water given occurrence, fitted to that month's observed
   pixels, which is a post-stratified form of the occurrence-based enhancement of Zhao and Gao (2018).

Output: data/interim/paper3/gsw/ (mask summary, occurrence histogram, monthly_area.csv; resumable).
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', GDAL_HTTP_MAX_RETRY='6',
                  GDAL_HTTP_RETRY_DELAY='3', GDAL_HTTP_TIMEOUT='120', VSI_CACHE='FALSE')

import cartopy.io.shapereader as shp
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from scipy import ndimage
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/interim/paper3/gsw'
OUT.mkdir(parents=True, exist_ok=True)
RES = 0.00025
W, E, S, N = -2.1, 0.7, 6.1, 9.3
DAM_LAT = 6.30
OCC = 'https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{}v1_4_2021.tif'
MON = 'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/MonthlyHistory/LATEST/tiles/{y}/{y}_{m:02d}/{y}_{m:02d}-0000280000-{c}.tif'
ROWS = int(round((10 - N) / RES)), int(round((N - S) / RES))
C_W = int(round((W + 10) / RES)), int(round(-W / RES))         # column offset and width in the 10W tile
C_E = 0, int(round(E / RES))                                   # same in the 0E tile
SHAPE = (ROWS[1], C_W[1] + C_E[1])
WIN_W = Window(C_W[0], ROWS[0], C_W[1], ROWS[1])
WIN_E = Window(C_E[0], ROWS[0], C_E[1], ROWS[1])
TRANSFORM = rasterio.Affine(RES, 0, W, 0, -RES, N)


def read_pair(u_w, u_e):
    with rasterio.open('/vsicurl/' + u_w) as r:
        a = r.read(1, window=WIN_W)
    with rasterio.open('/vsicurl/' + u_e) as r:
        b = r.read(1, window=WIN_E)
    return np.hstack([a, b])


def lake_mask():
    f = OUT / 'lake_mask.npz'
    if f.exists():
        z = np.load(f)
        return z['idx'], z['occ'], z['area']
    occ = read_pair(OCC.format('10W_10N'), OCC.format('0E_10N'))
    lakes = gpd.read_file(shp.natural_earth(resolution='10m', category='physical', name='lakes'))
    geom = lakes[lakes.name == 'Lake Volta'].buffer(0.05).union_all()
    inside = geometry_mask([geom.__geo_interface__], SHAPE, TRANSFORM, invert=True)
    lat = N - (np.arange(SHAPE[0]) + .5) * RES
    inside[lat < DAM_LAT, :] = False
    water = inside & (occ > 0) & (occ <= 100)
    lab, n = ndimage.label(water, structure=np.ones((3, 3), bool))
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    keep = lab == big
    del lab
    idx = np.flatnonzero(keep.ravel()).astype(np.int64)
    rows = idx // SHAPE[1]
    area = ((RES * 111.32) ** 2 * np.cos(np.radians(lat[rows]))).astype(np.float32)
    o = occ.ravel()[idx]
    np.savez_compressed(f, idx=idx, occ=o, area=area)
    print(f'components {n}; lake pixels {idx.size:,}; maximum water extent {area.sum():,.0f} km2', flush=True)
    return idx, o, area


idx, occ_m, area_m = lake_mask()
TOTAL = np.bincount(occ_m, weights=area_m, minlength=101)
pd.DataFrame({'occurrence': np.arange(101), 'area_km2': TOTAL}).to_csv(OUT / 'occurrence_area.csv', index=False)
EDGE = (occ_m >= 5) & (occ_m <= 95)
print(f'mask {area_m.sum():,.0f} km2; area with occurrence >= 50%: {TOTAL[50:].sum():,.0f} km2; >= 90%: {TOTAL[90:].sum():,.0f} km2', flush=True)


def one(y, m):
    v = read_pair(MON.format(y=y, m=m, c='0000680000'), MON.format(y=y, m=m, c='0000720000')).ravel()[idx]
    obs = v > 0
    Vb = np.bincount(occ_m[obs], weights=area_m[obs], minlength=101)
    Wb = np.bincount(occ_m[v == 2], weights=area_m[v == 2], minlength=101)
    o = np.arange(1, 101)
    ok = Vb[1:] > 0
    r = dict(year=y, month=m, valid_frac=float(area_m[obs].sum() / area_m.sum()), edge_valid_frac=float(area_m[obs & EDGE].sum() / area_m[EDGE].sum()),
             observed_water_km2=float(Wb.sum()), area_km2=np.nan)
    if ok.sum() >= 20 and Vb.sum() > 0:
        p = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip').fit(o[ok], Wb[1:][ok] / Vb[1:][ok], sample_weight=Vb[1:][ok]).predict(o)
        r['area_km2'] = float((TOTAL[1:] * p).sum())
        r['filled_frac'] = 1 - r['valid_frac']
    return r


f = OUT / 'monthly_area.csv'
done = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=['year', 'month'])
have = set(zip(done.year.astype(int), done.month.astype(int)))
todo = [(y, m) for y in range(1992, 2022) for m in range(1, 13) if (y, m) >= (1992, 10) and (y, m) not in have]
print(f'months to process: {len(todo)}', flush=True)
rows, fails = [], []
with ThreadPoolExecutor(3) as ex:
    fut = {ex.submit(one, y, m): (y, m) for y, m in todo}
    for k, x in enumerate(as_completed(fut)):
        try:
            rows.append(x.result())
        except Exception as e:
            fails.append((fut[x], str(e)[:90]))
        if (k + 1) % 15 == 0 or k + 1 == len(todo):
            pd.concat([done, pd.DataFrame(rows)]).sort_values(['year', 'month']).to_csv(f, index=False)
            print(f'{k + 1}/{len(todo)} months; failures {len(fails)}', flush=True)
if rows:
    pd.concat([done, pd.DataFrame(rows)]).sort_values(['year', 'month']).to_csv(f, index=False)
print('failed months:', fails[:8], len(fails), flush=True)
