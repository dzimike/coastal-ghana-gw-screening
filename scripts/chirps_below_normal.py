"""Regenerate data/processed/climate_indices/chirps_drought_stress_per_cell.csv from raw CHIRPS.

BN_i = percentage of the 300 months January 2000 - December 2024 in which monthly rainfall at the
CHIRPS v2.0 0.05-degree pixel nearest to the cell centroid was below that pixel's 1981-2010
calendar-month mean. chirps_mean_rain_mm is the mean annual rainfall over the same 300 months.
Running with --check compares against the existing file instead of overwriting it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/chirps/chirps-v2.0.monthly.nc'
OUT = ROOT / 'data/processed/climate_indices/chirps_drought_stress_per_cell.csv'

ap = argparse.ArgumentParser()
ap.add_argument('--check', action='store_true')
args = ap.parse_args()

ref = pd.read_csv(OUT)
ds = xr.open_dataset(RAW)
pr = ds[list(ds.data_vars)[0]]
if 'latitude' in pr.dims:
    pr = pr.rename({'latitude': 'lat', 'longitude': 'lon'})
P = pr.sel(lat=xr.DataArray(ref.centroid_lat.values, dims='c'),
           lon=xr.DataArray(ref.centroid_lon.values, dims='c'), method='nearest').load()
clim = P.sel(time=slice('1981', '2010')).groupby('time.month').mean('time')
sub = P.sel(time=slice('2000-01', '2024-12'))
assert sub.sizes['time'] == 300
bn = ((sub.groupby('time.month') - clim) < 0).sum('time') / 300 * 100
rain = sub.sum('time') / 25
new = ref[['cell_id', 'centroid_lat', 'centroid_lon']].copy()
new['chirps_below_normal_pct'] = bn.values
new['chirps_mean_rain_mm'] = rain.values
if args.check:
    print('max abs diff BN:', float(np.abs(new.chirps_below_normal_pct - ref.chirps_below_normal_pct).max()),
          ' mean rain:', float(np.abs(new.chirps_mean_rain_mm - ref.chirps_mean_rain_mm).max()))
else:
    new.to_csv(OUT, index=False)
    print('wrote', OUT)
