#!/usr/bin/env python3
"""Export the MintPy v2 LOS velocity to GeoTIFF (EPSG:32630, 80 m, v1 published grid).

Source is mintpy_v2/velocity.h5, which MintPy fits to timeseries_ERA5_demErr.h5 (the
ERA5- and DEM-error-corrected time series). The FILE_PATH attribute is checked so that
velocityERA5.h5 (the tropospheric delay rate, the field behind the withdrawn v1 raster)
can never be exported by mistake. Pixels with temporal coherence < 0.7 or average spatial
coherence < 0.5 are set to NaN; the second threshold removes decorrelated vegetated pixels
inland, which pass the temporal-coherence test but scatter by tens of mm/yr.

Outputs (data/processed/subsidence/, v1 files untouched):
  mintpy_velocity_los_v2.tif   LOS velocity, mm/yr (positive = towards the satellite)
  mintpy_velocity_vert_v2.tif  LOS / cos(incidence), mm/yr; assumes purely vertical motion
  mintpy_velocity_std_v2.tif   velocity standard deviation, mm/yr
"""
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
MINTPY_V2 = ROOT / 'data/processed/subsidence/mintpy_v2'
OUT_DIR = ROOT / 'data/processed/subsidence'
MIN_TEMP_COH = 0.7
MIN_SPATIAL_COH = 0.5


def main():
    with h5py.File(MINTPY_V2 / 'velocity.h5', 'r') as f:
        atr = dict(f.attrs)
        vel = f['velocity'][:] * 1000.0
        std = f['velocityStd'][:] * 1000.0
    src = Path(atr['FILE_PATH']).name
    if src != 'timeseries_ERA5_demErr.h5':
        raise SystemExit(f'velocity.h5 was fitted to {src}, expected timeseries_ERA5_demErr.h5')
    if int(atr['EPSG']) != 32630:
        raise SystemExit(f'unexpected EPSG {atr["EPSG"]}')

    with h5py.File(MINTPY_V2 / 'temporalCoherence.h5', 'r') as f:
        tcoh = f['temporalCoherence'][:]
    with h5py.File(MINTPY_V2 / 'avgSpatialCoh.h5', 'r') as f:
        scoh = f['coherence'][:]
    with h5py.File(MINTPY_V2 / 'inputs/geometryGeo.h5', 'r') as f:
        inc = f['incidenceAngle'][:]

    # MintPy writes 0 where a pixel was not inverted
    bad = (tcoh < MIN_TEMP_COH) | (scoh < MIN_SPATIAL_COH) | (vel == 0) | ~np.isfinite(vel) | (inc <= 0)
    vel = np.where(bad, np.nan, vel).astype('float32')
    std = np.where(bad, np.nan, std).astype('float32')
    vert = np.where(bad, np.nan, vel / np.cos(np.radians(inc))).astype('float32')

    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'width': vel.shape[1], 'height': vel.shape[0], 'crs': 'EPSG:32630',
        'transform': from_origin(float(atr['X_FIRST']), float(atr['Y_FIRST']),
                                 float(atr['X_STEP']), -float(atr['Y_STEP'])),
        'nodata': np.nan, 'compress': 'lzw', 'tiled': True,
    }
    tags = {'source': 'mintpy_v2/velocity.h5 <- timeseries_ERA5_demErr.h5',
            'period': f'{atr["START_DATE"]}-{atr["END_DATE"]}',
            'reference_yx': f'{atr["REF_Y"]},{atr["REF_X"]}',
            'mask': f'temporalCoherence >= {MIN_TEMP_COH} and avgSpatialCoh >= {MIN_SPATIAL_COH}',
            'units': 'mm/yr'}
    for name, arr in [('los', vel), ('vert', vert), ('std', std)]:
        out = OUT_DIR / f'mintpy_velocity_{name}_v2.tif'
        with rasterio.open(out, 'w', **profile) as dst:
            dst.write(arr, 1)
            dst.update_tags(**tags)
        print(f'{out.name}: {np.isfinite(arr).sum()} valid px, median {np.nanmedian(arr):.2f} mm/yr')


if __name__ == '__main__':
    main()
