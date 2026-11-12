"""Build the harmonised CGSRI (v2): identical component rules in every 10 km cell.

Changes from v1 (notebook 10):
- GWH: one regional value for all cells, max(-s, 0) of the regional GWSA Sen's slope (normalised
  to [0, 1] by its own maximum possible value is meaningless for a constant, so the raw decline in
  mm/yr is mapped to 0 when s >= 0). The regional slope is positive, so GWH = 0 everywhere.
- CS: normalised CHIRPS below-normal frequency (chirps_below_normal_pct) for every cell.
- SH proxy: mean(inverse elevation, normalised built-up fraction, GWH).
- E and V: unchanged.
- cgsri = primary four-component index (GWH, CS, E, V); cgsri5 = exploratory five-component index.

Optional --velocity: recompute InSAR cell statistics from a (new) MintPy LOS velocity GeoTIFF.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / 'data/processed/risk_index/cgsri_grid10km.gpkg'
V2 = ROOT / 'data/processed/risk_index/cgsri_grid10km_v2.gpkg'
GWSA = ROOT / 'data/processed/groundwater_anomalies/gwsa_by_region_monthly.csv'
CLASSES = ['Very low', 'Low', 'Moderate', 'High', 'Very high']


def mm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def regional_slope():
    g = pd.read_csv(GWSA, parse_dates=['time']).set_index('time').loc[:'2023-12', 'gwsa_mm'].dropna()
    t = (g.index - g.index[0]).days.values / 365.2425
    v = g.values
    return float(np.median([(v[j] - v[i]) / (t[j] - t[i]) for i in range(len(v)) for j in range(i + 1, len(v))]))


def insar_stats(grid, vel_path):
    vel = np.full(len(grid), np.nan)
    nval = np.zeros(len(grid), dtype=int)
    with rasterio.open(vel_path) as src:
        g = grid.to_crs(src.crs)
        nd = src.nodata if src.nodata is not None else -9999.0
        inside = g.intersects(box(*src.bounds))
        for i in np.where(inside)[0]:
            img, _ = rio_mask(src, [g.geometry.iloc[i].__geo_interface__], crop=True, all_touched=True)
            v = img[0].astype(float)
            v[v == nd] = np.nan
            v[np.abs(v) > 300] = np.nan
            nval[i] = int(np.isfinite(v).sum())
            if nval[i] >= 10:
                vel[i] = float(np.nanmean(v))
    return vel, nval


def quintiles(s):
    return pd.qcut(s.rank(method='first'), 5, labels=CLASSES).astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--velocity', type=Path, help='LOS velocity GeoTIFF (mm/yr) to replace v1 InSAR statistics')
    args = ap.parse_args()
    g = gpd.read_file(V1)
    if args.velocity:
        g['insar_vel_mmyr'], g['insar_n_valid'] = insar_stats(g, args.velocity)

    s_reg = regional_slope()
    g['gw_hazard'] = max(-s_reg, 0.0)
    g['climate_stress'] = mm(g['chirps_below_normal_pct'])

    has = g['insar_n_valid'] >= 10
    vc = g.loc[has, 'insar_vel_mmyr'].clip(-60, 20)
    haz = pd.Series(np.nan, index=g.index)
    haz[has] = 1 - (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)
    elev_inv = 1 - mm(g['elev_m'].clip(0, 200))
    g['sub_proxy'] = (elev_inv + mm(g['builtup_density_2020']) + g['gw_hazard']) / 3
    w = np.clip(g['insar_n_valid'] / 500.0, 0, 1)
    w = np.where(g['centroid_lat'] < 7.0, np.minimum(w, 0.70), w)
    w = np.where(has, w, 0.0)
    g['insar_hazard'] = haz
    g['insar_weight'] = w
    g['sub_hazard'] = (1 - w) * g['sub_proxy'] + w * haz.fillna(g['sub_proxy'])

    g['cgsri'] = (g['gw_hazard'] + g['climate_stress'] + g['exposure'] + g['vulnerability']) / 4
    g['cgsri5'] = (g['gw_hazard'] + g['sub_hazard'] + g['climate_stress'] + g['exposure'] + g['vulnerability']) / 5
    g['cgsri_class'] = quintiles(g['cgsri'])
    g['cgsri5_class'] = quintiles(g['cgsri5'])
    keep = [c for c in g.columns if c not in ('ugpi_equal', 'ugpi_class', 'ugpi_expert', 'ugpi_hydro_only',
                                              'n_DRT', 'n_GW', 'has_grace')]
    g = g[keep]
    g.to_file(V2, driver='GPKG')
    g.drop(columns='geometry').to_csv(V2.with_suffix('.csv'), index=False)
    print(f'regional Sen slope {s_reg:+.2f} mm/yr -> GWH {g.gw_hazard.iloc[0]:.3f} for all cells')
    for c in ['cgsri', 'cgsri5']:
        vh = g[c].rank(method='first') > len(g) - 91
        print(f'{c}: range {g[c].min():.3f}-{g[c].max():.3f}, mean {g[c].mean():.3f}; '
              f'VH pop {g.loc[vh, "population_2020"].sum() / 1e6:.2f}M')
    print('wrote', V2)


if __name__ == '__main__':
    main()
