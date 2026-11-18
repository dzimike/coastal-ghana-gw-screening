"""Regenerate Fig. 5 (urban growth) from the GHSL built-up grid and area time series.

Panel (c) sums GHSL built-up surface inside the 453 grid cells for each epoch and writes
data/processed/urban_growth/builtup_area_timeseries_grid453.csv. (The earlier
builtup_area_timeseries.csv summed the whole raster, which covers all of Ghana.)

Panel (c) is ticked only at the five GHSL epochs (2000, 2010, 2015, 2020, 2025), so the
axis no longer shows an empty 2005 position. Panels (a) and (b) follow notebook 12, cell 11.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UG, BOUND, FIGS = ROOT / 'data/processed/urban_growth', ROOT / 'data/raw/boundaries', ROOT / 'outputs/figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
MAP_XLIM, MAP_YLIM = (-3.5, 1.5), (4.0, 7.5)
CITIES = [(-0.187, 5.603, 'Accra'), (-0.016, 5.670, 'Tema'), (-1.246, 5.106, 'Cape Coast'),
          (-1.752, 4.894, 'Takoradi'), (0.627, 5.772, 'Ada'), (0.999, 5.916, 'Keta')]


def furniture(ax):
    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_xlabel('Longitude (°E)', fontsize=8)
    ax.set_ylabel('Latitude (°N)', fontsize=8)
    for lon, lat, nm in CITIES:
        ax.plot(lon, lat, 'k^', ms=4, zorder=5)
        ax.annotate(nm, (lon, lat), xytext=(4, 3), textcoords='offset points', fontsize=6, zorder=6)
    ax.annotate('N', xy=(0.96, 0.94), xycoords='axes fraction', fontsize=9, fontweight='bold', ha='center')
    ax.annotate('', xy=(0.96, 0.94), xytext=(0.96, 0.88), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=1.5))


ghana = gpd.read_file(BOUND / 'ghana_boundary.gpkg').to_crs('EPSG:4326')
g = gpd.read_file(UG / 'builtup_grid10km.gpkg').to_crs('EPSG:4326')
g_utm = gpd.read_file(UG / 'builtup_grid10km.gpkg').to_crs('EPSG:32630')
rows = []
for year in (2000, 2010, 2015, 2020, 2025):
    with rasterio.open(UG / f'ghsl_builtup_{year}_utm30n.tif') as src:
        arr = src.read(1).astype(float)
        inside = ~geometry_mask(g_utm.geometry, arr.shape, src.transform)
        rows.append(dict(year=year, built_area_km2=arr[inside].sum() / 1e6,
                         national_raster_km2=arr.sum() / 1e6))
ts = pd.DataFrame(rows)
ts['growth_km2_per_yr'] = ts.built_area_km2.diff() / ts.year.diff()
ts.round(2).to_csv(UG / 'builtup_area_timeseries_grid453.csv', index=False)
g['density_frac'] = g['builtup_m2_2020'] / (g['area_km2'] * 1e6)
g['growth_pct'] = g['builtup_growth_pct'].clip(0, 400)

fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
ax = axes[0]
g.plot(column='density_frac', ax=ax, cmap='Oranges', vmin=0, vmax=g['density_frac'].quantile(0.97),
       edgecolor='none', legend=True, legend_kwds={'label': 'Built-up fraction (2020)', 'shrink': 0.8})
ghana.boundary.plot(ax=ax, edgecolor='#555555', lw=0.5)
furniture(ax)
ax.set_title('(a) Built-up density 2020', fontsize=9)

ax = axes[1]
g.plot(column='growth_pct', ax=ax, cmap='RdYlGn_r', vmin=0, vmax=g['growth_pct'].quantile(0.95),
       edgecolor='none', missing_kwds={'color': 'lightgray'}, legend=True,
       legend_kwds={'label': 'Built-up growth 2000–2025 (%)', 'shrink': 0.8})
ghana.boundary.plot(ax=ax, edgecolor='#555555', lw=0.5)
furniture(ax)
ax.set_title('(b) Built-up growth rate 2000–2025 (%)', fontsize=9)

ax = axes[2]
yrs, vals = ts['year'].astype(int).tolist(), ts['built_area_km2'].tolist()
bars = ax.bar(yrs, vals, color='#CC6633', width=3.5, alpha=0.75)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, f'{v:,.0f}', ha='center', fontsize=7.5)
ax.set_xticks(yrs)
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax.set(xlabel='GHSL epoch', ylabel='Built-up area (km²)')
ax.set_title('(c) Built-up area, 2000–2025 (2025 extrapolated)', fontsize=9)
ax.text(0.03, 0.95, f'+{(vals[-1] - vals[0]) / vals[0] * 100:.0f}% in 25 years', transform=ax.transAxes,
        fontsize=8, va='top', bbox=dict(boxstyle='round', fc='#FFF3CD', alpha=0.9))

plt.tight_layout()
plt.savefig(FIGS / 'fig05_urban_growth.png')
plt.close()
print('Saved fig05;', dict(zip(yrs, [round(v) for v in vals])))
