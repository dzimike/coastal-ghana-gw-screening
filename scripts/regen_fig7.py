"""Regenerate Fig. 7 from the harmonised CGSRI (v2).

Fig. 7a now plots the subsidence proxy as implemented in notebook 10:
mean of inverse elevation (clipped 0-200 m), normalised GHSL built-up fraction, and GWH.
Fig. S6 maps GRACE coverage status (330 covered / 123 fallback cells) with Very High cells outlined.
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
BOUND, RI, FIGS = ROOT / 'data/raw/boundaries', ROOT / 'data/processed/risk_index', ROOT / 'outputs/figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
MAP_XLIM, MAP_YLIM = (-3.5, 1.5), (4.0, 7.5)
CITIES = [(-0.187, 5.603, 'Accra'), (-0.016, 5.670, 'Tema'), (-1.246, 5.106, 'Cape Coast'),
          (-1.752, 4.894, 'Takoradi'), (0.627, 5.772, 'Ada'), (0.999, 5.916, 'Keta')]

districts = gpd.read_file(BOUND / 'coastal_districts.gpkg').to_crs('EPSG:4326')
g = gpd.read_file(RI / 'cgsri_grid10km_v2.gpkg').to_crs('EPSG:4326')


def mm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


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


g['proxy'] = g['sub_proxy']
has = g['insar_n_valid'] >= 10

# ── Fig. 7 ─────────────────────────────────────────────────────────────────
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6))
g.plot(ax=ax, color='#EEEEEE', edgecolor='none')
g[~has].plot(ax=ax, column='proxy', cmap='YlOrRd', vmin=0, vmax=g['proxy'].max(), edgecolor='none',
             legend=True, legend_kwds={'label': 'Proxy subsidence hazard (0–1)', 'shrink': 0.8})
districts.boundary.plot(ax=ax, edgecolor='#555555', lw=0.35)
furniture(ax)
ax.set_title('(a) Proxy subsidence hazard, cells with <10 valid InSAR pixels (n=%d)\n'
             'mean of inverse elevation, built-up fraction and GW hazard score (= 0)' % (~has).sum(), fontsize=8)

g.plot(ax=ax2, color='#EEEEEE', edgecolor='none')
g[has].plot(ax=ax2, column='insar_vel_mmyr', cmap='RdBu_r', vmin=-25, vmax=5, edgecolor='none', legend=True,
            legend_kwds={'label': 'LOS velocity (mm yr⁻¹)', 'shrink': 0.8})
districts.boundary.plot(ax=ax2, edgecolor='#555555', lw=0.35)
furniture(ax2)
ax2.set_title('(b) ERA5-corrected SBAS LOS velocity (n=%d cells, ≥10 valid pixels)\n'
              'relative to reference pixel ~5.55°N, 0.42°W' % has.sum(), fontsize=8)
ax2.text(0.02, 0.04, 'Relative LOS rates, not vertical subsidence.\n'
                     'Uncorrected orbital ramp (~0.37 mm/yr per km eastward)\n'
                     'dominates differences between cells.',
         transform=ax2.transAxes, fontsize=6, va='bottom',
         bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85, lw=0.4))
plt.tight_layout()
plt.savefig(FIGS / 'fig07_subsidence_insar.png')
plt.close()
print('Saved fig07; proxy range %.3f-%.3f' % (g.loc[~has, 'proxy'].min(), g.loc[~has, 'proxy'].max()))
