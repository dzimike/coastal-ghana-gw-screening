"""Supplementary Figs S2 (components), S5 (Monte Carlo stability, primary index) and S8 (Lake Volta)."""
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / 'outputs/figures'
RC = ROOT / 'outputs/tables/revision_checks'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
g = gpd.read_file(ROOT / 'data/processed/risk_index/cgsri_grid10km_v2.gpkg').to_crs('EPSG:4326')
districts = gpd.read_file(ROOT / 'data/raw/boundaries/coastal_districts.gpkg').to_crs('EPSG:4326')
XL, YL = (-3.5, 1.5), (4.6, 7.2)


def frame(ax, title):
    districts.boundary.plot(ax=ax, edgecolor='#777777', lw=0.25)
    ax.set_xlim(XL)
    ax.set_ylim(YL)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


# S2 — components
comps = [('climate_stress', '(a) Climate Stress (CHIRPS below-normal frequency)'),
         ('exposure', '(b) Exposure'), ('vulnerability', '(c) Vulnerability'),
         ('sub_hazard', '(d) Subsidence Hazard (exploratory index only)')]
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
for ax, (c, t) in zip(axes.flat, comps):
    g.plot(ax=ax, column=c, cmap='YlOrRd', vmin=0, vmax=1, edgecolor='none', legend=True,
           legend_kwds={'shrink': 0.7})
    frame(ax, f'{t}\nmean {g[c].mean():.2f}, SD {g[c].std():.2f}')
fig.suptitle('Supplementary Fig. S2. CGSRI component scores (harmonised definitions). '
             'GW Hazard is 0 in every cell (regional GWSA trend is positive) and is not mapped.', fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / 'figS2_cgsri_components.png')
plt.close()

# S5 — Monte Carlo stability (primary)
s2 = pd.read_csv(ROOT / 'outputs/tables/supplementary_table_S2_sensitivity_analysis_v2.csv')
g = g.merge(s2[['cell_id', 'mc_vh_probability_primary']], on='cell_id')
vh = g[g['cgsri_class'] == 'Very high']
fig, ax = plt.subplots(figsize=(9, 5))
g.plot(ax=ax, column='mc_vh_probability_primary', cmap='viridis', vmin=0, vmax=1, edgecolor='none', legend=True,
       legend_kwds={'label': 'Proportion of 1,000 weightings in Very High tier', 'shrink': 0.7})
vh.boundary.plot(ax=ax, edgecolor='red', lw=0.6)
frame(ax, f'Supplementary Fig. S5. Monte Carlo stability of the primary CGSRI Very High tier '
          f'({int((vh.mc_vh_probability_primary >= 0.75).sum())} of 91 cells ≥ 0.75; red outlines = Very High at equal weights)')
plt.savefig(FIGS / 'figS5_monte_carlo_stability.png')
plt.close()

# S8 — Lake Volta
lv = pd.read_csv(RC / 'lake_volta_monthly.csv', index_col=0, parse_dates=True)
lv = lv[lv.lake_ewh_mm_fullsupply.notna()]
corr = lv.gwsa_mm - lv.lake_ewh_mm_fullsupply
full = pd.date_range(lv.index.min(), lv.index.max(), freq='MS')
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
a1.plot(lv.gwsa_mm.reindex(full), color='#4477AA', lw=1, label='GWSA (uncorrected)')
a1.plot(lv.lake_ewh_mm_fullsupply.reindex(full), color='#CC6677', lw=1, label='Lake Volta storage term (EWH, 8,502 km²)')
a1.plot(corr.reindex(full), color='#117733', lw=1.2, label='GWSA minus Lake Volta term')
a1.axhline(0, color='k', lw=0.5)
a1.set_ylabel('mm EWH')
a1.legend(fontsize=7, loc='upper left')
a1.set_title('(a) Regional GWSA and the Lake Volta storage term (anomalies vs 2004–2009 monthly climatology)', fontsize=8)
for s, c, lbl in [(lv.gwsa_mm, '#4477AA', 'uncorrected'), (corr, '#117733', 'corrected')]:
    z = (s - s.mean()) / s.std()
    a2.plot(z.reindex(full), color=c, lw=1, label=f'SGWA {lbl}')
for t in (-0.5, -1, -1.5, -2):
    a2.axhline(t, color='#999999', lw=0.4, ls=':')
a2.axhline(0, color='k', lw=0.5)
a2.set_ylabel('Standardised GWSA')
a2.legend(fontsize=7, loc='upper left')
a2.set_title('(b) Standardised GWSA over the matched months (April 2002 – March 2019); dotted lines: drought thresholds', fontsize=8)
fig.suptitle('Supplementary Fig. S8. Effect of removing Lake Volta storage (G-REALM V2 altimetry) from the regional GWSA', fontsize=9)
plt.tight_layout()
plt.savefig(FIGS / 'figS8_lake_volta_correction.png')
plt.close()
print('saved S2, S5, S8')
