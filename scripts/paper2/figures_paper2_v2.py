"""Figures for Paper 2 v2 (run build_uhgpi_v2.py first). Writes outputs/manuscript/paper2/v2/fig0*.png.

Figures are numbered in order of first citation in manuscript_paper2_v2.md.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from shapely.geometry import box

matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data/processed/lst/v2'
TAB = ROOT / 'outputs/tables/paper2_v2'
OUT = ROOT / 'outputs/manuscript/paper2/v2'
OUT.mkdir(parents=True, exist_ok=True)
NE_COAST = Path.home() / '.local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_coastline.shp'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'font.family': 'DejaVu Sans'})
SEA = '#e8f1f8'
CITIES = [(-0.186, 5.556, 'Accra', 'left'), (0.017, 5.668, 'Tema', 'left'), (-1.245, 5.105, 'Cape Coast', 'right'),
          (-1.734, 4.934, 'Sekondi-Takoradi', 'right'), (-0.637, 5.349, 'Winneba', 'right'),
          (0.635, 5.781, 'Ada Foah', 'left'), (1.003, 5.910, 'Keta', 'left'), (-2.237, 4.869, 'Axim', 'below'),
          (-1.995, 5.301, 'Tarkwa', 'right')]
EXTENT = (-3.35, 1.3, 4.65, 7.0)

grid = gpd.read_file(D / 'uhgpi_grid10km_v2.gpkg').to_crs(4326)
a = grid[grid.uhgpi.notna()].copy()
a['fdr_sig'] = a.fdr_sig.astype(str).str.lower().isin(['true', '1', '1.0'])
ghana = gpd.read_file(ROOT / 'data/raw/boundaries/ghana_boundary.gpkg').to_crs(4326)
coast = gpd.read_file(NE_COAST).clip(box(-3.5, 4.4, 1.5, 6.5))
regions = gpd.read_file(ROOT / 'data/raw/boundaries/gadm41_GHA.gpkg', layer='ADM_ADM_1').to_crs(4326)


def basemap(ax, cities=True, labels=True):
    ax.set_facecolor(SEA)
    ghana.plot(ax=ax, color='#f4f1ea', edgecolor='none', zorder=0)
    regions.boundary.plot(ax=ax, color='#b5b0a5', lw=0.3, zorder=3)
    ghana.boundary.plot(ax=ax, color='#555555', lw=0.6, zorder=3)
    coast.plot(ax=ax, color='#3b6e8f', lw=0.7, zorder=3)
    ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:])
    ax.set_aspect(1 / np.cos(np.radians(5.8)))
    ax.tick_params(labelsize=7)
    ax.set_xticks([-3, -2, -1, 0, 1]); ax.set_xticklabels(['3°W', '2°W', '1°W', '0°', '1°E'])
    ax.set_yticks([5, 5.5, 6, 6.5, 7]); ax.set_yticklabels(['5.0°N', '5.5°N', '6.0°N', '6.5°N', '7.0°N'])
    if cities:
        for lon, lat, name, ha in CITIES:
            ax.plot(lon, lat, 'o', ms=2.6, color='k', zorder=6)
            if labels:
                dx, dy, h = (0.05, 0.035, ha) if ha != 'below' else (0.0, -0.11, 'center')
                dx = -0.05 if ha == 'right' else dx
                ax.text(lon + dx, lat + dy, name, ha=h, fontsize=6.3, zorder=7,
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])


def scalebar_north(ax, x=0.62, y=0.07):
    lon0, lat0 = EXTENT[0] + (EXTENT[1] - EXTENT[0]) * x, EXTENT[2] + (EXTENT[3] - EXTENT[2]) * y
    dlon = 100 / (111.32 * np.cos(np.radians(lat0)))
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color='k', lw=2)
    ax.text(lon0 + dlon / 2, lat0 + 0.05, '100 km', ha='center', fontsize=6.5)
    ax.annotate('N', xy=(0.95, 0.93), xytext=(0.95, 0.80), xycoords='axes fraction', ha='center', fontsize=8,
                fontweight='bold', arrowprops=dict(arrowstyle='-|>', color='k', lw=1))


def label(ax, s):
    ax.set_title(s, loc='left', fontweight='bold', fontsize=9)


def save(fig, name, source):
    fig.text(0.01, 0.005, source, fontsize=6.3, color='#444444', ha='left', va='bottom')
    fig.savefig(OUT / name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


# ── Fig 1 study area ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.2, 4.6))
basemap(ax)
outside = grid[~grid.coastal.astype(bool)]
nolst = grid[grid.coastal.astype(bool) & grid.uhgpi.isna()]
a.plot(ax=ax, color='#c9dfb3', edgecolor='white', lw=0.3, zorder=2)
nolst.plot(ax=ax, color='#d9d9d9', edgecolor='white', lw=0.3, zorder=2)
outside.plot(ax=ax, facecolor='none', edgecolor='#9a9a9a', hatch='////', lw=0.3, zorder=2)
ax.legend(handles=[Patch(color='#c9dfb3', label=f'Analysis cells (n = {len(a)})'),
                   Patch(color='#d9d9d9', label=f'Within 50 km of sea, < 18 valid dry seasons (n = {len(nolst)})'),
                   Patch(facecolor='none', edgecolor='#9a9a9a', hatch='////',
                         label=f'Shared grid cells > 50 km from sea, excluded (n = {len(outside)})')],
          loc='upper left', fontsize=6.8, framealpha=0.95)
scalebar_north(ax)
save(fig, 'fig01_study_area.png', 'Grid: 10 km cells, EPSG:32630. Boundaries: GADM v4.1. Coastline: Natural Earth 10 m.')

# ── Fig 2 workflow ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.6, 4.6))
ax.set_xlim(0, 10.6); ax.set_ylim(0, 6); ax.axis('off')
FS = 6.4


def node(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.04', fc=fc, ec='#444444', lw=0.7))
    ax.text(x, y, text, ha='center', va='center', fontsize=FS)


def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#555555', lw=0.8))


C1, C2, C3, C4 = 1.0, 3.6, 6.25, 9.15
W1, W2, W3, W4 = 1.9, 2.15, 2.2, 2.55
inputs = [(5.05, 'MODIS MOD11A2 v6.1\ndaytime LST, 8-day, 1 km'),
          (3.8, 'GHSL built-up 2000–2025\nWorldPop 2020'),
          (2.55, 'CHIRPS v2.0 rainfall\n1981–2024'),
          (0.75, 'GRACE/GRACE-FO mascons,\nGLDAS Noah, G-REALM')]
proc = [(5.05, 'Dec–Feb seasons 2001–2023\n(≥ 3 composites per season),\nper-cell mean on 10 km grid'),
        (3.8, 'UGPI: mean of normalised\nbuilt-up fraction, population\ndensity, built-up growth'),
        (2.55, 'Climate stress: % of months\nbelow 1981–2010 mean\n(2000–2024)'),
        (0.75, 'Regional GWSA =\nTWSA − SMSA − CWSA;\nSGWA; Lake Volta removal')]
for (y, t), (_, t2) in zip(inputs, proc):
    node(C1, y, W1, 0.85, t, '#e8eef5')
    node(C2, y, W2, 0.95, t2, '#f3efe4')
    arrow(C1 + W1 / 2 + 0.05, y, C2 - W2 / 2 - 0.07, y)
node(C3, 5.55, W3, 0.6, 'LST trend: Sen slope, Hamed–Rao,\nBH-FDR (≥ 18 seasons)', '#fbe3d6')
node(C3, 4.65, W3, 0.6, 'LST anomaly 2019–2023\n(cell minus domain mean)', '#fbe3d6')
arrow(C2 + W2 / 2 + 0.05, 5.2, C3 - W3 / 2 - 0.07, 5.5)
arrow(C2 + W2 / 2 + 0.05, 4.9, C3 - W3 / 2 - 0.07, 4.7)
node(C4, 4.2, W4, 1.35, 'UHGPI: equal-weight mean of\nn(LST trend), n(LST anomaly),\nn(UGPI), n(climate stress);\nquintile classes, 264 cells\n≤ 50 km from the sea', '#f6d0c0')
arrow(C3 + W3 / 2 + 0.05, 5.5, C4 - W4 / 2 - 0.07, 4.65)
arrow(C3 + W3 / 2 + 0.05, 4.6, C4 - W4 / 2 - 0.07, 4.35)
arrow(C2 + W2 / 2 + 0.05, 3.8, C4 - W4 / 2 - 0.07, 4.0)
arrow(C2 + W2 / 2 + 0.05, 2.55, C4 - W4 / 2 - 0.07, 3.7)
node(C4, 2.2, W4, 1.0, 'Sensitivity: weights, drop-one,\nMonte Carlo, rank normalisation,\nresidual LST, full 453-cell grid', '#ede7f3')
arrow(C4, 3.5, C4, 2.73)
node(C3, 1.95, W3, 0.8, 'Co-location of warming trend\nand UGPI (Fisher; Moran\nspectral randomisation)', '#ede7f3')
arrow(C3 - 0.6, 4.33, C3 - 0.6, 2.38)
node(C3, 0.75, W3, 0.8, 'Temporal context only: annual\nSGWA vs dry-season domain\nLST anomaly', '#e3eedf')
arrow(C2 + W2 / 2 + 0.05, 0.75, C3 - W3 / 2 - 0.07, 0.75)
ax.text(C4, 0.75, 'GRACE has no spatial information\nat 10 km and is not an index\ncomponent', ha='center', va='center',
        fontsize=FS, style='italic', color='#555555')
save(fig, 'fig02_workflow.png', '')

# ── Fig 3 LST trend ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8.4, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.65, 1], width_ratios=[1, 1.25], hspace=0.32, wspace=0.25)
ax = fig.add_subplot(gs[0, :])
basemap(ax)
norm = TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.15)
a.plot(ax=ax, column='sen_slope', cmap='RdBu_r', norm=norm, edgecolor='white', lw=0.2, zorder=2)
a[a.fdr_sig.astype(bool)].plot(ax=ax, facecolor='none', edgecolor='k', lw=0.8, zorder=4)
sm = plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r')
cb = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.01)
cb.set_label('Dry-season LST trend (°C yr⁻¹)')
ax.legend(handles=[Patch(facecolor='none', edgecolor='k', label='FDR q < 0.05')], loc='upper left', fontsize=7)
scalebar_north(ax)
label(ax, '(a)')
ax = fig.add_subplot(gs[1, 0])
ax.hist(a.sen_slope, bins=np.arange(-0.08, 0.2, 0.01), color='#d6604d', edgecolor='white')
ax.axvline(0, color='k', lw=0.7); ax.axvline(a.sen_slope.median(), color='k', ls='--', lw=0.8,
                                              label=f'median {a.sen_slope.median():+.3f}')
ax.set_xlabel('Sen slope (°C yr⁻¹)'); ax.set_ylabel('Cells'); ax.legend(fontsize=7)
label(ax, '(b)')
ax = fig.add_subplot(gs[1, 1])
dom = pd.read_csv(D / 'domain_dry_season_lst.csv', index_col=0).iloc[:, 0]
ax.plot(dom.index, dom.values, '-o', ms=3, color='#b2182b')
t = dom.index.values
ax.plot(t, np.polyval(np.polyfit(t, dom.values, 1), t), color='k', lw=0.8, ls='--')
ax.set_xlabel('Dry season (Dec–Feb, labelled by Jan–Feb year)'); ax.set_ylabel('Domain-mean LST (°C)')
label(ax, '(c)')
save(fig, 'fig03_lst_trend.png', 'Data: MODIS MOD11A2 v6.1 daytime LST, 2001–2023; Hamed–Rao modified Mann–Kendall with Sen slope.')

# ── Fig 4 LST anomaly and urban contrast ────────────────────────────────────
fig = plt.figure(figsize=(8.4, 5.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.65, 1], hspace=0.34, wspace=0.28)
ax = fig.add_subplot(gs[0, :])
basemap(ax)
norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=6)
a.plot(ax=ax, column='lst_anom', cmap='RdBu_r', norm=norm, edgecolor='white', lw=0.2, zorder=2)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'), ax=ax, shrink=0.75, pad=0.01)
cb.set_label('Dry-season LST anomaly 2019–2023 (°C)')
scalebar_north(ax)
label(ax, '(a)')
ax = fig.add_subplot(gs[1, 0])
wb = pd.read_csv(TAB / 'ugpi_contrast_within_bands.csv')
x = np.arange(len(wb))
ax.bar(x - 0.2, wb['False'], 0.4, color='#92c5de', label='Other cells')
ax.bar(x + 0.2, wb['True'], 0.4, color='#d6604d', label='Top-quintile UGPI')
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(['0–10', '10–20', '20–30', '30–40', '40–50'])
ax.set_xlabel('Distance to sea (km)'); ax.set_ylabel('Mean LST anomaly (°C)'); ax.legend(fontsize=7)
label(ax, '(b)')
ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(a.chirps_mean_rain_mm, a.lst_anom, c=a.ugpi_urban_only, cmap='magma_r', s=9, vmin=0, vmax=0.3)
cb = fig.colorbar(sc, ax=ax, pad=0.01); cb.set_label('UGPI', fontsize=7)
ax.set_xlabel('Mean annual rainfall, CHIRPS (mm)'); ax.set_ylabel('LST anomaly (°C)')
label(ax, '(c)')
save(fig, 'fig04_lst_anomaly.png', 'Data: MODIS MOD11A2 v6.1; CHIRPS v2.0; UGPI from GHSL and WorldPop.')

# ── Fig 5 UGPI and climate stress ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
for ax, col, cmap, lab, ttl, vmax in [(axes[0], 'ugpi_urban_only', 'magma_r', 'UGPI', '(a)', 0.4),
                                      (axes[1], 'chirps_below_normal_pct', 'YlOrBr', 'Below-normal months (%)', '(b)', None)]:
    basemap(ax, labels=False)
    vmin = a[col].min() if vmax is None else 0
    vm = a[col].max() if vmax is None else vmax
    a.plot(ax=ax, column=col, cmap=cmap, vmin=vmin, vmax=vm, edgecolor='white', lw=0.2, zorder=2)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vm), cmap=cmap), ax=ax, shrink=0.8, pad=0.01)
    cb.set_label(lab, fontsize=7)
    label(ax, ttl)
save(fig, 'fig05_ugpi_climate.png', 'Data: GHSL GHS-BUILT-S R2023A; WorldPop 2020; CHIRPS v2.0 (2000–2024 vs 1981–2010 calendar-month means).')

# ── Fig 6 bivariate warming trend x UGPI ───────────────────────────────────
BIV = {(1, 1): '#e8e8e8', (2, 1): '#e4acac', (3, 1): '#c85a5a', (1, 2): '#b0d5df', (2, 2): '#ad9ea5',
       (3, 2): '#985356', (1, 3): '#64acbe', (2, 3): '#627f8c', (3, 3): '#574249'}
fig = plt.figure(figsize=(8.4, 4.7))
ax = fig.add_axes([0.0, 0.0, 0.82, 1])
basemap(ax)
a['bcol'] = [BIV[(int(t), int(u))] for t, u in zip(a.bivar_trend, a.bivar_ugpi)]
a.plot(ax=ax, color=a.bcol, edgecolor='white', lw=0.2, zorder=2)
scalebar_north(ax)
lx = fig.add_axes([0.85, 0.55, 0.13, 0.24])
for (t, u), c in BIV.items():
    lx.add_patch(plt.Rectangle((u - 1, t - 1), 1, 1, color=c))
lx.set_xlim(0, 3); lx.set_ylim(0, 3); lx.set_xticks([]); lx.set_yticks([])
lx.set_xlabel('UGPI tertile →', fontsize=7); lx.set_ylabel('Warming tertile →', fontsize=7, labelpad=2)
co = pd.read_csv(TAB / 'colocation_trend_ugpi.csv')
r = co.iloc[0]
fig.text(0.83, 0.42, f"Top-tertile pairs: {r.observed}\n(expected {r.expected};\nspatial-null p = {r.msr_p_more:.2f})",
         fontsize=6.8, va='top')
save(fig, 'fig06_bivariate_trend_ugpi.png', 'Tertiles over the 264 analysis cells. Data: MODIS MOD11A2 v6.1; GHSL; WorldPop.')

# ── Fig 7 UHGPI ─────────────────────────────────────────────────────────────
CL = ['Very low', 'Low', 'Moderate', 'High', 'Very high']
PAL = ['#d1e5f0', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d']
fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.6))
ax = axes[0]
basemap(ax)
for cl, c in zip(CL, PAL):
    a[a.uhgpi_class == cl].plot(ax=ax, color=c, edgecolor='#999999', lw=0.2, zorder=2)
ax.legend(handles=[Patch(color=c, label=cl) for cl, c in zip(CL, PAL)], title='UHGPI quintile', fontsize=7,
          title_fontsize=7, loc='upper left')
scalebar_north(ax)
label(ax, '(a)')
ax = axes[1]
basemap(ax, labels=False)
cmap = ListedColormap(['#f7f7f7', '#fde0c5', '#f4a582', '#d6604d', '#8b1a1a'])
nrm = BoundaryNorm([0, 0.1, 0.5, 0.75, 0.9, 1.0001], cmap.N)
a.plot(ax=ax, column='p_top_quintile_mc', cmap=cmap, norm=nrm, edgecolor='#999999', lw=0.2, zorder=2)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cmap), ax=ax, shrink=0.7, pad=0.01,
                  ticks=[0, 0.1, 0.5, 0.75, 0.9, 1.0])
cb.set_label('Share of 10,000 random weightings\nplacing the cell in the top quintile', fontsize=7)
label(ax, '(b)')
save(fig, 'fig07_uhgpi.png', 'Quintile classes are relative to the 264 analysis cells. Data: MODIS; GHSL; WorldPop; CHIRPS.')

# ── Fig 8 temporal context ──────────────────────────────────────────────────
ann = pd.read_csv(TAB / 'annual_series.csv', index_col=0)
lake = pd.read_csv(ROOT / 'outputs/tables/revision_checks/lake_volta_monthly.csv', parse_dates=['time']).set_index('time')
lc = lake.gwsa_mm - lake.lake_ewh_mm_fullsupply
lzm = (lc - lc.mean()) / lc.std()
cnt = lzm.groupby(lzm.index.year).count()
lz = lzm.groupby(lzm.index.year).mean()[cnt >= 6]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw=dict(width_ratios=[1.7, 1]))
ax = axes[0]
yrs = ann.loc[2002:2023].index
deficit = ann.loc[2002:2023, 'sgwa_calendar'] < -0.5
for y in yrs[deficit.values]:
    ax.axvspan(y - 0.5, y + 0.5, color='#fde0c5', zorder=0)
ax.plot(yrs, ann.loc[2002:2023, 'sgwa_calendar'], '-o', ms=3, color='#2166ac', label='Annual SGWA (uncorrected)')
ax.plot(lz.index, lz.values, '-s', ms=2.5, color='#67a9cf', label='Annual SGWA, Lake Volta removed')
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('SGWA (standard deviations)'); ax.set_xlabel('Year'); ax.set_ylim(-2.2, 2.2)
ax2 = ax.twinx()
ax2.bar(yrs, ann.loc[2002:2023, 'lst_anom_domain'], color='#d6604d', alpha=0.55, width=0.6, label='Dry-season LST anomaly')
ax2.set_ylabel('Dry-season LST anomaly (°C)')
ax2.set_ylim(-2.2, 2.2)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2 + [Patch(color='#fde0c5', label='SGWA < −0.5')], l1 + l2 + ['SGWA < −0.5'], fontsize=6.5, loc='upper left')
label(ax, '(a)')
ax = axes[1]
j = ann.loc[2002:2023].dropna()
ax.scatter(j.sgwa_calendar, j.lst_anom_domain, color='#555555', s=14)
for y, rr in j.iterrows():
    ax.text(rr.sgwa_calendar + 0.04, rr.lst_anom_domain, str(y)[2:], fontsize=5.5, color='#777777')
tp = pd.read_csv(TAB / 'temporal_gwsa_lst.csv').iloc[0]
ax.set_xlabel('Annual SGWA'); ax.set_ylabel('Dry-season LST anomaly (°C)')
ax.text(0.03, 0.95, f'r = {tp.pearson_r:+.2f}, p = {tp.p:.2f}, n = {int(tp.n)}', transform=ax.transAxes, fontsize=7, va='top')
label(ax, '(b)')
plt.tight_layout()
save(fig, 'fig08_temporal_context.png', 'Data: JPL GRACE/GRACE-FO RL06.3 mascons, GLDAS Noah v2.1, G-REALM V2 (to March 2019), MODIS MOD11A2 v6.1.')
