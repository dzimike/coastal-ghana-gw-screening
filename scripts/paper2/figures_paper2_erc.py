"""Figures for the Environmental Research Communications version of Paper 2.

Run build_uhgpi_v2.py and build_paper2_erc_checks.py first. Figures are drawn at their final width
(16 cm) with text of at least 8 pt; data sources are given in the captions, as IOP requires.
Writes PNG and PDF to outputs/manuscript/paper2/erc/.
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data/processed/lst/v2'
E = ROOT / 'data/processed/lst/erc'
TAB = ROOT / 'outputs/tables/paper2_v2'
TE = ROOT / 'outputs/tables/paper2_erc'
OUT = ROOT / 'outputs/manuscript/paper2/erc'
OUT.mkdir(parents=True, exist_ok=True)
NE_COAST = Path.home() / '.local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_coastline.shp'
W = 6.3
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
SEA = '#e8f1f8'
CITIES = [(-0.186, 5.556, 'Accra', 'left', .05, .05), (0.017, 5.668, 'Tema', 'left', .05, .06),
          (-1.245, 5.105, 'Cape Coast', 'center', 0, -.15), (-1.734, 4.934, 'Takoradi', 'center', .02, -.15),
          (-0.637, 5.349, 'Winneba', 'center', .05, -.15), (0.635, 5.781, 'Ada', 'center', .04, -.15),
          (1.003, 5.910, 'Keta', 'center', .08, -.15), (-2.237, 4.869, 'Axim', 'center', -.05, -.15),
          (-1.995, 5.301, 'Tarkwa', 'right', -.05, .04)]
EXTENT = (-3.35, 1.3, 4.62, 7.0)

grid = gpd.read_file(D / 'uhgpi_grid10km_v2.gpkg').to_crs(4326)
a = grid[grid.uhgpi.notna()].copy()
a['fdr_sig'] = a.fdr_sig.astype(str).str.lower().isin(['true', '1', '1.0'])
night = pd.read_csv(E / 'night_lst_by_cell.csv')
a = a.merge(night, on='cell_id', how='left')
ghana = gpd.read_file(ROOT / 'data/raw/boundaries/ghana_boundary.gpkg').to_crs(4326)
coast = gpd.read_file(NE_COAST).clip(box(-3.5, 4.4, 1.5, 6.5))
regions = gpd.read_file(ROOT / 'data/raw/boundaries/gadm41_GHA.gpkg', layer='ADM_ADM_1').to_crs(4326)


def basemap(ax, cities=True):
    ax.set_facecolor(SEA)
    ghana.plot(ax=ax, color='#f4f1ea', edgecolor='none', zorder=0)
    regions.boundary.plot(ax=ax, color='#b5b0a5', lw=0.3, zorder=3)
    ghana.boundary.plot(ax=ax, color='#555555', lw=0.6, zorder=3)
    coast.plot(ax=ax, color='#3b6e8f', lw=0.7, zorder=3)
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    ax.set_aspect(1 / np.cos(np.radians(5.8)))
    ax.set_xticks([-3, -2, -1, 0, 1])
    ax.set_xticklabels(['3°W', '2°W', '1°W', '0°', '1°E'])
    ax.set_yticks([5, 6, 7])
    ax.set_yticklabels(['5°N', '6°N', '7°N'])
    if cities:
        for lon, lat, name, ha, dx, dy in CITIES:
            ax.plot(lon, lat, 'o', ms=2.6, color='k', zorder=6)
            ax.text(lon + dx, lat + dy, name, ha=ha, fontsize=8, zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])


def scalebar_north(ax, x=0.70, y=0.86):
    lon0, lat0 = EXTENT[0] + (EXTENT[1] - EXTENT[0]) * x, EXTENT[2] + (EXTENT[3] - EXTENT[2]) * y
    dlon = 100 / (111.32 * np.cos(np.radians(lat0)))
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color='k', lw=2, zorder=8)
    ax.text(lon0 + dlon / 2, lat0 + 0.05, '100 km', ha='center', fontsize=8, zorder=8)
    ax.annotate('N', xy=(0.96, 0.97), xytext=(0.96, 0.80), xycoords='axes fraction', ha='center', fontsize=9,
                fontweight='bold', arrowprops=dict(arrowstyle='-|>', color='k', lw=1), zorder=8)


def label(ax, s):
    ax.set_title(s, loc='left', fontweight='bold', fontsize=9, pad=3)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


def cbar(fig, ax, norm, cmap, text, **kw):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.028, pad=0.015, **kw)
    cb.set_label(text)
    return cb


# Figure 1: study area
fig, ax = plt.subplots(figsize=(W, 3.5))
basemap(ax)
outside = grid[~grid.coastal.astype(bool)]
nolst = grid[grid.coastal.astype(bool) & grid.uhgpi.isna()]
a.plot(ax=ax, color='#c9dfb3', edgecolor='white', lw=0.3, zorder=2)
nolst.plot(ax=ax, color='#bdbdbd', edgecolor='white', lw=0.3, zorder=2)
outside.plot(ax=ax, facecolor='none', edgecolor='#9a9a9a', hatch='////', lw=0.3, zorder=2)
ax.legend(handles=[Patch(color='#c9dfb3', label=f'Analysis cells (n = {len(a)})'),
                   Patch(color='#bdbdbd', label=f'≤ 50 km from sea, < 18 valid seasons (n = {len(nolst)})'),
                   Patch(facecolor='none', edgecolor='#9a9a9a', hatch='////', label=f'> 50 km from sea, excluded (n = {len(outside)})')],
          loc='upper left', framealpha=0.95)
scalebar_north(ax, x=0.47, y=0.70)
save(fig, 'fig01_study_area')

# Figure 2: workflow
fig, ax = plt.subplots(figsize=(W, 4.7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.2)
ax.axis('off')


def node(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.03', fc=fc, ec='#444444', lw=0.7))
    ax.text(x, y, text, ha='center', va='center', fontsize=8)


def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#555555', lw=0.8))


C1, C2, C3 = 1.62, 5.0, 8.38
BW = 3.12
ys = [6.3, 4.85, 3.4, 1.3]
inp = ['MODIS Terra and Aqua\nday and night LST\n(8-day, 1 km)', 'GHSL built-up surface;\nWorldPop population', 'CHIRPS v2.0 rainfall\n(1981–2024)',
       'GRACE/GRACE-FO: JPL,\nCSR, GSFC; GLDAS: Noah,\nCLSM, VIC; G-REALM']
prc = ['Dec–Feb, 2001–2023:\nSen slope, Hamed–Rao,\nFDR; 2019–2023 anomaly', 'UGPI: normalised built-up\nfraction, population\ndensity, built-up growth',
       'Climate stress: % months\nbelow 1981–2010 mean\n(2000–2024)', '9 storage members;\n18 with Lake Volta\nremoved']
for y, t, u in zip(ys, inp, prc):
    node(C1, y, BW, 1.08, t, '#e8eef5')
    node(C2, y, BW, 1.08, u, '#f3efe4')
    arrow(C1 + BW / 2 + .03, y, C2 - BW / 2 - .04, y)
node(C3, 5.6, BW, 1.3, 'UHGPI: equal-weight\nmean of LST trend, LST\nanomaly, UGPI, climate\nstress (264 cells)', '#f6d0c0')
for y, yy in zip(ys[:3], (6.0, 5.6, 5.2)):
    arrow(C2 + BW / 2 + .03, y, C3 - BW / 2 - .04, yy)
node(C3, 3.45, BW, 1.3, 'Robustness: weights,\ndrop-one, Monte Carlo,\nAqua, end year, night,\ngrowth, mining, spatial null', '#ede7f3')
arrow(C3, 4.9, C3, 4.15)
node(C3, 1.3, BW, 1.08, 'Temporal context only;\nnot an index component', '#e3eedf')
arrow(C2 + BW / 2 + .03, 1.3, C3 - BW / 2 - .04, 1.3)
for x, t in ((C1, 'Input data'), (C2, 'Indicators'), (C3, 'Index and tests')):
    ax.text(x, 7.08, t, ha='center', fontweight='bold', fontsize=9)
save(fig, 'fig02_workflow')

# Figure 3: daytime LST trend
fig = plt.figure(figsize=(W, 6.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1], hspace=0.3, wspace=0.32)
ax = fig.add_subplot(gs[0, :])
basemap(ax)
norm = TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.15)
a.plot(ax=ax, column='sen_slope', cmap='RdBu_r', norm=norm, edgecolor='white', lw=0.2, zorder=2)
a[a.fdr_sig].plot(ax=ax, facecolor='none', edgecolor='k', lw=0.8, zorder=4)
cbar(fig, ax, norm, 'RdBu_r', 'Daytime LST trend (°C yr⁻¹)')
ax.legend(handles=[Patch(facecolor='none', edgecolor='k', label='FDR q < 0.05')], loc='upper left')
scalebar_north(ax)
label(ax, '(a)')
ax = fig.add_subplot(gs[1, 0])
ax.hist(a.sen_slope, bins=np.arange(-0.08, 0.2, 0.01), color='#d6604d', edgecolor='white')
ax.axvline(0, color='k', lw=0.7)
ax.axvline(a.sen_slope.median(), color='k', ls='--', lw=0.9, label=f'Median {a.sen_slope.median():+.3f}')
ax.set_xlabel('Sen slope (°C yr⁻¹)')
ax.set_ylabel('Cells')
ax.legend(frameon=False)
label(ax, '(b)')
ax = fig.add_subplot(gs[1, 1])
dom = pd.read_csv(D / 'domain_dry_season_lst.csv', index_col=0).iloc[:, 0]
ax.plot(dom.index, dom.values, '-o', ms=3, color='#b2182b')
t = dom.index.values
ax.plot(t, np.polyval(np.polyfit(t, dom.values, 1), t), color='k', lw=0.9, ls='--')
ax.set_xlabel('Dry season (year of Jan–Feb)')
ax.set_ylabel('Mean daytime LST (°C)')
label(ax, '(c)')
save(fig, 'fig03_lst_trend')

# Figure 4: recent LST and urban contrast
fig = plt.figure(figsize=(W, 6.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1], hspace=0.3, wspace=0.36)
ax = fig.add_subplot(gs[0, :])
basemap(ax)
norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=6)
a.plot(ax=ax, column='lst_anom', cmap='RdBu_r', norm=norm, edgecolor='white', lw=0.2, zorder=2)
cbar(fig, ax, norm, 'RdBu_r', 'Daytime LST anomaly (°C)')
scalebar_north(ax)
label(ax, '(a)')
ax = fig.add_subplot(gs[1, 0])
wb = pd.read_csv(TAB / 'ugpi_contrast_within_bands.csv')
x = np.arange(len(wb))
ax.bar(x - 0.2, wb['False'], 0.4, color='#92c5de', edgecolor='#333333', lw=0.4, label='Other cells')
ax.bar(x + 0.2, wb['True'], 0.4, color='#d6604d', edgecolor='#333333', lw=0.4, hatch='///', label='Top-quintile UGPI')
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(['0–10', '10–20', '20–30', '30–40', '40–50'])
ax.set_xlabel('Distance to sea (km)')
ax.set_ylabel('Mean LST anomaly (°C)')
ax.legend(frameon=False, loc='lower left')
label(ax, '(b)')
ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(a.chirps_mean_rain_mm, a.lst_anom, c=a.ugpi_urban_only, cmap='magma_r', s=9, vmin=0, vmax=0.3)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label('UGPI')
ax.set_xlabel('Mean annual rainfall (mm)')
ax.set_ylabel('LST anomaly (°C)')
label(ax, '(c)')
save(fig, 'fig04_lst_anomaly')

# Figure 5: night-time LST
fig = plt.figure(figsize=(W, 6.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1], hspace=0.3, wspace=0.32)
ax = fig.add_subplot(gs[0, :])
basemap(ax)
norm = TwoSlopeNorm(vmin=-1.6, vcenter=0, vmax=2.6)
a.plot(ax=ax, column='night_anom', cmap='RdBu_r', norm=norm, edgecolor='white', lw=0.2, zorder=2)
cbar(fig, ax, norm, 'RdBu_r', 'Night-time LST anomaly (°C)')
scalebar_north(ax)
label(ax, '(a)')
ax = fig.add_subplot(gs[1, 0])
nb = pd.read_csv(TE / 'night_ugpi_contrast_within_bands.csv')
ax.bar(x - 0.2, wb['diff_top_minus_rest'], 0.4, color='#d6604d', edgecolor='#333333', lw=0.4, label='Day')
ax.bar(x + 0.2, nb['diff'], 0.4, color='#4d4d7f', edgecolor='#333333', lw=0.4, hatch='///', label='Night')
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(['0–10', '10–20', '20–30', '30–40', '40–50'])
ax.set_xlabel('Distance to sea (km)')
ax.set_ylabel('Urban contrast (°C)')
ax.legend(frameon=False)
label(ax, '(b)')
ax = fig.add_subplot(gs[1, 1])
ad = pd.read_csv(TE / 'aqua_domain_series.csv', index_col=0)
ax.plot(ad.index, ad.terra_night - ad.terra_night.mean(), '--s', ms=3, color='#4d4d7f', label='Terra')
ax.plot(ad.index, ad.aqua_night - ad.aqua_night.mean(), '-^', ms=3, color='#1b7837', label='Aqua')
ax.axvspan(2019.5, 2023.5, color='#eeeeee', zorder=0)
ax.axhline(0, color='k', lw=0.5)
ax.set_xlim(2002.5, 2023.5)
ax.set_xlabel('Dry season (year of Jan–Feb)')
ax.set_ylabel('Night-time anomaly (°C)')
ax.legend(frameon=False, loc='upper left')
label(ax, '(c)')
save(fig, 'fig05_night_lst')

# Figure 6: UGPI and climate stress
fig, axes = plt.subplots(2, 1, figsize=(W, 6.4))
for ax, col, cmap, lab, ttl, vmax in [(axes[0], 'ugpi_urban_only', 'magma_r', 'UGPI', '(a)', 0.4),
                                      (axes[1], 'chirps_below_normal_pct', 'YlOrBr', 'Months below normal (%)', '(b)', None)]:
    basemap(ax, cities=False)
    vmin = a[col].min() if vmax is None else 0
    vm = a[col].max() if vmax is None else vmax
    a.plot(ax=ax, column=col, cmap=cmap, vmin=vmin, vmax=vm, edgecolor='white', lw=0.2, zorder=2)
    cbar(fig, ax, plt.Normalize(vmin, vm), cmap, lab)
    scalebar_north(ax)
    label(ax, ttl)
fig.subplots_adjust(hspace=0.18)
save(fig, 'fig06_ugpi_climate')

# Figure 7: bivariate warming x UGPI
BIV = {(1, 1): '#e8e8e8', (2, 1): '#e4acac', (3, 1): '#c85a5a', (1, 2): '#b0d5df', (2, 2): '#ad9ea5',
       (3, 2): '#985356', (1, 3): '#64acbe', (2, 3): '#627f8c', (3, 3): '#574249'}
fig = plt.figure(figsize=(W, 3.45))
ax = fig.add_axes([0.0, 0.0, 0.8, 1])
basemap(ax)
a['bcol'] = [BIV[(int(p), int(u))] for p, u in zip(a.bivar_trend, a.bivar_ugpi)]
a.plot(ax=ax, color=a.bcol, edgecolor='white', lw=0.2, zorder=2)
scalebar_north(ax)
lx = fig.add_axes([0.85, 0.42, 0.13, 0.26])
for (p, u), c in BIV.items():
    lx.add_patch(plt.Rectangle((u - 1, p - 1), 1, 1, color=c))
lx.set_xlim(0, 3)
lx.set_ylim(0, 3)
lx.set_xticks([])
lx.set_yticks([])
lx.set_xlabel('UGPI tertile →')
lx.set_ylabel('Warming tertile →', labelpad=2)
save(fig, 'fig07_bivariate_trend_ugpi')

# Figure 8: UHGPI
CL = ['Very low', 'Low', 'Moderate', 'High', 'Very high']
PAL = ['#d1e5f0', '#92c5de', '#f7f7f7', '#f4a582', '#b2182b']
fig, axes = plt.subplots(2, 1, figsize=(W, 6.4))
ax = axes[0]
basemap(ax)
for cl, c in zip(CL, PAL):
    a[a.uhgpi_class == cl].plot(ax=ax, color=c, edgecolor='#999999', lw=0.2, zorder=2)
ax.legend(handles=[Patch(facecolor=c, edgecolor='#777777', label=cl) for cl, c in zip(CL, PAL)], title='UHGPI quintile', loc='upper left', ncol=2)
scalebar_north(ax)
label(ax, '(a)')
ax = axes[1]
basemap(ax, cities=False)
cmap = ListedColormap(['#f7f7f7', '#fde0c5', '#f4a582', '#d6604d', '#8b1a1a'])
nrm = BoundaryNorm([0, 0.1, 0.5, 0.75, 0.9, 1.0001], cmap.N)
a.plot(ax=ax, column='p_top_quintile_mc', cmap=cmap, norm=nrm, edgecolor='#999999', lw=0.2, zorder=2)
cbar(fig, ax, nrm, cmap, 'Share of weightings in top quintile', ticks=[0, 0.1, 0.5, 0.75, 0.9, 1.0])
scalebar_north(ax)
label(ax, '(b)')
fig.subplots_adjust(hspace=0.18)
save(fig, 'fig08_uhgpi')

# Figure 9: regional storage and dry-season heat
s = pd.read_csv(TE / 'djf_sgwa_ensemble.csv', index_col=0)
tg = pd.read_csv(TE / 'temporal_gwsa_lst_ensemble.csv')
fig = plt.figure(figsize=(W, 5.3))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.36, wspace=0.3)
ax = fig.add_subplot(gs[0, :])
yrs = s.index.values
ax.bar(yrs, s.lst_anom, color='#f4a582', edgecolor='#a0522d', lw=0.4, width=0.65, label='Dry-season LST anomaly (°C)')
for col, lo, hi, c, ls, mk_, lab in (('uncorrected', 'unc_min', 'unc_max', '#2166ac', '-', 'o', 'Storage, Lake Volta included'),
                                     ('lake_corrected', 'cor_min', 'cor_max', '#1b7837', '--', 's', 'Storage, Lake Volta removed')):
    y = s.reindex(range(2003, 2024))
    ax.fill_between(y.index, y[lo], y[hi], color=c, alpha=0.14, lw=0)
    ax.plot(y.index, y[col], ls=ls, marker=mk_, ms=3, color=c, label=lab)
ax.axhline(0, color='k', lw=0.5)
ax.set_xlim(2002.4, 2023.6)
ax.set_xticks(range(2003, 2024, 2))
ax.set_ylabel('Standardised storage\nanomaly; LST (°C)')
ax.set_xlabel('Dry season (year of Jan–Feb)')
ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=7.6, handlelength=1.8, columnspacing=1.0)
label(ax, '(a)')


def detr(v):
    t = v.index.values.astype(float)
    return v - np.polyval(np.polyfit(t, v.values, 1), t)


def rtext(setname, form):
    r = tg[(tg.member == 'ensemble mean') & (tg.set == setname) & (tg.pairing == 'Dec-Feb') & (tg.form == form)].iloc[0]
    return f'r = {r.r:+.2f}, p = {r.p:.2f}'.replace('-', '−')


for k, (form, ttl) in enumerate((('raw', '(b)'), ('detrended', '(c)'))):
    ax = fig.add_subplot(gs[1, k])
    for col, setname, c, m in (('uncorrected', 'uncorrected', '#2166ac', 'o'), ('lake_corrected', 'lake-corrected', '#1b7837', 's')):
        xv, yv = (s[col], s.lst_anom) if form == 'raw' else ((detr(s[col]) - detr(s[col]).mean()) / detr(s[col]).std(), detr(s.lst_anom))
        ax.scatter(xv, yv, s=13, color=c, marker=m, alpha=0.85)
    ax.axhline(0, color='k', lw=0.4)
    ax.axvline(0, color='k', lw=0.4)
    ax.set_xlabel('Storage anomaly' + (' (detrended)' if form == 'detrended' else ''))
    ax.set_ylabel('LST anomaly (°C)' if form == 'raw' else 'Detrended LST (°C)')
    ax.legend(handles=[Line2D([], [], color='#2166ac', marker='o', ls='none', ms=4, label=f"Lake included: {rtext('uncorrected', form)}"),
                       Line2D([], [], color='#1b7837', marker='s', ls='none', ms=4, label=f"Lake removed: {rtext('lake-corrected', form)}")],
              loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=True, framealpha=0.85, edgecolor='none', fontsize=7.6, handletextpad=0.2, borderpad=0.2)
    ax.set_ylim(top=ax.get_ylim()[1] + 0.6)
    label(ax, ttl)
save(fig, 'fig09_groundwater_context')

# Supplementary Figure S1: end-year sensitivity
dd = pd.read_csv(TE / 'trend_end_year_sensitivity.csv')
fig, axes = plt.subplots(1, 2, figsize=(W, 2.7))
for band, c, ls, m in (('day', '#b2182b', '-', 'o'), ('night', '#4d4d7f', '--', 's')):
    k = dd[dd.band == band]
    axes[0].plot(k.end, k.median_slope, ls=ls, marker=m, color=c, label=band.capitalize())
    axes[1].plot(k.end, k.fdr_cells, ls=ls, marker=m, color=c, label=band.capitalize())
for ax, yl, ttl in ((axes[0], 'Median Sen slope (°C yr⁻¹)', '(a)'), (axes[1], 'Cells with FDR q < 0.05', '(b)')):
    ax.set_xticks(range(2019, 2024))
    ax.set_xlabel('Final dry season of the record')
    ax.set_ylabel(yl)
    ax.legend(frameon=False)
    label(ax, ttl)
axes[0].set_ylim(0, 0.07)
plt.tight_layout()
save(fig, 'figS1_end_year_sensitivity')

# Supplementary Figure S2: Terra and Aqua
ad = pd.read_csv(TE / 'aqua_domain_series.csv', index_col=0)
fig = plt.figure(figsize=(W, 4.8))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.3)
for k, (band, ttl) in enumerate((('day', '(a)'), ('night', '(b)'))):
    ax = fig.add_subplot(gs[0, k])
    ax.plot(ad.index, ad[f'terra_{band}'] - ad[f'terra_{band}'].mean(), '--s', ms=3, color='#b2182b', label='Terra')
    ax.plot(ad.index, ad[f'aqua_{band}'] - ad[f'aqua_{band}'].mean(), '-^', ms=3, color='#1b7837', label='Aqua')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvspan(2019.5, 2023.5, color='#eeeeee', zorder=0)
    ax.set_xlim(2002.5, 2023.5)
    ax.set_xlabel('Dry season')
    ax.set_ylabel(f'{band.capitalize()} LST anomaly (°C)')
    ax.legend(frameon=False, loc='upper left')
    label(ax, ttl)
ax = fig.add_subplot(gs[1, :])
for band, c, ls, m in (('day', '#b2182b', '-', 'o'), ('night', '#4d4d7f', '--', 's')):
    d_ = ad[f'terra_{band}'] - ad[f'aqua_{band}']
    ax.plot(d_.index, d_ - d_.loc[:2019].mean(), ls=ls, marker=m, ms=3, color=c, label=band.capitalize())
ax.axhline(0, color='k', lw=0.5)
ax.axvspan(2019.5, 2023.5, color='#eeeeee', zorder=0)
ax.set_xlim(2002.5, 2023.5)
ax.set_xticks(range(2003, 2024, 2))
ax.set_xlabel('Dry season (year of Jan–Feb)')
ax.set_ylabel('Terra minus Aqua,\nrelative to 2003–2019 (°C)')
ax.legend(frameon=False, ncol=2, loc='upper left')
label(ax, '(c)')
save(fig, 'figS2_terra_aqua')
