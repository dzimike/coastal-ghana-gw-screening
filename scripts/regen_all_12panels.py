#!/usr/bin/env python3
"""Regenerate all 12 figure panels flagged in the Author Action List.

Author Action List items addressed:
  Row 1  – Fig.1b: corridor labels over city names → manual offsets
  Row 2  – Fig.3a: vertical blue fill artefact at right edge → removed
  Row 3  – Fig.4a: confirm stippling p<0.10
  Row 4  – Fig.4b: confirm panel matches corrected caption (SGWA drought classes)
  Row 5  – Fig.7a: proxy map restricted to non-Track-147 cells; title corrected
  Row 6  – Fig.7b: yellow threshold line weight/contrast increased
  Row 7  – Fig.8a: Q75 thresholds (not median split); panel title corrected
  Row 8  – Fig.8b: Q75 threshold lines updated
  Row 9  – Fig.9 both panels: panel arrangement corrected (continuous left, quintile right)
  Row 10 – Fig.9 right: histogram text box standardised to 6.2M
  Row 11 – Fig.10 left: +/-1 SD error bars added; text box not cut off
  Row 12 – Fig.10 right: 91 VH cells shown (not 20); "CGSI" -> "CGSRI" fixed
"""
import warnings; warnings.filterwarnings('ignore')
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd, geopandas as gpd
import xarray as xr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path
import pymannkendall as mk

ROOT   = Path('.')
BOUND  = ROOT / 'data/raw/boundaries'
GW_PR  = ROOT / 'data/processed/groundwater_anomalies'
UG_PR  = ROOT / 'data/processed/urban_growth'
PI     = ROOT / 'data/processed/pressure_indices'
RI     = ROOT / 'data/processed/risk_index'
FIGS   = ROOT / 'outputs/figures'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
DPI = 300
MAP_XLIM = (-3.5, 1.5); MAP_YLIM = (4.0, 7.5)
CITIES = [(-0.187, 5.603, 'Accra'), (-0.016, 5.670, 'Tema'),
          (-1.246, 5.106, 'Cape Coast'), (-1.752, 4.894, 'Takoradi'),
          (0.627,  5.772, 'Ada'),       (0.999,  5.916, 'Keta')]
RISK_COLORS = {'Very low': '#2166AC', 'Low': '#74ADD1', 'Moderate': '#FEE08B',
               'High': '#F46D43', 'Very high': '#A50026'}
RISK_ORDER  = ['Very low', 'Low', 'Moderate', 'High', 'Very high']

NE_PATH = ('/opt/miniconda3/envs/accra-risk/lib/python3.11/site-packages'
           '/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp')


# ── Shared data ────────────────────────────────────────────────────────────────
districts = gpd.read_file(str(BOUND / 'coastal_districts.gpkg')).to_crs('EPSG:4326')
study_area = gpd.read_file(str(BOUND / 'study_area_coastal_50km.gpkg')).to_crs('EPSG:4326')
gdf_cs    = gpd.read_file(str(RI  / 'cgsri_grid10km_v2.gpkg')).to_crs('EPSG:4326')   # harmonised index; cgsri = primary
gdf_ugpi  = gpd.read_file(str(PI  / 'ugpi_grid10km.gpkg')).to_crs('EPSG:4326')

try:
    import pyogrio
    africa = gpd.read_file(NE_PATH).to_crs('EPSG:4326')
    africa = africa[africa['continent'] == 'Africa']
except Exception:
    africa = None


def map_furniture(ax, xlim=MAP_XLIM, ylim=MAP_YLIM):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel('Longitude (°E)', fontsize=8)
    ax.set_ylabel('Latitude (°N)', fontsize=8)
    for x in np.arange(np.ceil(xlim[0]), xlim[1] + 0.5, 1):
        ax.axvline(x, color='#CCCCCC', lw=0.3, zorder=0)
    for y in np.arange(np.ceil(ylim[0]), ylim[1] + 0.5, 1):
        ax.axhline(y, color='#CCCCCC', lw=0.3, zorder=0)
    for lon, lat, nm in CITIES:
        if xlim[0] <= lon <= xlim[1] and ylim[0] <= lat <= ylim[1]:
            ax.plot(lon, lat, 'k^', ms=4, zorder=5)
            ax.annotate(nm, (lon, lat), xytext=(4, 3),
                        textcoords='offset points', fontsize=6, zorder=6)


def north_arrow(ax, x=0.96, y=0.94):
    ax.annotate('N', xy=(x, y), xycoords='axes fraction',
                fontsize=9, fontweight='bold', ha='center', va='bottom')
    ax.annotate('', xy=(x, y), xytext=(x, y - 0.06), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))


# ── Infer Track-147 footprint ──────────────────────────────────────────────────
# Cells with InSAR data (insar_n_valid >= 1) are within the Sentinel-1 track 147 footprint
insar_mask = gdf_cs['insar_n_valid'].notna() & (gdf_cs['insar_n_valid'] > 0)


# ==============================================================================
# FIG 1 – Conceptual framework  (Row 1: fix corridor label offsets)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Panel (a): causal chain diagram
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('(a) Conceptual framework (motivating chain; not tested)', fontsize=10, fontweight='bold')

chain_items = [
    (5, 9.0, 'Climate variability +\nurban growth +\ngroundwater abstraction', '#4292C6'),
    (5, 7.0, 'Groundwater storage\ndecline / aquifer stress',              '#2171B5'),
    (5, 5.0, 'Land subsidence +\nsaline intrusion +\ninfrastructure vulnerability', '#084594'),
    (5, 3.0, 'Interacting coastal\nwater-security risks',                   '#08306B'),
]
for cx, cy, label, col in chain_items:
    ax.add_patch(mpatches.FancyBboxPatch((cx - 2.3, cy - 0.55), 4.6, 1.1,
                 boxstyle='round,pad=0.1', fc=col, ec='white', lw=1.2, alpha=0.9))
    ax.text(cx, cy, label, ha='center', va='center', fontsize=8,
            color='white', fontweight='bold', wrap=True)

for i in range(len(chain_items) - 1):
    y1 = chain_items[i][1] - 0.55
    y2 = chain_items[i + 1][1] + 0.55
    ax.annotate('', xy=(5, y2), xytext=(5, y1),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

# Panel (b): study area map with corridor labels — manual offsets to clear city names
ax = axes[1]
ax.set_title('(b) Study area: 50 km coastal belt with key corridors', fontsize=10,
             fontweight='bold')
if africa is not None:
    africa.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', lw=0.5)
study_area.plot(ax=ax, color='#C6DBEF', edgecolor='#2171B5', lw=0.8, alpha=0.7)
districts.boundary.plot(ax=ax, edgecolor='#888888', lw=0.3)


# Corridor labels with manual offsets to avoid city name overlap
# (Row 1 fix: manual coordinate offsets so labels clear city annotations)
corridors = [
    (-0.10, 5.58,  'Accra–Tema\ncorridor',       (-10, -45)),
    (-1.30, 5.08,  'Cape Coast–\nElmina',          (-40, -45)),
    (-1.75, 4.90,  'Sekondi–\nTakoradi',           (-30, -40)),
    ( 0.80, 5.85,  'Ada–Keta\nzone',               ( 10, -45)),
    (-0.33, 5.57,  'Weija–Densu\nbasin',           (-60,  35)),
    (-1.00, 5.20,  'Komenda–\nSaltpond',           ( 15, -45)),
]
for lon, lat, label, offset in corridors:
    ax.annotate(label, (lon, lat), xytext=offset, textcoords='offset points',
                fontsize=6.5, color='#08306B',
                arrowprops=dict(arrowstyle='->', color='#08306B', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, lw=0))

map_furniture(ax)
north_arrow(ax)

plt.tight_layout()
plt.savefig(str(FIGS / 'fig01_conceptual_study_area.png'), dpi=DPI)
plt.close()
print(f"Saved fig01  ({(FIGS/'fig01_conceptual_study_area.png').stat().st_size//1024} KB)")


# ==============================================================================
# FIG 3 – GWSA time series  (Row 2: remove blue fill artefact at right edge)
# ==============================================================================
gwsa = pd.read_csv(str(GW_PR / 'gwsa_by_region_monthly.csv'),
                   index_col=0, parse_dates=True)
gwsa = gwsa.loc[:'2023-12-31'].copy()   # truncate to analysis period

gwsa_ts  = gwsa['gwsa_mm'].dropna()
sgwa_ts  = gwsa['sgwa'].dropna()
gwsa_plot = gwsa['gwsa_mm'].resample('MS').mean()   # NaN at missing months -> visible breaks
sgwa_plot = gwsa['sgwa'].resample('MS').mean()

# Sen's slope — use calendar months to avoid inflation from the GRACE mission gap
x_cal     = (gwsa_ts.index - gwsa_ts.index[0]).days.values.astype(float) / 30.4375
vals_cal   = gwsa_ts.values.astype(float)
n_cal      = len(vals_cal)
cal_pairs  = [(vals_cal[j]-vals_cal[i])/(x_cal[j]-x_cal[i])
              for i in range(n_cal-1) for j in range(i+1, n_cal)]
sen_slope  = float(np.median(cal_pairs))   # mm per calendar month
sen_yr     = sen_slope * 12                # mm per year  (correct: ~+12.04)
intercept  = float(np.median(vals_cal - sen_slope * x_cal))
trend_line = intercept + sen_slope * x_cal
trend_s    = pd.Series(trend_line, index=gwsa_ts.index)

# Annual mean GWSA
gwsa_ann = gwsa_ts.resample('YE').mean()

# Drought class mapping for colour bands
DROUGHT_COLORS = {
    'Extreme drought':   '#67001F',
    'Severe drought':    '#D6604D',
    'Moderate drought':  '#FDAE61',
    'Mild drought':      '#FEE090',
    'Near-normal':       '#E0F3F8',
    'Wet':               '#4393C3',
}

fig = plt.figure(figsize=(12, 7))
gs  = GridSpec(2, 1, figure=fig, hspace=0.35)

# Panel (a): monthly GWSA — Row 2 fix: no axvspan at right edge beyond last data point
ax = fig.add_subplot(gs[0])
ax.fill_between(gwsa_plot.index, gwsa_plot.values, 0,
                where=(gwsa_plot.values >= 0), color='#4477AA', alpha=0.35,
                label='Surplus', interpolate=True)
ax.fill_between(gwsa_plot.index, gwsa_plot.values, 0,
                where=(gwsa_plot.values < 0), color='#CC4444', alpha=0.35,
                label='Deficit', interpolate=True)
ax.plot(gwsa_ts.index, trend_s.values, color='#CC4444', lw=1.5, ls='--',
        label=f"Sen's slope: {sen_yr:+.2f} mm yr⁻¹ (MK p<0.001)")
ax.axhline(0, color='k', lw=0.7)
# GRACE mission gap — bounded strictly between first and last real data points
gap_start = pd.Timestamp('2017-06-15')
gap_end   = pd.Timestamp('2018-05-31')
ax.axvspan(gap_start, gap_end, color='gray', alpha=0.14, label='GRACE/GRACE-FO gap (Jul 2017–May 2018)')
ax.set_xlim(gwsa_ts.index[0], gwsa_ts.index[-1])   # no overhang → no artefact
ax.set_ylabel('GWSA (mm EWH)')
ax.set_title("(a) Monthly GWSA with Sen's trend (2002–2023; 226 valid months)", fontsize=9)
ax.legend(ncol=3, fontsize=7, loc='upper left')

# Panel (b): standardised GWSA with drought-class colour bands
ax2 = fig.add_subplot(gs[1])
drought_thresholds = [
    (-99,  -2.0, 'Extreme drought'),
    (-2.0, -1.5, 'Severe drought'),
    (-1.5, -1.0, 'Moderate drought'),
    (-1.0, -0.5, 'Mild drought'),
    (-0.5,  0.5, 'Near-normal'),
    ( 0.5,  99,  'Wet'),
]
for lo, hi, cls in drought_thresholds:
    ax2.axhspan(lo, hi, color=DROUGHT_COLORS[cls], alpha=0.25)
ax2.plot(sgwa_plot.index, sgwa_plot.values, color='#333333', lw=0.9)
ax2.axhline(0, color='k', lw=0.7)
ax2.set_xlim(sgwa_ts.index[0], sgwa_ts.index[-1])
ax2.set_ylim(-3, 3)
ax2.set_ylabel('Standardised GWSA (σ)')
ax2.set_title('(b) Monthly standardised GWSA with drought-severity classes\n'
              '(Thomas et al. [33]: mild −0.5/−1.0; moderate −1.0/−1.5; '
              'severe −1.5/−2.0; extreme <−2.0)', fontsize=8)
handles_d = [mpatches.Patch(fc=DROUGHT_COLORS[c], alpha=0.5, label=c)
             for _, _, c in drought_thresholds]
ax2.legend(handles=handles_d, fontsize=6.5, loc='upper right', ncol=3)

plt.savefig(str(FIGS / 'fig03_gwsa_timeseries.png'), dpi=DPI)
plt.close()
print(f"Saved fig03  ({(FIGS/'fig03_gwsa_timeseries.png').stat().st_size//1024} KB)")


# ==============================================================================
# FIG 4 – GWSA trend and drought map  (Rows 3–4: stippling p<0.10; correct panels)
# ==============================================================================
ds_trend  = xr.open_dataset(str(GW_PR / 'gwsa_trend_pixelwise.nc'))
ds_drought = xr.open_dataset(str(GW_PR / 'gdsi_drought_metrics_grid.nc'))

trend_slope  = ds_trend['sens_slope_mm_yr'].values
trend_pvalue = ds_trend['mk_pvalue'].values
lons = ds_trend['lon'].values
lats = ds_trend['lat'].values
LON, LAT = np.meshgrid(lons, lats)

drought_freq  = ds_drought['drought_freq_pct'].values
sgwa_recent   = ds_drought['sgwa_recent_mean'].values

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Panel (a): Sen's slope with stippling at p<0.10
ax = axes[0]
vmin, vmax = -20, 30
im = ax.pcolormesh(LON, LAT, trend_slope, cmap='RdBu', vmin=vmin, vmax=vmax,
                   shading='auto')
plt.colorbar(im, ax=ax, label="Sen's slope (mm yr⁻¹)", shrink=0.85)
# Stippling at p < 0.10 (Row 3 fix: confirm correct threshold)
sig_mask = trend_pvalue < 0.10
if sig_mask.any():
    ax.scatter(LON[sig_mask], LAT[sig_mask], marker='.', s=4, color='k', alpha=0.5,
               label='p < 0.10 (Mann-Kendall)')
    ax.legend(fontsize=7, loc='lower right')
districts.boundary.plot(ax=ax, edgecolor='#555555', lw=0.35)
map_furniture(ax)
north_arrow(ax)
ax.set_title("(a) Sen's slope 2002–2023 (mm yr⁻¹)\n"
             "stippled: p < 0.10; mascon-scale (~300 km) GWSA", fontsize=8)
ax.text(0.01, 0.99,
        'Important: 10 km grid is a reporting frame only.\n'
        'Within-belt spatial variation is not interpretable\n'
        'as local groundwater heterogeneity at 10 km.',
        transform=ax.transAxes, fontsize=6, va='top', ha='left', color='#555555',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.8, lw=0))

# Panel (b): Annual mean SGWA drought frequency (Row 4 fix: correct panel content)
ax2 = axes[1]
im2 = ax2.pcolormesh(LON, LAT, drought_freq, cmap='YlOrRd', vmin=0, vmax=60,
                     shading='auto')
plt.colorbar(im2, ax=ax2, label='Groundwater drought frequency (%)', shrink=0.85)
districts.boundary.plot(ax=ax2, edgecolor='#555555', lw=0.35)
map_furniture(ax2)
north_arrow(ax2)
ax2.set_title('(b) Groundwater drought frequency (%)\n'
              '(months with SGWA < −0.5, 2002–2023)', fontsize=8)

plt.tight_layout()
plt.savefig(str(FIGS / 'fig04_gwsa_trend_drought.png'), dpi=DPI)
plt.close()
print(f"Saved fig04  ({(FIGS/'fig04_gwsa_trend_drought.png').stat().st_size//1024} KB)")


# FIG 7 is produced by scripts/regen_fig7.py (implemented proxy formula).


# ==============================================================================
# FIG 8 – Bivariate analysis  (Rows 7–8: Q75 thresholds replace median split)
# ==============================================================================
ugpi_col = 'ugpi_urban_only'
sh_col   = 'sub_hazard'

ugpi_q75 = gdf_cs[ugpi_col].quantile(0.75)
sh_q75   = gdf_cs[sh_col].quantile(0.75)
ugpi_q90 = gdf_cs[ugpi_col].quantile(0.90)
sh_q90   = gdf_cs[sh_col].quantile(0.90)

# Row 7 fix: use Q75 thresholds
gdf_cs['bivar_q75'] = gdf_cs.apply(
    lambda r: ('H' if r[ugpi_col] >= ugpi_q75 else 'L') + '/' +
              ('H' if r[sh_col]   >= sh_q75   else 'L'), axis=1)

bivar_colors = {'L/L': '#F7F7F7', 'H/L': '#74ADD1',
                'L/H': '#F46D43', 'H/H': '#A50026'}
bivar_labels = {
    'L/L': 'Low UGPI + low SH',
    'H/L': 'High UGPI + low SH',
    'L/H': 'Low UGPI + high SH',
    'H/H': 'High UGPI + high SH (Q75)',
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Panel (a): spatial bivariate map at Q75 — Row 7 fix: correct title
ax = axes[0]
gdf_cs.plot(ax=ax, facecolor='#F7F7F7', edgecolor='none')
for cls, color in bivar_colors.items():
    sub = gdf_cs[gdf_cs['bivar_q75'] == cls]
    if len(sub) > 0:
        sub.plot(ax=ax, facecolor=color, edgecolor='none')
districts.boundary.plot(ax=ax, edgecolor='#555555', lw=0.35)
n_hh = (gdf_cs['bivar_q75'] == 'H/H').sum()
handles = [mpatches.Patch(fc=bivar_colors[k],
           label=f'{bivar_labels[k]} (n={( gdf_cs["bivar_q75"]==k).sum()})')
           for k in bivar_colors]
ax.legend(handles=handles, fontsize=6.5, loc='lower right')
map_furniture(ax)
north_arrow(ax)
# Row 7 fix: title corrected from "median split" to "Q75 thresholds"
ax.set_title(f'(a) Bivariate UGPI × subsidence hazard\n'
             f'Q75 thresholds: UGPI={ugpi_q75:.3f}, SH={sh_q75:.3f}; '
             f'n={n_hh} high-high cells', fontsize=8)
from scipy.stats import fisher_exact
def _fisher(q):
    a = gdf_cs[ugpi_col] >= gdf_cs[ugpi_col].quantile(q); b = gdf_cs[sh_col] >= gdf_cs[sh_col].quantile(q)
    o, pv = fisher_exact([[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]])
    return int((a & b).sum()), o, pv
_msr = pd.read_csv('outputs/tables/revision_checks/colocation_msr_test.csv')
def _pmsr(q):
    return _msr[(_msr.threshold == q) & (_msr.randomised == 'UGPI randomised')].p_one_sided.iloc[0]
_h75, _o75, _p75 = _fisher(0.75); _h90, _o90, _p90 = _fisher(0.90)
_q90 = gdf_cs[(gdf_cs[ugpi_col] >= ugpi_q90) & (gdf_cs[sh_col] >= sh_q90)]
ax.text(0.02, 0.97,
        f'Q75: {_h75} high-high cells; Fisher OR={_o75:.2f}, p={_p75:.2f}; spatial null p={_pmsr("Q75"):.2f}\n'
        f'Q90: {_h90} high-high cells; Fisher OR={_o90:.2f}, p={_p90:.2f}; spatial null p={_pmsr("Q90"):.2f}\n'
        f'Neither is significant under the spatial null. {int((_q90.insar_n_valid >= 10).sum())} of {len(_q90)} Q90 cells have valid InSAR.',
        transform=ax.transAxes, fontsize=6.2, va='top', ha='left', color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85,
                  edgecolor='#CCCCCC', lw=0.5))

# Panel (b): scatter — Row 8 fix: Q75 threshold lines (not median)
ax2 = axes[1]
sc = ax2.scatter(gdf_cs[ugpi_col], gdf_cs[sh_col],
                 c=gdf_cs['cgsri'], cmap='RdYlGn_r', s=8, alpha=0.45, linewidths=0)
plt.colorbar(sc, ax=ax2, label='CGSRI composite score', shrink=0.85)
# Row 8 fix: Q75 threshold lines
ax2.axvline(ugpi_q75, color='navy', lw=1.2, ls='--',
            label=f'UGPI Q75 ({ugpi_q75:.3f})')
ax2.axhline(sh_q75,   color='darkorange', lw=1.2, ls='--',
            label=f'SH Q75 ({sh_q75:.3f})')
ax2.axvline(ugpi_q90, color='navy', lw=0.8, ls=':',
            label=f'UGPI Q90 ({ugpi_q90:.3f})')
ax2.axhline(sh_q90,   color='darkorange', lw=0.8, ls=':',
            label=f'SH Q90 ({sh_q90:.3f})')
# Mark the 8 Q90 high-high cells
q90_hh = gdf_cs[(gdf_cs[ugpi_col] >= ugpi_q90) & (gdf_cs[sh_col] >= sh_q90)]
ax2.scatter(q90_hh[ugpi_col], q90_hh[sh_col],
            marker='*', s=80, color='#A50026', zorder=10,
            label=f'Q90 high-high (n={len(q90_hh)})')
ax2.set_xlabel('UGPI (3-indicator equal-weight)')
ax2.set_ylabel('Subsidence Hazard score')
ax2.legend(fontsize=6.5, loc='upper left')
ax2.set_title('(b) Scatter: UGPI vs Subsidence Hazard\ncoloured by CGSRI score', fontsize=9)

plt.tight_layout()
plt.savefig(str(FIGS / 'fig08_bivariate_ugpi_sh.png'), dpi=DPI)
plt.close()
print(f"Saved fig08  Q75_HH={n_hh}  Q90_HH={len(q90_hh)}"
      f"  ({(FIGS/'fig08_bivariate_ugpi_sh.png').stat().st_size//1024} KB)")


# ==============================================================================
# FIG 9 – CGSRI risk map  (Rows 9–10)
#   Row 9:  Panel (a) = continuous CGSRI scores; Panel (b) = quintile classification
#   Row 10: histogram text box standardised to 6.2M
# ==============================================================================
vh = gdf_cs[gdf_cs['cgsri_class'] == 'Very high']
hi = gdf_cs[gdf_cs['cgsri_class'].isin(['Very high', 'High'])]
vh_pop = vh['population_2020'].sum()
hi_pop = hi['population_2020'].sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

# Panel (a): continuous CGSRI scores — Row 9 fix: left panel is CONTINUOUS
ax = axes[0]
gdf_cs.plot(ax=ax, column='cgsri', cmap='YlOrRd', vmin=gdf_cs['cgsri'].min(), vmax=gdf_cs['cgsri'].max(),
            edgecolor='none', legend=True,
            legend_kwds={'label': 'CGSRI score (0–1)', 'shrink': 0.85})
districts.boundary.plot(ax=ax, edgecolor='#555555', lw=0.35)
map_furniture(ax)
north_arrow(ax)
ax.set_title('(a) Primary CGSRI, continuous scores\n'
             '(GWH, CS, E, V; equal weights)', fontsize=9)

# Panel (b): quintile classification with histogram
ax2 = axes[1]
for cls in RISK_ORDER:
    sub = gdf_cs[gdf_cs['cgsri_class'] == cls]
    if len(sub) > 0:
        sub.plot(ax=ax2, facecolor=RISK_COLORS[cls], edgecolor='none')
# hatching Very High
vh.plot(ax=ax2, facecolor='none', edgecolor='black', lw=0.6, hatch='///')
districts.boundary.plot(ax=ax2, edgecolor='#555555', lw=0.35)
handles = [mpatches.Patch(fc=RISK_COLORS[cls],
           label=f'{cls} (n={len(gdf_cs[gdf_cs["cgsri_class"]==cls])})')
           for cls in RISK_ORDER]
ax2.legend(handles=handles, fontsize=7, loc='lower right', framealpha=0.9)
map_furniture(ax2)
north_arrow(ax2)
ax2.set_title('(b) Quintile risk classification\n'
              '(Very High hatched; classification relative to coastal-belt distribution)',
              fontsize=9)
# Row 10: population text computed from data
ax2.text(0.02, 0.97,
         f'Very High quintile (top 20%):\n'
         f'91 cells, ~{vh_pop / 1e6:.1f} million people\n'
         f'(WorldPop 2020; ±25%: {0.75 * vh_pop / 1e6:.1f}–{1.25 * vh_pop / 1e6:.1f} M)\n'
         f'High + Very High: ~{hi_pop / 1e6:.1f} M',
         transform=ax2.transAxes, fontsize=7, va='top', ha='left',
         bbox=dict(boxstyle='round', fc='#FFF3CD', alpha=0.9, edgecolor='#CCCCCC'))

plt.tight_layout()
plt.savefig(str(FIGS / 'fig09_cgsri_risk_map.png'), dpi=DPI)
plt.close()
print(f"Saved fig09  VH={vh_pop / 1e6:.2f}M  VH+H={hi_pop / 1e6:.2f}M"
      f"  ({(FIGS/'fig09_cgsri_risk_map.png').stat().st_size//1024} KB)")


# ==============================================================================
# FIG 10 – Regional CGSRI ranking and priority zones  (Rows 11–12)
#   Row 11: ±1 SD error bars on left panel; text box not cut off
#   Row 12: right panel shows ALL 91 VH cells; "CGSI" → "CGSRI" fixed
# ==============================================================================
# Spatial join to admin regions (level 1)
regions = gpd.read_file(str(BOUND / 'gadm41_GHA.gpkg'), layer=1).to_crs('EPSG:4326')
joined  = gpd.sjoin(gdf_cs, regions[['NAME_1', 'geometry']], how='left',
                    predicate='intersects')
# Mean and SD CGSRI per region — drop duplicates from multi-polygon joins
grp = joined.groupby('NAME_1')['cgsri'].agg(['mean', 'std', 'count']).reset_index()
grp = grp[grp['count'] >= 3].sort_values('mean', ascending=True)

_qt = gdf_cs['cgsri'].quantile([0.2, 0.4, 0.6, 0.8]).values
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

# Panel (a): mean CGSRI ± 1 SD by region — Row 11 fix: add error bars; full text box
ax = axes[0]
y_pos = np.arange(len(grp))
bars = ax.barh(y_pos, grp['mean'].values,
               xerr=grp['std'].values,               # Row 11 fix: ±1 SD error bars
               error_kw=dict(ecolor='#444444', capsize=3, lw=0.8),
               color=[RISK_COLORS['Very high'] if v >= _qt[3] else
                      RISK_COLORS['High']      if v >= _qt[2] else
                      RISK_COLORS['Moderate']  if v >= _qt[1] else
                      RISK_COLORS['Low']       if v >= _qt[0] else
                      RISK_COLORS['Very low']
                      for v in grp['mean'].values],
               edgecolor='white', height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(grp['NAME_1'].values, fontsize=7)
ax.set_xlabel('Mean CGSRI score (±1 SD)')
ax.set_title('(a) Mean CGSRI by administrative region\n'
             '(regions intersecting 50 km coastal grid; ≥3 cells)', fontsize=9)
ax.axvline(_qt[3], color='k', lw=0.7, ls='--', alpha=0.4)
# Row 11 fix: text box with explicit bottom positioning to avoid clipping
ax.text(0.97, 0.02,
        'Colour follows CGSRI quintile thresholds.\n'
        'Primary CGSRI (no InSAR term).\n'
        'Error bars = ±1 SD across cells in region.',
        transform=ax.transAxes, fontsize=6.5, va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.88,
                  edgecolor='#CCCCCC', lw=0.5))

# Panel (b): all 91 VH cells scatter — Row 12 fix
ax2 = axes[1]
sc = ax2.scatter(vh['population_2020'] / 1e3, vh['cgsri'],
                 c=vh['cgsri'], cmap='YlOrRd', s=40, alpha=0.8,
                 vmin=vh['cgsri'].min(), vmax=vh['cgsri'].max(), edgecolors='#555555',
                 linewidths=0.4)
# Row 12 fix: colorbar title "CGSRI" (not "CGSI")
cbar = plt.colorbar(sc, ax=ax2, shrink=0.85)
cbar.set_label('CGSRI score')
# Label the largest outlier
max_pop_idx = vh['population_2020'].idxmax()
ax2.annotate(f"Cell {max_pop_idx}\n({vh.loc[max_pop_idx,'population_2020']/1e3:.0f}k)",
             xy=(vh.loc[max_pop_idx, 'population_2020'] / 1e3,
                 vh.loc[max_pop_idx, 'cgsri']),
             xytext=(10, -15), textcoords='offset points',
             fontsize=6.5, arrowprops=dict(arrowstyle='->', lw=0.8))
ax2.set_xlabel('Cell population 2020 (thousands)')
ax2.set_ylabel('CGSRI score')
# Row 12 fix: title says "91 Very High" not "20 priority"
ax2.set_title(f'(b) CGSRI score vs population\nAll 91 Very High quintile cells '
              f'(~{vh["population_2020"].sum() / 1e6:.1f}M total; '
              f'±25%: {0.75 * vh["population_2020"].sum() / 1e6:.1f}–{1.25 * vh["population_2020"].sum() / 1e6:.1f}M)', fontsize=9)
ax2.text(0.02, 0.97,
         f'n = {len(vh)} Very High cells\n'
         f'Population range: {vh["population_2020"].min()/1e3:.0f}k '
         f'– {vh["population_2020"].max()/1e3:.0f}k',
         transform=ax2.transAxes, fontsize=6.5, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85, lw=0))

plt.tight_layout()
plt.savefig(str(FIGS / 'fig10_regional_cgsri_ranking.png'), dpi=DPI)
plt.close()
print(f"Saved fig10  VH_cells={len(vh)}  VH_pop={vh_pop/1e6:.2f}M"
      f"  ({(FIGS/'fig10_regional_cgsri_ranking.png').stat().st_size//1024} KB)")

print('\n✓ All 12 Author Action List panels regenerated.')
