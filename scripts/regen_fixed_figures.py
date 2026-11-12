#!/usr/bin/env python3
"""Regenerate figs 01, 02, 06, 08, 09 with reviewer corrections applied."""
import warnings; warnings.filterwarnings('ignore')
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats
import pymannkendall as mk  # Sen's slope from MK test (mm/month intervals)
from pathlib import Path

ROOT  = Path('.')
BOUND = ROOT / 'data/raw/boundaries'
PI    = ROOT / 'data/processed/pressure_indices'
RI    = ROOT / 'data/processed/risk_index'
FIGS  = ROOT / 'outputs/figures'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
DPI = 300
MAP_XLIM = (-3.5, 1.5); MAP_YLIM = (4.0, 7.5)
CITIES = [(-0.187,5.603,'Accra'), (-0.016,5.670,'Tema'), (-1.246,5.106,'Cape Coast'),
          (-1.752,4.894,'Takoradi'), (0.627,5.772,'Ada'), (0.999,5.916,'Keta')]
RISK_COLORS = {'Very low':'#2166AC','Low':'#74ADD1','Moderate':'#FEE08B',
               'High':'#F46D43','Very high':'#A50026'}
RISK_ORDER  = ['Very low','Low','Moderate','High','Very high']

NE_PATH = ('/opt/miniconda3/envs/accra-risk/lib/python3.11/site-packages'
           '/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp')

def map_furniture(ax, xlim=MAP_XLIM, ylim=MAP_YLIM):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel('Longitude (°E)', fontsize=8)
    ax.set_ylabel('Latitude (°N)', fontsize=8)
    for x in np.arange(np.ceil(xlim[0]), xlim[1]+0.5, 1):
        ax.axvline(x, color='#CCCCCC', lw=0.3, zorder=0)
    for y in np.arange(np.ceil(ylim[0]), ylim[1]+0.5, 1):
        ax.axhline(y, color='#CCCCCC', lw=0.3, zorder=0)
    for lon, lat, nm in CITIES:
        if xlim[0] <= lon <= xlim[1] and ylim[0] <= lat <= ylim[1]:
            ax.plot(lon, lat, 'k^', ms=4, zorder=5)
            ax.annotate(nm, (lon,lat), xytext=(4,3),
                        textcoords='offset points', fontsize=6, zorder=6)

def north_arrow(ax, x=0.96, y=0.94):
    ax.annotate('N', xy=(x,y), xycoords='axes fraction',
                fontsize=9, fontweight='bold', ha='center', va='bottom')
    ax.annotate('', xy=(x,y), xytext=(x,y-0.06), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))


# ── Load shared data ─────────────────────────────────────────
ghana     = gpd.read_file(str(BOUND/'ghana_boundary.gpkg')).to_crs('EPSG:4326')
coast50   = gpd.read_file(str(BOUND/'study_area_coastal_50km.gpkg')).to_crs('EPSG:4326')
districts = gpd.read_file(str(BOUND/'coastal_districts.gpkg')).to_crs('EPSG:4326')
grid10    = gpd.read_file(str(BOUND/'analysis_grid_10km.gpkg')).to_crs('EPSG:4326')

world  = gpd.read_file(NE_PATH)
africa = world[world['continent'] == 'Africa']

print('Boundaries loaded.')


# ── FIG 01: Study area — fix Africa inset ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))

ax = axes[0]
ghana.plot(ax=ax, facecolor='#F5F0E8', edgecolor='black', lw=0.8)
coast50.plot(ax=ax, facecolor='#C6EFCE', edgecolor='#538135', lw=0.8, alpha=0.55)
for lon, lat, nm in [(-0.187,5.603,'Accra'),(-1.624,6.688,'Kumasi'),(-1.752,4.894,'Takoradi')]:
    ax.plot(lon, lat, 'r*', ms=7, zorder=5)
    ax.annotate(nm, (lon,lat), xytext=(4,4), textcoords='offset points', fontsize=8)
ax.set_xlim(-3.8, 1.6); ax.set_ylim(4.5, 11.3)
ax.set_title('(a) Ghana — study area location', fontsize=9)
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')

axins = ax.inset_axes([0.02, 0.50, 0.35, 0.47])
axins.set_facecolor('#C8E0F0')
africa.plot(ax=axins, facecolor='#E8D5A3', edgecolor='#888888', lw=0.3)
axins.add_patch(Rectangle((-3.8,4.5), 5.4, 2.5,
                facecolor='#CC4444', alpha=0.75, edgecolor='red', lw=1.2, zorder=3))
axins.set_xlim(-20, 55); axins.set_ylim(-35, 40)
axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for sp in axins.spines.values(): sp.set_linewidth(0.5)

ax = axes[1]
districts.plot(ax=ax, facecolor='#E8EEF4', edgecolor='#888888', lw=0.5)
coast50.boundary.plot(ax=ax, edgecolor='#538135', lw=1.5, linestyle='--',
                      label='50 km coastal buffer')
grid10.plot(ax=ax, facecolor='none', edgecolor='#BBBBBB', lw=0.2, alpha=0.6)
map_furniture(ax)
north_arrow(ax)
ax.set_title('(b) Coastal study area and 10 km analysis grid', fontsize=9)
ax.legend(loc='lower right', fontsize=7)
for lon, lat, nm in [(-0.10,5.70,'Accra–Tema\ncorridor'), (-1.75,5.05,'Sekondi–\nTakoradi'),
                     (0.80,6.05,'Ada coast'), (-1.25,5.15,'Cape Coast')]:
    ax.text(lon, lat, nm, fontsize=5.5, color='#2c5f2e', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))

plt.tight_layout()
fig.savefig(str(FIGS/'fig01_study_area.png'), dpi=DPI)
plt.close()
print(f"Saved fig01  ({(FIGS/'fig01_study_area.png').stat().st_size//1024} KB)")


# ── FIG 02: GWSA — replace OLS label with Sen's slope ───────
df_g = pd.read_csv('data/interim/time_series/gwsa_monthly.csv')
df_g['time'] = pd.to_datetime(df_g['time'])
gwsa_ts = df_g.set_index('time')['gwsa_mm']

mk_res = mk.original_test(gwsa_ts.dropna())
sen_month = mk_res.slope            # mm/month from pymannkendall
sen_yr    = sen_month * 12          # mm/year to match manuscript
# compute months since start for each timestamp (including NaN months)
months_idx = np.array([(t - gwsa_ts.index[0]).days / 30.4375
                        for t in gwsa_ts.index])
# intercept: median(y) - slope * median(x) using non-NaN values
clean_mask = gwsa_ts.notna()
intercept  = float(gwsa_ts[clean_mask].median()) - sen_month * float(months_idx[clean_mask.values].mean())
trend_line = sen_month * months_idx + intercept

gwsa_ann       = gwsa_ts.resample('YE').mean()
gwsa_clim_mean = gwsa_ts.groupby(gwsa_ts.index.month).mean()
gwsa_clim_std  = gwsa_ts.groupby(gwsa_ts.index.month).std()

fig = plt.figure(figsize=(11, 7))
gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax = fig.add_subplot(gs[0, :])
ax.fill_between(gwsa_ts.index, np.where(gwsa_ts<0, gwsa_ts, 0), 0,
                color='#CC4444', alpha=0.4, label='Deficit')
ax.fill_between(gwsa_ts.index, np.where(gwsa_ts>=0, gwsa_ts, 0), 0,
                color='#4477AA', alpha=0.4, label='Surplus')
ax.plot(gwsa_ts.index, gwsa_ts.rolling(12,center=True).mean(),
        color='navy', lw=1.6, label='12-month mean')
ax.plot(gwsa_ts.index, trend_line, color='#CC4444', lw=1.2, ls='--',
        label=f"Sen's slope: {sen_yr:+.1f} mm yr⁻¹ (MK p<0.001)")
ax.axhline(0, color='k', lw=0.7)
ax.axvspan(pd.Timestamp('2017-06-01'), pd.Timestamp('2018-06-01'),
           color='gray', alpha=0.14, label='GRACE mission gap')
ax.set_ylabel('GWSA (mm)')
ax.set_title("(a) Monthly GWSA with Sen's trend and 12-month running mean", fontsize=9)
ax.legend(ncol=3, fontsize=7, loc='upper left')

ax2 = fig.add_subplot(gs[1, 0])
bar_c = ['#CC4444' if v<0 else '#4477AA' for v in gwsa_ann.values]
ax2.bar(gwsa_ann.index.year, gwsa_ann.values, color=bar_c, width=0.7, edgecolor='white')
ax2.axhline(0, color='k', lw=0.7)
ax2.set_ylabel('Annual mean GWSA (mm)')
ax2.set_title('(b) Annual mean GWSA', fontsize=9)
ax2.tick_params(axis='x', rotation=45)

ax3 = fig.add_subplot(gs[1, 1])
months = np.arange(1, 13)
ax3.bar(months, gwsa_clim_mean.values, color='steelblue', width=0.7, alpha=0.7)
ax3.errorbar(months, gwsa_clim_mean.values, yerr=gwsa_clim_std.values,
             fmt='none', ecolor='navy', capsize=3, lw=0.8)
ax3.axhline(0, color='k', lw=0.7)
ax3.set_xticks(months)
ax3.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax3.set_ylabel('GWSA (mm)')
ax3.set_title('(c) Mean seasonal cycle (±1 SD)', fontsize=9)

plt.savefig(str(FIGS/'fig02_gwsa_timeseries.png'), dpi=DPI)
plt.close()
print(f"Saved fig02  Sen={sen_yr:.2f} mm/yr  ({(FIGS/'fig02_gwsa_timeseries.png').stat().st_size//1024} KB)")


# ── FIG 06: UGPI — use urban-only 3-indicator version ───────
gdf_ugpi = gpd.read_file(str(PI/'ugpi_grid10km.gpkg')).to_crs('EPSG:4326')
ugpi_col = 'ugpi_urban_only'   # 3-indicator: n_BD + n_PD + n_BUGR (manuscript definition)
n_cols_ugpi = ['n_BD', 'n_PD', 'n_BUGR']   # only the three urban indicators

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

ax = axes[0]
gdf_ugpi.plot(column=ugpi_col, ax=ax, cmap='RdYlGn_r',
              vmin=0, vmax=gdf_ugpi[ugpi_col].quantile(0.98),
              edgecolor='none', legend=True,
              legend_kwds={'label':'UGPI (0–1, high = more pressure)','shrink':0.8})
districts.boundary.plot(ax=ax, edgecolor='#444444', lw=0.4)
map_furniture(ax)
north_arrow(ax)
ax.set_title('(a) Urban Groundwater Pressure Index (3-indicator)', fontsize=9)

ax = axes[1]
ind_means = gdf_ugpi[n_cols_ugpi].mean().sort_values()
short_lbl = [c.replace('n_','').upper() for c in ind_means.index]
colors_b  = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(ind_means)))
ax.barh(range(len(ind_means)), ind_means.values, color=colors_b)
ax.set_yticks(range(len(ind_means)))
ax.set_yticklabels(short_lbl)
ax.axvline(0.5, color='k', lw=0.7, ls='--', alpha=0.4)
for i, v in enumerate(ind_means.values):
    ax.text(v+0.01, i, f'{v:.3f}', va='center', fontsize=7)
ax.set_xlabel('Mean normalised indicator value')
ax.set_title('(b) Indicator contributions (study-area mean)', fontsize=9)
ax.text(0.98, 0.04,
        'UGPI = mean(n_BD, n_PD, n_BUGR) — equal weights.\n'
        'Building density (BD) and population density (PD)\n'
        'are dominant; built-up growth rate (BUGR) reflects\n'
        'peri-urban expansion (see sensitivity analysis, Table S1).',
        transform=ax.transAxes, fontsize=6.5, va='bottom', ha='right',
        color='#444444',
        bbox=dict(boxstyle='round,pad=0.3', fc='#F9F9F9', alpha=0.85,
                  edgecolor='#CCCCCC', lw=0.5))

plt.tight_layout()
plt.savefig(str(FIGS/'fig06_ugpi_map.png'), dpi=DPI)
plt.close()
print(f"Saved fig06  ({(FIGS/'fig06_ugpi_map.png').stat().st_size//1024} KB)")


# ── FIG 08: Bivariate — distinguish proxy n=35 from InSAR n=8
gdf_cs   = gpd.read_file(str(RI/'cgsri_grid10km.gpkg')).to_crs('EPSG:4326')
ugpi_med = gdf_cs['ugpi_urban_only'].median()
sub_med  = gdf_cs['sub_hazard'].median()
gdf_cs['bivar'] = gdf_cs.apply(
    lambda r: ('H' if r['ugpi_urban_only']>=ugpi_med else 'L') + '/' +
              ('H' if r['sub_hazard'] >=sub_med  else 'L'), axis=1)
bivar_colors = {'L/L':'#F7F7F7','H/L':'#74ADD1','L/H':'#F46D43','H/H':'#A50026'}
bivar_labels = {
    'L/L': 'Low UGPI + low subsidence proxy',
    'H/L': 'High UGPI + low subsidence proxy',
    'L/H': 'Low UGPI + high subsidence proxy',
    'H/H': 'HIGH UGPI + HIGH subsidence proxy',
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

ax = axes[0]
gdf_cs.plot(ax=ax, facecolor='#F7F7F7', edgecolor='none')
for cls, color in bivar_colors.items():
    sub = gdf_cs[gdf_cs['bivar']==cls]
    if len(sub) > 0:
        sub.plot(ax=ax, facecolor=color, edgecolor='none')
districts.boundary.plot(ax=ax, edgecolor='#555555', lw=0.4)
n_hh = len(gdf_cs[gdf_cs['bivar']=='H/H'])
handles = [mpatches.Patch(fc=bivar_colors[k],
           label=f'{bivar_labels[k]} (n={len(gdf_cs[gdf_cs["bivar"]==k])})')
           for k in bivar_colors]
ax.legend(handles=handles, fontsize=6.5, loc='lower right')
map_furniture(ax)
north_arrow(ax)
ax.set_title('(a) Bivariate UGPI (3-indicator) × subsidence proxy (median split)', fontsize=9)
ax.text(0.02, 0.97,
        f'Subsidence proxy = composite of elevation, building\n'
        f'density and GW demand (n={n_hh} co-located cells).\n'
        f'InSAR S1 velocity co-location: 8 cells\n'
        f'(Accra–Tema corridor; see Fig. 7b).',
        transform=ax.transAxes, fontsize=6.2, va='top', ha='left', color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85,
                  edgecolor='#CCCCCC', lw=0.5))

ax = axes[1]
sc = ax.scatter(gdf_cs['ugpi_urban_only'], gdf_cs['sub_hazard'],
                c=gdf_cs['cgsri'], cmap='RdYlGn_r', s=8, alpha=0.45, linewidths=0)
plt.colorbar(sc, ax=ax, label='CGSRI composite score', shrink=0.85)
ax.axvline(ugpi_med, color='navy', lw=0.9, ls='--', alpha=0.6,
           label=f'UGPI median ({ugpi_med:.3f})')
ax.axhline(sub_med,  color='orange', lw=0.9, ls='--', alpha=0.6,
           label=f'Sub median ({sub_med:.3f})')
ax.set_xlabel('UGPI — Urban Groundwater Pressure Index (3-indicator, normalised)')
ax.set_ylabel('Subsidence proxy hazard (normalised)')
ax.legend(fontsize=7)
ax.set_title('(b) Scatter: UGPI vs subsidence proxy\ncoloured by CGSRI composite', fontsize=9)
ax.text(0.01, 0.99,
        'UGPI = mean(n_BD, n_PD, n_BUGR):\n'
        'building density, population density, built-up growth rate',
        transform=ax.transAxes, fontsize=5.5, va='top', ha='left', color='#333333',
        bbox=dict(boxstyle='round', fc='white', alpha=0.80, edgecolor='none'))

plt.tight_layout()
plt.savefig(str(FIGS/'fig08_bivariate_gw_subsidence.png'), dpi=DPI)
plt.close()
print(f"Saved fig08  ({(FIGS/'fig08_bivariate_gw_subsidence.png').stat().st_size//1024} KB)")


# ── FIG 09: CGSRI risk map — correct population annotation ──
fig = plt.figure(figsize=(13, 7))
gs  = GridSpec(1, 3, figure=fig, wspace=0.08, width_ratios=[2.8,0.12,1.1])

ax_map = fig.add_subplot(gs[0])
gdf_cs.plot(ax=ax_map, facecolor='#EEEEEE', edgecolor='none')
for cls in RISK_ORDER:
    sub = gdf_cs[gdf_cs['cgsri_class']==cls]
    if len(sub) > 0:
        sub.plot(ax=ax_map, fc=RISK_COLORS[cls], edgecolor='none')
vh = gdf_cs[gdf_cs['cgsri_class']=='Very high']
if len(vh) > 0:
    vh.plot(ax=ax_map, fc='none', edgecolor='black', lw=0.5, hatch='///')
districts.boundary.plot(ax=ax_map, edgecolor='#555555', lw=0.4)
handles = [mpatches.Patch(fc=RISK_COLORS[cls],
           label=f'{cls} (n={len(gdf_cs[gdf_cs["cgsri_class"]==cls])})')
           for cls in RISK_ORDER]
ax_map.legend(handles=handles, fontsize=7.5, loc='lower right', framealpha=0.9)
map_furniture(ax_map)
north_arrow(ax_map)
ax_map.set_title('CGSRI risk classification (quintiles)\n(hatching = Very High)', fontsize=9)

ax_cb = fig.add_subplot(gs[1])
cmap_r = mcolors.LinearSegmentedColormap.from_list(
    'risk', [RISK_COLORS[c] for c in RISK_ORDER])
norm_r = mcolors.Normalize(vmin=gdf_cs['cgsri'].min(), vmax=gdf_cs['cgsri'].max())
sm = plt.cm.ScalarMappable(cmap=cmap_r, norm=norm_r); sm.set_array([])
plt.colorbar(sm, cax=ax_cb, label='CGSRI value')

ax_hist = fig.add_subplot(gs[2])
bins = np.linspace(gdf_cs['cgsri'].min(), gdf_cs['cgsri'].max(), 25)
for cls in RISK_ORDER:
    vals = gdf_cs[gdf_cs['cgsri_class']==cls]['cgsri'].values
    if len(vals) > 0:
        ax_hist.hist(vals, bins=bins, fc=RISK_COLORS[cls], label=cls, alpha=0.88)
ax_hist.set_xlabel('CGSRI value')
ax_hist.set_ylabel('N grid cells')
ax_hist.set_title('CGSRI distribution\nby risk class', fontsize=8)
ax_hist.legend(fontsize=6.5)

vh_pop = gdf_cs[gdf_cs['cgsri_class']=='Very high']['population_2020'].sum()
hi_pop = gdf_cs[gdf_cs['cgsri_class'].isin(['Very high','High'])]['population_2020'].sum()
ax_hist.text(0.03, 0.38,
             f'Very High (top quintile):\n~{vh_pop/1e6:.1f}M people\n'
             f'(Very High + High: {hi_pop/1e6:.1f}M)',
             transform=ax_hist.transAxes, fontsize=7,
             bbox=dict(boxstyle='round', fc='#FFF3CD', alpha=0.9))

plt.savefig(str(FIGS/'fig09_cgsri_risk_map.png'), dpi=DPI)
plt.close()
print(f"Saved fig09  VH={vh_pop/1e6:.2f}M  VH+H={hi_pop/1e6:.2f}M"
      f"  ({(FIGS/'fig09_cgsri_risk_map.png').stat().st_size//1024} KB)")

print('\nAll 5 figures complete.')
