"""Figures for Paper 3 (v2). Run after build_paper3_v2.py. Output: outputs/manuscript/paper3/v2/fig0*.png"""
from pathlib import Path

import cartopy.io.shapereader as shp
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw'
PRO = ROOT / 'data/processed/paper3'
TAB = ROOT / 'outputs/tables/paper3_v2'
OUT = ROOT / 'outputs/manuscript/paper3/v2'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 300})
C_RAW, C_COR, C_LAKE, C_GAP = '#b2182b', '#2166ac', '#1b7837', '#d9d9d9'
GAP = (pd.Timestamp('2017-07-01'), pd.Timestamp('2018-05-31'))
LAT, LON = (4.0, 11.5), (-3.5, 1.5)

lakes = gpd.read_file(shp.natural_earth(resolution='10m', category='physical', name='lakes'))
volta = lakes[lakes.name == 'Lake Volta']
countries = gpd.read_file(shp.natural_earth(resolution='10m', category='cultural', name='admin_0_countries'))
ghana = countries[countries.ADM0_A3 == 'GHA']
PLACES = {'Accra': (-0.186, 5.556), 'Kumasi': (-1.624, 6.688), 'Tamale': (-0.853, 9.404),
          'Akosombo Dam': (0.059, 6.300), 'Bui Dam': (-2.236, 8.278)}

jpl = xr.open_dataset(RAW / 'grace/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04CRI.nc')
jpl = jpl.assign_coords(lon=((jpl.lon + 180) % 360 - 180)).sortby('lon')
box = dict(lat=slice(*LAT), lon=slice(*LON))
mask = jpl.land_mask.sel(**box) == 1


def break_gaps(s, max_days=70):
    """Insert NaN where consecutive observations are more than ~2 months apart, so lines break only at long gaps."""
    idx = s.index
    gaps = idx[1:][(idx[1:] - idx[:-1]).days > max_days]
    return pd.concat([s, pd.Series(np.nan, index=gaps - pd.Timedelta(days=1))]).sort_index()


def gap(ax):
    ax.axvspan(*GAP, color=C_GAP, alpha=0.6, lw=0, zorder=0)


def base_map(ax, extent=(-4.2, 2.2, 3.8, 12.0)):
    countries.plot(ax=ax, color='#f2f2f2', edgecolor='#999999', lw=0.4)
    ghana.plot(ax=ax, color='#e6e6e6', edgecolor='#555555', lw=0.8)
    volta.plot(ax=ax, color='#6baed6', edgecolor='#2171b5', lw=0.4)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')


def places(ax, which=PLACES):
    for n, (x, y) in which.items():
        mk = '^' if 'Dam' in n else 'o'
        ax.plot(x, y, mk, ms=4, color='k', zorder=5)
        ax.annotate(n, (x, y), xytext=(4, 3), textcoords='offset points', fontsize=7, zorder=6)


# ---------------------------------------------------------------- Fig 1 domain
def fig1():
    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    base_map(ax)
    for la in mask.lat.values:
        for lo in mask.lon.values:
            if bool(mask.sel(lat=la, lon=lo)):
                ax.add_patch(Rectangle((lo - 0.25, la - 0.25), 0.5, 0.5, fill=False, ec='#e08214', lw=0.5, zorder=3))
    ax.add_patch(Rectangle((LON[0], LAT[0]), LON[1] - LON[0], LAT[1] - LAT[0], fill=False, ec='k', lw=1.0, ls='--', zorder=4))
    places(ax)
    ax.legend(handles=[Patch(fc='none', ec='#e08214', label='0.5° GRACE land cells (n = 123)'),
                       Patch(fc='none', ec='k', ls='--', label='Averaging domain 4.0–11.5°N, 3.5°W–1.5°E'),
                       Patch(fc='#6baed6', ec='#2171b5', label='Lake Volta (≈8,500 km² at full supply)'),
                       Line2D([], [], marker='^', ls='', color='k', ms=5, label='Dams')],
              loc='lower left', fontsize=6.5, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / 'fig01_domain.png')
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2 lake level and lake storage vs GWSA
def fig2():
    d = pd.read_csv(PRO / 'lake_volta_grealm_10day.csv', parse_dates=['date'])
    ens = pd.read_csv(PRO / 'gwsa_ensemble_summary.csv', index_col=0, parse_dates=True)
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 5.6), gridspec_kw={'height_ratios': [1, 1.25]})
    cols = {'TOPEX': '#999999', 'JASN1': '#66c2a5', 'JASN2': '#fc8d62', 'JASN3': '#8da0cb', 'SEN6A': '#e78ac3'}
    for m, g in d.groupby('mission'):
        a.plot(g.date, g.h, '.', ms=2, color=cols.get(m, 'k'), label=m.replace('JASN', 'Jason-').replace('SEN6A', 'Sentinel-6A'))
    a.set_ylabel('Lake Volta level (m,\nrelative to reference)')
    a.set_title('(a) G-REALM 10-day Lake Volta levels, 1992–2026', loc='left', fontsize=9)
    a.legend(ncol=5, fontsize=6.5, markerscale=3, loc='upper left', frameon=False)
    a.axvspan(pd.Timestamp('2002-04-01'), pd.Timestamp('2025-12-31'), color='#fff7bc', alpha=0.6, lw=0, zorder=0)
    t = ens.index
    gap(b)
    b.fill_between(t, ens.gwsa_raw_min, ens.gwsa_raw_max, color=C_RAW, alpha=0.18, lw=0)
    b.plot(t, ens.gwsa_raw_mean, color=C_RAW, lw=1.1, label='GRACE–GLDAS GWSA (9-member mean and range)')
    b.fill_between(t, ens.lake_A6000, ens.lake_A8502, color=C_LAKE, alpha=0.35, lw=0)
    b.plot(t, ens.lake_A8502, color=C_LAKE, lw=1.1, label='Lake Volta storage anomaly (6,000–8,502 km²)')
    b.axhline(0, color='k', lw=0.5)
    b.set_ylabel('Storage anomaly (mm EWH,\ndomain mean)')
    b.set_title('(b) Regional GWSA and Lake Volta storage, April 2002 – December 2025', loc='left', fontsize=9)
    b.legend(fontsize=7, loc='upper left', frameon=False)
    for ax in (a, b):
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(OUT / 'fig02_lake_vs_gwsa.png')
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3 JPL pixel footprint of the lake
def fig3():
    tw = (jpl.lwe_thickness * jpl.scale_factor).sel(**box) * 10
    tw = tw.assign_coords(time=pd.DatetimeIndex(tw.time.values).to_period('M').to_timestamp())
    tw = tw.isel(time=~pd.Index(tw.time.values).duplicated()).sel(time=slice('2002-04-01', '2025-12-01'))
    clim = tw.sel(time=slice('2004', '2009')).groupby('time.month').mean()
    an = (tw.groupby('time.month') - clim).where(mask)
    lk = pd.read_csv(PRO / 'lake_volta_storage_ewh.csv', index_col=0, parse_dates=True).A8502
    lk = lk.reindex(pd.DatetimeIndex(an.time.values))
    ok = lk.notna().values
    A = an.values[ok]
    L = lk.values[ok]
    Az = A - A.mean(0)
    Lz = L - L.mean()
    r = (Az * Lz[:, None, None]).sum(0) / (Lz ** 2).sum()      # regression slope of cell TWSA on domain-mean lake EWH
    yrs = (pd.DatetimeIndex(an.time.values).year + (pd.DatetimeIndex(an.time.values).dayofyear - 0.5) / 365.25).values
    X = an.values
    slope = np.full(X.shape[1:], np.nan)
    from scipy import stats
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            y = X[:, i, j]
            m = np.isfinite(y)
            if m.sum() > 100:
                slope[i, j] = stats.theilslopes(y[m], yrs[m])[0]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.6))
    lon, lat = an.lon.values, an.lat.values
    for ax, data, cmap, vmin, vmax, lab, ttl in [
            (axs[0], r, 'YlOrBr', 0, 3, 'mm per mm of domain-mean lake storage', '(a) TWSA regression on lake storage'),
            (axs[1], slope, 'YlGnBu', 0, 30, 'mm yr⁻¹', '(b) TWSA trend, 2002–2025')]:
        base_map(ax)
        pm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(np.where(mask.values, data, np.nan)), cmap=cmap,
                           vmin=vmin, vmax=vmax, shading='nearest', alpha=0.9, zorder=2)
        volta.boundary.plot(ax=ax, color='k', lw=0.6, zorder=4)
        ax.set_title(ttl, fontsize=8.5, loc='left')
        fig.colorbar(pm, ax=ax, shrink=0.75, label=lab)
        places(ax, {k: v for k, v in PLACES.items() if k in ('Accra', 'Bui Dam', 'Akosombo Dam')})
    axs[1].set_ylabel('')
    fig.tight_layout()
    fig.savefig(OUT / 'fig03_jpl_lake_footprint.png')
    plt.close(fig)
    return r, slope


# ---------------------------------------------------------------- Fig 4 drought index before / after
def fig4():
    Z = pd.read_csv(PRO / 'gdsi_ensemble_mean.csv', index_col=0, parse_dates=True)
    ag = pd.read_csv(PRO / 'drought_member_agreement.csv', index_col=0, parse_dates=True)
    ep = pd.read_csv(TAB / 'episodes_ensemble_mean.csv')
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    for ax, k, col, ttl in [(axs[0], 'raw', C_RAW, '(a) Without lake correction'),
                            (axs[1], 'corrected', C_COR, '(b) Lake Volta storage removed')]:
        gap(ax)
        z = break_gaps(Z[f'{k}_z'].dropna())
        zd = break_gaps(Z[f'{k}_zdetr'].dropna())
        ax.plot(z.index, z, color=col, lw=1.1, label='Standardised GWSA')
        ax.plot(zd.index, zd, color='k', lw=0.8, ls='--', label='Standardised GWSA, detrended')
        for _, e in ep[ep.series == f'{k}_z'].iterrows():
            ax.axvspan(pd.Timestamp(e.start), pd.Timestamp(e.end) + pd.offsets.MonthEnd(0), color=col, alpha=0.08, lw=0)
        ax2 = ax.twinx()
        ax2.fill_between(ag.index, 0, ag[k], step='mid', color='#636363', alpha=0.3, lw=0, label='Share of 9 members with z < −0.5')
        ax2.set_ylim(0, 4)
        ax2.set_yticks([0, 0.5, 1])
        ax2.set_ylabel('Share of members\nwith z < −0.5', fontsize=7)
        ax2.spines['top'].set_visible(False)
        ax.axhline(-0.5, color='grey', lw=0.5, ls=':')
        ax.axhline(0, color='k', lw=0.4)
        ax.axvspan(pd.Timestamp('2015-01-01'), pd.Timestamp('2016-12-31'), fc='none', ec='#e08214', lw=1.0, ls='-')
        ax.set_ylim(-3, 3.5)
        ax.set_ylabel('GDSI (z)')
        ax.set_title(ttl, loc='left', fontsize=9)
    axs[0].legend(fontsize=7, loc='upper left', frameon=False)
    axs[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(OUT / 'fig05_gdsi_before_after.png')
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5 trends by member
def fig5():
    tr = pd.read_csv(TAB / 'trends_by_member.csv')
    cases = [('none', 'Uncorrected', C_RAW), ('A8502_half_recovery', '0.5 × lake (8,502 km²)', '#f4a582'),
             ('A6000', 'Lake removed (6,000 km²)', '#67a9cf'), ('A8502', 'Lake removed (8,502 km²)', C_COR),
             ('A8502+Bui_gross_12.57', '+ Bui (12.57 km³)', '#542788')]
    pers = ['2002-04..2025-12', '2010-01..2025-12']
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True)
    members = sorted(tr.member.unique())
    for ax, per in zip(axs, pers):
        for k, (case, lab, col) in enumerate(cases):
            g = tr[(tr.period == per) & (tr.lake == case)].set_index('member').reindex(members)
            y = np.arange(len(members)) + (k - 2) * 0.16
            sig = g.p < 0.05
            ax.scatter(g.slope[sig], y[sig.values], color=col, s=16, label=lab, zorder=3)
            ax.scatter(g.slope[~sig], y[~sig.values], facecolor='white', edgecolor=col, s=16, zorder=3)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_yticks(np.arange(len(members)))
        ax.set_yticklabels([m.replace('-', ' × ') for m in members], fontsize=7)
        ax.set_xlabel('GWSA trend (mm yr⁻¹)')
        ax.set_title(per.replace('..', ' – ').replace('-', '/', 0), loc='left', fontsize=9)
        ax.grid(axis='x', color='#eeeeee')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.99, 0.065, 'Filled: Hamed–Rao p < 0.05; open: not significant', ha='right', fontsize=6.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / 'fig04_trends_by_member.png')
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6 climate linkage heatmap
def fig6():
    cl = pd.read_csv(TAB / 'climate_linkage.csv')
    cl = cl[cl.lag == 0]
    drivers = ['PDSI', 'SPI-1', 'SPI-3', 'SPI-6', 'SPI-12', 'SPI-24']
    targets = [('TWSA', 'TWSA'), ('Lake storage', 'Lake Volta storage'), ('GWSA raw', 'GWSA, uncorrected'),
               ('GWSA lake half', 'GWSA − 0.5 × lake (8,502 km²)'), ('GWSA lake A6000', 'GWSA − lake (6,000 km²)'),
               ('GWSA lake A8502', 'GWSA − lake (8,502 km²)')]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, (col, pcol, ttl) in zip(axs, [('r', 'p_eff', '(a) Monthly series'), ('r_detr', 'p_detr', '(b) Linear trends removed')]):
        M = np.full((len(targets), len(drivers)), np.nan)
        P = np.full_like(M, np.nan)
        for i, (t, _) in enumerate(targets):
            g = cl[cl.target == t].set_index('driver').reindex(drivers)
            M[i] = g[col].values
            P[i] = g[pcol].values
        im = ax.imshow(M, cmap='RdBu', vmin=-0.9, vmax=0.9, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f'{M[i, j]:+.2f}' + ('*' if P[i, j] < 0.05 else ''), ha='center', va='center',
                        fontsize=6.5, color='white' if abs(M[i, j]) > 0.55 else 'k')
        ax.set_xticks(range(len(drivers)))
        ax.set_xticklabels(drivers, fontsize=7)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels([lab for _, lab in targets], fontsize=7)
        ax.set_title(ttl, loc='left', fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.colorbar(im, ax=axs, shrink=0.8, label='Pearson r (lag 0)')
    fig.text(0.01, 0.01, '* p < 0.05 using the effective sample size (lag-1 autocorrelation of both series)', fontsize=6.5)
    fig.savefig(OUT / 'fig07_climate_linkage.png', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- Fig 7 robustness of drought windows
def fig7():
    wr = pd.read_csv(TAB / 'episode_windows_by_member.csv')
    windows = list(dict.fromkeys(wr.window))
    cases = [('none', 'Uncorrected', C_RAW), ('half', '0.5 × lake (8,502 km²)', '#f4a582'),
             ('A6000', 'Lake removed (6,000 km²)', '#67a9cf'), ('A8502', 'Lake removed (8,502 km²)', C_COR)]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for ax, col, ttl in [(axs[0], 'z', '(a) Standardised GWSA'), (axs[1], 'zdetr', '(b) Standardised detrended GWSA')]:
        for k, (case, lab, c) in enumerate(cases):
            for i, w in enumerate(windows):
                v = wr[(wr.window == w) & (wr.lake == case)][col]
                y = i + (k - 1.5) * 0.18
                ax.plot([v.min(), v.max()], [y, y], color=c, lw=1.2)
                ax.plot(v.median(), y, 'o', color=c, ms=4, label=lab if i == 0 else None)
        ax.axvline(-0.5, color='grey', ls=':', lw=0.8)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_yticks(range(len(windows)))
        ax.set_yticklabels([w.replace('..', ' – ') for w in windows], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('Mean z over the window')
        ax.set_title(ttl, loc='left', fontsize=9)
        ax.grid(axis='x', color='#eeeeee')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.99, 0.065, 'Dot: median of 9 GWSA members; line: range', ha='right', fontsize=6.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / 'fig06_episode_robustness.png')
    plt.close(fig)



# ---------------------------------------------------------------- Supplementary figures
def figS1():
    sc = pd.read_csv(PRO / 'seasonal_climatology.csv', index_col=0)
    sty = {'TWS (3-solution mean)': ('k', '-'), 'SM Noah': ('#1b9e77', '-'), 'SM CLSM': ('#d95f02', '-'),
           'SM VIC': ('#7570b3', '-'), 'Lake A8502': (C_LAKE, '--'), 'Lake A6000': (C_LAKE, ':')}
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for k, (c, ls) in sty.items():
        ax.plot(sc.index, sc[k], color=c, ls=ls, lw=1.4, marker='o', ms=3, label=k.replace('A8502', '(8,502 km²)').replace('A6000', '(6,000 km²)'))
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.set_ylabel('Mean seasonal anomaly (mm EWH)')
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / 'figS1_seasonal_cycle.png')
    plt.close(fig)


def figS2():
    ts = pd.read_csv(TAB / 'trend_start_end_sensitivity.csv')
    cols = {'uncorrected': C_RAW, 'lake 6,000 km2': '#67a9cf', 'lake 8,502 km2': C_COR}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, v, xl in [(axs[0], 'start', 'Start year (end December 2025)'), (axs[1], 'end', 'End year (start April 2002)')]:
        for nm, c in cols.items():
            g = ts[(ts.series == nm) & (ts.vary == v)]
            ax.plot(g.year, g.slope, color=c, lw=1.2, label=nm.replace('km2', 'km²'))
            ax.scatter(g.year[g.p < 0.05], g.slope[g.p < 0.05], color=c, s=14, zorder=3)
            ax.scatter(g.year[g.p >= 0.05], g.slope[g.p >= 0.05], facecolor='white', edgecolor=c, s=14, zorder=3)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel(xl)
        ax.grid(color='#eeeeee')
    axs[0].set_ylabel('Ensemble-mean GWSA trend (mm yr⁻¹)')
    axs[0].legend(fontsize=7, frameon=False)
    fig.text(0.99, 0.01, 'Filled: Hamed–Rao p < 0.05', ha='right', fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / 'figS2_trend_start_end.png')
    plt.close(fig)


def figS3():
    U = pd.read_csv(PRO / 'jpl_domain_uncertainty.csv', index_col=0, parse_dates=True)
    ens = pd.read_csv(PRO / 'gwsa_ensemble_summary.csv', index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    gap(ax)
    ax.fill_between(U.index, U.independent, U.correlated, color='#636363', alpha=0.35, lw=0, label='JPL formal 1σ, domain mean (independent–correlated)')
    ax.plot(ens.index, (ens.gwsa_raw_max - ens.gwsa_raw_min) / 2, color=C_RAW, lw=1, label='Half-range of the 9 GWSA members')
    ax.plot(ens.index, (ens.lake_A8502 - ens.lake_A6000).abs(), color=C_LAKE, lw=1, label='Difference between lake-area bounds')
    ax.plot(ens.index, ens.lake_A8502.abs(), color=C_LAKE, lw=0.8, ls=':', label='|Lake storage anomaly| (8,502 km²)')
    ax.set_ylabel('mm EWH')
    ax.legend(fontsize=6.5, frameon=False, loc='upper left')
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(OUT / 'figS3_uncertainty_sources.png')
    plt.close(fig)


if __name__ == '__main__':
    fig1()
    fig2()
    r, slope = fig3()
    print('Fig 3: slope range', np.nanmin(r), np.nanmax(r), 'trend range', np.nanmin(slope), np.nanmax(slope))
    fig4()
    fig5()
    fig6()
    fig7()
    figS1()
    figS2()
    figS3()
    print('figures written to', OUT)
