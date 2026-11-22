"""Figures for the Journal of Hydrology: Regional Studies version of Paper 3.

Reuses the map helpers of figures_paper3_v2.py and draws every figure with the level-dependent Lake Volta correction as the
main case. Run build_lake_hypsometry.py and build_paper3_curve_results.py first. Output: outputs/manuscript/paper3/jhrs/.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

import figures_paper3_v2 as base

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/manuscript/paper3/jhrs'
OUT.mkdir(parents=True, exist_ok=True)
base.OUT = OUT
OP = ROOT / 'data/processed/paper3_jhrs'
OT = ROOT / 'outputs/tables/paper3_jhrs'
PRO0 = ROOT / 'data/processed/paper3'
C_RAW, C_COR, C_LAKE, C_SCL, C_HALF, C_BUI = '#b2182b', '#2166ac', '#1b7837', '#67a9cf', '#f4a582', '#542788'
_map = base.base_map
LABELS = {'GHANA': (-1.15, 7.45), 'TOGO': (1.05, 8.45), 'BURKINA FASO': (-1.4, 11.28), "CÔTE D'IVOIRE": (-3.3, 8.75), 'BENIN': (1.86, 9.9)}


def map_with_furniture(ax, extent=(-4.2, 2.2, 3.8, 12.0)):
    _map(ax, extent)
    x0, y0 = 0.72, 4.42
    dlon = 100 / (111.32 * np.cos(np.radians(y0)))
    ax.plot([x0, x0 + dlon], [y0, y0], color='k', lw=2, solid_capstyle='butt', zorder=8)
    ax.text(x0 + dlon / 2, y0 + 0.1, '100 km', ha='center', fontsize=7, zorder=8)
    ax.annotate('N', xy=(0.94, 0.98), xytext=(0.94, 0.89), xycoords='axes fraction', ha='center', fontsize=8,
                fontweight='bold', arrowprops=dict(arrowstyle='-|>', color='k', lw=0.9), zorder=8)


base.base_map = map_with_furniture


def fig1():
    base.fig1()
    # redraw with country labels on the same layout
    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    base.base_map(ax)
    for la in base.mask.lat.values:
        for lo in base.mask.lon.values:
            if bool(base.mask.sel(lat=la, lon=lo)):
                ax.add_patch(base.Rectangle((lo - .25, la - .25), .5, .5, fill=False, ec='#e08214', lw=.5, zorder=3))
    ax.add_patch(base.Rectangle((base.LON[0], base.LAT[0]), base.LON[1] - base.LON[0], base.LAT[1] - base.LAT[0], fill=False, ec='k', lw=1, ls='--', zorder=4))
    base.places(ax)
    for name, (x, y) in LABELS.items():
        ax.text(x, y, name, ha='center', va='center', fontsize=6.8 if name != 'GHANA' else 8.5, color='#444444',
                fontweight='bold' if name == 'GHANA' else 'normal', zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax.legend(handles=[base.Patch(fc='none', ec='#e08214', label='0.5° GRACE land cells (n = 123)'),
                       base.Patch(fc='none', ec='k', ls='--', label='Averaging domain'),
                       base.Patch(fc='#6baed6', ec='#2171b5', label='Lake Volta'),
                       base.Line2D([], [], marker='^', ls='', color='k', ms=5, label='Dams')],
              loc='lower left', fontsize=7, frameon=True, framealpha=.95)
    fig.tight_layout()
    fig.savefig(OUT / 'fig01_domain.png')
    plt.close(fig)


def fig2_curve():
    k = pd.read_csv(PRO0 / 'lake_volta_level_area_pairs.csv', index_col=0, parse_dates=True)
    c = pd.read_csv(PRO0 / 'lake_volta_hypsometric_curve.csv')
    c = c[c.h_m.between(-6.3, 7.1)]
    t = pd.read_csv(OP / 'lake_storage_terms.csv', index_col=0, parse_dates=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1, 1.35]))
    a.scatter(k.h, k.area_km2 / 1000, s=13, color='#555555', alpha=.75, lw=0, label=f'Landsat months (n = {len(k)})', zorder=3)
    inside = c.h_m.between(k.h.min(), k.h.max())
    a.plot(c.h_m[inside], c.area_km2[inside] / 1000, color=C_LAKE, lw=1.8, label='Fitted A(h)', zorder=4)
    a.plot(c.h_m[c.h_m < k.h.min()], c.area_km2[c.h_m < k.h.min()] / 1000, color=C_LAKE, lw=1.2, ls=':', zorder=4)
    a.plot(c.h_m[c.h_m > k.h.max()], c.area_km2[c.h_m > k.h.max()] / 1000, color=C_LAKE, lw=1.2, ls=':', zorder=4)
    for v, ls, lab in ((8.502, '--', '8,502 km² (full supply)'), (6.0, '-.', '6,000 km²')):
        a.axhline(v, color='#777777', lw=.9, ls=ls)
        a.text(-6.15, v + .07, lab, fontsize=7, color='#444444')
    a.set_xlim(-6.4, 7.2)
    a.set_ylim(4, 9)
    a.set_xlabel('Lake level (m, relative to reference)')
    a.set_ylabel('Lake area (10³ km²)')
    a.legend(loc='lower right', fontsize=7, frameon=False)
    a.set_title('(a) Level–area relationship', loc='left', fontsize=9)
    base.gap(b)
    b.plot(t.index, t.A8502, color='#777777', lw=.9, ls='--', label='Constant 8,502 km²')
    b.plot(t.index, t.A6000, color='#777777', lw=.9, ls='-.', label='Constant 6,000 km²')
    b.fill_between(t.index, t.JRC, t.JRC_scaled, color=C_LAKE, alpha=.25, lw=0, label='Level-dependent, scaled to full supply')
    b.plot(t.index, t.JRC, color=C_LAKE, lw=1.3, label='Level-dependent (main case)')
    b.axhline(0, color='k', lw=.5)
    b.set_ylabel('Lake storage anomaly (mm)')
    b.xaxis.set_major_locator(mdates.YearLocator(4))
    b.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    b.legend(loc='upper left', fontsize=6.5, frameon=False, ncol=2)
    b.set_ylim(-125, 245)
    b.set_title('(b) Lake storage anomaly, domain mean', loc='left', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'fig02_level_area.png')
    plt.close(fig)


def fig3_lake():
    d = pd.read_csv(PRO0 / 'lake_volta_grealm_10day.csv', parse_dates=['date'])
    e = pd.read_csv(PRO0 / 'gwsa_ensemble_summary.csv', index_col=0, parse_dates=True)
    l = pd.read_csv(OP / 'lake_storage_terms.csv', index_col=0, parse_dates=True)
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 5.6), gridspec_kw={'height_ratios': [1, 1.25]})
    cols = {'TOPEX': '#999999', 'JASN1': '#66c2a5', 'JASN2': '#fc8d62', 'JASN3': '#8da0cb', 'SEN6A': '#e78ac3'}
    for m, g in d.groupby('mission'):
        a.plot(g.date, g.h, '.', ms=2, color=cols.get(m, 'k'), label=m.replace('JASN', 'Jason-').replace('SEN6A', 'Sentinel-6A'))
    a.set_ylabel('Lake Volta level (m,\nrelative to reference)')
    a.set_title('(a) G-REALM 10-day Lake Volta levels, 1992–2026', loc='left', fontsize=9)
    a.legend(ncol=5, fontsize=6.5, markerscale=3, loc='upper left', frameon=False)
    a.axvspan(pd.Timestamp('2002-04-01'), pd.Timestamp('2025-12-31'), color='#fff7bc', alpha=.6, lw=0, zorder=0)
    base.gap(b)
    b.fill_between(e.index, e.gwsa_raw_min, e.gwsa_raw_max, color=C_RAW, alpha=.18, lw=0)
    b.plot(e.index, e.gwsa_raw_mean, color=C_RAW, lw=1.1, label='GRACE–GLDAS GWSA (9-member mean and range)')
    b.plot(l.index, l.JRC, color=C_LAKE, lw=1.3, ls='--', label='Lake Volta storage anomaly (level-dependent)')
    b.axhline(0, color='k', lw=.5)
    b.set_ylabel('Storage anomaly (mm EWH,\ndomain mean)')
    b.set_title('(b) Regional GWSA and Lake Volta storage, April 2002 – December 2025', loc='left', fontsize=9)
    b.legend(fontsize=7, loc='upper left', frameon=False)
    for ax in (a, b):
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(OUT / 'fig03_lake_vs_gwsa.png')
    plt.close(fig)


def fig4_footprint():
    from scipy import stats
    jpl, box, mask = base.jpl, base.box, base.mask
    tw = (jpl.lwe_thickness * jpl.scale_factor).sel(**box) * 10
    tw = tw.assign_coords(time=pd.DatetimeIndex(tw.time.values).to_period('M').to_timestamp())
    tw = tw.isel(time=~pd.Index(tw.time.values).duplicated()).sel(time=slice('2002-04-01', '2025-12-01'))
    clim = tw.sel(time=slice('2004', '2009')).groupby('time.month').mean()
    an = (tw.groupby('time.month') - clim).where(mask)
    lk = pd.read_csv(OP / 'lake_storage_terms.csv', index_col=0, parse_dates=True).JRC.reindex(pd.DatetimeIndex(an.time.values))
    ok = lk.notna().values
    A, L = an.values[ok], lk.values[ok]
    r = ((A - A.mean(0)) * (L - L.mean())[:, None, None]).sum(0) / ((L - L.mean()) ** 2).sum()
    tm = pd.DatetimeIndex(an.time.values)
    yrs = (tm.year + (tm.dayofyear - .5) / 365.25).values
    X = an.values
    slope = np.full(X.shape[1:], np.nan)
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            m = np.isfinite(X[:, i, j])
            if m.sum() > 100:
                slope[i, j] = stats.theilslopes(X[:, i, j][m], yrs[m])[0]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.6))
    lon, lat = an.lon.values, an.lat.values
    for ax, data, cmap, vmin, vmax, lab, ttl in ((axs[0], r, 'YlOrBr', 0, 4.5, 'mm per mm of domain-mean lake storage', '(a) TWSA regression on lake storage'),
                                                 (axs[1], slope, 'YlGnBu', 0, 30, 'mm yr⁻¹', '(b) TWSA trend, 2002–2025')):
        base.base_map(ax)
        pm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(np.where(mask.values, data, np.nan)), cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest', alpha=.9, zorder=2)
        base.volta.boundary.plot(ax=ax, color='k', lw=.6, zorder=4)
        ax.set_title(ttl, fontsize=8.5, loc='left')
        fig.colorbar(pm, ax=ax, shrink=.75, label=lab)
        base.places(ax, {k: v for k, v in base.PLACES.items() if k in ('Accra', 'Bui Dam', 'Akosombo Dam')})
    axs[1].set_ylabel('')
    fig.tight_layout()
    fig.savefig(OUT / 'fig04_jpl_lake_footprint.png')
    plt.close(fig)
    print(f'Fig. 4: maximum cell regression slope {np.nanmax(np.where(mask.values, r, np.nan)):.2f}; maximum TWSA trend {np.nanmax(np.where(mask.values, slope, np.nan)):.1f} mm/yr; '
          f'median trend {np.nanmedian(np.where(mask.values, slope, np.nan)):.1f}; IQR {np.nanpercentile(np.where(mask.values, slope, np.nan), 25):.1f}-{np.nanpercentile(np.where(mask.values, slope, np.nan), 75):.1f}')


def fig5_trends():
    tr = pd.read_csv(OT / 'trends_by_member.csv')
    cases = [('none', 'Uncorrected', C_RAW, 'o'), ('half', '0.5 × lake', C_HALF, 'v'), ('JRC', 'Level-dependent', C_COR, 's'),
             ('JRC_scaled', 'Scaled to full supply', C_SCL, 'D'), ('JRC+Bui_gross', '+ Bui (12.57 km³)', C_BUI, '^')]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True)
    members = sorted(tr.member.unique())
    for ax, per, ttl in zip(axs, ('2002-04..2025-12', '2010-01..2025-12'), ('(a) April 2002 – December 2025', '(b) January 2010 – December 2025')):
        for k, (case, lab, col, mk) in enumerate(cases):
            g = tr[(tr.period == per) & (tr.lake == case)].set_index('member').reindex(members)
            y = np.arange(len(members)) + (k - 2) * .16
            sig = g.p < .05
            ax.scatter(g.slope[sig], y[sig.values], color=col, marker=mk, s=16, label=lab, zorder=3)
            ax.scatter(g.slope[~sig], y[~sig.values], facecolor='white', edgecolor=col, marker=mk, s=16, zorder=3)
        ax.axvline(0, color='k', lw=.5)
        ax.set_yticks(np.arange(len(members)))
        ax.set_yticklabels([m.replace('-', ' × ') for m in members], fontsize=7)
        ax.set_xlabel('GWSA trend (mm yr⁻¹)')
        ax.set_title(ttl, loc='left', fontsize=9)
        ax.grid(axis='x', color='#eeeeee')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(.5, .025))
    fig.text(.5, .005, 'Filled: Hamed–Rao p < 0.05; open: not significant', ha='center', fontsize=7)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(OUT / 'fig05_trends_by_member.png')
    plt.close(fig)


def fig6_gdsi():
    Z = pd.read_csv(OP / 'gdsi_ensemble_mean.csv', index_col=0, parse_dates=True)
    ag = pd.read_csv(OP / 'drought_member_agreement.csv', index_col=0, parse_dates=True)
    ep = pd.read_csv(OT / 'episodes_ensemble_mean.csv')
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    for ax, k, col, ttl in ((axs[0], 'raw', C_RAW, '(a) Without lake correction'), (axs[1], 'corrected', C_COR, '(b) Level-dependent Lake Volta storage removed')):
        base.gap(ax)
        z, zd = base.break_gaps(Z[f'{k}_z'].dropna()), base.break_gaps(Z[f'{k}_zdetr'].dropna())
        ax.plot(z.index, z, color=col, lw=1.1, label='Standardised GWSA')
        ax.plot(zd.index, zd, color='k', lw=.8, ls='--', label='Standardised GWSA, detrended')
        for _, e in ep[ep.series == f'{k}_z'].iterrows():
            ax.axvspan(pd.Timestamp(e.start), pd.Timestamp(e.end) + pd.offsets.MonthEnd(0), color=col, alpha=.08, lw=0)
        ax.bar(ag.index, ag[k] * .9, bottom=-3, width=31, color='#636363', alpha=.35, lw=0, label='Share of 9 members with z < −0.5 (bar height 0.9 = all)')
        ax.axhline(-.5, color='grey', lw=.5, ls=':')
        ax.axhline(0, color='k', lw=.4)
        ax.axvspan(pd.Timestamp('2015-01-01'), pd.Timestamp('2016-12-31'), fc='none', ec='#e08214', lw=1)
        ax.set_ylim(-3, 3.5)
        ax.set_ylabel('GDSI (z)')
        ax.set_title(ttl, loc='left', fontsize=9)
    axs[0].legend(fontsize=7, loc='upper left', frameon=False)
    axs[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(OUT / 'fig06_gdsi_before_after.png')
    plt.close(fig)


def fig7_windows():
    wr = pd.read_csv(OT / 'episode_windows_by_member.csv')
    windows = list(dict.fromkeys(wr.window))
    cases = [('none', 'Uncorrected', C_RAW, 'o'), ('half', '0.5 × lake', C_HALF, 'v'), ('JRC', 'Level-dependent', C_COR, 's'), ('JRC_scaled', 'Scaled to full supply', C_SCL, 'D')]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.7), sharey=True)
    for ax, col, ttl in ((axs[0], 'z', '(a) Standardised GWSA'), (axs[1], 'zdetr', '(b) Standardised detrended GWSA')):
        for k, (case, lab, c, mk) in enumerate(cases):
            for i, w in enumerate(windows):
                v = wr[(wr.window == w) & (wr.lake == case)][col]
                y = i + (k - 1.5) * .18
                ax.plot([v.min(), v.max()], [y, y], color=c, lw=1.2)
                ax.plot(v.median(), y, mk, color=c, ms=4, label=lab if i == 0 else None)
        ax.axvline(-.5, color='grey', ls=':', lw=.8)
        ax.axvline(0, color='k', lw=.5)
        ax.set_yticks(range(len(windows)))
        ax.set_yticklabels([w.replace('..', ' – ') for w in windows], fontsize=7)
        ax.set_xlabel('Mean z over the window')
        ax.set_title(ttl, loc='left', fontsize=9)
        ax.grid(axis='x', color='#eeeeee')
    axs[0].invert_yaxis()
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(.5, .025))
    fig.text(.5, .005, 'Marker: median of 9 GWSA members; line: range', ha='center', fontsize=7)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(OUT / 'fig07_episode_robustness.png')
    plt.close(fig)


def fig8_climate():
    cl = pd.read_csv(OT / 'climate_linkage.csv')
    cl = cl[cl.lag == 0]
    drivers = ['PDSI', 'SPI-1', 'SPI-3', 'SPI-6', 'SPI-12', 'SPI-24']
    targets = [('TWSA', 'TWSA'), ('Lake storage', 'Lake Volta storage'), ('GWSA raw', 'GWSA, uncorrected'), ('GWSA half', 'GWSA − 0.5 × lake'),
               ('GWSA JRC', 'GWSA − lake (level-dependent)'), ('GWSA JRC_scaled', 'GWSA − lake (scaled)')]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, (col, pcol, ttl) in zip(axs, (('r', 'p_eff', '(a) Monthly series'), ('r_detr', 'p_detr', '(b) Linear trends removed'))):
        M, P = np.full((len(targets), len(drivers)), np.nan), np.full((len(targets), len(drivers)), np.nan)
        for i, (t, _) in enumerate(targets):
            g = cl[cl.target == t].set_index('driver').reindex(drivers)
            M[i], P[i] = g[col].values, g[pcol].values
        im = ax.imshow(M, cmap='RdBu', vmin=-.9, vmax=.9, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f'{M[i, j]:+.2f}' + ('*' if P[i, j] < .05 else ''), ha='center', va='center', fontsize=6.5, color='white' if abs(M[i, j]) > .55 else 'k')
        ax.set_xticks(range(len(drivers)))
        ax.set_xticklabels(drivers, fontsize=7)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels([x for _, x in targets], fontsize=7)
        ax.set_title(ttl, loc='left', fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.colorbar(im, ax=axs, shrink=.8, label='Pearson r (lag 0)')
    fig.text(.01, .01, '* p < 0.05 using the effective sample size (lag-1 autocorrelation of both series)', fontsize=6.5)
    fig.savefig(OUT / 'fig08_climate_linkage.png', bbox_inches='tight')
    plt.close(fig)


def figS1():
    sc = pd.read_csv(OP / 'seasonal_climatology.csv', index_col=0)
    sty = {'TWS (3-solution mean)': ('k', '-', 'o'), 'SM Noah': ('#1b9e77', '-', 's'), 'SM CLSM': ('#d95f02', '-', '^'), 'SM VIC': ('#7570b3', '-', 'D'), 'Lake (level-dependent)': (C_LAKE, '--', 'v')}
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for k, (c, ls, m) in sty.items():
        ax.plot(sc.index, sc[k], color=c, ls=ls, lw=1.4, marker=m, ms=3, label=k)
    ax.axhline(0, color='k', lw=.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(list('JFMAMJJASOND'))
    ax.set_ylabel('Mean seasonal anomaly (mm EWH)')
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / 'figS1_seasonal_cycle.png')
    plt.close(fig)


def figS2():
    ts = pd.read_csv(OT / 'trend_start_end_sensitivity.csv')
    cols = {'none': ('Uncorrected', C_RAW, 'o'), 'JRC': ('Level-dependent', C_COR, 's'), 'JRC_scaled': ('Scaled to full supply', C_SCL, 'D'), 'A8502': ('Constant 8,502 km²', '#777777', '^')}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, v, xl in ((axs[0], 'start', 'Start year (end December 2025)'), (axs[1], 'end', 'End year (start April 2002)')):
        for nm, (lab, c, m) in cols.items():
            g = ts[(ts.series == nm) & (ts.vary == v)]
            ax.plot(g.year, g.slope, color=c, lw=1.2, label=lab)
            ax.scatter(g.year[g.p < .05], g.slope[g.p < .05], color=c, marker=m, s=14, zorder=3)
            ax.scatter(g.year[g.p >= .05], g.slope[g.p >= .05], facecolor='white', edgecolor=c, marker=m, s=14, zorder=3)
        ax.axhline(0, color='k', lw=.5)
        ax.set_xlabel(xl)
        ax.grid(color='#eeeeee')
    axs[0].set_ylabel('Ensemble-mean GWSA trend (mm yr⁻¹)')
    axs[0].legend(fontsize=7, frameon=False)
    fig.text(.99, .01, 'Filled: Hamed–Rao p < 0.05', ha='right', fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / 'figS2_trend_start_end.png')
    plt.close(fig)


def figS3():
    U = pd.read_csv(PRO0 / 'jpl_domain_uncertainty.csv', index_col=0, parse_dates=True)
    e = pd.read_csv(PRO0 / 'gwsa_ensemble_summary.csv', index_col=0, parse_dates=True)
    l = pd.read_csv(OP / 'lake_storage_terms.csv', index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    base.gap(ax)
    ax.fill_between(U.index, U.independent, U.correlated, color='#636363', alpha=.35, lw=0, label='JPL formal 1σ, domain mean (independent–correlated)')
    ax.plot(e.index, (e.gwsa_raw_max - e.gwsa_raw_min) / 2, color=C_RAW, lw=1, label='Half-range of the 9 GWSA members')
    ax.plot(l.index, (l.JRC_scaled - l.JRC).abs(), color=C_LAKE, lw=1, label='Difference, level-dependent and scaled lake terms')
    ax.plot(l.index, l.JRC.abs(), color=C_LAKE, lw=.8, ls=':', label='|Lake storage anomaly| (level-dependent)')
    ax.set_ylabel('mm EWH')
    ax.legend(fontsize=6.5, frameon=False, loc='upper left')
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(OUT / 'figS3_uncertainty_sources.png')
    plt.close(fig)


def graphical_abstract():
    """Three small multiples on one axis, read as an equation: GRACE groundwater - lake storage = corrected groundwater."""
    ens = pd.read_csv(base.PRO / 'gwsa_ensemble_summary.csv', index_col=0, parse_dates=True)
    t = (ens.index - ens.index[0]).days.values / 365.2425
    det = lambda v: v - np.polyval(np.polyfit(t, v.values, 1), t)
    terms = pd.read_csv(OP / 'lake_storage_terms.csv', index_col=0, parse_dates=True).reindex(ens.index)
    raw, lake, l8 = det(ens.gwsa_raw_mean), det(terms.JRC), det(terms.JRC_scaled)
    l6 = lake
    INK, MUTED, RAW, LAKE, COR, WIN = '#222222', '#666666', '#c8553d', '#1f8a5b', '#2a6fb0', '#f0e6c8'
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': .5, 'xtick.major.width': .5, 'ytick.major.width': .5})
    fig = plt.figure(figsize=(5.23, 2.12))
    gs = fig.add_gridspec(1, 3, left=.075, right=.995, top=.80, bottom=.13, wspace=.27)
    win = (pd.Timestamp('2015-01-01'), pd.Timestamp('2016-12-31'))
    axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
    for ax in axes:
        ax.axvspan(*win, color=WIN, lw=0, zorder=0)
        ax.axvspan(*base.GAP, color='#ececec', lw=0, zorder=0)
        ax.axhline(0, color='#888888', lw=.5)
        ax.xaxis.set_major_locator(mdates.YearLocator(10, month=1, day=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlim(pd.Timestamp('2002-01-01'), pd.Timestamp('2026-03-01'))
        ax.set_ylim(-175, 175)
        ax.set_yticks([-150, 0, 150])
        ax.tick_params(length=2, pad=1.5, labelsize=6.5, colors=MUTED)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    a, b, c = axes
    a.set_ylabel('Detrended storage (mm)', fontsize=6.5, color=MUTED, labelpad=1)
    a.plot(base.break_gaps(raw).index, base.break_gaps(raw).values, color=RAW, lw=1)
    b.fill_between(ens.index, l6, l8, color=LAKE, alpha=.35, lw=0)
    b.plot(base.break_gaps(lake).index, base.break_gaps(lake).values, color=LAKE, lw=1)
    c.fill_between(ens.index, raw - l6, raw - l8, color=COR, alpha=.25, lw=0)
    cm = raw - lake
    c.plot(base.break_gaps(cm).index, base.break_gaps(cm).values, color=COR, lw=1)
    for ax, head, sub in ((a, 'GRACE "groundwater"', 'lake not removed'), (b, 'Lake Volta storage', 'Landsat + altimetry'),
                          (c, 'Groundwater', 'lake removed')):
        ax.text(0, 1.20, head, transform=ax.transAxes, fontsize=7.8, fontweight='bold', color=INK, va='bottom')
        ax.text(0, 1.09, sub, transform=ax.transAxes, fontsize=6.5, color=MUTED, va='bottom')
    tx = pd.Timestamp('2016-01-01')
    for ax, txt, y, ty in ((a, '2015–16\ndeficit', raw.loc['2016'].min(), -140), (b, 'Reservoir\ndrawdown', lake.loc['2016'].min(), -140),
                           (c, 'No deficit', cm.loc['2015':'2016'].max(), 150)):
        ax.annotate(txt, xy=(tx, y), xytext=(pd.Timestamp('2008-01-01'), ty), fontsize=6.5, color=INK, ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', lw=.5, color=INK, shrinkB=1))
    p0, p1, p2 = [ax.get_position() for ax in axes]
    for l, r, sym in ((p0.x1, p1.x0, '−'), (p1.x1, p2.x0, '=')):
        fig.text((l + r) / 2 - .006, .47, sym, ha='center', va='center', fontsize=12, fontweight='bold', color=INK)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'graphical_abstract_paper3.{ext}', dpi=600)
    fig.savefig(OUT / 'graphical_abstract_paper3.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    plt.rcParams.update({'font.size': 9})
    w = pd.Timestamp('2015-01-01'), pd.Timestamp('2016-12-31')
    print('2015-16 mean detrended mm: raw', round(raw.loc[w[0]:w[1]].mean(), 1), 'lake', round(lake.loc[w[0]:w[1]].mean(), 1), 'corrected', round(cm.loc[w[0]:w[1]].mean(), 1))


if __name__ == '__main__':
    fig1()
    fig2_curve()
    fig3_lake()
    fig4_footprint()
    fig5_trends()
    fig6_gdsi()
    fig7_windows()
    fig8_climate()
    figS1()
    figS2()
    figS3()
    graphical_abstract()
    print('figures written to', OUT)
