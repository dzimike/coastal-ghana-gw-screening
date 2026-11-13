#!/usr/bin/env python3
"""InSAR diagnostics, comparisons and supplementary figures for the MintPy v2 velocity.

Replaces the InSAR parts of regenerate_supp_figures.py and revision_checks.py, which read the
withdrawn v1 field (velocityERA5.h5, the tropospheric delay rate) or hard-coded its values.
Every number here is computed from data/processed/subsidence/mintpy_v2 and the v2 export
(mintpy_velocity_los_v2.tif: temporal coherence >= 0.7 and average spatial coherence >= 0.5).

Outputs
  outputs/tables/revision_checks_v2/insar_v2_summary.csv     key statistics quoted in the text
  outputs/tables/supplementary_table_S1_insar_coherence_v2.csv
  data/processed/subsidence/gnss_insar_comparison_v2.csv
  data/processed/subsidence/hq_pixel_consistency_v2.csv      internal consistency (not validation)
  outputs/figures/figS1_insar_diagnostics.png, figS3_ps_sbas_comparison.png,
  outputs/figures/figS4_gnss_comparison.png, figS7_connected_components.png
"""
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from mintpy.utils import utils0 as ut
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / 'data/processed/subsidence'
MP = SUB / 'mintpy_v2'
MP1 = SUB / 'mintpy'
VEL_TIF = SUB / 'mintpy_velocity_los_v2.tif'
GRID = ROOT / 'data/processed/risk_index/cgsri_grid10km_v2.gpkg'
S1_OLD = ROOT / 'outputs/tables/supplementary_table_S1_insar_coherence.csv'
TAB = ROOT / 'outputs/tables'
FIGS = ROOT / 'outputs/figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

TO_UTM = Transformer.from_crs('EPSG:4326', 'EPSG:32630', always_xy=True)
POINTS = {'Kotoka Airport': (5.60, -0.19)}  # assumed stable in v1; kept for comparison
summary = []


def note(key, value, desc):
    summary.append(dict(key=key, value=value, description=desc))
    print(f'{key:32s} {value!s:>12}  {desc}')


def dates_of(f):
    return [datetime.strptime(d.decode(), '%Y%m%d') for d in f['date'][:]]


def fit_velocity(path):
    """Least-squares linear rate (mm/yr) of every pixel of a MintPy time-series file."""
    with h5py.File(path, 'r') as f:
        d = dates_of(f)
        t = np.array([(x - d[0]).days / 365.25 for x in d])
        p = np.linalg.pinv(np.c_[t, np.ones_like(t)])[0]
        ts = f['timeseries']
        v = np.zeros(ts.shape[1:], np.float32)
        for r in range(0, ts.shape[1], 250):
            v[r:r + 250] = np.tensordot(p, ts[:, r:r + 250, :], axes=1) * 1000
    return v


def rowcol(src, lat, lon):
    e, n = TO_UTM.transform(lon, lat)
    return src.index(e, n)


def sample(vel, src, lat, lon, radii=(500, 1000, 2000)):
    """Median of valid pixels within the smallest radius that holds >= 5 of them."""
    r0, c0 = rowcol(src, lat, lon)
    yy, xx = np.ogrid[:vel.shape[0], :vel.shape[1]]
    d2 = ((yy - r0) ** 2 + (xx - c0) ** 2) * 80.0 ** 2
    for rad in radii:
        v = vel[(d2 <= rad ** 2) & np.isfinite(vel)]
        if v.size >= 5:
            return float(np.median(v)), int(v.size), rad
    return np.nan, 0, radii[-1]


def temporal_components(pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    return len({find(x) for x in list(parent)})


def network(stack):
    with h5py.File(stack, 'r') as f:
        d12 = f['date'][:].astype(str).tolist()
        keep = f['dropIfgram'][:]
    return [tuple(p) for p, k in zip(d12, keep) if k]


def main():
    with rasterio.open(VEL_TIF) as src:
        vel = src.read(1)
        profile = src.profile
        transform = src.transform
    with h5py.File(MP / 'temporalCoherence.h5', 'r') as f:
        tcoh = f['temporalCoherence'][:]
    with h5py.File(MP / 'avgSpatialCoh.h5', 'r') as f:
        scoh = f['coherence'][:]
    with h5py.File(MP / 'inputs/geometryGeo.h5', 'r') as f:
        inc = f['incidenceAngle'][:]
        azi = f['azimuthAngle'][:]
    with h5py.File(MP / 'velocity.h5', 'r') as f:
        atr = dict(f.attrs)
        vstd = f['velocityStd'][:] * 1000
    valid = np.isfinite(vel)
    src = rasterio.open(VEL_TIF)

    # ── network ──────────────────────────────────────────────────────────────
    p1 = network(MP1 / 'inputs/ifgramStack.h5')
    p2 = network(MP / 'inputs/ifgramStack.h5')
    note('v1_pairs', len(p1), 'interferograms used in v1')
    note('v1_temporal_components', temporal_components(p1), 'disconnected temporal subsets in v1')
    note('v2_pairs', len(p2), 'interferograms used in v2 (20220915 excluded)')
    note('v2_dates', len({d for p in p2 for d in p}), 'acquisitions in v2')
    note('v2_temporal_components', temporal_components(p2), 'temporal subsets in v2')
    note('v2_period', f'{atr["START_DATE"]}-{atr["END_DATE"]}', 'time span')

    # ── coverage and noise ───────────────────────────────────────────────────
    note('valid_px', int(valid.sum()), 'pixels kept (tc>=0.7, avgSpatialCoh>=0.5)')
    note('valid_px_tc_only', int(((tcoh >= 0.7) & (vstd > 0)).sum()), 'pixels with tc>=0.7 alone')
    note('los_median', round(float(np.nanmedian(vel)), 2), 'median LOS velocity, mm/yr')
    note('los_iqr', round(float(np.subtract(*np.nanpercentile(vel, [75, 25]))), 2), 'IQR LOS velocity, mm/yr')
    note('los_p5_p95', '%.1f/%.1f' % tuple(np.nanpercentile(vel, [5, 95])), '5th/95th percentile, mm/yr')
    note('vstd_median', round(float(np.median(vstd[valid])), 2), 'median formal velocity s.d., mm/yr')

    # velocity before/after corrections (same network, same mask)
    v_raw = fit_velocity(MP / 'timeseries.h5')
    v_era = fit_velocity(MP / 'timeseries_ERA5.h5')
    for k, a in [('sd_raw', v_raw), ('sd_era5', v_era), ('sd_era5_demerr', vel)]:
        note(k, round(float(np.nanstd(a[valid])), 2), 'spatial s.d. of LOS velocity, mm/yr')

    # planar ramp over the kept pixels
    rr, cc = np.where(valid)
    sel = slice(None, None, 7)
    e = (transform.c + (cc[sel] + 0.5) * transform.a) / 1000
    n = (transform.f + (rr[sel] + 0.5) * transform.e) / 1000
    y = vel[rr[sel], cc[sel]]
    A = np.c_[e - e.mean(), n - n.mean(), np.ones_like(e)]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - np.var(y - A @ coef) / np.var(y)
    note('ramp_east', round(coef[0] * 100, 2), 'planar gradient, mm/yr per 100 km eastward')
    note('ramp_north', round(coef[1] * 100, 2), 'planar gradient, mm/yr per 100 km northward')
    note('ramp_r2', round(float(r2), 3), 'variance explained by the plane')
    note('corr_height', round(float(np.corrcoef(vel[valid], h5py.File(MP / 'inputs/geometryGeo.h5', 'r')['height'][:][valid])[0, 1]), 3),
         'correlation of LOS velocity with DEM height')
    with h5py.File(MP / 'velocityERA5.h5', 'r') as f:
        vera5 = f['velocity'][:] * 1000
    note('corr_era5_rate', round(float(np.corrcoef(vel[valid], vera5[valid])[0, 1]), 3),
         'correlation with the ERA5 delay rate (v1 raster: 0.88)')

    for name, (lat, lon) in POINTS.items():
        v, k, rad = sample(vel, src, lat, lon)
        note(f'{name.split()[0].lower()}_los', round(v, 2), f'{name} LOS, median of {k} px within {rad} m')

    # ── GNSS comparison (relative to ACRA) ───────────────────────────────────
    gn = pd.read_csv(SUB / 'gnss_insar_validation_v2.csv')[['code', 'lat', 'lon', 'Ve', 'sVe', 'Vn', 'sVn', 'Vu', 'sVu', 'span']]
    rows = []
    for _, s in gn.iterrows():
        r, c = rowcol(src, s.lat, s.lon)
        i, a = float(inc[r, c]), float(azi[r, c])
        ue, un, uu = (ut.enu2los(1, 0, 0, i, az_angle=a), ut.enu2los(0, 1, 0, i, az_angle=a), ut.enu2los(0, 0, 1, i, az_angle=a))
        v, k, rad = sample(vel, src, s.lat, s.lon)
        rows.append(dict(code=s.code, inc_deg=round(i, 2), az_deg=round(a, 2), u_e=ue, u_n=un, u_u=uu,
                         gnss_los=ue * s.Ve + un * s.Vn + uu * s.Vu,
                         s_gnss_los=np.sqrt((ue * s.sVe) ** 2 + (un * s.sVn) ** 2 + (uu * s.sVu) ** 2),
                         insar_los_rel=v, insar_n_px=k, insar_radius_m=rad))
    G = gn.merge(pd.DataFrame(rows), on='code')
    a = G.iloc[0]
    G['insar_los_abs'] = G.insar_los_rel - a.insar_los_rel + a.gnss_los   # tie InSAR to GNSS at ACRA
    G['vu_insar'] = (G.insar_los_abs - G.u_e * G.Ve - G.u_n * G.Vn) / G.u_u
    G['residual_vu'] = G.vu_insar - G.Vu
    G['within_2sigma'] = G.residual_vu.abs() <= 2 * G.sVu
    G.round(4).to_csv(SUB / 'gnss_insar_comparison_v2.csv', index=False)
    print(G[['code', 'u_e', 'u_n', 'u_u', 'insar_los_rel', 'insar_n_px', 'insar_radius_m', 'Vu', 'sVu', 'vu_insar', 'residual_vu']].round(2))
    t = G.iloc[1:].dropna(subset=['residual_vu'])
    note('gnss_n_compared', len(t), 'stations compared (ACRA is the tie point)')
    note('gnss_resid_rms', round(float(np.sqrt((t.residual_vu ** 2).mean())), 2), 'RMS Vu residual, mm/yr')
    note('gnss_within_2sigma', int(t.within_2sigma.sum()), 'stations with |residual| <= 2 sigma(GNSS Vu)')

    # ── per-cell statistics (Table S1) and high-quality pixel consistency ────
    g = gpd.read_file(GRID).to_crs(profile['crs'])
    hq = valid & (tcoh >= 0.9) & (scoh >= 0.8)
    stack = np.stack([vel, np.where(valid, tcoh, np.nan), np.where(hq, vel, np.nan)]).astype('float32')
    prof = dict(profile, count=3, nodata=np.nan)
    recs = []
    with MemoryFile() as mf, mf.open(**prof) as ds:
        ds.write(stack)
        for i in np.where(g.insar_n_valid >= 10)[0]:
            img, _ = rio_mask(ds, [g.geometry.iloc[i].__geo_interface__], crop=True, all_touched=True, nodata=np.nan)
            v, tc, h = (img[j][np.isfinite(img[j])] for j in range(3))
            recs.append(dict(cell_id=g.cell_id.iloc[i], centroid_lat_deg=g.centroid_lat.iloc[i],
                             centroid_lon_deg=g.centroid_lon.iloc[i], n_valid_pixels=v.size,
                             mean_temporal_coherence=tc.mean(), mean_LOS_velocity_mmyr=v.mean(),
                             std_LOS_velocity_mmyr=v.std(ddof=1), cgsri_blending_weight=g.insar_weight.iloc[i],
                             n_hq_pixels=h.size, hq_mean_vel=h.mean() if h.size >= 10 else np.nan))
    T = pd.DataFrame(recs)
    lc = pd.read_csv(S1_OLD)[['cell_id', 'dominant_land_cover_class']]
    T = T.merge(lc, on='cell_id', how='left').fillna({'dominant_land_cover_class': 'not classified'})
    cols = ['cell_id', 'centroid_lat_deg', 'centroid_lon_deg', 'dominant_land_cover_class', 'n_valid_pixels',
            'mean_temporal_coherence', 'mean_LOS_velocity_mmyr', 'std_LOS_velocity_mmyr', 'cgsri_blending_weight']
    T[cols].round(4).to_csv(TAB / 'supplementary_table_S1_insar_coherence_v2.csv', index=False)
    assert np.allclose(T.mean_LOS_velocity_mmyr.values,
                       g.set_index('cell_id').loc[T.cell_id, 'insar_vel_mmyr'].values, atol=1e-3)
    note('cells_insar', len(T), 'grid cells with >= 10 valid pixels')
    note('cells_insar_vel_range', '%.1f/%.1f' % (T.mean_LOS_velocity_mmyr.min(), T.mean_LOS_velocity_mmyr.max()),
         'min/max cell-mean LOS velocity, mm/yr')
    H = T.dropna(subset=['hq_mean_vel'])
    H[['cell_id', 'n_valid_pixels', 'n_hq_pixels', 'mean_LOS_velocity_mmyr', 'hq_mean_vel']].round(4).to_csv(
        SUB / 'hq_pixel_consistency_v2.csv', index=False)
    res = H.hq_mean_vel - H.mean_LOS_velocity_mmyr
    note('hq_cells', len(H), 'cells with >= 10 high-quality pixels (tc>=0.9, avgSpatialCoh>=0.8)')
    note('hq_r', round(float(np.corrcoef(H.hq_mean_vel, H.mean_LOS_velocity_mmyr)[0, 1]), 3), 'r, HQ vs all-pixel cell mean')
    note('hq_rmse', round(float(np.sqrt((res ** 2).mean())), 2), 'RMS difference, mm/yr')

    pd.DataFrame(summary).to_csv(TAB / 'revision_checks_v2/insar_v2_summary.csv', index=False)

    # ── Fig. S1 ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    bins = np.linspace(-60, 60, 81)
    for a_, lab, col in [(v_raw, 'No correction', '#bbbbbb'), (v_era, 'ERA5', '#92c5de'), (vel, 'ERA5 + DEM error', '#2166ac')]:
        ax[0, 0].hist(np.clip(a_[valid], -60, 60), bins=bins, density=True, histtype='stepfilled' if col == '#2166ac' else 'step',
                      color=col, lw=1.4, alpha=0.8, label=f'{lab} (s.d. {np.nanstd(a_[valid]):.1f})')
    ax[0, 0].set(xlabel='LOS velocity (mm yr⁻¹)', ylabel='Probability density')
    ax[0, 0].legend(fontsize=8)
    lat_rows = np.array([Transformer.from_crs('EPSG:32630', 'EPSG:4326', always_xy=True)
                         .transform(transform.c + 1000 * 80, transform.f + (r + 0.5) * transform.e)[1] for r in range(vel.shape[0])])
    edges = np.arange(np.floor(lat_rows.min() * 5) / 5, lat_rows.max() + 0.2, 0.2)
    cnt = np.histogram(lat_rows, bins=edges, weights=valid.sum(1))[0]
    ax[0, 1].barh((edges[:-1] + edges[1:]) / 2, cnt / 1e3, height=0.18, color='#4393c3')
    ax[0, 1].set(xlabel='Valid pixels (×10³)', ylabel='Latitude (°N)')
    im = ax[1, 0].imshow(np.where(valid, vel, np.nan), cmap='RdBu', vmin=-20, vmax=20, interpolation='nearest')
    ax[1, 0].plot(int(atr['REF_X']), int(atr['REF_Y']), 'k*', ms=8)
    ax[1, 0].set(xlabel='Column', ylabel='Row')
    plt.colorbar(im, ax=ax[1, 0], shrink=0.8, label='LOS velocity (mm yr⁻¹)')
    with h5py.File(MP / 'timeseries_ERA5_demErr.h5', 'r') as f:
        d = dates_of(f)
        for cls, col in [('Urban/built-up', '#e07b39'), ('Mixed cropland/settlement', '#1f6db5'), ('Cropland/savanna', '#4dac26')]:
            sub = T[T.dominant_land_cover_class == cls]
            if sub.empty:
                continue
            row = sub.sort_values('n_valid_pixels').iloc[-1]
            r0, c0 = rowcol(src, row.centroid_lat_deg, row.centroid_lon_deg)
            r0, c0 = max(r0 - 62, 0), max(c0 - 62, 0)
            m = valid[r0:r0 + 125, c0:c0 + 125]
            ts = f['timeseries'][:, r0:r0 + 125, c0:c0 + 125][:, m].mean(1) * 1000
            ax[1, 1].plot(d, ts - ts[0], lw=1, color=col, label=f'{cls} ({row.cell_id}, {m.sum()} px)')
    ax[1, 1].axhline(0, color='k', lw=0.5, ls='--')
    ax[1, 1].set(xlabel='Date', ylabel='Cumulative LOS displacement (mm)')
    ax[1, 1].legend(fontsize=7)
    for a_, lab in zip(ax.flat, 'abcd'):
        a_.set_title(f'({lab})', loc='left', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGS / 'figS1_insar_diagnostics.png')
    plt.close()

    # ── Fig. S3: high-quality pixels vs all valid pixels ─────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    a1.scatter(H.mean_LOS_velocity_mmyr, H.hq_mean_vel, s=18, c='#2166ac')
    lim = np.nanmax(np.abs(np.r_[H.mean_LOS_velocity_mmyr, H.hq_mean_vel])) * 1.1
    a1.plot([-lim, lim], [-lim, lim], 'k--', lw=0.6)
    a1.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel='Cell mean, all valid pixels (mm yr⁻¹)',
           ylabel='Cell mean, high-quality pixels (mm yr⁻¹)')
    a2.hist(res, bins=20, color='#92c5de')
    a2.set(xlabel='High-quality − all-pixel mean (mm yr⁻¹)', ylabel='Cells')
    for a_, lab in zip((a1, a2), 'ab'):
        a_.set_title(f'({lab})', loc='left', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGS / 'figS3_ps_sbas_comparison.png')
    plt.close()

    # ── Fig. S4: GNSS ────────────────────────────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(G))
    a1.bar(x - 0.2, G.gnss_los - a.gnss_los, 0.38, yerr=G.s_gnss_los, color='#999999', label='GNSS')
    a1.bar(x + 0.2, G.insar_los_rel - a.insar_los_rel, 0.38, color='#2166ac', label='InSAR')
    a1.set(xticks=x, xticklabels=G.code, ylabel='LOS velocity relative to ACRA (mm yr⁻¹)')
    a1.legend(fontsize=8)
    a2.bar(x - 0.2, G.Vu, 0.38, yerr=G.sVu, color='#999999', label='GNSS Vu')
    a2.bar(x + 0.2, G.vu_insar, 0.38, color='#2166ac', label='InSAR Vu (GNSS horizontals)')
    a2.set(xticks=x, xticklabels=G.code, ylabel='Vertical velocity (mm yr⁻¹)')
    a2.legend(fontsize=8)
    for a_, lab in zip((a1, a2), 'ab'):
        a_.axhline(0, color='k', lw=0.5)
        a_.set_title(f'({lab})', loc='left', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGS / 'figS4_gnss_comparison.png')
    plt.close()

    # ── Fig. S7: v1 vs v2 temporal networks ──────────────────────────────────
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for a_, pairs, mp, lab in [(axs[0], p1, MP1, 'v1'), (axs[1], p2, MP, 'v2')]:
        dd = sorted({x for p in pairs for x in p})
        with h5py.File(mp / 'timeseries.h5', 'r') as f:   # per-date perpendicular baseline
            base = dict(zip(f['date'][:].astype(str), f['bperp'][:]))
        for d1, d2 in pairs:
            if d1 in base and d2 in base:
                a_.plot([datetime.strptime(d1, '%Y%m%d'), datetime.strptime(d2, '%Y%m%d')], [base[d1], base[d2]],
                        lw=0.5, color='#2166ac', alpha=0.6)
        a_.plot([datetime.strptime(x, '%Y%m%d') for x in dd if x in base], [base[x] for x in dd if x in base], 'k.', ms=3)
        name = 'Final network' if lab == 'v2' else 'Initial network'
        a_.set_title(f'({"ab"[lab == "v2"]}) {name}', loc='left', fontweight='bold')
        print(f'{name}: {len(pairs)} interferograms, {len(dd)} dates, '
              f'{temporal_components(pairs)} connected subset(s)')
        a_.set_xlabel('Date')
    axs[0].set_ylabel('Perpendicular baseline (m)')
    plt.tight_layout()
    plt.savefig(FIGS / 'figS7_connected_components.png')
    plt.close()
    print('figures written to', FIGS)


if __name__ == '__main__':
    main()
