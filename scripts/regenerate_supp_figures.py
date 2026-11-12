#!/usr/bin/env python3
"""
Regenerate Supplementary Figures S1–S4 for Paper 1 (STOTEN submission).

Changes vs. original figures
- Removed fig.suptitle() from all four figures
- Replaced long descriptive ax.set_title() text with concise panel-letter labels
- S3: removed bottom footnote text block
- S4: removed bottom formula note box
- Font sizes increased throughout (axis labels 12 pt, tick labels 10 pt,
  panel labels 13 pt bold)

Outputs (overwrite in-place):
  outputs/manuscript/figS1_insar_diagnostics.png
  outputs/manuscript/figS2_cgsri_components.png
  outputs/manuscript/figS3_ps_sbas_comparison.png
  outputs/manuscript/figS4_gnss_insar_validation.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import geopandas as gpd
import h5py
from pathlib import Path
from pyproj import Transformer

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parents[1]
MINTPY = BASE / "data/processed/subsidence/mintpy"
SUB    = BASE / "data/processed/subsidence"
RISK   = BASE / "data/processed/risk_index"
BDRY   = BASE / "data/raw/boundaries"
COHTAB = BASE / "outputs/tables/supplementary_table_S1_insar_coherence.csv"
OUTDIR = BASE / "outputs/manuscript"
DPI    = 300

# ── shared style ──────────────────────────────────────────────────────────────
FS_PANEL  = 13   # panel letter labels  e.g. "(a)"
FS_LABEL  = 12   # axis labels
FS_TICK   = 10   # tick labels / legend
FS_ANNOT  =  9   # in-plot annotations

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_PANEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_TICK,
})

# ── UTM-30N  ↔  WGS84 transformer ────────────────────────────────────────────
to_wgs84  = Transformer.from_crs("EPSG:32630", "EPSG:4326", always_xy=True)
to_utm30n = Transformer.from_crs("EPSG:4326",  "EPSG:32630", always_xy=True)

# MintPy raster origin / step (EPSG:32630)
X0, Y0   = 684400.0, 822800.0   # easting/northing of pixel (0,0)
DX, DY   = 80.0,     -80.0      # cell size (m); Y decreases downward


def latlon_to_rowcol(lat_deg, lon_deg):
    """Convert WGS84 lat/lon to MintPy raster row, col (nearest integer)."""
    e, n = to_utm30n.transform(lon_deg, lat_deg)
    col  = int(round((e - X0) / DX))
    row  = int(round((n - Y0) / DY))
    return row, col


def rowcol_to_latlon(row, col):
    """Convert raster row/col to WGS84 lat/lon."""
    e = X0 + col * DX
    n = Y0 + row * DY
    lon, lat = to_wgs84.transform(e, n)
    return lat, lon


# =============================================================================
# Figure S1 — InSAR velocity diagnostics (4 panels)
# =============================================================================
def make_figS1():
    print("Generating S1 …")

    # ── load masks and velocity arrays ────────────────────────────────────────
    with h5py.File(MINTPY / "temporalCoherence.h5") as f:
        tc = f["temporalCoherence"][:]
    valid = tc >= 0.7                              # (3518, 1927) bool

    # ERA5-corrected velocity (the stored product)
    with h5py.File(MINTPY / "velocityERA5.h5") as f:
        vel_era = f["velocity"][:] * 1e3           # m/yr → mm/yr

    # Pre-ERA5 velocity: approximate by (last - first) / dt from uncorrected timeseries
    with h5py.File(MINTPY / "timeseries.h5") as f:
        raw_d   = [d.decode() for d in f["date"][:]]
        ts_fst  = f["timeseries"][0][:]  * 1e3    # mm
        ts_lst  = f["timeseries"][-1][:] * 1e3    # mm
    from datetime import datetime as _dt
    dt_yr   = (_dt.strptime(raw_d[-1], "%Y%m%d") -
               _dt.strptime(raw_d[0],  "%Y%m%d")).days / 365.25
    vel_raw = (ts_lst - ts_fst) / dt_yr            # mm/yr

    vel_raw_v = vel_raw[valid]
    vel_era_v = vel_era[valid]

    # ── panel (b): valid-pixel count by 0.2° latitude band ───────────────────
    n_rows = valid.shape[0]
    # northing of each row centre, convert to lat
    northings = Y0 + np.arange(n_rows) * DY
    _, lats_row = to_wgs84.transform(
        np.full(n_rows, X0 + 963 * DX),  # mid-column
        northings
    )
    row_counts = valid.sum(axis=1)   # valid pixels per row

    lat_min, lat_max = lats_row.min(), lats_row.max()
    edges = np.arange(np.floor(lat_min / 0.2) * 0.2,
                      np.ceil( lat_max / 0.2) * 0.2 + 0.001, 0.2)
    band_labels = (edges[:-1] + edges[1:]) / 2
    band_counts = np.zeros(len(band_labels), dtype=np.int64)
    for r in range(n_rows):
        lat_r = lats_row[r]
        idx   = np.searchsorted(edges, lat_r, side="right") - 1
        if 0 <= idx < len(band_counts):
            band_counts[idx] += int(row_counts[r])

    # ── panel (c): LOS velocity at named reference / stable points ───────────
    # Values from manuscript Section 3.4 / 4.2 (ERA5-corrected, relative LOS)
    ref_pts = {
        "SBAS reference\n(5.55°N, −0.42°E)":  0.0,
        "Kotoka Airport\n(5.60°N, −0.19°E)":  +9.0,
        "Rock outcrop A\n(7.1°N)":             +1.4,
        "Rock outcrop B\n(7.3°N)":             -1.9,
    }
    pt_colors = ["#636363", "#e07b39", "#2166ac", "#2166ac"]

    # ── panel (d): time series for 3 representative cells ────────────────────
    # Cells chosen from coherence table by land-cover class
    coh_df = pd.read_csv(COHTAB)
    urban_row  = coh_df[coh_df.dominant_land_cover_class == "Urban/built-up"].sort_values(
        "n_valid_pixels_coh07", ascending=False).iloc[0]
    mixed_row  = coh_df[coh_df.dominant_land_cover_class == "Mixed cropland/settlement"].sort_values(
        "n_valid_pixels_coh07", ascending=False).iloc[0]
    crop_row   = coh_df[coh_df.dominant_land_cover_class == "Cropland/savanna"].sort_values(
        "n_valid_pixels_coh07", ascending=False).iloc[0]
    ts_cells = [
        ("Urban", urban_row,  "#e07b39"),
        ("Mixed cropland", mixed_row,  "#1f6db5"),
        ("Cropland/savanna", crop_row, "#4dac26"),
    ]

    with h5py.File(MINTPY / "timeseries_ERA5.h5") as f:
        raw_dates = [d.decode() for d in f["date"][:]]
        dates = pd.to_datetime(raw_dates, format="%Y%m%d")
        half  = 62   # pixels each side of cell centroid (≈10 km / 80 m / 2)
        ts_data = []
        for label, row_meta, col in ts_cells:
            lat  = row_meta.centroid_lat_deg
            lon  = row_meta.centroid_lon_deg
            rr, cc = latlon_to_rowcol(lat, lon)
            r0, r1 = max(0, rr - half), min(3517, rr + half + 1)
            c0, c1 = max(0, cc - half), min(1926, cc + half + 1)
            patch_ts    = f["timeseries"][:, r0:r1, c0:c1]    # (T, H, W)
            patch_valid = valid[r0:r1, c0:c1]                  # (H, W)
            n_valid_pix = patch_valid.sum()
            if n_valid_pix == 0:
                ts_mean = np.zeros(len(dates))
            else:
                masked = np.where(patch_valid[np.newaxis], patch_ts, np.nan)
                ts_mean = np.nanmean(masked.reshape(len(dates), -1), axis=1) * 1e3  # m→mm
                # Remove outlier dates (phase-unwrapping spikes)
                med  = np.nanmedian(ts_mean)
                mad  = np.nanmedian(np.abs(ts_mean - med)) * 1.4826
                clip = max(mad * 5, 150)          # at least ±150 mm window
                ts_mean = np.where(np.abs(ts_mean - med) > clip, np.nan, ts_mean)
                # Linear interpolation over flagged dates
                idx = np.arange(len(ts_mean))
                good = np.isfinite(ts_mean)
                if good.sum() > 2:
                    ts_mean = np.interp(idx, idx[good], ts_mean[good])
            ts_data.append((label, col, ts_mean))

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                           left=0.09, right=0.97, top=0.95, bottom=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) velocity histograms before / after ERA5
    # clip to ±100 mm/yr for legibility; both distributions are centred near 0
    bins = np.linspace(-100, 100, 61)
    ax_a.hist(vel_raw_v.clip(-100, 100), bins=bins, density=True,
              color="#aec7e8", alpha=0.75, label="Before ERA5 correction", linewidth=0)
    ax_a.hist(vel_era_v.clip(-100, 100), bins=bins, density=True,
              color="#1f6db5", alpha=0.75, label="After ERA5 correction",  linewidth=0)
    raw_std = np.std(vel_raw_v); era_std = np.std(vel_era_v)
    ax_a.text(0.97, 0.96,
              f"Before: σ = {raw_std:.0f} mm yr⁻¹\nAfter:   σ = {era_std:.0f} mm yr⁻¹",
              transform=ax_a.transAxes, fontsize=FS_ANNOT, va="top", ha="right",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#aaa"))
    ax_a.set_xlabel("LOS velocity (mm yr⁻¹)")
    ax_a.set_ylabel("Probability density")
    ax_a.legend(loc="upper left")
    ax_a.set_title("(a)", fontweight="bold", loc="left", pad=4)
    ax_a.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # (b) valid pixel count by 0.2° latitude band
    ax_b.barh(band_labels, band_counts / 1e3, height=0.18,
              color="#4393c3", edgecolor="white", linewidth=0.4)
    ax_b.set_xlabel("Valid pixels (×10³)")
    ax_b.set_ylabel("Latitude (°N)")
    ax_b.set_title("(b)", fontweight="bold", loc="left", pad=4)
    ax_b.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax_b.grid(axis="x", lw=0.4, alpha=0.5)

    # (c) LOS velocity at reference and stable-area points
    names  = list(ref_pts.keys())
    values = list(ref_pts.values())
    colors = [("tomato" if v > 3 else "steelblue" if v < -1 else "#888888")
              for v in values]
    bars = ax_c.bar(range(len(names)), values, color=colors,
                    edgecolor="white", linewidth=0.5, width=0.55)
    ax_c.axhline(0, color="k", lw=0.8)
    ax_c.set_xticks(range(len(names)))
    ax_c.set_xticklabels(names, fontsize=8.5)
    ax_c.set_ylabel("ERA5-corrected LOS velocity (mm yr⁻¹)")
    ax_c.set_title("(c)", fontweight="bold", loc="left", pad=4)
    for bar, val in zip(bars, values):
        offset = 0.25 if val >= 0 else -0.8
        ax_c.text(bar.get_x() + bar.get_width()/2, val + offset,
                  f"{val:+.1f}", ha="center", va="bottom" if val>=0 else "top",
                  fontsize=FS_ANNOT)
    ax_c.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # (d) displacement time series
    for label, color, ts in ts_data:
        ax_d.plot(dates, ts, lw=1.2, label=label, color=color)
    ax_d.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax_d.set_xlabel("Date")
    ax_d.set_ylabel("Cumulative LOS displacement (mm)")
    ax_d.legend()
    ax_d.set_title("(d)", fontweight="bold", loc="left", pad=4)
    ax_d.xaxis.set_major_locator(mticker.MaxNLocator(6))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax_d.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax_d.grid(axis="y", lw=0.4, alpha=0.4)

    out = OUTDIR / "figS1_insar_diagnostics.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# =============================================================================
# Figure S2 — CGSRI component contribution maps (5 panels)
# =============================================================================
def make_figS2():
    print("Generating S2 …")

    gdf_risk = gpd.read_file(RISK / "cgsri_grid10km.gpkg")
    ghana    = gpd.read_file(BDRY / "ghana_boundary.gpkg").to_crs(epsg=4326)
    coast50  = gpd.read_file(BDRY / "study_area_coastal_50km.gpkg").to_crs(epsg=4326)
    if "geometry" not in gdf_risk.columns:
        grid = gpd.read_file(BDRY / "analysis_grid_10km.gpkg").to_crs(epsg=4326)
        gdf_risk = grid.merge(
            pd.read_csv(RISK / "cgsri_grid10km.csv"), on="cell_id", how="inner")
    else:
        gdf_risk = gdf_risk.to_crs(epsg=4326)

    components = [
        ("gw_hazard",      "(a) GW Hazard",         "#1f6db5"),
        ("sub_hazard",     "(b) Subsidence Hazard",  "#7b3294"),
        ("climate_stress", "(c) Climate Stress",     "#e07b39"),
        ("exposure",       "(d) Exposure",           "#b2182b"),
        ("vulnerability",  "(e) Vulnerability",      "#4dac26"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    cmap = "YlOrRd"

    for ax, (col, title, col_color) in zip(axes, components):
        ghana.plot(ax=ax, facecolor="#f5f0e8", edgecolor="#888", lw=0.5, zorder=0)
        coast50.plot(ax=ax, facecolor="none", edgecolor="#538135",
                     lw=0.6, alpha=0.5, zorder=1)
        gdf_risk.plot(column=col, ax=ax, cmap=cmap, vmin=0, vmax=1,
                      legend=False, zorder=2, linewidth=0, alpha=0.85)
        ax.set_xlim(-3.5, 1.3)
        ax.set_ylim(4.2, 7.5)
        ax.set_title(title, fontweight="bold", fontsize=FS_PANEL,
                     color=col_color, pad=5)
        ax.set_xlabel("°E", fontsize=FS_TICK)
        ax.set_ylabel("°N" if ax is axes[0] else "", fontsize=FS_TICK)
        ax.tick_params(labelsize=FS_TICK - 1)
        vals = gdf_risk[col].dropna()
        ax.annotate(f"μ={vals.mean():.2f}\nσ={vals.std():.2f}",
                    xy=(0.04, 0.04), xycoords="axes fraction",
                    fontsize=FS_ANNOT, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # shared horizontal colorbar below all panels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.15, wspace=0.08)
    cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.025])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Normalised score [0–1]", fontsize=FS_TICK)
    cb.ax.tick_params(labelsize=FS_ANNOT)

    out = OUTDIR / "figS2_cgsri_components.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# =============================================================================
# Figure S3 — PS vs SBAS internal consistency (2 panels)
# =============================================================================
def make_figS3():
    print("Generating S3 …")

    df = pd.read_csv(SUB / "ps_vs_sbas_comparison_matched.csv")

    sbas = df["sbas_mean_vel"].values
    ps   = df["ps_mean_vel"].values
    coh  = df["ps_mean_coh"].values
    resid = ps - sbas

    r    = np.corrcoef(sbas, ps)[0, 1]
    rmse = np.sqrt(np.mean(resid ** 2))
    bias = resid.mean()

    # coherence quartile colouring
    q25, q75 = np.percentile(coh, [25, 75])
    def coh_color(c):
        if c < q25:      return "#d73027"
        elif c < 0.875:  return "#f4a582"
        elif c < q75:    return "#92c5de"
        else:            return "#4575b4"

    colors_sc = [coh_color(c) for c in coh]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5),
                                      gridspec_kw={"wspace": 0.32})

    # (a) scatter PS vs SBAS
    ax_a.scatter(sbas, ps, c=colors_sc, s=32, alpha=0.85, zorder=3,
                 edgecolors="white", linewidths=0.3)
    lim = max(abs(sbas).max(), abs(ps).max()) * 1.08
    ax_a.plot([-lim, lim], [-lim, lim], "k-",  lw=1.2, zorder=2, label="1:1")
    ax_a.plot([-lim, lim], [-lim + rmse, lim + rmse], "k--", lw=0.8, alpha=0.5)
    ax_a.plot([-lim, lim], [-lim - rmse, lim - rmse], "k--", lw=0.8, alpha=0.5,
              label=f"±RMSE ({rmse:.1f} mm/yr)")
    ax_a.set_xlim(-lim, lim);  ax_a.set_ylim(-lim, lim)
    ax_a.set_xlabel("SBAS mean LOS velocity (mm yr⁻¹)")
    ax_a.set_ylabel("PS-candidate mean LOS velocity (mm yr⁻¹)")
    ax_a.set_title("(a)", fontweight="bold", loc="left", pad=4)

    # coherence legend patches
    patches = [
        mpatches.Patch(color="#d73027", label="Coh Q1 (low)"),
        mpatches.Patch(color="#f4a582", label="Coh Q2"),
        mpatches.Patch(color="#92c5de", label="Coh Q3"),
        mpatches.Patch(color="#4575b4", label="Coh Q4 (high)"),
    ]
    # annotate outlier C034R011
    if "C034R011" in df["cell_id"].values:
        idx = df.index[df["cell_id"] == "C034R011"][0]
        ax_a.annotate("C034R011",
                      xy=(sbas[idx], ps[idx]),
                      xytext=(sbas[idx] + 2, ps[idx] - 4),
                      fontsize=FS_ANNOT,
                      arrowprops=dict(arrowstyle="-", color="#555", lw=0.8),
                      bbox=dict(fc="#ffffcc", boxstyle="round,pad=0.2",
                                ec="#888", lw=0.5))

    # stats box — bottom left, below the data cloud
    ax_a.text(0.04, 0.04,
              f"r = {r:.3f}  (p < 0.001)\nRMSE = {rmse:.1f} mm yr⁻¹\n"
              f"Bias = {bias:+.2f} mm yr⁻¹\nn = {len(df)} matched cells",
              transform=ax_a.transAxes, fontsize=FS_ANNOT,
              va="bottom", ha="left",
              bbox=dict(boxstyle="round", fc="white", alpha=0.88, ec="#aaa"))
    # coherence-colour legend — lower right
    leg1 = ax_a.legend(handles=patches, fontsize=FS_ANNOT, loc="lower right",
                       title="Spatial coherence", title_fontsize=FS_ANNOT)
    # line-style legend — upper left
    ax_a.legend(fontsize=FS_ANNOT, loc="upper left")
    ax_a.add_artist(leg1)
    ax_a.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax_a.axvline(0, color="k", lw=0.4, alpha=0.3)

    # (b) residual histogram
    bins_r = np.linspace(resid.min() - 0.5, resid.max() + 0.5, 25)
    ax_b.hist(resid, bins=bins_r, color="#4393c3", edgecolor="white",
              linewidth=0.5, zorder=2)
    ax_b.axvline(0,    color="k",        lw=1.0, zorder=3)
    ax_b.axvline(bias, color="#b2182b",  lw=1.5, ls="--", zorder=3,
                 label=f"Mean bias = {bias:+.2f} mm yr⁻¹")
    ax_b.axvline( rmse, color="#636363", lw=1.0, ls=":",  zorder=3,
                 label=f"±RMSE = {rmse:.1f} mm yr⁻¹")
    ax_b.axvline(-rmse, color="#636363", lw=1.0, ls=":",  zorder=3)
    ax_b.set_xlabel("PS − SBAS residual (mm yr⁻¹)")
    ax_b.set_ylabel("Number of 10 km cells")
    ax_b.set_title("(b)", fontweight="bold", loc="left", pad=4)
    ax_b.legend(fontsize=FS_ANNOT)
    ax_b.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax_b.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    out = OUTDIR / "figS3_ps_sbas_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# =============================================================================
# Figure S4 — GNSS vs InSAR vertical land motion comparison (4 panels)
# =============================================================================
def make_figS4():
    print("Generating S4 …")

    gdf = pd.read_csv(SUB / "gnss_insar_validation_v2.csv")
    stations = gdf["code"].tolist()

    # LOS unit vector (ascending, incidence 38°, heading −12°)
    e_E, e_N, e_U = 0.603, 0.128, 0.788

    # Relative ΔLOS (relative to ACRA anchor)
    gnss_abs_los  = gdf["GNSS_LOS"].values
    insar_abs_los = gdf["InSAR_LOS_abs"].values
    gnss_ref      = gnss_abs_los[0]    # ACRA
    insar_ref     = insar_abs_los[0]

    gnss_dlos  = gnss_abs_los  - gnss_ref
    insar_dlos = insar_abs_los - insar_ref
    gnss_1sig  = gdf["sGNSS_LOS"].values

    gnss_vu    = gdf["Vu"].values
    gnss_svu   = gdf["sVu"].values
    insar_vu   = gdf["Vu_from_InSAR"].values
    spans      = gdf["span"].values
    resid_vu   = gdf["residual_Vu"].values
    within_2s  = np.abs(resid_vu) <= 2 * gnss_svu

    # colours per station
    st_colors = ["#1f6db5", "#4dac26", "#e07b39", "#7b3294"]

    x    = np.arange(len(stations))
    w    = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.32})
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # (a) ΔLOS relative to ACRA
    ax_a.bar(x - w/2, gnss_dlos,  width=w, color=st_colors, label="GNSS ΔLOS",
             alpha=0.85, zorder=2)
    ax_a.bar(x + w/2, insar_dlos, width=w, color=st_colors, label="InSAR ΔLOS",
             hatch="///", alpha=0.6, zorder=2)
    ax_a.errorbar(x - w/2, gnss_dlos, yerr=gnss_1sig * 1,
                  fmt="none", color="k", capsize=4, lw=1.2, zorder=3,
                  label="GNSS 1σ")
    ax_a.axhline(0, color="k", lw=0.7)
    ax_a.set_xticks(x); ax_a.set_xticklabels(stations)
    ax_a.set_ylabel("ΔLOS relative to ACRA (mm yr⁻¹)")
    ax_a.set_title("(a)", fontweight="bold", loc="left", pad=4)
    # explicit y-limits so KUKU insar_dlos bar (≈−12.3) is fully visible
    all_a = np.concatenate([gnss_dlos, insar_dlos, gnss_dlos + gnss_1sig, gnss_dlos - gnss_1sig])
    ax_a.set_ylim(np.nanmin(all_a) * 1.18, np.nanmax(all_a) * 1.35 + 0.5)
    ax_a.text(0.04, 0.96, "ACRA is anchor: ΔLOS = 0 by definition",
              transform=ax_a.transAxes, fontsize=FS_ANNOT,
              va="top", color="#636363", style="italic")
    handles, labels = ax_a.get_legend_handles_labels()
    ax_a.legend(handles, labels, fontsize=FS_ANNOT, loc="lower left")
    ax_a.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # (b) Absolute Vu comparison
    ax_b.bar(x - w/2, gnss_vu,  width=w, color=st_colors, alpha=0.85, zorder=2,
             label="GNSS Vᵤ")
    ax_b.bar(x + w/2, insar_vu, width=w, color=st_colors, hatch="///",
             alpha=0.6, zorder=2, label="InSAR-derived Vᵤ")
    ax_b.errorbar(x - w/2, gnss_vu, yerr=gnss_svu,
                  fmt="none", color="k", capsize=4, lw=1.2, zorder=3)
    ax_b.axhline(0, color="k", lw=0.7)
    ax_b.set_xticks(x); ax_b.set_xticklabels(stations)
    ax_b.set_ylabel("Vertical land motion Vᵤ (mm yr⁻¹)")
    ax_b.set_title("(b)", fontweight="bold", loc="left", pad=4)
    # explicit y-limits so KUKU insar_vu bar (≈−14.8) is fully visible
    all_b = np.concatenate([gnss_vu, insar_vu, gnss_vu + gnss_svu, gnss_vu - gnss_svu])
    ax_b.set_ylim(np.nanmin(all_b) * 1.18, np.nanmax(all_b) * 1.6 + 0.5)
    ax_b.legend(fontsize=FS_ANNOT)
    ax_b.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # (c) Residual Vu (InSAR − GNSS)
    resid_colors = ["#1a9641" if w2s else "#d7191c" for w2s in within_2s]
    ax_c.bar(x, resid_vu, color=resid_colors, alpha=0.85, zorder=2)
    ax_c.errorbar(x, resid_vu, yerr=2 * gnss_svu,
                  fmt="none", color="k", capsize=5, lw=1.2, zorder=3,
                  label="2σ GNSS uncertainty")
    ax_c.axhline(0, color="k", lw=0.7)
    ax_c.set_xticks(x); ax_c.set_xticklabels(stations)
    ax_c.set_ylabel("Residual Vᵤ (InSAR − GNSS)  mm yr⁻¹")
    ax_c.set_title("(c)", fontweight="bold", loc="left", pad=4)
    # explicit y-limits so KUKU residual + 2σ error bar fully visible, plus span labels below
    all_c_lo = np.nanmin(resid_vu - 2 * gnss_svu)
    all_c_hi = np.nanmax(resid_vu + 2 * gnss_svu)
    c_bot = all_c_lo * 1.2 - 2.5   # extra room for span labels
    c_top = max(all_c_hi * 1.4, 6)
    ax_c.set_ylim(c_bot, c_top)
    for i, sp in enumerate(spans):
        ax_c.text(i, c_bot + 0.5, f"{sp:.1f} yr",
                  ha="center", va="bottom", fontsize=FS_ANNOT, color="#555")
    ax_c.legend(fontsize=FS_ANNOT)
    ax_c.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # (d) Validation summary table
    ax_d.axis("off")
    ax_d.set_title("(d)", fontweight="bold", loc="left", pad=4)
    col_labels = ["Site", "GNSS Vᵤ\n(mm/yr)", "GNSS σ\n(mm/yr)",
                  "Span\n(yr)", "InSAR Vᵤ\n(mm/yr)",
                  "Residual\n(mm/yr)", "≤2σ?"]
    col_widths = [0.09, 0.17, 0.15, 0.10, 0.18, 0.17, 0.14]
    table_data = []
    for i, row in gdf.iterrows():
        w2s_sym  = "✓" if within_2s[i] else "✗"
        table_data.append([
            row["code"],
            f"{row['Vu']:.2f}",
            f"±{row['sVu']:.2f}",
            f"{row['span']:.1f}",
            f"{row['Vu_from_InSAR']:.2f}",
            f"{row['residual_Vu']:+.2f}",
            w2s_sym,
        ])
    tbl = ax_d.table(
        cellText=table_data, colLabels=col_labels,
        loc="upper center", cellLoc="center",
        bbox=[0.0, 0.15, 1.0, 0.82],
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FS_ANNOT + 0.5)
    # header styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2166ac")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # row colour alternation + within-2σ highlight
    for i, row in enumerate(table_data):
        bg = "#f7f7f7" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)
        w2s_cell = tbl[i + 1, len(col_labels) - 1]
        if row[-1] == "✓":
            w2s_cell.set_facecolor("#c7e9c0")
        else:
            w2s_cell.set_facecolor("#fcbba1")

    out = OUTDIR / "figS4_gnss_insar_validation.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# =============================================================================
if __name__ == "__main__":
    make_figS1()
    make_figS2()
    make_figS3()
    make_figS4()
    print("Done.")
