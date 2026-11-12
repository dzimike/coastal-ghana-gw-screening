"""
src/visualization.py
====================
Publication-quality figure and map generation for the Ghana groundwater-
subsidence project (Stage 12 of the pipeline).

All functions produce figures suitable for journal submission:
  - 300 DPI minimum resolution
  - Explicit axis labels with units
  - Scale bars and north arrows on all maps
  - Data source annotations at figure foot
  - British English labels throughout
  - Colourblind-accessible palettes where possible (Wong 2011)

Public API
----------
    plot_timeseries(series_dict, ...)
    plot_anomaly_bar_chart(series, ...)
    plot_trend_map(gdf, value_col, ...)
    plot_bivariate_map(gdf, x_col, y_col, ...)
    plot_risk_map(gdf, risk_col, ...)
    export_publication_figure(fig, path, ...)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

try:
    import geopandas as gpd  # noqa: F401
    _GPD_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GPD_AVAILABLE = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-wide style constants
# ---------------------------------------------------------------------------
FIGURE_DPI: int = 300
FONT_FAMILY: str = "DejaVu Sans"
FONT_SIZE_BASE: int = 10
FONT_SIZE_TITLE: int = 12
FONT_SIZE_LABEL: int = 10
FONT_SIZE_TICK: int = 8
FONT_SIZE_ANNOT: int = 8
DATA_SOURCE_COLOUR: str = "#636363"

# Diverging colourmap for anomalies (blue = positive/wet, red = negative/dry)
DIVERGING_CMAP: str = "RdBu"
# Sequential palette for risk maps
RISK_CMAP: str = "YlOrRd"

# Colourblind-safe discrete palette (Wong 2011, 8 colours)
CB_PALETTE: list[str] = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
]

# GDSI drought class colours (diverging blue → red)
DROUGHT_COLOURS: dict[str, str] = {
    "Extremely wet":               "#084594",
    "Moderately wet":              "#4292c6",
    "Mildly wet":                  "#9ecae1",
    "Near normal":                 "#d9d9d9",
    "Mild groundwater drought":    "#fdae6b",
    "Moderate groundwater drought": "#e6550d",
    "Severe groundwater drought":  "#a63603",
    "Extreme groundwater drought": "#7f0000",
    "No data":                     "#f7f7f7",
}

# Discrete risk class colours (green → red)
RISK_COLOURS: dict[str, str] = {
    "Very low":  "#1a9850",
    "Low":       "#91cf60",
    "Moderate":  "#ffffbf",
    "High":      "#fc8d59",
    "Very high": "#d73027",
}


def _apply_base_style() -> None:
    """Apply consistent rcParams for all project figures."""
    mpl.rcParams.update({
        "font.family":       FONT_FAMILY,
        "font.size":         FONT_SIZE_BASE,
        "axes.titlesize":    FONT_SIZE_TITLE,
        "axes.labelsize":    FONT_SIZE_LABEL,
        "xtick.labelsize":   FONT_SIZE_TICK,
        "ytick.labelsize":   FONT_SIZE_TICK,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.dpi":       FIGURE_DPI,
    })


# ---------------------------------------------------------------------------
# 1. Time series plot
# ---------------------------------------------------------------------------
def plot_timeseries(
    series_dict: dict[str, pd.Series],
    title: str = "Groundwater Storage Anomaly — Coastal Ghana",
    ylabel: str = "Anomaly (mm EWH)",
    xlabel: str = "Date",
    figsize: tuple[float, float] = (12, 5),
    highlight_periods: list[tuple[str, str, str]] | None = None,
    add_zero_line: bool = True,
    add_trend_line: bool = False,
    data_source: str = "GRACE / GRACE-FO mascon solutions (NASA)",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot one or more monthly time series on a shared axis.

    Parameters
    ----------
    series_dict : dict
        Mapping of legend label → pd.Series with DatetimeIndex.
        Example: {"TWSA": series_twsa, "GWSA": series_gwsa}
    title : str
        Figure title.
    ylabel : str
        Y-axis label (include units).
    xlabel : str
        X-axis label.
    figsize : tuple
        Figure size in inches (width, height).
    highlight_periods : list of (start_iso, end_iso, label) or None
        Shade time periods of interest (e.g. drought episodes).
    add_zero_line : bool
        Draw a horizontal dashed line at y = 0 (default True).
    add_trend_line : bool
        Overlay a Sen's slope linear trend for the first series (default False).
    data_source : str
        Data source annotation shown at figure foot.
    output_path : str, Path, or None
        If provided, save the figure via ``export_publication_figure``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_base_style()
    fig, ax = plt.subplots(figsize=figsize)

    for idx, (label, series) in enumerate(series_dict.items()):
        colour = CB_PALETTE[idx % len(CB_PALETTE)]
        ax.plot(
            series.index, series.values,
            colour=colour, linewidth=1.4, label=label, zorder=3,
        )
        # Shade positive / negative regions for single-series plots
        if len(series_dict) == 1:
            ax.fill_between(
                series.index, 0, series.values,
                where=(series.values >= 0),
                colour="#2171b5", alpha=0.15, zorder=2,
            )
            ax.fill_between(
                series.index, 0, series.values,
                where=(series.values < 0),
                colour="#cb181d", alpha=0.15, zorder=2,
            )

    # Optional Sen's slope trend line for the first series
    if add_trend_line and series_dict:
        first = next(iter(series_dict.values())).dropna()
        x_num = np.arange(len(first))
        # Median of all pairwise slopes
        slopes = [
            (first.iloc[j] - first.iloc[i]) / (j - i)
            for i in range(len(x_num) - 1)
            for j in range(i + 1, len(x_num))
        ]
        m = float(np.median(slopes))
        b = float(np.median(first.values - m * x_num))
        ax.plot(
            first.index, m * x_num + b,
            colour="#d73027", linewidth=1.5, linestyle="--",
            label=f"Trend (Sen's slope: {m * 12:+.2f} mm/yr)", zorder=4,
        )

    if add_zero_line:
        ax.axhline(0, colour="#252525", linewidth=0.8, linestyle="--", zorder=1)

    # Highlight shaded periods (e.g. droughts or political events)
    if highlight_periods:
        for hp_start, hp_end, hp_label in highlight_periods:
            ax.axvspan(
                pd.Timestamp(hp_start), pd.Timestamp(hp_end),
                alpha=0.12, colour="#fc8d59", zorder=0, label=hp_label,
            )

    # GRACE mission gap
    ax.axvspan(
        pd.Timestamp("2017-07-01"), pd.Timestamp("2018-06-01"),
        alpha=0.18, colour="#d9d9d9", zorder=0, label="GRACE mission gap",
    )

    ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=10)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.legend(fontsize=FONT_SIZE_ANNOT, framealpha=0.9, edgecolor="#bdbdbd")
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))

    _add_data_source(fig, data_source)
    plt.tight_layout()

    if output_path:
        export_publication_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Anomaly bar chart
# ---------------------------------------------------------------------------
def plot_anomaly_bar_chart(
    series: pd.Series,
    title: str = "Monthly Groundwater Storage Anomaly",
    ylabel: str = "Anomaly (mm EWH)",
    xlabel: str = "Date",
    figsize: tuple[float, float] = (14, 5),
    positive_colour: str = "#2171b5",
    negative_colour: str = "#cb181d",
    data_source: str = "GRACE / GRACE-FO mascon solutions (NASA)",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Produce a two-colour bar chart of monthly anomalies.

    Positive bars (wetter than baseline) are shown in blue; negative
    bars (drier than baseline) in red. The GRACE mission gap is shaded.

    Parameters
    ----------
    series : pd.Series
        Monthly anomaly values with DatetimeIndex, in mm EWH.
    title, ylabel, xlabel : str
    figsize : tuple
    positive_colour, negative_colour : str
        Hex colour codes for positive and negative bars.
    data_source : str
    output_path : str, Path, or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_base_style()
    fig, ax = plt.subplots(figsize=figsize)

    colours = np.where(series.values >= 0, positive_colour, negative_colour)
    ax.bar(
        series.index, series.values,
        color=colours, width=25, align="center", zorder=2,
    )
    ax.axhline(0, colour="#252525", linewidth=0.7, zorder=3)

    # GRACE mission gap
    ax.axvspan(
        pd.Timestamp("2017-07-01"), pd.Timestamp("2018-06-01"),
        alpha=0.25, colour="#d9d9d9", zorder=0,
        label="GRACE mission gap (Jul 2017 – Jun 2018)",
    )

    legend_handles = [
        mpatches.Patch(colour=positive_colour, label="Positive anomaly (wetter than baseline)"),
        mpatches.Patch(colour=negative_colour, label="Negative anomaly (drier than baseline)"),
        mpatches.Patch(colour="#d9d9d9", alpha=0.8, label="GRACE mission gap"),
    ]
    ax.legend(handles=legend_handles, fontsize=FONT_SIZE_ANNOT,
              framealpha=0.9, edgecolor="#bdbdbd")

    ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=10)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)

    _add_data_source(fig, data_source)
    plt.tight_layout()

    if output_path:
        export_publication_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Trend map (choropleth)
# ---------------------------------------------------------------------------
def plot_trend_map(
    gdf,
    value_col: str,
    title: str = "Groundwater Storage Trend — Coastal Ghana",
    cmap: str = DIVERGING_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "Sen's slope (mm/year)",
    figsize: tuple[float, float] = (10, 9),
    boundary_gdf=None,
    data_source: str = "GRACE / GRACE-FO; GLDAS NOAH (NASA)",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Produce a choropleth map of a continuous spatial variable.

    Suitable for trend maps (Sen's slope), subsidence velocity maps,
    rainfall anomaly maps, and pressure index maps.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon GeoDataFrame with ``value_col`` and geometry in UTM CRS.
    value_col : str
        Column to map.
    title : str
    cmap : str
        Matplotlib colourmap. Use diverging maps for anomalies or trends.
    vmin, vmax : float or None
        Colour scale limits. Defaults to 2nd–98th percentile.
    cbar_label : str
        Colourbar label (include units).
    figsize : tuple
    boundary_gdf : GeoDataFrame or None
        Optional national boundary overlay.
    data_source : str
    output_path : str, Path, or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not _GPD_AVAILABLE:
        raise ImportError("geopandas is required for map plots.")

    _apply_base_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    vals = gdf[value_col].dropna()
    if vmin is None:
        vmin = float(np.percentile(vals, 2))
    if vmax is None:
        vmax = float(np.percentile(vals, 98))

    # Symmetric limits for diverging maps centred on zero
    is_diverging = cmap in {"RdBu", "RdBu_r", "BrBG", "BrBG_r", "PuOr", "RdYlBu", "RdYlGn"}
    if is_diverging and vmin < 0 < vmax:
        v_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -v_abs, v_abs
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    gdf.plot(
        column=value_col, ax=ax, cmap=cmap, norm=norm,
        edgecolor="#bdbdbd", linewidth=0.3, legend=False,
        missing_kwds={
            "color": "#f0f0f0", "edgecolor": "#bdbdbd",
            "linewidth": 0.3, "label": "No data",
        },
    )

    if boundary_gdf is not None:
        boundary_gdf.to_crs(gdf.crs).plot(
            ax=ax, facecolor="none", edgecolor="#252525", linewidth=1.2,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(cbar_label, fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    _add_map_furniture(ax, gdf)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=10)
    ax.set_xlabel("Easting (m) — EPSG:32630", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Northing (m) — EPSG:32630", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    _add_data_source(fig, data_source)
    plt.tight_layout()

    if output_path:
        export_publication_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Bivariate map (groundwater stress × subsidence)
# ---------------------------------------------------------------------------
def plot_bivariate_map(
    gdf,
    x_col: str,
    y_col: str,
    title: str = "Bivariate Map: Groundwater Stress × Land Subsidence",
    x_label: str = "Groundwater stress",
    y_label: str = "Subsidence rate",
    n_classes: int = 3,
    figsize: tuple[float, float] = (12, 9),
    boundary_gdf=None,
    data_source: str = "GRACE / GRACE-FO; OPERA-DISP (NASA JPL)",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Produce a bivariate choropleth map showing two variables simultaneously.

    Uses a 3 × 3 bivariate colour matrix (Stevens 2015 scheme):
    - Blue axis (x): groundwater stress (higher = more stressed)
    - Red axis (y): subsidence rate (higher = more subsidence)
    - Purple: high on both (highest compound risk)

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon GeoDataFrame with ``x_col`` and ``y_col``.
    x_col : str
        First variable column (e.g. normalised groundwater stress, [0, 1]).
    y_col : str
        Second variable column (e.g. normalised subsidence rate, [0, 1]).
    title : str
    x_label, y_label : str
        Axis labels for the inset bivariate legend.
    n_classes : int
        Classes per dimension. 3 is strongly recommended for readability.
    figsize : tuple
    boundary_gdf : GeoDataFrame or None
    data_source : str
    output_path : str, Path, or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not _GPD_AVAILABLE:
        raise ImportError("geopandas is required for map plots.")

    _apply_base_style()

    # 3 × 3 bivariate palette (y rows × x columns, bottom-left = low/low)
    bivariate_3x3 = [
        "#e8e8e8", "#ace4e4", "#5ac8c8",   # y=Low:  x=Low, Mid, High
        "#dfb0d6", "#a5add3", "#5698b9",   # y=Mid
        "#be64ac", "#8c62aa", "#3b4994",   # y=High: purple when both high
    ]

    def _bin_column(col: pd.Series, n: int) -> pd.Series:
        try:
            return pd.qcut(col, q=n, labels=False, duplicates="drop")
        except Exception:
            return pd.cut(col, bins=n, labels=False, include_lowest=True)

    gdf_work = gdf.copy()
    gdf_work["_xc"] = _bin_column(gdf_work[x_col].fillna(gdf_work[x_col].median()), n_classes)
    gdf_work["_yc"] = _bin_column(gdf_work[y_col].fillna(gdf_work[y_col].median()), n_classes)

    def _pick_colour(row) -> str:
        xi = int(row["_xc"]) if pd.notna(row["_xc"]) else 0
        yi = int(row["_yc"]) if pd.notna(row["_yc"]) else 0
        if n_classes == 3:
            return bivariate_3x3[yi * 3 + xi]
        # Generic greyscale fallback for other n
        combined = (xi + yi) / (2 * (n_classes - 1))
        g = int(combined * 200 + 55)
        return f"#{g:02x}{g:02x}{g:02x}"

    gdf_work["_colour"] = gdf_work.apply(_pick_colour, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf_work.plot(color=gdf_work["_colour"], ax=ax,
                  edgecolor="#bdbdbd", linewidth=0.3)

    if boundary_gdf is not None:
        boundary_gdf.to_crs(gdf.crs).plot(
            ax=ax, facecolor="none", edgecolor="#252525", linewidth=1.2,
        )

    if n_classes == 3:
        _add_bivariate_legend(fig, bivariate_3x3, x_label, y_label)

    _add_map_furniture(ax, gdf_work)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=10)
    ax.set_xlabel("Easting (m) — EPSG:32630", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Northing (m) — EPSG:32630", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    _add_data_source(fig, data_source)
    plt.tight_layout()

    if output_path:
        export_publication_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Risk map (classified choropleth)
# ---------------------------------------------------------------------------
def plot_risk_map(
    gdf,
    risk_col: str = "cgsri_class",
    score_col: str | None = "cgsri",
    title: str = "Coastal Groundwater-Subsidence Risk Index — Coastal Ghana",
    risk_labels: list[str] | None = None,
    figsize: tuple[float, float] = (11, 10),
    boundary_gdf=None,
    highlight_hotspots: bool = True,
    hotspot_threshold: float = 0.75,
    data_source: str = (
        "GRACE / GRACE-FO; GLDAS; OPERA-DISP; Sentinel-1; "
        "Google Open Buildings; WorldPop (NASA / ESA / Google)"
    ),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Produce a classified risk map with a discrete legend.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon GeoDataFrame with risk class column in UTM CRS.
    risk_col : str
        Column containing risk class labels (string).
    score_col : str or None
        Column containing continuous risk scores.
        Used to identify and outline hotspots.
    title : str
    risk_labels : list of str or None
        Ordered risk class labels (ascending risk).
        Default: ["Very low", "Low", "Moderate", "High", "Very high"].
    figsize : tuple
    boundary_gdf : GeoDataFrame or None
    highlight_hotspots : bool
        If True and ``score_col`` is provided, outline the top
        ``1 − hotspot_threshold`` fraction of cells with a dashed border.
    hotspot_threshold : float
        Score quantile above which cells are highlighted (default 0.75).
    data_source : str
    output_path : str, Path, or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not _GPD_AVAILABLE:
        raise ImportError("geopandas is required for map plots.")

    _apply_base_style()

    if risk_labels is None:
        risk_labels = ["Very low", "Low", "Moderate", "High", "Very high"]

    # Merge project palette with any custom labels
    colour_map = {**RISK_COLOURS}
    fallback = ["#1a9850", "#91cf60", "#ffffbf", "#fc8d59", "#d73027"]
    for i, lbl in enumerate(risk_labels):
        if lbl not in colour_map:
            colour_map[lbl] = fallback[i % len(fallback)]

    gdf_work = gdf.copy()
    gdf_work["_colour"] = gdf_work[risk_col].map(colour_map).fillna("#f0f0f0")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf_work.plot(color=gdf_work["_colour"], ax=ax,
                  edgecolor="#bdbdbd", linewidth=0.3)

    # Hotspot outline
    if highlight_hotspots and score_col and score_col in gdf_work.columns:
        thresh_val = gdf_work[score_col].quantile(hotspot_threshold)
        hotspots = gdf_work[gdf_work[score_col] >= thresh_val]
        if len(hotspots) > 0:
            hotspots.plot(
                ax=ax, facecolor="none", edgecolor="#7f0000",
                linewidth=2.0, linestyle="--",
            )

    if boundary_gdf is not None:
        boundary_gdf.to_crs(gdf.crs).plot(
            ax=ax, facecolor="none", edgecolor="#252525", linewidth=1.3,
        )

    # Discrete legend
    legend_patches = [
        mpatches.Patch(colour=colour_map.get(lbl, "#f0f0f0"), label=lbl)
        for lbl in risk_labels
    ]
    if highlight_hotspots and score_col and score_col in gdf_work.columns:
        legend_patches.append(
            Line2D(
                [0], [0], colour="#7f0000", linewidth=2, linestyle="--",
                label=f"High-priority hotspot (top {int((1 - hotspot_threshold) * 100)} %)",
            )
        )
    ax.legend(
        handles=legend_patches,
        title="Risk class", title_fontsize=FONT_SIZE_ANNOT,
        fontsize=FONT_SIZE_ANNOT, framealpha=0.95,
        edgecolor="#bdbdbd", loc="upper left",
    )

    _add_map_furniture(ax, gdf_work)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=10)
    ax.set_xlabel("Easting (m) — EPSG:32630", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Northing (m) — EPSG:32630", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    _add_data_source(fig, data_source)
    plt.tight_layout()

    if output_path:
        export_publication_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Export publication figure
# ---------------------------------------------------------------------------
def export_publication_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = FIGURE_DPI,
    formats: list[str] | None = None,
    bbox_inches: str = "tight",
) -> list[Path]:
    """
    Save a figure at publication resolution, avoiding silent overwrites.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or Path
        Target file path. Extension sets format unless ``formats`` overrides.
    dpi : int
        Output resolution (default 300 DPI).
    formats : list of str or None
        List of format extensions to export (e.g. ["png", "pdf"]).
        If None, the extension in ``path`` is used.
    bbox_inches : str
        Passed to ``fig.savefig`` (default "tight").

    Returns
    -------
    list of Path
        Paths of all files written.

    Notes
    -----
    If the target path already exists, a version suffix is appended
    (e.g. ``figure_v2.png``) rather than overwriting the existing file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = [path.suffix.lstrip(".") or "png"]

    written = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        if out.exists():
            stem = out.stem
            counter = 2
            while out.exists():
                out = out.parent / f"{stem}_v{counter}{out.suffix}"
                counter += 1
            log.warning("File already exists; saving to: %s", out)

        fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches, facecolor="white")
        log.info("Figure saved: %s  (%d DPI)", out, dpi)
        written.append(out)

    return written


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _add_map_furniture(ax: plt.Axes, gdf) -> None:
    """Add a scale bar and north arrow to a projected map axis."""
    xmin, ymin, xmax, ymax = gdf.total_bounds
    width  = xmax - xmin
    height = ymax - ymin

    # Choose a round-number scale bar length
    scale_m = 50_000
    for candidate in [10_000, 25_000, 50_000, 100_000]:
        if width / candidate >= 3:
            scale_m = candidate
            break

    scale_x0 = xmax - scale_m * 1.4
    scale_y0 = ymin + height * 0.04
    ax.plot(
        [scale_x0, scale_x0 + scale_m], [scale_y0, scale_y0],
        color="#252525", linewidth=2.5, solid_capstyle="butt",
    )
    ax.text(
        scale_x0 + scale_m / 2,
        scale_y0 + height * 0.012,
        f"{scale_m // 1000:g} km",
        ha="center", va="bottom",
        fontsize=FONT_SIZE_ANNOT, color="#252525",
    )

    # North arrow
    arrow_x   = xmax - width * 0.06
    arrow_tip  = ymax - height * 0.05
    arrow_tail = arrow_tip - height * 0.06
    ax.annotate(
        "", xy=(arrow_x, arrow_tip), xytext=(arrow_x, arrow_tail),
        arrowprops=dict(arrowstyle="->", color="#252525", lw=2),
    )
    ax.text(
        arrow_x, arrow_tail - height * 0.015, "N",
        ha="center", va="top",
        fontsize=FONT_SIZE_LABEL + 2, fontweight="bold", color="#252525",
    )


def _add_data_source(fig: plt.Figure, source: str) -> None:
    """Add a small data source note at the bottom of the figure."""
    fig.text(
        0.5, -0.01,
        f"Data sources: {source}  |  CRS: EPSG:32630 (WGS 84 / UTM Zone 30N)",
        ha="center", va="bottom",
        fontsize=6, color=DATA_SOURCE_COLOUR,
    )


def _add_bivariate_legend(
    fig: plt.Figure,
    colours_3x3: list[str],
    x_label: str,
    y_label: str,
    position: tuple[float, float] = (0.73, 0.07),
    size: float = 0.11,
) -> None:
    """Embed a 3 × 3 bivariate colour legend as a figure inset."""
    ax_leg = fig.add_axes([position[0], position[1], size, size])
    n = 3
    matrix = np.array(colours_3x3).reshape(n, n)

    for yi in range(n):
        for xi in range(n):
            rect = mpatches.FancyBboxPatch(
                (xi / n, yi / n), 1 / n, 1 / n,
                boxstyle="square,pad=0",
                linewidth=0.5, edgecolor="white",
                facecolor=matrix[yi, xi],
            )
            ax_leg.add_patch(rect)

    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.set_xticks([0.5])
    ax_leg.set_yticks([0.5])
    ax_leg.set_xticklabels([x_label], fontsize=6)
    ax_leg.set_yticklabels([y_label], fontsize=6, rotation=90, va="center")
    ax_leg.tick_params(length=0)
    for spine in ax_leg.spines.values():
        spine.set_visible(False)
    ax_leg.set_title("Bivariate legend", fontsize=6, pad=3)
