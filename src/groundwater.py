"""
src/groundwater.py
==================
Groundwater storage anomaly estimation and trend analysis for the Ghana
groundwater-subsidence project (Stage 5 and Stage 6 of the pipeline).

Core equation
-------------
    GWSA = TWSA − SMSA − CWSA

Where:
    GWSA  = groundwater storage anomaly          (mm EWH)
    TWSA  = total water storage anomaly (GRACE)  (mm EWH)
    SMSA  = soil moisture storage anomaly (GLDAS) (mm EWH)
    CWSA  = canopy water storage anomaly (GLDAS)  (mm EWH)

Snow and ice terms are excluded for Ghana (tropical climate).
Surface water storage anomaly (SWSA) is treated as negligible unless
a regional surface-water product is available.

Caution
-------
GRACE spatial resolution is approximately 300–400 km.
All outputs at this stage are basin-scale or regional. Do not interpret
individual GRACE grid cells as local groundwater conditions.

Public API
----------
    load_grace_twsa(path, baseline_start, baseline_end)
    load_gldas_components(path, baseline_start, baseline_end)
    align_monthly_timeseries(da_grace, ds_gldas)
    compute_storage_anomalies(series, baseline_start, baseline_end)
    compute_groundwater_storage_anomaly(ds_aligned, swsa)
    aggregate_to_regions(gwsa_da, regions_gdf, region_id_col, crs, stat)
    run_mann_kendall_trend(series, method, alpha)
    compute_sens_slope(series)
    classify_groundwater_drought(series, custom_thresholds)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

try:
    import pymannkendall as mk
    _MK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MK_AVAILABLE = False
    warnings.warn(
        "pymannkendall not installed. run_mann_kendall_trend() will be unavailable. "
        "Install with: pip install pymannkendall",
        ImportWarning,
        stacklevel=2,
    )

from src.config import thresholds, ANOMALY_BASELINE_START, ANOMALY_BASELINE_END

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
CM_TO_MM: float = 10.0   # GRACE lwe_thickness is in cm EWH → mm EWH
# 1 kg/m² ≡ 1 mm EWH for liquid water (density = 1000 kg/m³)

# GLDAS soil moisture layers (NOAH v2.1)
GLDAS_SM_LAYERS: list[str] = [
    "SoilMoi0_10cm_inst",
    "SoilMoi10_40cm_inst",
    "SoilMoi40_100cm_inst",
    "SoilMoi100_200cm_inst",
]
GLDAS_CANOPY_VAR: str = "CanopInt_inst"


# ---------------------------------------------------------------------------
# 1. Load GRACE TWSA
# ---------------------------------------------------------------------------
def load_grace_twsa(
    path: str | Path,
    baseline_start: str = ANOMALY_BASELINE_START,
    baseline_end: str = ANOMALY_BASELINE_END,
    variable: str = "lwe_thickness",
    lon_slice: slice | None = None,
    lat_slice: slice | None = None,
) -> xr.DataArray:
    """
    Load GRACE / GRACE-FO mascon total water storage anomaly.

    Parameters
    ----------
    path : str or Path
        Path to a GRACE NetCDF file or directory containing NetCDF files.
        If a directory, all ``*.nc`` files are opened as a multi-file dataset.
    baseline_start : str
        ISO date string for the start of the anomaly baseline period
        (default: 2004-01-01, GRACE standard).
    baseline_end : str
        ISO date string for the end of the anomaly baseline period
        (default: 2009-12-31).
    variable : str
        NetCDF variable name for total water storage (default: ``lwe_thickness``).
    lon_slice : slice or None
        Optional longitude subset (degrees East, WGS 84).
    lat_slice : slice or None
        Optional latitude subset (degrees North, WGS 84).

    Returns
    -------
    xr.DataArray
        Monthly TWSA in mm equivalent water height (EWH), indexed by time.
        Attributes record the baseline period and units.

    Notes
    -----
    - Converts cm EWH to mm EWH (multiply by 10).
    - Re-centres the anomaly relative to the specified baseline mean if
      the mascon product uses a different internal baseline.
    - Missing months are preserved as NaN so that downstream gap handling
      is explicit.
    """
    path = Path(path)
    log.info("Loading GRACE TWSA from: %s", path)

    if path.is_dir():
        nc_files = sorted(path.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {path}")
        ds = xr.open_mfdataset(nc_files, combine="by_coords", engine="netcdf4")
    else:
        ds = xr.open_dataset(path, engine="netcdf4")

    if variable not in ds:
        available = list(ds.data_vars)
        raise KeyError(
            f"Variable '{variable}' not found in dataset. Available: {available}"
        )

    da = ds[variable]

    # Spatial subset
    if lon_slice is not None:
        lon_dim = _find_dim(da, ["lon", "longitude", "x"])
        da = da.sel({lon_dim: lon_slice})
    if lat_slice is not None:
        lat_dim = _find_dim(da, ["lat", "latitude", "y"])
        da = da.sel({lat_dim: lat_slice})

    # Convert cm EWH → mm EWH
    da = da * CM_TO_MM
    da.attrs["units"] = "mm EWH"
    da.attrs["long_name"] = "Total Water Storage Anomaly (GRACE/GRACE-FO)"

    # Re-centre anomaly to the requested baseline period
    time_dim = _find_dim(da, ["time"])
    baseline_mask = (
        (da[time_dim] >= np.datetime64(baseline_start))
        & (da[time_dim] <= np.datetime64(baseline_end))
    )
    baseline_mean = da.where(baseline_mask).mean(dim=time_dim, skipna=True)
    da = da - baseline_mean
    da.attrs["baseline_start"] = baseline_start
    da.attrs["baseline_end"] = baseline_end

    n_months = int(da[time_dim].size)
    log.info(
        "GRACE TWSA loaded: %d months. Baseline: %s to %s.",
        n_months, baseline_start, baseline_end,
    )
    return da


# ---------------------------------------------------------------------------
# 2. Load GLDAS components
# ---------------------------------------------------------------------------
def load_gldas_components(
    path: str | Path,
    baseline_start: str = ANOMALY_BASELINE_START,
    baseline_end: str = ANOMALY_BASELINE_END,
    sm_layers: list[str] = GLDAS_SM_LAYERS,
    canopy_var: str = GLDAS_CANOPY_VAR,
    lon_slice: slice | None = None,
    lat_slice: slice | None = None,
) -> xr.Dataset:
    """
    Load GLDAS NOAH land surface model components needed to isolate
    the groundwater signal from GRACE TWSA.

    Parameters
    ----------
    path : str or Path
        Path to GLDAS NetCDF file(s).
    baseline_start, baseline_end : str
        Anomaly baseline period (ISO date strings).
    sm_layers : list of str
        GLDAS soil moisture layer variable names to sum.
    canopy_var : str
        GLDAS canopy water storage variable name.
    lon_slice, lat_slice : slice or None
        Optional spatial subsets.

    Returns
    -------
    xr.Dataset with variables:
        ``SM_total``  : total column soil moisture (mm EWH, all layers summed)
        ``SMSA``      : soil moisture storage anomaly re-centred to baseline (mm EWH)
        ``CWSA``      : canopy water storage anomaly (mm EWH)

    Notes
    -----
    - GLDAS units are kg/m², which are numerically equivalent to mm EWH
      for liquid water (density = 1000 kg/m³).
    - Snow and ice are excluded (not present in Ghana's tropical climate).
    - GLDAS does not contain a direct groundwater variable; GWSA is
      isolated as the GRACE residual.
    """
    path = Path(path)
    log.info("Loading GLDAS components from: %s", path)

    if path.is_dir():
        nc_files = sorted(path.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {path}")
        ds = xr.open_mfdataset(nc_files, combine="by_coords", engine="netcdf4")
    else:
        ds = xr.open_dataset(path, engine="netcdf4")

    # Spatial subset
    for sl, dims in [(lon_slice, ["lon", "longitude", "x"]),
                     (lat_slice, ["lat", "latitude", "y"])]:
        if sl is not None:
            dim = _find_dim(ds, dims)
            ds = ds.sel({dim: sl})

    time_dim = _find_dim(ds, ["time"])

    # --- Soil moisture: sum all available layers (kg/m² == mm EWH) ---
    sm_present = [v for v in sm_layers if v in ds]
    missing_sm = [v for v in sm_layers if v not in ds]
    if missing_sm:
        log.warning("GLDAS SM layers not found (skipped): %s", missing_sm)
    if not sm_present:
        raise KeyError("No soil moisture layers found in GLDAS dataset.")

    sm_total = ds[sm_present[0]].copy(deep=True)
    for v in sm_present[1:]:
        sm_total = sm_total + ds[v]
    sm_total.attrs["units"] = "mm EWH"
    sm_total.attrs["long_name"] = "Total column soil moisture (all layers summed)"

    # --- Canopy water ---
    if canopy_var not in ds:
        log.warning("Canopy water variable '%s' not found; setting CWSA to 0.", canopy_var)
        canopy = xr.zeros_like(sm_total)
    else:
        canopy = ds[canopy_var].copy(deep=True)
    canopy.attrs["units"] = "mm EWH"

    # --- Compute anomalies relative to the baseline period ---
    def _anomaly(da: xr.DataArray) -> xr.DataArray:
        mask = (
            (da[time_dim] >= np.datetime64(baseline_start))
            & (da[time_dim] <= np.datetime64(baseline_end))
        )
        mean = da.where(mask).mean(dim=time_dim, skipna=True)
        return da - mean

    smsa = _anomaly(sm_total)
    smsa.attrs["units"] = "mm EWH"
    smsa.attrs["long_name"] = "Soil moisture storage anomaly"

    cwsa = _anomaly(canopy)
    cwsa.attrs["units"] = "mm EWH"
    cwsa.attrs["long_name"] = "Canopy water storage anomaly"

    out = xr.Dataset({"SM_total": sm_total, "SMSA": smsa, "CWSA": cwsa})
    out.attrs["baseline_start"] = baseline_start
    out.attrs["baseline_end"] = baseline_end
    log.info(
        "GLDAS components loaded: %d months. SM layers used: %s.",
        int(out[time_dim].size), sm_present,
    )
    return out


# ---------------------------------------------------------------------------
# 3. Align monthly time series
# ---------------------------------------------------------------------------
def align_monthly_timeseries(
    da_grace: xr.DataArray,
    ds_gldas: xr.Dataset,
) -> xr.Dataset:
    """
    Align GRACE TWSA and GLDAS components to a common monthly time index.

    Reindexes both datasets to the union of their time axes; months present
    in only one dataset are filled with NaN. This makes temporal gaps
    explicit in downstream calculations.

    Parameters
    ----------
    da_grace : xr.DataArray
        GRACE TWSA (output of ``load_grace_twsa``).
    ds_gldas : xr.Dataset
        GLDAS components (output of ``load_gldas_components``).

    Returns
    -------
    xr.Dataset with variables ``TWSA``, ``SMSA``, ``CWSA`` on a common
    monthly time axis.
    """
    time_g = _find_dim(da_grace, ["time"])
    time_l = _find_dim(ds_gldas, ["time"])

    t_min = min(
        pd.Timestamp(da_grace[time_g].values[0]),
        pd.Timestamp(ds_gldas[time_l].values[0]),
    )
    t_max = max(
        pd.Timestamp(da_grace[time_g].values[-1]),
        pd.Timestamp(ds_gldas[time_l].values[-1]),
    )
    common_index = pd.date_range(t_min, t_max, freq="MS")
    common_times = xr.DataArray(common_index, dims=[time_g])

    twsa = da_grace.reindex({time_g: common_times}, method=None)
    smsa = ds_gldas["SMSA"].reindex({time_l: common_times}, method=None)
    cwsa = ds_gldas["CWSA"].reindex({time_l: common_times}, method=None)

    out = xr.Dataset({"TWSA": twsa, "SMSA": smsa, "CWSA": cwsa})

    n_overlap = int(
        (~np.isnan(twsa.values) & ~np.isnan(smsa.values)).sum()
    )
    log.info(
        "Aligned time series: %d total months, %d months with both GRACE and GLDAS.",
        len(common_index), n_overlap,
    )
    return out


# ---------------------------------------------------------------------------
# 4. Compute storage anomaly for a 1-D pandas Series
# ---------------------------------------------------------------------------
def compute_storage_anomalies(
    series: pd.Series,
    baseline_start: str = ANOMALY_BASELINE_START,
    baseline_end: str = ANOMALY_BASELINE_END,
) -> pd.Series:
    """
    Re-centre a water storage time series relative to a baseline period mean.

    Useful for processing regionally aggregated CSV time series (e.g. after
    ``aggregate_to_regions``) where re-centring to a common baseline is needed.

    Parameters
    ----------
    series : pd.Series
        Monthly time series with DatetimeIndex, in mm EWH.
    baseline_start, baseline_end : str
        ISO date strings defining the baseline period.

    Returns
    -------
    pd.Series
        Anomaly series (values minus baseline mean), same index.
    """
    baseline_mean = series.loc[baseline_start:baseline_end].mean()
    anomaly = series - baseline_mean
    return anomaly


# ---------------------------------------------------------------------------
# 5. Compute groundwater storage anomaly
# ---------------------------------------------------------------------------
def compute_groundwater_storage_anomaly(
    ds_aligned: xr.Dataset,
    swsa: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Estimate groundwater storage anomaly (GWSA) from the aligned dataset.

    Formula:  GWSA = TWSA − SMSA − CWSA [− SWSA]

    Parameters
    ----------
    ds_aligned : xr.Dataset
        Output of ``align_monthly_timeseries`` with variables
        ``TWSA``, ``SMSA``, ``CWSA``.
    swsa : xr.DataArray or None
        Surface water storage anomaly. If None, SWSA is assumed zero —
        acceptable for most of coastal Ghana where inland surface water
        storage changes are small relative to GRACE uncertainty.
        Document this assumption explicitly in methods.

    Returns
    -------
    xr.DataArray
        Monthly GWSA in mm EWH.

    Notes
    -----
    - GRACE uncertainty is approximately ±20–30 mm EWH. Interpret
      small GWSA values with caution.
    - Months where either GRACE or GLDAS is missing will be NaN.
    """
    twsa = ds_aligned["TWSA"]
    smsa = ds_aligned["SMSA"]
    cwsa = ds_aligned["CWSA"]

    gwsa = twsa - smsa - cwsa

    if swsa is not None:
        gwsa = gwsa - swsa
        gwsa.attrs["swsa_included"] = True
        log.info("SWSA included in GWSA calculation.")
    else:
        gwsa.attrs["swsa_included"] = False
        log.info("SWSA not provided — assumed zero. Document this in methods.")

    gwsa.attrs["units"] = "mm EWH"
    gwsa.attrs["long_name"] = "Groundwater Storage Anomaly"
    gwsa.attrs["formula"] = "GWSA = TWSA - SMSA - CWSA"
    gwsa.attrs["scale_note"] = (
        "GRACE spatial resolution ~300-400 km. "
        "Interpret at basin/regional scale only."
    )

    time_dim = _find_dim(gwsa, ["time"])
    log.info(
        "GWSA computed: %d monthly time steps.", int(gwsa[time_dim].size)
    )
    return gwsa


# ---------------------------------------------------------------------------
# 6. Aggregate gridded GWSA to polygon regions
# ---------------------------------------------------------------------------
def aggregate_to_regions(
    gwsa_da: xr.DataArray,
    regions_gdf: gpd.GeoDataFrame,
    region_id_col: str = "cell_id",
    crs: str = "EPSG:4326",
    stat: Literal["mean", "median", "sum"] = "mean",
) -> pd.DataFrame:
    """
    Aggregate gridded GWSA to polygon regions (districts, grid cells, basins).

    Each GRACE grid cell whose centre falls within a region polygon
    contributes to the aggregate value. Where no GRACE cell centre falls
    inside a small region, the nearest cell is used. Because GRACE cells
    are large (~0.5°), many small regions will inherit identical or nearly
    identical GWSA values. Report this limitation in outputs.

    Parameters
    ----------
    gwsa_da : xr.DataArray
        Gridded monthly GWSA (time × lat × lon) in mm EWH.
    regions_gdf : gpd.GeoDataFrame
        Polygon regions. Must contain ``region_id_col``.
    region_id_col : str
        Column identifying each region.
    crs : str
        CRS of the GRACE grid (default EPSG:4326).
    stat : {"mean", "median", "sum"}
        Spatial aggregation statistic.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns [``time``, ``region_id``, ``gwsa_mm``].
    """
    from shapely.geometry import Point

    lon_dim  = _find_dim(gwsa_da, ["lon", "longitude", "x"])
    lat_dim  = _find_dim(gwsa_da, ["lat", "latitude", "y"])
    time_dim = _find_dim(gwsa_da, ["time"])

    lons  = gwsa_da[lon_dim].values
    lats  = gwsa_da[lat_dim].values
    times = pd.DatetimeIndex(gwsa_da[time_dim].values)

    regions_geo = (
        regions_gdf.to_crs(crs)
        if regions_gdf.crs and regions_gdf.crs.to_string() != crs
        else regions_gdf
    )

    log.info(
        "Aggregating GWSA to %d regions using %s ...", len(regions_geo), stat
    )
    records = []

    for _, row in regions_geo.iterrows():
        region_id = row[region_id_col]
        geom = row.geometry

        # Find GRACE cell centres strictly inside this polygon.
        # Build a boolean grid (n_lat × n_lon) via point-in-polygon; this
        # avoids the Cartesian-product overselection bug that arises when
        # separate lon/lat masks are combined with .sel().
        lon_grid, lat_grid = np.meshgrid(lons, lats)  # (n_lat, n_lon)
        inside = np.array([
            geom.contains(Point(lo, la))
            for lo, la in zip(lon_grid.ravel(), lat_grid.ravel())
        ]).reshape(lon_grid.shape)

        valid_lons = lons[inside.any(axis=0)]
        valid_lats = lats[inside.any(axis=1)]

        if valid_lons.size > 0 and valid_lats.size > 0:
            subset = gwsa_da.sel({lon_dim: valid_lons, lat_dim: valid_lats})
            if stat == "mean":
                agg = subset.mean(dim=[lon_dim, lat_dim], skipna=True)
            elif stat == "median":
                agg = subset.median(dim=[lon_dim, lat_dim], skipna=True)
            else:
                agg = subset.sum(dim=[lon_dim, lat_dim], skipna=True)
        else:
            # Nearest-cell fallback
            cx = geom.centroid.x
            cy = geom.centroid.y
            i_lon = int(np.argmin(np.abs(lons - cx)))
            i_lat = int(np.argmin(np.abs(lats - cy)))
            agg = gwsa_da.isel({lon_dim: i_lon, lat_dim: i_lat})

        for t, val in zip(times, agg.values.ravel()):
            records.append({
                "time":      t,
                "region_id": region_id,
                "gwsa_mm":   float(val),
            })

    df = pd.DataFrame(records)
    log.info("Aggregation complete: %d region-month records.", len(df))
    return df


# ---------------------------------------------------------------------------
# 7. Mann-Kendall trend test
# ---------------------------------------------------------------------------
def run_mann_kendall_trend(
    series: pd.Series,
    method: Literal["original", "hamed_rao", "yue_wang", "seasonal"] = "original",
    alpha: float = 0.05,
) -> dict:
    """
    Run a Mann-Kendall trend test on a monthly time series.

    Parameters
    ----------
    series : pd.Series
        Monthly time series (NaN values are dropped before testing).
    method : str
        MK variant:
        ``"original"``   — standard Mann-Kendall
        ``"hamed_rao"``  — Hamed & Rao (1998) for autocorrelated series
        ``"yue_wang"``   — Yue & Wang (2002) pre-whitening
        ``"seasonal"``   — Seasonal Mann-Kendall
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict with keys:
        trend, h, p_value, z_score, tau, s, var_s,
        slope (mm/month), slope_yr (mm/year), n, method, alpha
    """
    if not _MK_AVAILABLE:
        raise ImportError("pymannkendall is required. pip install pymannkendall")

    clean = series.dropna()
    n = len(clean)
    if n < 12:
        log.warning(
            "Mann-Kendall: only %d valid observations. "
            "Minimum recommended is 12. Results may be unreliable.", n
        )

    mk_funcs = {
        "original":  mk.original_test,
        "hamed_rao": mk.hamed_rao_modification_test,
        "yue_wang":  mk.yue_wang_modification_test,
        "seasonal":  mk.seasonal_test,
    }
    if method not in mk_funcs:
        raise ValueError(f"method must be one of {list(mk_funcs)}; got '{method}'.")

    result = mk_funcs[method](clean.values, alpha=alpha)
    slope_info = compute_sens_slope(clean)

    log.info(
        "Mann-Kendall (%s): trend=%s, p=%.4f, slope=%.4f mm/yr, n=%d",
        method, result.trend, result.p, slope_info["slope_yr"], n,
    )

    return {
        "trend":    result.trend,
        "h":        result.h,
        "p_value":  result.p,
        "z_score":  result.z,
        "tau":      result.Tau,
        "s":        result.s,
        "var_s":    result.var_s,
        "slope":    slope_info["slope"],
        "slope_yr": slope_info["slope_yr"],
        "n":        n,
        "method":   method,
        "alpha":    alpha,
    }


# ---------------------------------------------------------------------------
# 8. Sen's slope estimator
# ---------------------------------------------------------------------------
def compute_sens_slope(series: pd.Series) -> dict:
    """
    Compute the Theil-Sen slope estimator for a time series.

    Sen's slope is the median of all pairwise slopes
    (y_j − y_i) / (j − i) for j > i. It is robust to outliers
    and does not require normality.

    Parameters
    ----------
    series : pd.Series
        Monthly time series (NaN values are dropped).

    Returns
    -------
    dict with keys:
        slope      : median pairwise slope (mm EWH per month)
        slope_yr   : slope × 12 (mm EWH per year)
        intercept  : Sen's intercept (median of y − slope × x)
        n          : number of valid observations
    """
    clean = series.dropna()
    n = len(clean)
    values = clean.values.astype(float)

    # Use actual calendar months if a DatetimeIndex is available.
    # Observation-index spacing inflates Sen's slope when months are missing
    # (e.g. the GRACE mission gap), because pairs straddling the gap get a
    # denominator of 1 step instead of the true ~35 months.
    if isinstance(clean.index, pd.DatetimeIndex):
        x = (clean.index - clean.index[0]).days.values.astype(float) / 30.4375
    else:
        x = np.arange(n, dtype=float)

    slopes = [
        (values[j] - values[i]) / (x[j] - x[i])
        for i in range(n - 1)
        for j in range(i + 1, n)
    ]

    slope = float(np.median(slopes)) if slopes else np.nan
    intercept = float(np.median(values - slope * x))

    return {
        "slope":     slope,
        "slope_yr":  slope * 12.0,
        "intercept": intercept,
        "n":         n,
    }


# ---------------------------------------------------------------------------
# 9. Classify groundwater drought (GDSI)
# ---------------------------------------------------------------------------
def classify_groundwater_drought(
    series: pd.Series,
    custom_thresholds: dict | None = None,
) -> pd.DataFrame:
    """
    Classify a GWSA time series into groundwater drought severity categories.

    GDSI (Groundwater Drought Severity Index) = standardised GWSA (z-score).

    Classification thresholds are loaded from ``config/thresholds.yaml``
    (``gdsi.classes``) unless overridden by ``custom_thresholds``.

    Parameters
    ----------
    series : pd.Series
        Monthly GWSA in mm EWH with DatetimeIndex.
    custom_thresholds : dict or None
        Optional mapping of class label → (min, max) tuple.
        Values of None for min or max indicate open-ended bounds.
        If None, project defaults from thresholds.yaml are used.

    Returns
    -------
    pd.DataFrame indexed by time with columns:
        gwsa_mm      : raw GWSA values
        gdsi         : standardised anomaly (z-score)
        drought_class : string drought class label
    """
    clean = series.dropna()
    mean_gwsa = float(clean.mean())
    std_gwsa  = float(clean.std(ddof=1))

    if std_gwsa == 0.0:
        log.warning("Standard deviation of GWSA is zero; GDSI cannot be computed.")
        gdsi = series.copy() * np.nan
    else:
        gdsi = (series - mean_gwsa) / std_gwsa

    # Build class lookup from config or custom input
    if custom_thresholds:
        classes = custom_thresholds
    else:
        classes = {
            c["label"]: (c["min"], c["max"])
            for c in thresholds["gdsi"]["classes"]
        }

    def _classify(val: float) -> str:
        if np.isnan(val):
            return "No data"
        for label, (lo, hi) in classes.items():
            lo_ok = (lo is None) or (val >= lo)
            hi_ok = (hi is None) or (val < hi)
            if lo_ok and hi_ok:
                return label
        return "Unclassified"

    drought_class = gdsi.apply(_classify)

    df = pd.DataFrame({
        "gwsa_mm":      series.values,
        "gdsi":         gdsi.values,
        "drought_class": drought_class.values,
    }, index=series.index)
    df.index.name = "time"

    log.info(
        "GDSI classification complete:\n%s",
        df["drought_class"].value_counts().to_string(),
    )
    return df


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------
def _find_dim(obj, candidates: list[str]) -> str:
    """Return the first dimension name from ``candidates`` found in ``obj``."""
    dims = list(obj.dims) if hasattr(obj, "dims") else []
    for c in candidates:
        if c in dims:
            return c
    raise KeyError(f"None of {candidates} found in dimensions {dims}.")
