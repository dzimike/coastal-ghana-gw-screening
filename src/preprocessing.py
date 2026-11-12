"""
src/preprocessing.py
====================
Spatial harmonisation and preprocessing utilities for the Ghana
groundwater-subsidence project (Stage 4 of the pipeline).

All processed spatial datasets are standardised to:
  - CRS         : EPSG:32630  (WGS 84 / UTM Zone 30N)
  - Study extent: coastal Ghana 30–50 km buffer
  - Nodata      : preserved and documented
  - File format : GeoTIFF (COG) for rasters, GeoPackage for vectors

Public API
----------
Raster functions
    reproject_raster(src, dst, target_crs, resampling, nodata)
    resample_raster(src, dst, target_res_m, resampling)
    align_to_reference(src, dst, ref_path, resampling)
    mask_to_study_area(src, dst, mask_gdf)
    harmonize_raster(src, dst, mask_gdf, target_res_m, resampling)
    merge_dem_tiles(tile_dir, dst, target_crs, mask_gdf)
    stack_rasters_to_netcdf(raster_paths, dates, var_name, dst)
    export_cog(src_array, dst, profile)

Vector functions
    repair_vector_geometries(gdf)
    remove_duplicate_features(gdf, subset_cols)
    standardize_column_names(gdf)
    reproject_vector(gdf, target_crs)
    clip_vector_to_study_area(gdf, mask_gdf)
    harmonize_vector(src, dst, mask_gdf, target_crs)

Quality-control functions
    check_crs_consistency(paths, target_crs)
    detect_temporal_gaps(dates, freq)
    validate_raster_grid(paths, tolerance_m)
    run_qc_report(raster_paths, vector_paths, target_crs)

Batch helpers
    batch_harmonize_rasters(src_dir, dst_dir, mask_gdf, pattern,
                             target_res_m, resampling)
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.warp import (
    calculate_default_transform,
    reproject as rio_reproject,
)
import xarray as xr

from src.utils import ensure_dir

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_CRS: str = "EPSG:32630"
NODATA_FLOAT: float = -9999.0
NODATA_INT: int = -9999
DEFAULT_RESAMPLING: Resampling = Resampling.bilinear

ResamplingName = Literal[
    "nearest", "bilinear", "cubic", "cubic_spline",
    "lanczos", "average", "mode", "gauss",
]

_RESAMPLING_MAP: dict[str, Resampling] = {
    "nearest":      Resampling.nearest,
    "bilinear":     Resampling.bilinear,
    "cubic":        Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos":      Resampling.lanczos,
    "average":      Resampling.average,
    "mode":         Resampling.mode,
    "gauss":        Resampling.gauss,
}


def _resampling(name: str | Resampling) -> Resampling:
    if isinstance(name, Resampling):
        return name
    return _RESAMPLING_MAP.get(name.lower(), DEFAULT_RESAMPLING)


# ---------------------------------------------------------------------------
# 1. Raster reprojection
# ---------------------------------------------------------------------------
def reproject_raster(
    src_path: str | Path,
    dst_path: str | Path,
    target_crs: str = TARGET_CRS,
    resampling: str | Resampling = "bilinear",
    nodata: float | None = None,
) -> Path:
    """
    Reproject a raster file to ``target_crs``.

    Parameters
    ----------
    src_path : str or Path
        Input raster file.
    dst_path : str or Path
        Output raster file.  Parent directories are created if needed.
    target_crs : str
        Target CRS as an EPSG string (default: ``"EPSG:32630"``).
    resampling : str or Resampling
        Resampling algorithm (default: bilinear).
    nodata : float or None
        Override nodata value.  If None, the source nodata is used.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)
    dst_crs = CRS.from_string(target_crs)
    algo = _resampling(resampling)

    with rasterio.open(src_path) as src:
        src_nodata = nodata if nodata is not None else src.nodata
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=src_nodata,
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                rio_reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=algo,
                    src_nodata=src_nodata,
                    dst_nodata=src_nodata,
                )

    log.info("Reprojected %s → %s  [%s]", src_path.name, dst_path.name, target_crs)
    return dst_path


# ---------------------------------------------------------------------------
# 2. Raster resampling (change pixel resolution)
# ---------------------------------------------------------------------------
def resample_raster(
    src_path: str | Path,
    dst_path: str | Path,
    target_res_m: float,
    resampling: str | Resampling = "bilinear",
) -> Path:
    """
    Resample a raster to a new pixel size (metres).

    The input raster must already be in a projected CRS with metre units
    (e.g. EPSG:32630).  Reproject first if needed.

    Parameters
    ----------
    src_path : str or Path
        Input raster (projected CRS, metre units).
    dst_path : str or Path
        Output raster.
    target_res_m : float
        Target pixel size in metres (e.g. 1000 for 1 km, 25000 for 25 km).
    resampling : str or Resampling
        Resampling algorithm.

    Returns
    -------
    Path
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)
    algo = _resampling(resampling)

    with rasterio.open(src_path) as src:
        scale_x = src.transform.a / target_res_m
        scale_y = abs(src.transform.e) / target_res_m
        new_width  = max(1, round(src.width  / scale_x))
        new_height = max(1, round(src.height / scale_y))

        new_transform = rasterio.transform.from_bounds(
            *src.bounds, new_width, new_height
        )
        profile = src.profile.copy()
        profile.update(
            width=new_width,
            height=new_height,
            transform=new_transform,
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                data = src.read(band_idx)
                resampled = np.empty((new_height, new_width), dtype=data.dtype)
                rio_reproject(
                    source=data,
                    destination=resampled,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=algo,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata,
                )
                dst.write(resampled, band_idx)

    log.info(
        "Resampled %s → %s  (%.0f m pixels, %d×%d)",
        src_path.name, dst_path.name, target_res_m, new_width, new_height,
    )
    return dst_path


# ---------------------------------------------------------------------------
# 3. Snap a raster to a reference grid
# ---------------------------------------------------------------------------
def align_to_reference(
    src_path: str | Path,
    dst_path: str | Path,
    ref_path: str | Path,
    resampling: str | Resampling = "bilinear",
) -> Path:
    """
    Warp ``src_path`` to exactly match the grid of ``ref_path``
    (same CRS, pixel size, origin, and extent).

    Both rasters should already share the same CRS.

    Parameters
    ----------
    src_path : str or Path
        Input raster to align.
    dst_path : str or Path
        Output raster aligned to reference.
    ref_path : str or Path
        Reference raster that defines the target grid.
    resampling : str or Resampling

    Returns
    -------
    Path
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)
    algo = _resampling(resampling)

    with rasterio.open(ref_path) as ref:
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_width     = ref.width
        dst_height    = ref.height
        dst_nodata    = ref.nodata

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata if src.nodata is not None else dst_nodata
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            nodata=src_nodata,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                rio_reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=algo,
                    src_nodata=src_nodata,
                    dst_nodata=src_nodata,
                )

    log.info("Aligned %s to reference grid of %s", src_path.name, Path(ref_path).name)
    return dst_path


# ---------------------------------------------------------------------------
# 4. Mask / clip raster to study area
# ---------------------------------------------------------------------------
def mask_to_study_area(
    src_path: str | Path,
    dst_path: str | Path,
    mask_gdf: gpd.GeoDataFrame,
    all_touched: bool = False,
    crop: bool = True,
) -> Path:
    """
    Clip a raster to the study-area boundary and apply a pixel mask.

    The ``mask_gdf`` should already be in the same CRS as the raster.
    If CRS differs, the mask is reprojected automatically.

    Parameters
    ----------
    src_path : str or Path
        Input raster.
    dst_path : str or Path
        Output raster, clipped and masked.
    mask_gdf : GeoDataFrame
        Study-area boundary polygon(s).
    all_touched : bool
        If True, include pixels that touch the boundary (not only those
        whose centre falls inside).  Default False.
    crop : bool
        If True, crop the raster extent to the mask bounding box.

    Returns
    -------
    Path
    """
    from rasterio.mask import mask as rio_mask

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)

    with rasterio.open(src_path) as src:
        if mask_gdf.crs != src.crs:
            mask_gdf = mask_gdf.to_crs(src.crs)

        shapes = [geom.__geo_interface__ for geom in mask_gdf.geometry]
        masked, transform = rio_mask(
            src, shapes,
            crop=crop,
            all_touched=all_touched,
            nodata=src.nodata,
        )
        profile = src.profile.copy()
        profile.update(
            height=masked.shape[1],
            width=masked.shape[2],
            transform=transform,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(masked)

    log.info("Masked %s → %s", src_path.name, dst_path.name)
    return dst_path


# ---------------------------------------------------------------------------
# 5. Full raster harmonisation pipeline
# ---------------------------------------------------------------------------
def harmonize_raster(
    src_path: str | Path,
    dst_path: str | Path,
    mask_gdf: gpd.GeoDataFrame,
    target_crs: str = TARGET_CRS,
    target_res_m: float | None = None,
    resampling: str | Resampling = "bilinear",
    nodata: float | None = None,
) -> Path:
    """
    Full harmonisation pipeline for a raster dataset.

    Steps applied in order:
      1. Reproject to ``target_crs``.
      2. Resample to ``target_res_m`` (if specified).
      3. Clip and mask to ``mask_gdf``.

    The output is saved to ``dst_path``.

    Parameters
    ----------
    src_path : str or Path
    dst_path : str or Path
    mask_gdf : GeoDataFrame
        Study-area boundary in any CRS (reprojected automatically).
    target_crs : str
        Target CRS (default: ``"EPSG:32630"``).
    target_res_m : float or None
        Target pixel resolution in metres.  If None, source resolution
        is preserved after reprojection.
    resampling : str or Resampling
    nodata : float or None
        Override nodata value.

    Returns
    -------
    Path
        Absolute path of the final harmonised file.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)

    tmp_files: list[Path] = []
    try:
        current = src_path

        # Step 1: reproject if needed
        with rasterio.open(current) as src:
            src_crs_str = src.crs.to_string() if src.crs else ""
        target_crs_str = CRS.from_string(target_crs).to_string()

        if src_crs_str != target_crs_str or nodata is not None:
            tmp_reproj = Path(tempfile.mktemp(suffix="_reproj.tif"))
            tmp_files.append(tmp_reproj)
            if src_crs_str != target_crs_str:
                current = reproject_raster(
                    current, tmp_reproj, target_crs=target_crs,
                    resampling=resampling, nodata=nodata,
                )
            else:
                _copy_with_nodata(current, tmp_reproj, nodata)
                current = tmp_reproj

        # Step 2: resample if requested
        if target_res_m is not None:
            tmp_res = Path(tempfile.mktemp(suffix="_res.tif"))
            tmp_files.append(tmp_res)
            current = resample_raster(
                current, tmp_res,
                target_res_m=target_res_m, resampling=resampling,
            )

        # Step 3: mask to study area
        current = mask_to_study_area(current, dst_path, mask_gdf)

    finally:
        for tmp in tmp_files:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    log.info("Harmonised %s → %s", src_path.name, dst_path.name)
    return dst_path


def _copy_with_nodata(src_path: Path, dst_path: Path, nodata: float | None) -> None:
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(src.read())


# ---------------------------------------------------------------------------
# 6. Merge and mosaic DEM tiles
# ---------------------------------------------------------------------------
def merge_dem_tiles(
    tile_dir: str | Path,
    dst_path: str | Path,
    target_crs: str = TARGET_CRS,
    mask_gdf: gpd.GeoDataFrame | None = None,
    pattern: str = "*.tif",
    resampling: str | Resampling = "bilinear",
) -> Path:
    """
    Merge multiple DEM tiles into a single mosaic, reproject, and
    optionally clip to the study area.

    Parameters
    ----------
    tile_dir : str or Path
        Directory containing DEM tile GeoTIFFs.
    dst_path : str or Path
        Output mosaic GeoTIFF.
    target_crs : str
        Target CRS (default: ``"EPSG:32630"``).
    mask_gdf : GeoDataFrame or None
        If provided, clip mosaic to this boundary.
    pattern : str
        Glob pattern for tile selection (default ``"*.tif"``).
    resampling : str or Resampling

    Returns
    -------
    Path
    """
    tile_dir = Path(tile_dir)
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)

    tile_paths = sorted(tile_dir.glob(pattern))
    if not tile_paths:
        raise FileNotFoundError(f"No DEM tiles found matching {tile_dir / pattern}")

    log.info("Merging %d DEM tiles ...", len(tile_paths))
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = rio_merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=1,
        )
        tmp_mosaic = Path(tempfile.mktemp(suffix="_mosaic.tif"))
        with rasterio.open(tmp_mosaic, "w", **profile) as out:
            out.write(mosaic)
    finally:
        for ds in datasets:
            ds.close()

    # Reproject
    tmp_reproj = Path(tempfile.mktemp(suffix="_reproj.tif"))
    reproject_raster(tmp_mosaic, tmp_reproj, target_crs=target_crs,
                     resampling=resampling)
    tmp_mosaic.unlink(missing_ok=True)

    # Optionally clip
    if mask_gdf is not None:
        mask_to_study_area(tmp_reproj, dst_path, mask_gdf)
        tmp_reproj.unlink(missing_ok=True)
    else:
        tmp_reproj.rename(dst_path)

    log.info("DEM mosaic saved → %s", dst_path.name)
    return dst_path


# ---------------------------------------------------------------------------
# 7. Stack time-series rasters into a single NetCDF
# ---------------------------------------------------------------------------
def stack_rasters_to_netcdf(
    raster_paths: list[str | Path],
    dates: list[str],
    var_name: str,
    dst_path: str | Path,
    units: str = "",
    long_name: str = "",
) -> Path:
    """
    Stack a time-ordered list of single-band GeoTIFFs into a NetCDF4 file.

    The rasters must share the same CRS, extent, and resolution.
    Dates are stored as the ``time`` dimension.

    Parameters
    ----------
    raster_paths : list of str or Path
        Ordered list of raster files (one per time step).
    dates : list of str
        ISO date strings matching the raster order (e.g. ``"2002-04-01"``).
    var_name : str
        Name of the data variable in the output NetCDF (e.g. ``"gwsa"``).
    dst_path : str or Path
        Output NetCDF4 file.
    units : str
        Units string written to variable attributes (e.g. ``"mm EWH"``).
    long_name : str
        Long descriptive name for the variable.

    Returns
    -------
    Path
    """
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)

    if len(raster_paths) != len(dates):
        raise ValueError("raster_paths and dates must have the same length.")

    arrays, x_coords, y_coords, crs_wkt = [], None, None, None

    for path in raster_paths:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            if x_coords is None:
                rows = np.arange(src.height)
                cols = np.arange(src.width)
                xs = src.transform.c + (cols + 0.5) * src.transform.a
                ys = src.transform.f + (rows + 0.5) * src.transform.e
                x_coords, y_coords = xs, ys
                crs_wkt = src.crs.to_wkt() if src.crs else ""
        arrays.append(data)

    stack = np.stack(arrays, axis=0)  # (time, y, x)
    time_index = pd.DatetimeIndex(dates)

    ds = xr.Dataset(
        {var_name: xr.DataArray(
            stack,
            dims=["time", "y", "x"],
            coords={"time": time_index, "y": y_coords, "x": x_coords},
            attrs={"units": units, "long_name": long_name or var_name,
                   "crs_wkt": crs_wkt or ""},
        )},
    )
    ds.attrs["Conventions"] = "CF-1.8"
    ds.to_netcdf(dst_path, format="NETCDF4")

    log.info(
        "Stacked %d rasters → %s  [var=%s, %d time steps]",
        len(raster_paths), dst_path.name, var_name, len(dates),
    )
    return dst_path


# ---------------------------------------------------------------------------
# 8. Export Cloud Optimised GeoTIFF
# ---------------------------------------------------------------------------
def export_cog(
    src_array: np.ndarray,
    dst_path: str | Path,
    profile: dict,
    overview_levels: list[int] | None = None,
) -> Path:
    """
    Write a numpy array as a Cloud Optimised GeoTIFF (COG).

    Parameters
    ----------
    src_array : ndarray
        2-D (single band) or 3-D (bands, rows, cols) array.
    dst_path : str or Path
        Output COG file.
    profile : dict
        Rasterio profile dict specifying CRS, transform, nodata, dtype, etc.
    overview_levels : list of int or None
        Decimation factors for internal overviews
        (default: [2, 4, 8, 16, 32]).

    Returns
    -------
    Path
    """
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)

    if overview_levels is None:
        overview_levels = [2, 4, 8, 16, 32]

    if src_array.ndim == 2:
        src_array = src_array[np.newaxis, ...]

    tmp_path = Path(tempfile.mktemp(suffix="_tmp.tif"))
    cog_profile = {**profile, "driver": "GTiff", "count": src_array.shape[0]}

    with rasterio.open(tmp_path, "w", **cog_profile) as tmp:
        tmp.write(src_array)
        tmp.build_overviews(overview_levels, Resampling.average)
        tmp.update_tags(ns="rio_overview", resampling="average")

    cog_meta = {
        **cog_profile,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "predictor": 2,
        "copy_src_overviews": True,
    }
    with rasterio.open(tmp_path) as tmp:
        with rasterio.open(dst_path, "w", **cog_meta) as cog:
            cog.write(tmp.read())
            cog.build_overviews(overview_levels, Resampling.average)

    tmp_path.unlink(missing_ok=True)
    log.info("Exported COG → %s", dst_path.name)
    return dst_path


# ---------------------------------------------------------------------------
# 9. Vector geometry repair
# ---------------------------------------------------------------------------
def repair_vector_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fix invalid geometries in a GeoDataFrame.

    Uses ``buffer(0)`` as the canonical fix for self-intersecting rings.
    Rows with null or unfixable geometries are dropped with a warning.

    Parameters
    ----------
    gdf : GeoDataFrame

    Returns
    -------
    GeoDataFrame
        Copy with repaired geometries.
    """
    gdf = gdf.copy()
    null_mask    = gdf.geometry.isna()
    n_null = null_mask.sum()

    if n_null:
        log.warning("Dropping %d features with null geometry.", n_null)
        gdf = gdf[~null_mask].copy()

    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = invalid_mask.sum()

    if n_invalid:
        log.info("Repairing %d invalid geometries via buffer(0).", n_invalid)
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)
        still_invalid = ~gdf.geometry.is_valid
        if still_invalid.any():
            log.warning(
                "%d geometries remain invalid after buffer(0) — dropping.",
                still_invalid.sum(),
            )
            gdf = gdf[~still_invalid].copy()

    return gdf


# ---------------------------------------------------------------------------
# 10. Remove duplicate vector features
# ---------------------------------------------------------------------------
def remove_duplicate_features(
    gdf: gpd.GeoDataFrame,
    subset_cols: list[str] | None = None,
    keep: str = "first",
) -> gpd.GeoDataFrame:
    """
    Remove duplicate rows from a GeoDataFrame.

    If ``subset_cols`` is None, duplicates are identified by geometry WKT.

    Parameters
    ----------
    gdf : GeoDataFrame
    subset_cols : list of str or None
        Attribute columns to use for duplicate detection.
        If None, geometry is used.
    keep : str
        Which duplicate to keep: ``"first"`` or ``"last"``.

    Returns
    -------
    GeoDataFrame
    """
    before = len(gdf)
    if subset_cols:
        gdf = gdf.drop_duplicates(subset=subset_cols, keep=keep)
    else:
        wkt = gdf.geometry.apply(lambda g: g.wkt if g else "")
        gdf = gdf[~wkt.duplicated(keep=keep)].copy()
    n_removed = before - len(gdf)
    if n_removed:
        log.info("Removed %d duplicate features.", n_removed)
    return gdf


# ---------------------------------------------------------------------------
# 11. Standardise column names
# ---------------------------------------------------------------------------
def standardize_column_names(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Lower-case all column names and replace spaces / hyphens with underscores.

    Parameters
    ----------
    gdf : GeoDataFrame

    Returns
    -------
    GeoDataFrame
    """
    gdf = gdf.copy()
    gdf.columns = [
        re.sub(r"[\s\-]+", "_", col).lower()
        for col in gdf.columns
    ]
    return gdf


# ---------------------------------------------------------------------------
# 12. Reproject vector data
# ---------------------------------------------------------------------------
def reproject_vector(
    gdf: gpd.GeoDataFrame,
    target_crs: str = TARGET_CRS,
) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to ``target_crs`` if not already there.

    Parameters
    ----------
    gdf : GeoDataFrame
    target_crs : str

    Returns
    -------
    GeoDataFrame
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS. Set it before reprojecting.")
    if gdf.crs.to_string() == CRS.from_string(target_crs).to_string():
        return gdf
    return gdf.to_crs(target_crs)


# ---------------------------------------------------------------------------
# 13. Clip vector to study area
# ---------------------------------------------------------------------------
def clip_vector_to_study_area(
    gdf: gpd.GeoDataFrame,
    mask_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Clip features in ``gdf`` to the union of ``mask_gdf`` polygons.

    Both GeoDataFrames are reprojected to the same CRS before clipping.

    Parameters
    ----------
    gdf : GeoDataFrame
        Features to clip.
    mask_gdf : GeoDataFrame
        Clip boundary.

    Returns
    -------
    GeoDataFrame
        Clipped features in the CRS of ``gdf``.
    """
    if mask_gdf.crs != gdf.crs:
        mask_gdf = mask_gdf.to_crs(gdf.crs)
    clipped = gpd.clip(gdf, mask_gdf)
    log.info("Clipped %d → %d features.", len(gdf), len(clipped))
    return clipped.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 14. Full vector harmonisation pipeline
# ---------------------------------------------------------------------------
def harmonize_vector(
    src_path: str | Path,
    dst_path: str | Path,
    mask_gdf: gpd.GeoDataFrame,
    target_crs: str = TARGET_CRS,
    layer: str | int | None = None,
) -> Path:
    """
    Full harmonisation pipeline for a vector dataset.

    Steps:
      1. Read file (optionally a specific layer from a GeoPackage).
      2. Repair invalid geometries.
      3. Remove duplicates (by geometry).
      4. Standardise column names.
      5. Reproject to ``target_crs``.
      6. Clip to ``mask_gdf``.
      7. Write to ``dst_path`` as GeoPackage.

    Parameters
    ----------
    src_path : str or Path
    dst_path : str or Path
        Output GeoPackage path.
    mask_gdf : GeoDataFrame
        Study-area mask.
    target_crs : str
    layer : str, int, or None
        Layer name or index for multi-layer files (e.g. GADM GeoPackage).

    Returns
    -------
    Path
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)

    read_kwargs: dict = {"filename": src_path}
    if layer is not None:
        read_kwargs["layer"] = layer
    gdf = gpd.read_file(**read_kwargs)

    gdf = repair_vector_geometries(gdf)
    gdf = remove_duplicate_features(gdf)
    gdf = standardize_column_names(gdf)
    gdf = reproject_vector(gdf, target_crs)
    gdf = clip_vector_to_study_area(gdf, mask_gdf.to_crs(target_crs))

    layer_name = dst_path.stem
    gdf.to_file(dst_path, driver="GPKG", layer=layer_name)
    log.info(
        "Harmonised vector %s → %s  (%d features)",
        src_path.name, dst_path.name, len(gdf),
    )
    return dst_path


# ---------------------------------------------------------------------------
# 15. QC: CRS consistency check
# ---------------------------------------------------------------------------
def check_crs_consistency(
    paths: list[str | Path],
    target_crs: str = TARGET_CRS,
) -> dict[str, bool]:
    """
    Check whether all raster / vector files share a consistent CRS.

    Parameters
    ----------
    paths : list of str or Path
        File paths to check (rasters or vectors).
    target_crs : str
        Expected CRS string.

    Returns
    -------
    dict
        Mapping ``{filename: True/False}`` where True means CRS matches.
    """
    target = CRS.from_string(target_crs)
    results: dict[str, bool] = {}

    for p in paths:
        p = Path(p)
        try:
            if p.suffix.lower() in {".tif", ".tiff", ".nc", ".vrt"}:
                with rasterio.open(p) as ds:
                    file_crs = ds.crs
            else:
                gdf = gpd.read_file(p, rows=0)
                file_crs = gdf.crs

            match = file_crs is not None and CRS(file_crs) == target
            results[p.name] = match
            if not match:
                log.warning(
                    "CRS mismatch: %s  (found %s, expected %s)",
                    p.name, file_crs, target_crs,
                )
        except Exception as exc:
            log.error("Could not read CRS from %s: %s", p.name, exc)
            results[p.name] = False

    n_ok = sum(results.values())
    log.info("CRS check: %d/%d files match %s", n_ok, len(paths), target_crs)
    return results


# ---------------------------------------------------------------------------
# 16. QC: Detect temporal gaps in a monthly time series
# ---------------------------------------------------------------------------
def detect_temporal_gaps(
    dates: list[str] | pd.DatetimeIndex,
    freq: str = "MS",
) -> list[str]:
    """
    Identify missing months in a time series.

    Parameters
    ----------
    dates : list of str or DatetimeIndex
        Observed time steps (monthly).
    freq : str
        Pandas frequency string for the expected cadence
        (default ``"MS"`` = month start).

    Returns
    -------
    list of str
        Missing month strings in ``"YYYY-MM"`` format.
        Empty list if the series is complete.
    """
    observed   = pd.DatetimeIndex(dates).normalize()
    full_range = pd.date_range(observed.min(), observed.max(), freq=freq)
    missing    = full_range.difference(observed)

    if missing.empty:
        log.info("Temporal completeness check: no gaps detected.")
    else:
        preview = ", ".join(missing.strftime("%Y-%m").tolist()[:10])
        if len(missing) > 10:
            preview += " ..."
        log.warning(
            "Temporal gaps detected (%d missing months): %s",
            len(missing), preview,
        )

    return missing.strftime("%Y-%m").tolist()


# ---------------------------------------------------------------------------
# 17. QC: Validate raster grid alignment
# ---------------------------------------------------------------------------
def validate_raster_grid(
    paths: list[str | Path],
    tolerance_m: float = 1.0,
) -> dict[str, bool]:
    """
    Check whether a list of rasters share the same grid (CRS, resolution,
    origin, and extent) within a tolerance.

    Parameters
    ----------
    paths : list of str or Path
        Raster file paths.
    tolerance_m : float
        Maximum acceptable difference in origin or pixel size (metres).

    Returns
    -------
    dict
        ``{filename: aligned}`` where ``aligned`` is True if the raster
        matches the first file in the list.
    """
    results: dict[str, bool] = {}
    ref_profile: dict | None = None

    for p in paths:
        p = Path(p)
        try:
            with rasterio.open(p) as src:
                profile = {
                    "crs":      src.crs,
                    "res_x":    src.transform.a,
                    "res_y":    abs(src.transform.e),
                    "origin_x": src.transform.c,
                    "origin_y": src.transform.f,
                }
                if ref_profile is None:
                    ref_profile = profile
                    results[p.name] = True
                    continue

                ok = (
                    profile["crs"] == ref_profile["crs"]
                    and abs(profile["res_x"]    - ref_profile["res_x"])    <= tolerance_m
                    and abs(profile["res_y"]    - ref_profile["res_y"])    <= tolerance_m
                    and abs(profile["origin_x"] - ref_profile["origin_x"]) <= tolerance_m
                    and abs(profile["origin_y"] - ref_profile["origin_y"]) <= tolerance_m
                )
                results[p.name] = ok
                if not ok:
                    log.warning(
                        "Grid mismatch: %s  (ref: %s)",
                        p.name, Path(paths[0]).name,
                    )
        except Exception as exc:
            log.error("Could not validate %s: %s", p.name, exc)
            results[p.name] = False

    n_ok = sum(results.values())
    log.info("Grid alignment: %d/%d rasters aligned.", n_ok, len(paths))
    return results


# ---------------------------------------------------------------------------
# 18. QC: Full report
# ---------------------------------------------------------------------------
def run_qc_report(
    raster_paths: list[str | Path] | None = None,
    vector_paths: list[str | Path] | None = None,
    target_crs: str = TARGET_CRS,
) -> dict:
    """
    Run all QC checks and return a summary dictionary.

    Parameters
    ----------
    raster_paths : list of str or Path or None
    vector_paths : list of str or Path or None
    target_crs : str

    Returns
    -------
    dict
        Keys: ``"crs_rasters"``, ``"crs_vectors"``, ``"grid_alignment"``.
        Each value is the result dict from the corresponding check function.
    """
    report: dict = {}

    if raster_paths:
        report["crs_rasters"]    = check_crs_consistency(raster_paths, target_crs)
        report["grid_alignment"] = validate_raster_grid(raster_paths)
    if vector_paths:
        report["crs_vectors"] = check_crs_consistency(vector_paths, target_crs)

    n_crs_ok = (
        sum(report.get("crs_rasters", {}).values())
        + sum(report.get("crs_vectors", {}).values())
    )
    n_total   = len(raster_paths or []) + len(vector_paths or [])
    n_grid_ok = sum(report.get("grid_alignment", {}).values())

    log.info(
        "QC summary: CRS %d/%d OK, grid alignment %d/%d OK",
        n_crs_ok, n_total,
        n_grid_ok, len(raster_paths or []),
    )
    return report


# ---------------------------------------------------------------------------
# 19. Batch raster harmonisation
# ---------------------------------------------------------------------------
def batch_harmonize_rasters(
    src_dir: str | Path,
    dst_dir: str | Path,
    mask_gdf: gpd.GeoDataFrame,
    pattern: str = "*.tif",
    target_crs: str = TARGET_CRS,
    target_res_m: float | None = None,
    resampling: str | Resampling = "bilinear",
    overwrite: bool = False,
) -> list[Path]:
    """
    Run ``harmonize_raster()`` on all files in ``src_dir`` matching
    ``pattern``.

    Parameters
    ----------
    src_dir : str or Path
        Source directory.
    dst_dir : str or Path
        Destination directory.
    mask_gdf : GeoDataFrame
        Study-area mask.
    pattern : str
        Glob pattern for file selection (default ``"*.tif"``).
    target_crs : str
    target_res_m : float or None
    resampling : str or Resampling
    overwrite : bool
        If False (default), skip files that already exist in ``dst_dir``.

    Returns
    -------
    list of Path
        Paths to all output files (processed + already-existing).
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    ensure_dir(dst_dir)

    src_files = sorted(src_dir.glob(pattern))
    if not src_files:
        log.warning("No files found in %s matching '%s'.", src_dir, pattern)
        return []

    outputs: list[Path] = []
    for src in src_files:
        dst = dst_dir / src.name
        if dst.exists() and not overwrite:
            log.debug("Skipping (already exists): %s", dst.name)
            outputs.append(dst)
            continue
        try:
            harmonize_raster(
                src, dst, mask_gdf,
                target_crs=target_crs,
                target_res_m=target_res_m,
                resampling=resampling,
            )
            outputs.append(dst)
        except Exception as exc:
            log.error("Failed to harmonise %s: %s", src.name, exc)

    log.info(
        "Batch harmonisation complete: %d/%d files processed to %s",
        len(outputs), len(src_files), dst_dir,
    )
    return outputs
