"""
Process MODIS MOD11A2 v6.1 HDF4 files.
Steps:
  1. Extract LST_Day_1km and QC_Day for all h17v08 and h18v08 granules (mosaicked per date).
  2. Apply scale (×0.02 − 273.15), mask invalid pixels (QC bits 0-1 != 00).
  3. Reproject from MODIS sinusoidal to EPSG:32630, resample to 1 km.
  4. Clip to coastal Ghana study area.
  5. Compute monthly means from 8-day composites.
  6. Aggregate to 10 km analysis grid.
  7. Compute dry-season (Dec–Feb) annual means and Mann-Kendall trends (v1; superseded by
     build_uhgpi_v2.py, which uses lst_8day_by_cell.csv, 2001–2023, >= 3 composites per season,
     >= 18 seasons and the Hamed–Rao test).
  8. Export processed LST grids and time series.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import pymannkendall as mk
from pathlib import Path
from pyhdf.SD import SD, SDC
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
RAW_DIR   = BASE / "data/raw/modis_lst"
OUT_DIR   = BASE / "data/processed/lst"
GRID_PATH = BASE / "data/processed/risk_index/cgsri_grid10km.gpkg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── study area bbox (WGS84) ────────────────────────────────────────────────
BBOX_WGS84 = (-3.8, 4.4, 1.6, 7.2)   # lon_min, lat_min, lon_max, lat_max
TARGET_CRS = "EPSG:32630"
LST_FILL   = 0   # MODIS fill value for LST

# ── MODIS sinusoidal projection ────────────────────────────────────────────
SINU_PROJ = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
    "+a=6371007.181 +b=6371007.181 +units=m +no_defs"
)

# ── parse date from granule name ───────────────────────────────────────────
def granule_date(path: Path) -> pd.Timestamp:
    """Extract acquisition date from MOD11A2.A<YYYYDDD>... filename."""
    m = re.search(r"A(\d{4})(\d{3})", path.name)
    if not m:
        raise ValueError(f"Cannot parse date from {path.name}")
    year, doy = int(m.group(1)), int(m.group(2))
    return pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1)


# ── read a single HDF4 granule ─────────────────────────────────────────────
def read_hdf4_lst(path: Path):
    """Return (lst_arr float32 K, nrows, ncols, ulx, uly, pixel_size)."""
    hdf = SD(str(path), SDC.READ)

    lst_ds  = hdf.select("LST_Day_1km")
    qc_ds   = hdf.select("QC_Day")
    attrs   = lst_ds.attributes()
    meta    = hdf.attributes()

    lst = lst_ds[:].astype(np.float32)
    qc  = qc_ds[:].astype(np.uint8)
    hdf.end()

    scale = attrs.get("scale_factor", 0.02)
    fill  = attrs.get("_FillValue", 0)

    # mask invalid: fill value or QC bits 0-1 != 00 (mandatory QA)
    valid = (lst != fill) & ((qc & 0x03) == 0)
    lst[~valid] = np.nan
    lst[valid]  = lst[valid] * scale - 273.15   # K → °C

    pixel_size = 926.625433  # metres per pixel at 1 km
    # Coordinates are computed in main() from the tile identifier in the filename,
    # not from the HDF4 metadata, to ensure correctness across h17v08 and h18v08.
    return lst, pixel_size


def reproject_to_utm(lst_arr, ulx, uly, pixel_size, out_path: Path):
    """Write reprojected GeoTIFF (EPSG:32630, 1 km) for the study area."""
    nrows, ncols = lst_arr.shape
    src_transform = rasterio.transform.from_origin(ulx, uly, pixel_size, pixel_size)

    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=nrows, width=ncols,
            count=1, dtype="float32",
            crs=SINU_PROJ,
            transform=src_transform,
            nodata=np.nan,
        ) as src_ds:
            src_ds.write(lst_arr, 1)

            dst_transform, dst_w, dst_h = calculate_default_transform(
                SINU_PROJ, TARGET_CRS, ncols, nrows, left=ulx, bottom=uly - nrows*pixel_size,
                right=ulx + ncols*pixel_size, top=uly,
            )

            kwargs = src_ds.meta.copy()
            kwargs.update({"crs": TARGET_CRS, "transform": dst_transform,
                           "width": dst_w, "height": dst_h})

            with rasterio.open(out_path, "w", **kwargs) as dst:
                reproject(
                    source=rasterio.band(src_ds, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src_transform,
                    src_crs=SINU_PROJ,
                    dst_transform=dst_transform,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.bilinear,
                )


# ── aggregate one GeoTIFF to 10 km grid cells ─────────────────────────────
def aggregate_to_grid(tif_path: Path, grid: gpd.GeoDataFrame) -> pd.Series:
    """Return mean LST per grid cell_id."""
    from rasterstats import zonal_stats
    grid_utm = grid.to_crs(TARGET_CRS)
    stats = zonal_stats(grid_utm, str(tif_path), stats=["mean"], nodata=np.nan)
    return pd.Series([s["mean"] for s in stats], index=grid["cell_id"])


# ── main processing loop ───────────────────────────────────────────────────
def mosaic_tiles(tile_paths: list, out_path: Path):
    """Mosaic two reprojected tiles (h17v08 + h18v08) into one GeoTIFF."""
    from rasterio.merge import merge as rio_merge
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = rio_merge(datasets, nodata=np.nan)
    meta = datasets[0].meta.copy()
    meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2],
                 "transform": transform})
    for ds in datasets:
        ds.close()
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic)


def main():
    h17_files = sorted(RAW_DIR.glob("MOD11A2.A*.h17v08*.hdf"))
    h18_files = sorted(RAW_DIR.glob("MOD11A2.A*.h18v08*.hdf"))

    # Build date-keyed dict for each tile
    h17_by_date = {granule_date(p): p for p in h17_files}
    h18_by_date = {granule_date(p): p for p in h18_files}

    all_dates = sorted(set(h17_by_date) | set(h18_by_date))
    print(f"h17v08 granules: {len(h17_files)}, h18v08: {len(h18_files)}")
    print(f"Total unique dates: {len(all_dates)}")

    if not all_dates:
        print("No granules found. Run download scripts first.")
        return

    grid = gpd.read_file(GRID_PATH)
    records = []

    for date in tqdm(all_dates, desc="Processing granules"):
        tif_path = OUT_DIR / f"lst_{date.strftime('%Y%m%d')}.tif"

        if not tif_path.exists():
            try:
                tile_tifs = []
                for tile_id, by_date in [("h17v08", h17_by_date), ("h18v08", h18_by_date)]:
                    if date in by_date:
                        h_val = 17 if tile_id == "h17v08" else 18
                        v_val = 8
                        tile_tif = OUT_DIR / f"lst_{date.strftime('%Y%m%d')}_{tile_id}.tif"
                        if not tile_tif.exists():
                            lst_arr, pix = read_hdf4_lst(by_date[date])
                            tile_size = pix * 1200
                            ulx_tile = (h_val - 18) * tile_size
                            uly_tile = (9 - v_val) * tile_size
                            reproject_to_utm(lst_arr, ulx_tile, uly_tile, pix, tile_tif)
                        tile_tifs.append(tile_tif)

                if len(tile_tifs) == 2:
                    mosaic_tiles(tile_tifs, tif_path)
                    for t in tile_tifs:
                        t.unlink(missing_ok=True)
                elif len(tile_tifs) == 1:
                    tile_tifs[0].rename(tif_path)
            except Exception as e:
                print(f"  SKIP {date}: {e}")
                continue

        if tif_path.exists():
            cell_means = aggregate_to_grid(tif_path, grid)
            row = {"date": date}
            row.update(cell_means.to_dict())
            records.append(row)

    if not records:
        print("No data processed.")
        return

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df.to_csv(OUT_DIR / "lst_8day_by_cell.csv", index=False)
    print(f"Saved 8-day LST table: {df.shape}")

    # ── monthly means ──────────────────────────────────────────────────────
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    cell_cols = [c for c in df.columns if c not in ("date", "year", "month")]
    monthly = df.groupby(["year", "month"])[cell_cols].mean().reset_index()
    monthly.to_csv(OUT_DIR / "lst_monthly_by_cell.csv", index=False)
    print(f"Saved monthly LST table: {monthly.shape}")

    # ── dry-season means (Dec–Feb) ─────────────────────────────────────────
    # Assign Dec to the following year's dry season
    df["ds_year"] = df["year"].where(df["month"] != 12, df["year"] + 1)
    dry = df[df["month"].isin([12, 1, 2])]
    dry_annual = dry.groupby("ds_year")[cell_cols].mean().reset_index()
    dry_annual.to_csv(OUT_DIR / "lst_dry_season_annual.csv", index=False)
    print(f"Saved dry-season annual table: {dry_annual.shape}")

    # ── Mann-Kendall trend per cell ────────────────────────────────────────
    trend_rows = []
    for cell in cell_cols:
        ts = dry_annual[cell].dropna()
        if len(ts) < 10:
            continue
        result = mk.original_test(ts)
        sens   = mk.sens_slope(ts)
        trend_rows.append({
            "cell_id":   cell,
            "mk_trend":  result.trend,
            "mk_z":      result.z,
            "mk_p":      result.p,
            "sens_slope_degC_yr": sens.slope,
        })

    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(OUT_DIR / "lst_trend_by_cell.csv", index=False)
    print(f"Saved trend table: {trend_df.shape}")
    print(trend_df["mk_trend"].value_counts())


if __name__ == "__main__":
    main()
