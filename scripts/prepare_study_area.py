"""
prepare_study_area.py
=====================
Stage 1 of the Ghana groundwater-subsidence pipeline.

Purpose
-------
Defines the spatial extent of the study by:
  1. Loading the Ghana national boundary (GADM Level 0).
  2. Extracting or deriving the Ghana coastline.
  3. Generating 30 km and 50 km inland coastal buffer zones.
  4. Selecting coastal districts (GADM Level 2) intersecting the buffers.
  5. Creating analysis grids at 10 km and 25 km resolution.
  6. Exporting all outputs as GeoPackage files to data/raw/boundaries/.
  7. Producing a study-area overview map.

Usage
-----
  python scripts/prepare_study_area.py

Requirements
------------
  geopandas, shapely, rasterio, pyproj, matplotlib, pyyaml, tqdm

  If GADM files are not yet downloaded, the script will prompt you to
  download them manually and place them in data/raw/boundaries/.

Outputs
-------
  data/raw/boundaries/ghana_boundary.gpkg
  data/raw/boundaries/ghana_coastline.gpkg
  data/raw/boundaries/study_area_coastal_30km.gpkg
  data/raw/boundaries/study_area_coastal_50km.gpkg
  data/raw/boundaries/coastal_districts.gpkg
  data/raw/boundaries/analysis_grid_10km.gpkg
  data/raw/boundaries/analysis_grid_25km.gpkg
  outputs/maps/study_area_overview.png

Notes
-----
- All outputs are projected to EPSG:32630 (WGS 84 / UTM Zone 30N).
- Buffers are generated from the coastline (not the national boundary),
  so they extend inland from the coast.
- Grid cells are square, aligned to a common origin, and clipped to
  the 50 km coastal buffer.
- Run this script before any other pipeline stage.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
from shapely.geometry import box, MultiPolygon, Polygon
from shapely.ops import unary_union
from pyproj import CRS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve project root and load config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config() -> tuple[dict, dict, dict]:
    """Load paths, study area, and thresholds config files."""
    paths_file = PROJECT_ROOT / "config" / "paths.yaml"
    sa_file = PROJECT_ROOT / "config" / "study_area.yaml"

    with open(paths_file) as f:
        paths = yaml.safe_load(f)
    with open(sa_file) as f:
        sa = yaml.safe_load(f)

    return paths, sa


# ---------------------------------------------------------------------------
# Helper: resolve relative path against project root
# ---------------------------------------------------------------------------
def rp(relative: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / relative


# ---------------------------------------------------------------------------
# Step 1: Load Ghana boundary
# ---------------------------------------------------------------------------
def load_ghana_boundary(boundaries_dir: Path) -> gpd.GeoDataFrame:
    """
    Load the Ghana national boundary (GADM Level 0).

    The function looks for a GeoPackage or Shapefile named gadm41_GHA_0.*
    or ghana_boundary.* in `boundaries_dir`. If not found, it raises a
    FileNotFoundError with download instructions.

    Parameters
    ----------
    boundaries_dir : Path
        Directory containing raw boundary files.

    Returns
    -------
    GeoDataFrame in EPSG:4326.
    """
    candidates = list(boundaries_dir.glob("gadm41_GHA_0.*")) + \
                 list(boundaries_dir.glob("gadm_410_GHA.*")) + \
                 list(boundaries_dir.glob("ghana_boundary.*"))

    # Filter to supported formats
    supported = {".gpkg", ".shp", ".geojson", ".json"}
    candidates = [p for p in candidates if p.suffix.lower() in supported]

    # Fall back to the combined multi-layer GADM GeoPackage
    multi_gpkg = boundaries_dir / "gadm41_GHA.gpkg"
    if not candidates and multi_gpkg.exists():
        log.info("Using multi-layer GADM GeoPackage: %s (layer ADM_ADM_0)", multi_gpkg)
        gdf = gpd.read_file(multi_gpkg, layer="ADM_ADM_0")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        log.info("Loaded Ghana boundary: %d feature(s)", len(gdf))
        return gdf

    if not candidates:
        msg = (
            f"Ghana boundary file not found in {boundaries_dir}.\n"
            "Please download GADM Ghana Level 0 from:\n"
            "  https://gadm.org/download_country.html\n"
            "Place the file (GeoPackage or Shapefile) in:\n"
            f"  {boundaries_dir}\n"
            "Then re-run this script."
        )
        raise FileNotFoundError(msg)

    path = candidates[0]
    log.info("Loading Ghana boundary from: %s", path)
    gdf = gpd.read_file(path)

    # Ensure geographic CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    log.info("Loaded Ghana boundary: %d feature(s)", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Step 2: Extract coastline
# ---------------------------------------------------------------------------
def extract_coastline(boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Derive the Ghana coastline from the national boundary polygon.

    The coastline is approximated as the southern exterior boundary
    of the national polygon. For a more accurate coastline, replace
    this with an OSM-derived coastal polygon.

    Parameters
    ----------
    boundary_gdf : GeoDataFrame
        Ghana national boundary in EPSG:4326.

    Returns
    -------
    GeoDataFrame containing the coastline as a LineString in EPSG:4326.
    """
    log.info("Extracting coastline from national boundary ...")

    union_geom = unary_union(boundary_gdf.geometry)

    # Extract exterior ring(s)
    if union_geom.geom_type == "Polygon":
        coastline_geom = union_geom.exterior
    elif union_geom.geom_type == "MultiPolygon":
        from shapely.ops import linemerge
        exteriors = [p.exterior for p in union_geom.geoms]
        coastline_geom = linemerge(exteriors)
    else:
        coastline_geom = union_geom.boundary

    coastline_gdf = gpd.GeoDataFrame(
        {"geometry": [coastline_geom], "source": ["GADM boundary exterior"]},
        crs="EPSG:4326",
    )
    log.info("Coastline extracted successfully.")
    return coastline_gdf


# ---------------------------------------------------------------------------
# Step 3: Generate coastal buffers
# ---------------------------------------------------------------------------
def generate_coastal_buffers(
    boundary_gdf: gpd.GeoDataFrame,
    buffer_30km_m: int = 30_000,
    buffer_50km_m: int = 50_000,
    utm_crs: str = "EPSG:32630",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Generate 30 km and 50 km inland coastal buffer zones.

    Buffers are generated from the **southern coastline only**, not the
    entire national perimeter.  The previous implementation buffered the
    full exterior boundary of Ghana (all four sides), which produced a
    frame-like band around the whole country instead of a coastal strip.

    Correct method
    --------------
    1. Project the national boundary to UTM.
    2. Extract the full exterior boundary line.
    3. Clip that line to a coastal bounding box (y < COASTAL_Y_MAX_UTM,
       i.e. roughly 6.5 °N) to keep only the southern coastal edge.
    4. Buffer the coastal line segment outward by 30 km / 50 km.
    5. Clip the resulting buffer polygon back to the national boundary so
       the study area stays inside Ghana.

    Ghana's actual coastline lies entirely below ~5.2 °N.  Setting the
    clip threshold at 6.5 °N (≈ 718 000 m northing in EPSG:32630) safely
    isolates the coast while excluding the northern and lateral land borders
    with Burkina Faso, Côte d'Ivoire, and Togo.

    Parameters
    ----------
    boundary_gdf : GeoDataFrame
        Ghana national boundary in EPSG:4326.
    buffer_30km_m : int
        30 km buffer distance in metres.
    buffer_50km_m : int
        50 km buffer distance in metres.
    utm_crs : str
        Target projected CRS (default EPSG:32630).

    Returns
    -------
    Tuple of (buffer_30km GeoDataFrame, buffer_50km GeoDataFrame) in UTM CRS.
    """
    # ------------------------------------------------------------------
    # Coastal clip threshold in UTM northing metres.
    # 6.5 °N ≈ 718 500 m in EPSG:32630.  Adding a small margin keeps any
    # near-coastal boundary vertices that sit fractionally above 6.5 °N.
    # ------------------------------------------------------------------
    COASTAL_Y_MAX_UTM = 720_000   # ~6.5 °N in EPSG:32630

    log.info("Projecting boundary to %s for buffering ...", utm_crs)
    boundary_utm = boundary_gdf.to_crs(utm_crs)
    boundary_union = unary_union(boundary_utm.geometry)

    # ------------------------------------------------------------------
    # Extract the full exterior boundary line of Ghana
    # ------------------------------------------------------------------
    if boundary_union.geom_type == "Polygon":
        full_boundary_line = boundary_union.exterior
    else:
        from shapely.ops import linemerge
        full_boundary_line = linemerge(
            [p.exterior for p in boundary_union.geoms]
        )

    # ------------------------------------------------------------------
    # Clip to the coastal zone: keep only segments below ~6.5 °N
    # ------------------------------------------------------------------
    minx, miny, maxx, _ = boundary_union.bounds
    # Use a generous W/E/S margin so no coastal vertex is accidentally clipped
    coastal_clip_box = box(
        minx - 100_000,   # west: well beyond Ghana's western edge
        miny - 50_000,    # south: well below Ghana's coast
        maxx + 100_000,   # east: well beyond Ghana's eastern edge
        COASTAL_Y_MAX_UTM,  # north: ~6.5 °N ceiling
    )
    coastline_only = full_boundary_line.intersection(coastal_clip_box)

    if coastline_only.is_empty:
        raise RuntimeError(
            "Coastal line extraction returned an empty geometry.  "
            "Check COASTAL_Y_MAX_UTM and the boundary CRS."
        )

    log.info(
        "Coastal boundary segment extracted (y < %.0f m ≈ 6.5 °N).",
        COASTAL_Y_MAX_UTM,
    )

    # ------------------------------------------------------------------
    # Buffer the coastline and clip to Ghana
    # ------------------------------------------------------------------
    log.info("Generating 30 km coastal buffer ...")
    buf_30_clipped = coastline_only.buffer(buffer_30km_m).intersection(boundary_union)

    log.info("Generating 50 km coastal buffer ...")
    buf_50_clipped = coastline_only.buffer(buffer_50km_m).intersection(boundary_union)

    gdf_30 = gpd.GeoDataFrame(
        {"geometry": [buf_30_clipped],
         "buffer_km": [30],
         "description": ["30 km inland coastal buffer, coastal Ghana"]},
        crs=utm_crs,
    )
    gdf_50 = gpd.GeoDataFrame(
        {"geometry": [buf_50_clipped],
         "buffer_km": [50],
         "description": ["50 km inland coastal buffer, coastal Ghana"]},
        crs=utm_crs,
    )

    # Sanity check: centroid of buf50 should be well south of 8 °N
    buf50_centroid_y = buf_50_clipped.centroid.y
    buf50_area_km2 = buf_50_clipped.area / 1e6
    ghana_area_km2 = boundary_union.area / 1e6
    log.info(
        "50 km buffer: area = %.0f km²  (%.1f %% of Ghana), centroid y = %.0f m",
        buf50_area_km2, 100 * buf50_area_km2 / ghana_area_km2, buf50_centroid_y,
    )
    if buf50_area_km2 > 0.25 * ghana_area_km2:
        log.warning(
            "50 km buffer covers > 25 %% of Ghana (%.0f km²).  "
            "Expected ~25 000–35 000 km².  Check COASTAL_Y_MAX_UTM.",
            buf50_area_km2,
        )

    log.info("Coastal buffers generated.")
    return gdf_30, gdf_50


# ---------------------------------------------------------------------------
# Step 4: Select coastal districts
# ---------------------------------------------------------------------------
def select_coastal_districts(
    boundaries_dir: Path,
    buffer_gdf: gpd.GeoDataFrame,
    utm_crs: str = "EPSG:32630",
) -> gpd.GeoDataFrame:
    """
    Select GADM Level 2 districts that intersect the coastal buffer.

    Parameters
    ----------
    boundaries_dir : Path
        Directory containing GADM Level 2 data.
    buffer_gdf : GeoDataFrame
        Coastal buffer zone (50 km recommended) in UTM CRS.
    utm_crs : str
        Target projected CRS.

    Returns
    -------
    GeoDataFrame of coastal districts in UTM CRS.
    """
    candidates = list(boundaries_dir.glob("gadm41_GHA_2.*")) + \
                 list(boundaries_dir.glob("gadm_410_GHA_2.*")) + \
                 list(boundaries_dir.glob("ghana_districts.*"))

    supported = {".gpkg", ".shp", ".geojson", ".json"}
    candidates = [p for p in candidates if p.suffix.lower() in supported]

    # Fall back to the combined multi-layer GADM GeoPackage
    multi_gpkg = boundaries_dir / "gadm41_GHA.gpkg"
    if not candidates and multi_gpkg.exists():
        log.info("Using multi-layer GADM GeoPackage: %s (layer ADM_ADM_2)", multi_gpkg)
        districts = gpd.read_file(multi_gpkg, layer="ADM_ADM_2")
        if districts.crs is None:
            districts = districts.set_crs("EPSG:4326")
        districts_utm = districts.to_crs(utm_crs)
        buffer_union = unary_union(buffer_gdf.geometry)
        coastal = districts_utm[districts_utm.intersects(buffer_union)].copy()
        coastal = coastal.reset_index(drop=True)
        log.info("Selected %d coastal districts.", len(coastal))
        return coastal

    if not candidates:
        log.warning(
            "GADM Level 2 (districts) file not found in %s. "
            "Download from https://gadm.org/ and re-run to include district selection.",
            boundaries_dir,
        )
        return gpd.GeoDataFrame(columns=["geometry"], crs=utm_crs)

    path = candidates[0]
    log.info("Loading districts from: %s", path)
    districts = gpd.read_file(path)

    if districts.crs is None:
        districts = districts.set_crs("EPSG:4326")
    districts_utm = districts.to_crs(utm_crs)

    # Spatial join: select districts intersecting the buffer
    buffer_union = unary_union(buffer_gdf.geometry)
    coastal = districts_utm[districts_utm.intersects(buffer_union)].copy()
    coastal = coastal.reset_index(drop=True)
    log.info("Selected %d coastal districts.", len(coastal))
    return coastal


# ---------------------------------------------------------------------------
# Step 5: Create analysis grids
# ---------------------------------------------------------------------------
def create_analysis_grid(
    clip_gdf: gpd.GeoDataFrame,
    cell_size_m: int,
    utm_crs: str = "EPSG:32630",
) -> gpd.GeoDataFrame:
    """
    Create a regular square grid clipped to the study area boundary.

    Parameters
    ----------
    clip_gdf : GeoDataFrame
        Polygon defining the grid extent (typically 50 km buffer).
    cell_size_m : int
        Grid cell size in metres (e.g. 10 000 or 25 000).
    utm_crs : str
        Projected CRS for the grid.

    Returns
    -------
    GeoDataFrame of grid cell polygons with row/column IDs.
    """
    log.info("Creating %d m analysis grid ...", cell_size_m)

    clip_union = unary_union(clip_gdf.to_crs(utm_crs).geometry)
    xmin, ymin, xmax, ymax = clip_union.bounds

    # Snap to a regular grid origin
    xmin_snap = np.floor(xmin / cell_size_m) * cell_size_m
    ymin_snap = np.floor(ymin / cell_size_m) * cell_size_m
    xmax_snap = np.ceil(xmax / cell_size_m) * cell_size_m
    ymax_snap = np.ceil(ymax / cell_size_m) * cell_size_m

    cols = np.arange(xmin_snap, xmax_snap, cell_size_m)
    rows = np.arange(ymin_snap, ymax_snap, cell_size_m)

    cells = []
    for col_idx, x0 in enumerate(cols):
        for row_idx, y0 in enumerate(rows):
            cell = box(x0, y0, x0 + cell_size_m, y0 + cell_size_m)
            if cell.intersects(clip_union):
                cells.append({
                    "geometry": cell,
                    "col_id": col_idx,
                    "row_id": row_idx,
                    "cell_id": f"C{col_idx:03d}R{row_idx:03d}",
                    "cell_size_km": cell_size_m // 1000,
                    "area_km2": (cell_size_m / 1000) ** 2,
                })

    grid_gdf = gpd.GeoDataFrame(cells, crs=utm_crs)
    # Clip cells to the study boundary
    grid_gdf["geometry"] = grid_gdf.geometry.intersection(clip_union)
    grid_gdf = grid_gdf[~grid_gdf.geometry.is_empty].reset_index(drop=True)
    log.info("Grid created: %d cells at %d km resolution.", len(grid_gdf), cell_size_m // 1000)
    return grid_gdf


# ---------------------------------------------------------------------------
# Step 6: Export to GeoPackage
# ---------------------------------------------------------------------------
def export_gpkg(gdf: gpd.GeoDataFrame, path: Path, layer_name: str) -> None:
    """Save a GeoDataFrame to GeoPackage, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG", layer=layer_name)
    log.info("Saved: %s  (%d features)", path, len(gdf))


# ---------------------------------------------------------------------------
# Step 7: Study area overview map
# ---------------------------------------------------------------------------
def plot_study_area_map(
    boundary_gdf: gpd.GeoDataFrame,
    buffer_30_gdf: gpd.GeoDataFrame,
    buffer_50_gdf: gpd.GeoDataFrame,
    grid_10_gdf: gpd.GeoDataFrame,
    districts_gdf: gpd.GeoDataFrame | None,
    output_path: Path,
    utm_crs: str = "EPSG:32630",
) -> None:
    """
    Produce a publication-quality study area overview map.

    Parameters
    ----------
    boundary_gdf, buffer_30_gdf, buffer_50_gdf, grid_10_gdf : GeoDataFrames
    districts_gdf : GeoDataFrame or None
    output_path : Path
        PNG output path.
    utm_crs : str
    """
    log.info("Generating study area overview map ...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Convert all to UTM for plotting
    boundary_utm = boundary_gdf.to_crs(utm_crs)
    buf50_utm = buffer_50_gdf.to_crs(utm_crs)
    buf30_utm = buffer_30_gdf.to_crs(utm_crs)
    grid10_utm = grid_10_gdf.to_crs(utm_crs)

    # Plot layers
    buf50_utm.plot(ax=ax, color="#c6dbef", edgecolor="#4292c6",
                   linewidth=0.8, alpha=0.5, label="50 km coastal buffer")
    buf30_utm.plot(ax=ax, color="#9ecae1", edgecolor="#2171b5",
                   linewidth=1.0, alpha=0.6, label="30 km coastal buffer")

    if districts_gdf is not None and len(districts_gdf) > 0:
        districts_gdf.to_crs(utm_crs).plot(
            ax=ax, facecolor="none", edgecolor="#636363",
            linewidth=0.5, label="Coastal districts",
        )

    grid10_utm.plot(ax=ax, facecolor="none", edgecolor="#bdbdbd",
                    linewidth=0.3, alpha=0.6, label="10 km analysis grid")

    boundary_utm.plot(ax=ax, facecolor="none", edgecolor="#252525",
                      linewidth=1.5, label="Ghana national boundary")

    # Labels for key sub-areas (approximate UTM Zone 30N coordinates)
    sub_areas = {
        "Greater Accra\n/ Tema": (776_000, 617_000),
        "Weija-Densu": (744_000, 617_000),
        "Ada Coast": (842_000, 639_000),
        "Keta-Anloga": (886_000, 656_000),
        "Cape Coast": (649_000, 572_000),
        "Sekondi-\nTakoradi": (602_000, 546_000),
    }
    for label, (x, y) in sub_areas.items():
        ax.annotate(
            label, xy=(x, y), fontsize=7, ha="center",
            color="#252525",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.7),
        )

    # Formatting
    ax.set_title(
        "Study Area: Coastal Ghana — Groundwater, Subsidence, and Urban Growth\n"
        "30 km and 50 km coastal buffers with 10 km analysis grid",
        fontsize=11, pad=12,
    )
    ax.set_xlabel("Easting (m) — EPSG:32630", fontsize=9)
    ax.set_ylabel("Northing (m) — EPSG:32630", fontsize=9)
    ax.tick_params(labelsize=8)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize=8,
              framealpha=0.9, edgecolor="#bdbdbd")

    # Scale bar (approximate: 50 km)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    scale_x0 = xlims[0] + (xlims[1] - xlims[0]) * 0.65
    scale_y0 = ylims[0] + (ylims[1] - ylims[0]) * 0.04
    ax.plot(
        [scale_x0, scale_x0 + 50_000], [scale_y0, scale_y0],
        color="black", linewidth=2,
    )
    ax.text(
        scale_x0 + 25_000, scale_y0 + (ylims[1] - ylims[0]) * 0.01,
        "50 km", ha="center", va="bottom", fontsize=8,
    )

    # North arrow
    ax.annotate(
        "N", xy=(0.96, 0.90), xycoords="axes fraction",
        fontsize=14, fontweight="bold", ha="center",
    )
    ax.annotate(
        "", xy=(0.96, 0.96), xytext=(0.96, 0.90),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )

    # Data source note
    fig.text(
        0.5, 0.01,
        "Data sources: GADM v4.1 administrative boundaries | CRS: EPSG:32630 (WGS 84 / UTM Zone 30N)",
        ha="center", fontsize=7, color="#636363",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Study area map saved to: %s", output_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 65)
    log.info("Stage 1: Define Study Area — coastal Ghana")
    log.info("=" * 65)

    paths, sa_config = load_config()
    utm_crs = f"EPSG:{sa_config['crs']['epsg']}"
    buf_30_m = sa_config["coastal_buffers"]["primary_m"]
    buf_50_m = sa_config["coastal_buffers"]["extended_m"]
    grid_10_m = sa_config["analysis_grids"]["fine_m"]
    grid_25_m = sa_config["analysis_grids"]["coarse_m"]

    boundaries_dir = rp(paths["raw"]["boundaries"])
    boundaries_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Ghana boundary
    # ------------------------------------------------------------------
    try:
        ghana_gdf = load_ghana_boundary(boundaries_dir)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    export_gpkg(
        ghana_gdf.to_crs(utm_crs),
        boundaries_dir / "ghana_boundary.gpkg",
        layer_name="ghana_boundary",
    )

    # ------------------------------------------------------------------
    # 2. Extract coastline
    # ------------------------------------------------------------------
    coastline_gdf = extract_coastline(ghana_gdf)
    export_gpkg(
        coastline_gdf.to_crs(utm_crs),
        boundaries_dir / "ghana_coastline.gpkg",
        layer_name="ghana_coastline",
    )

    # ------------------------------------------------------------------
    # 3. Generate coastal buffers
    # ------------------------------------------------------------------
    buf_30_gdf, buf_50_gdf = generate_coastal_buffers(
        ghana_gdf, buf_30_m, buf_50_m, utm_crs
    )
    export_gpkg(
        buf_30_gdf,
        boundaries_dir / "study_area_coastal_30km.gpkg",
        layer_name="study_area_30km",
    )
    export_gpkg(
        buf_50_gdf,
        boundaries_dir / "study_area_coastal_50km.gpkg",
        layer_name="study_area_50km",
    )

    # ------------------------------------------------------------------
    # 4. Select coastal districts
    # ------------------------------------------------------------------
    coastal_districts = select_coastal_districts(boundaries_dir, buf_50_gdf, utm_crs)
    if len(coastal_districts) > 0:
        export_gpkg(
            coastal_districts,
            boundaries_dir / "coastal_districts.gpkg",
            layer_name="coastal_districts",
        )

    # ------------------------------------------------------------------
    # 5. Create analysis grids
    # ------------------------------------------------------------------
    grid_10_gdf = create_analysis_grid(buf_50_gdf, grid_10_m, utm_crs)
    export_gpkg(
        grid_10_gdf,
        boundaries_dir / "analysis_grid_10km.gpkg",
        layer_name="grid_10km",
    )

    grid_25_gdf = create_analysis_grid(buf_50_gdf, grid_25_m, utm_crs)
    export_gpkg(
        grid_25_gdf,
        boundaries_dir / "analysis_grid_25km.gpkg",
        layer_name="grid_25km",
    )

    # ------------------------------------------------------------------
    # 6. Study area map
    # ------------------------------------------------------------------
    map_output = rp(paths["outputs"]["maps"]) / "study_area_overview.png"
    plot_study_area_map(
        boundary_gdf=ghana_gdf,
        buffer_30_gdf=buf_30_gdf,
        buffer_50_gdf=buf_50_gdf,
        grid_10_gdf=grid_10_gdf,
        districts_gdf=coastal_districts if len(coastal_districts) > 0 else None,
        output_path=map_output,
        utm_crs=utm_crs,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=" * 65)
    log.info("Stage 1 complete. Outputs written to data/raw/boundaries/")
    log.info("  ghana_boundary.gpkg")
    log.info("  ghana_coastline.gpkg")
    log.info("  study_area_coastal_30km.gpkg")
    log.info("  study_area_coastal_50km.gpkg")
    if len(coastal_districts) > 0:
        log.info("  coastal_districts.gpkg  (%d districts)", len(coastal_districts))
    log.info("  analysis_grid_10km.gpkg  (%d cells)", len(grid_10_gdf))
    log.info("  analysis_grid_25km.gpkg  (%d cells)", len(grid_25_gdf))
    log.info("  outputs/maps/study_area_overview.png")
    log.info("=" * 65)
    log.info(
        "Next step: Stage 2 — build data inventory "
        "(notebooks/01_data_inventory.ipynb)"
    )


if __name__ == "__main__":
    main()
