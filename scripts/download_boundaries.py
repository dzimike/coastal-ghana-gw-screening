"""
scripts/download_boundaries.py
================================
Download ancillary boundary, elevation, population, and building footprint
datasets for the Ghana groundwater-subsidence project.

All datasets in this script are open access — no authentication required.

Usage
-----
    python scripts/download_boundaries.py              # download everything
    python scripts/download_boundaries.py --gadm-only
    python scripts/download_boundaries.py --dem-only
    python scripts/download_boundaries.py --pop-only
    python scripts/download_boundaries.py --buildings-only

Outputs
-------
  data/raw/boundaries/           — GADM Ghana GeoPackage
  data/raw/boundaries/dem/       — Copernicus GLO-30 DEM tiles (~21 tiles)
  data/raw/population/           — WorldPop Ghana 100m GeoTIFFs
  data/raw/buildings/            — Google Open Buildings Ghana Parquet/CSV

After downloading, run:
    python scripts/prepare_study_area.py   (Stage 1 — requires GADM)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import paths, rp
from src.data_download import (
    download_copernicus_dem,
    download_gadm_ghana,
    download_google_open_buildings_ghana,
    download_worldpop_ghana,
    get_ghana_dem_tiles,
)
from src.utils import setup_logging

log = setup_logging(
    log_file=rp("outputs") / "logs" / "download_boundaries.log"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ancillary datasets (GADM, DEM, WorldPop, buildings)."
    )
    parser.add_argument("--gadm-only",      action="store_true")
    parser.add_argument("--dem-only",       action="store_true")
    parser.add_argument("--pop-only",       action="store_true")
    parser.add_argument("--buildings-only", action="store_true")
    parser.add_argument(
        "--gadm-version", default="4.1",
        help="GADM version string (default: 4.1).",
    )
    parser.add_argument(
        "--pop-years", nargs="+", type=int,
        default=[2000, 2005, 2010, 2015, 2020],
        help="WorldPop years to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine which products to download
    download_all = not any([
        args.gadm_only, args.dem_only,
        args.pop_only,  args.buildings_only,
    ])
    do_gadm      = download_all or args.gadm_only
    do_dem       = download_all or args.dem_only
    do_pop       = download_all or args.pop_only
    do_buildings = download_all or args.buildings_only

    log.info("=" * 65)
    log.info("Ancillary Data Download Script")
    log.info("  No authentication required for any dataset.")
    log.info("=" * 65)

    # ------------------------------------------------------------------
    # 1. GADM Ghana administrative boundaries
    # ------------------------------------------------------------------
    if do_gadm:
        log.info("-" * 50)
        log.info("Downloading GADM Ghana v%s ...", args.gadm_version)
        log.info(
            "  Contains all administrative levels (0=national, 1=region, "
            "2=district, 3=sub-district) in a single GeoPackage (~25 MB)."
        )
        gadm_file = download_gadm_ghana(
            dest_dir=rp(paths["raw"]["boundaries"]),
            version=args.gadm_version,
        )
        log.info("GADM saved: %s", gadm_file)
        log.info("")
        log.info(
            "  ✓  You can now run:  python scripts/prepare_study_area.py"
        )

    # ------------------------------------------------------------------
    # 2. Copernicus GLO-30 DEM tiles
    # ------------------------------------------------------------------
    if do_dem:
        log.info("-" * 50)
        tiles = get_ghana_dem_tiles()
        log.info(
            "Downloading %d Copernicus GLO-30 DEM tiles for Ghana ...", len(tiles)
        )
        log.info(
            "  Coverage: N04–N11, W004–E001 (30m resolution, ~500 MB total)"
        )
        log.info(
            "  Note: Some ocean/sea tiles may return 404 and will be skipped."
        )
        dem_files = download_copernicus_dem(
            dest_dir=rp(paths["raw"]["boundaries"]) / "dem",
            tiles=tiles,
        )
        log.info("DEM download complete: %d tiles.", len(dem_files))

    # ------------------------------------------------------------------
    # 3. WorldPop 100m population Ghana
    # ------------------------------------------------------------------
    if do_pop:
        log.info("-" * 50)
        log.info("Downloading WorldPop Ghana 100m population ...")
        log.info("  Years: %s", args.pop_years)
        log.info("  Each file ~40–80 MB.")
        pop_files = download_worldpop_ghana(
            dest_dir=rp(paths["raw"]["population"]),
            years=args.pop_years,
        )
        log.info("WorldPop complete: %d files.", len(pop_files))

    # ------------------------------------------------------------------
    # 4. Google Open Buildings (Ghana temporal)
    # ------------------------------------------------------------------
    if do_buildings:
        log.info("-" * 50)
        log.info("Downloading Google Open Buildings (Ghana temporal dataset) ...")
        log.info(
            "  The Ghana Parquet file contains building footprints with "
            "confidence scores and temporal estimates."
        )
        log.info("  Expected size: ~200–500 MB")
        gob_files = download_google_open_buildings_ghana(
            dest_dir=rp(paths["raw"]["buildings"]),
        )
        log.info("Google Open Buildings complete: %d files.", len(gob_files))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=" * 65)
    log.info("Ancillary download script finished.")
    log.info("")
    log.info("Suggested next steps:")
    if do_gadm:
        log.info("  1. python scripts/prepare_study_area.py")
    log.info("  2. python scripts/download_grace_gldas.py")
    log.info("  3. python scripts/download_chirps.py --terraclimate")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
