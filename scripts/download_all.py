"""
scripts/download_all.py
========================
Master orchestrator: download all datasets for the Ghana groundwater-
subsidence project in the recommended order.

Usage
-----
    python scripts/download_all.py

    # Skip datasets already downloaded:
    python scripts/download_all.py --skip-grace --skip-gldas

    # Dry run (print what would be downloaded, no actual downloads):
    python scripts/download_all.py --dry-run

Prerequisites
-------------
  Run FIRST:  python scripts/setup_earthdata_credentials.py
  This configures ~/.netrc for NASA Earthdata authentication.

Download order and estimated sizes
-----------------------------------
  Step 1  GADM boundaries          ~25 MB    (open, fast)
  Step 2  Copernicus GLO-30 DEM    ~500 MB   (open, fast)
  Step 3  WorldPop population      ~300 MB   (open, fast)
  Step 4  Google Open Buildings    ~400 MB   (open, moderate)
  Step 5  CHIRPS rainfall          ~5 GB     (open, moderate — 43 years)
  Step 6  TerraClimate             ~8 GB     (open, slow — 5 vars × 43 years)
  Step 7  GRACE / GRACE-FO         ~200 MB   (NASA Earthdata, requires login)
  Step 8  GLDAS NOAH               ~6 GB     (NASA Earthdata, requires login)

  Total estimated: ~20 GB
  Estimated download time on a 50 Mbps connection: 1–3 hours

Skipping datasets
-----------------
Use individual flags to skip datasets already on disk or not needed:
  --skip-boundaries  --skip-dem  --skip-pop  --skip-buildings
  --skip-chirps  --skip-terraclimate  --skip-grace  --skip-gldas
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import paths, rp
from src.data_download import (
    authenticate_earthdata,
    download_chirps_monthly,
    download_copernicus_dem,
    download_gadm_ghana,
    download_gldas_noah,
    download_google_open_buildings_ghana,
    download_grace_mascon_bulk,
    download_terraclimate,
    download_worldpop_ghana,
    get_ghana_dem_tiles,
)
from src.utils import check_netrc, setup_logging

# Create logs directory before setting up file handler
(rp("outputs") / "logs").mkdir(parents=True, exist_ok=True)

log = setup_logging(
    log_file=rp("outputs") / "logs" / "download_all.log"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download all project datasets in recommended order."
    )
    p.add_argument("--skip-boundaries",    action="store_true")
    p.add_argument("--skip-dem",           action="store_true")
    p.add_argument("--skip-pop",           action="store_true")
    p.add_argument("--skip-buildings",     action="store_true")
    p.add_argument("--skip-chirps",        action="store_true")
    p.add_argument("--skip-terraclimate",  action="store_true")
    p.add_argument("--skip-grace",         action="store_true")
    p.add_argument("--skip-gldas",         action="store_true")
    p.add_argument(
        "--auth", default="netrc",
        choices=["netrc", "environment", "interactive"],
        help="NASA Earthdata authentication strategy.",
    )
    p.add_argument(
        "--grace-start", default="2002-04",
        help="GRACE download start (YYYY-MM).",
    )
    p.add_argument(
        "--grace-end", default="2024-12",
        help="GRACE download end (YYYY-MM).",
    )
    p.add_argument(
        "--chirps-start", type=int, default=1981,
        help="First year for CHIRPS (default: 1981).",
    )
    p.add_argument(
        "--chirps-end", type=int, default=2024,
        help="Last year for CHIRPS (default: 2024).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without downloading anything.",
    )
    return p.parse_args()


def _step(n: int, name: str, skip: bool, dry_run: bool) -> bool:
    """Log step header. Return True if step should run."""
    log.info("")
    log.info("=" * 65)
    if skip:
        log.info("STEP %d: %s  [SKIPPED]", n, name)
        log.info("=" * 65)
        return False
    if dry_run:
        log.info("STEP %d: %s  [DRY RUN — no download]", n, name)
        log.info("=" * 65)
        return False
    log.info("STEP %d: %s", n, name)
    log.info("=" * 65)
    return True


def main() -> None:
    args = parse_args()
    t_start = time.time()

    log.info("=" * 65)
    log.info("  Ghana Groundwater-Subsidence Project")
    log.info("  Master Data Download Script")
    log.info("=" * 65)

    if args.dry_run:
        log.info("  DRY RUN MODE — nothing will be downloaded.")

    # ------------------------------------------------------------------ 1
    if _step(1, "GADM Administrative Boundaries", args.skip_boundaries, args.dry_run):
        download_gadm_ghana(dest_dir=rp(paths["raw"]["boundaries"]))

    # ------------------------------------------------------------------ 2
    if _step(2, "Copernicus GLO-30 DEM", args.skip_dem, args.dry_run):
        download_copernicus_dem(
            dest_dir=rp(paths["raw"]["boundaries"]) / "dem",
            tiles=get_ghana_dem_tiles(),
        )

    # ------------------------------------------------------------------ 3
    if _step(3, "WorldPop 100m Population", args.skip_pop, args.dry_run):
        download_worldpop_ghana(
            dest_dir=rp(paths["raw"]["population"]),
            years=[2000, 2005, 2010, 2015, 2020],
        )

    # ------------------------------------------------------------------ 4
    if _step(4, "Google Open Buildings (Ghana)", args.skip_buildings, args.dry_run):
        download_google_open_buildings_ghana(
            dest_dir=rp(paths["raw"]["buildings"]),
        )

    # ------------------------------------------------------------------ 5
    if _step(5, "CHIRPS v2.0 Monthly Rainfall", args.skip_chirps, args.dry_run):
        years = list(range(args.chirps_start, args.chirps_end + 1))
        download_chirps_monthly(
            dest_dir=rp(paths["raw"]["chirps"]),
            years=years,
            africa_only=True,
        )

    # ------------------------------------------------------------------ 6
    if _step(6, "TerraClimate (PDSI + water balance)", args.skip_terraclimate, args.dry_run):
        years = list(range(args.chirps_start, args.chirps_end + 1))
        download_terraclimate(
            dest_dir=rp(paths["raw"]["terraclimate"]),
            variables=["PDSI", "ppt", "pet", "aet", "soil"],
            years=years,
        )

    # ------------------------------------------------------------------ 7 & 8 (NASA Earthdata)
    need_nasa = not (args.skip_grace and args.skip_gldas)
    if need_nasa and not args.dry_run:
        log.info("")
        log.info("Verifying NASA Earthdata credentials ...")
        if not check_netrc():
            log.error(
                "NASA Earthdata credentials not found in ~/.netrc.\n"
                "Run: python scripts/setup_earthdata_credentials.py"
            )
            log.warning("Skipping GRACE and GLDAS downloads.")
            args.skip_grace = True
            args.skip_gldas = True
        else:
            try:
                authenticate_earthdata(args.auth)
            except RuntimeError as exc:
                log.error("Authentication failed: %s", exc)
                args.skip_grace = True
                args.skip_gldas = True

    if _step(7, "GRACE / GRACE-FO Mascon Solutions", args.skip_grace, args.dry_run):
        log.info(
            "Note: GRACE mission gap (Jul 2017 – Jun 2018) will produce "
            "no files for that period — this is expected."
        )
        download_grace_mascon_bulk(
            dest_dir=rp(paths["raw"]["grace"]),
            temporal=(args.grace_start, args.grace_end),
            auth_strategy=args.auth,
        )

    if _step(8, "GLDAS NOAH 0.25° Monthly v2.1", args.skip_gldas, args.dry_run):
        log.info(
            "Note: GLDAS is ~6 GB. Ensure GES DISC app is approved:\n"
            "  https://urs.earthdata.nasa.gov/approve_app"
            "?client_id=e2WVk8Pw6weeLUKZYOxvTQ"
        )
        download_gldas_noah(
            dest_dir=rp(paths["raw"]["gldas"]),
            temporal=(args.grace_start, args.grace_end),
            auth_strategy=args.auth,
        )

    # ------------------------------------------------------------------ Done
    elapsed = time.time() - t_start
    log.info("")
    log.info("=" * 65)
    log.info("All download steps complete.  Elapsed: %.0f minutes.", elapsed / 60)
    log.info("")
    log.info("Suggested next steps:")
    log.info("  1. python scripts/prepare_study_area.py")
    log.info("     (generates study boundary and analysis grids)")
    log.info("")
    log.info("  2. Inspect downloaded files:")
    log.info("     ls data/raw/grace/")
    log.info("     ls data/raw/gldas/")
    log.info("     ls data/raw/chirps/")
    log.info("")
    log.info("  3. Open notebooks/03_grace_gldas_groundwater_anomaly.ipynb")
    log.info("     (Stage 5 — GWSA estimation)")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
