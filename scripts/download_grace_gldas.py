"""
scripts/download_grace_gldas.py
================================
Download GRACE / GRACE-FO mascon solutions and GLDAS NOAH monthly data
from NASA Earthdata for the Ghana groundwater-subsidence project.

Usage
-----
    python scripts/download_grace_gldas.py [--grace-only | --gldas-only]

Prerequisites
-------------
1. NASA Earthdata account: https://urs.earthdata.nasa.gov/users/new
2. Credentials in ~/.netrc  (run: python scripts/setup_earthdata_credentials.py)
3. GES DISC app approved in Earthdata profile (required for GLDAS):
   https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ

Outputs
-------
  data/raw/grace/        — GRACE + GRACE-FO monthly NetCDF files
  data/raw/gldas/        — GLDAS NOAH monthly NetCDF files

Notes
-----
- GRACE mission gap: July 2017 – June 2018 (no data; handled in processing).
- GLDAS v2.1 covers 2000-01 to present. Download from 2002-01 to align
  with GRACE start, or from 2000-01 for longer soil moisture baseline.
- Estimated download sizes:
    GRACE (2002–2023):  ~100–300 MB
    GLDAS (2002–2023):  ~3–8 GB (monthly NetCDF4 files)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import paths, rp, study_area
from src.data_download import (
    authenticate_earthdata,
    download_gldas_noah,
    download_grace_mascon_bulk,
)
from src.utils import setup_logging

log = setup_logging(
    log_file=rp("outputs") / "logs" / "download_grace_gldas.log"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GRACE/GRACE-FO mascon and GLDAS NOAH data."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--grace-only", action="store_true",
        help="Download only GRACE / GRACE-FO mascon files.",
    )
    group.add_argument(
        "--gldas-only", action="store_true",
        help="Download only GLDAS NOAH monthly files.",
    )
    parser.add_argument(
        "--grace-start", default="2002-04",
        help="GRACE download start month (YYYY-MM, default: 2002-04).",
    )
    parser.add_argument(
        "--grace-end", default="2023-12",
        help="GRACE download end month (YYYY-MM, default: 2023-12).",
    )
    parser.add_argument(
        "--gldas-start", default="2002-01",
        help="GLDAS download start month (YYYY-MM, default: 2002-01).",
    )
    parser.add_argument(
        "--gldas-end", default="2023-12",
        help="GLDAS download end month (YYYY-MM, default: 2023-12).",
    )
    parser.add_argument(
        "--auth", default="netrc",
        choices=["netrc", "environment", "interactive"],
        help="Earthdata authentication strategy (default: netrc).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 65)
    log.info("GRACE / GLDAS Download Script")
    log.info("=" * 65)

    # Resolve output directories from config
    grace_dir = rp(paths["raw"]["grace"])
    gldas_dir = rp(paths["raw"]["gldas"])

    # Verify authentication before starting
    log.info("Verifying NASA Earthdata authentication ...")
    try:
        authenticate_earthdata(args.auth)
    except RuntimeError as exc:
        log.error("%s", exc)
        log.error(
            "Run  python scripts/setup_earthdata_credentials.py  "
            "to configure credentials."
        )
        sys.exit(1)

    # --- GRACE / GRACE-FO ---
    if not args.gldas_only:
        log.info("-" * 50)
        log.info("Downloading GRACE / GRACE-FO mascon solutions ...")
        log.info("  Destination : %s", grace_dir)
        log.info("  Period      : %s to %s", args.grace_start, args.grace_end)
        log.info("")
        log.info(
            "  NOTE: GRACE mission gap (Jul 2017 – Jun 2018) will produce "
            "no granules for that period. This is expected."
        )
        log.info("-" * 50)

        grace_files = download_grace_mascon_bulk(
            dest_dir=grace_dir,
            temporal=(args.grace_start, args.grace_end),
            auth_strategy=args.auth,
        )
        log.info("GRACE complete: %d files in %s", len(grace_files), grace_dir)

    # --- GLDAS NOAH ---
    if not args.grace_only:
        log.info("-" * 50)
        log.info("Downloading GLDAS NOAH 0.25° Monthly v2.1 ...")
        log.info("  Destination : %s", gldas_dir)
        log.info("  Period      : %s to %s", args.gldas_start, args.gldas_end)
        log.info("")
        log.info(
            "  NOTE: GLDAS files are ~3–8 GB total. This may take "
            "30–90 minutes depending on your connection speed."
        )
        log.info("")
        log.info(
            "  IMPORTANT: GES DISC application must be approved in your "
            "Earthdata profile before GLDAS downloads will succeed.\n"
            "  → https://urs.earthdata.nasa.gov/approve_app"
            "?client_id=e2WVk8Pw6weeLUKZYOxvTQ"
        )
        log.info("-" * 50)

        gldas_files = download_gldas_noah(
            dest_dir=gldas_dir,
            temporal=(args.gldas_start, args.gldas_end),
            auth_strategy=args.auth,
        )
        log.info("GLDAS complete: %d files in %s", len(gldas_files), gldas_dir)

    log.info("=" * 65)
    log.info("Download script finished.")
    log.info("Next step: python scripts/download_chirps.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
