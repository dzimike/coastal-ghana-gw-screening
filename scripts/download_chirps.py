"""
scripts/download_chirps.py
===========================
Download CHIRPS v2.0 monthly precipitation NetCDF files for the
Ghana groundwater-subsidence project.

CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
is open access — no account or authentication required.

Usage
-----
    python scripts/download_chirps.py
    python scripts/download_chirps.py --start-year 2000 --end-year 2023
    python scripts/download_chirps.py --global       # download global product

Outputs
-------
  data/raw/chirps/  — Annual NetCDF files, one per year
                       e.g. chirps-v2.0.2020.months.nc

Notes
-----
- Each file contains 12 monthly layers (~100–150 MB per year for Africa).
- The Africa subset is downloaded by default (--global flag for world).
- Data available from 1981 to approximately 1–2 months before present.
- TerraClimate PDSI is downloaded by the same script family
  (see download_boundaries.py --terraclimate flag).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import paths, rp
from src.data_download import download_chirps_monthly, download_terraclimate
from src.utils import setup_logging

log = setup_logging(
    log_file=rp("outputs") / "logs" / "download_chirps.log"
)


def parse_args() -> argparse.Namespace:
    current_year = datetime.date.today().year
    parser = argparse.ArgumentParser(
        description="Download CHIRPS v2.0 monthly rainfall and (optionally) TerraClimate."
    )
    parser.add_argument(
        "--start-year", type=int, default=1981,
        help="First year to download (default: 1981).",
    )
    parser.add_argument(
        "--end-year", type=int, default=current_year - 1,
        help=f"Last year to download (default: {current_year - 1}).",
    )
    parser.add_argument(
        "--global", dest="global_product", action="store_true",
        help="Download the global CHIRPS product instead of Africa only.",
    )
    parser.add_argument(
        "--terraclimate", action="store_true",
        help="Also download TerraClimate PDSI, ppt, pet, aet, and soil variables.",
    )
    parser.add_argument(
        "--tc-vars", nargs="+",
        default=["PDSI", "ppt", "pet", "aet", "soil"],
        help="TerraClimate variables to download (used with --terraclimate).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = list(range(args.start_year, args.end_year + 1))

    log.info("=" * 65)
    log.info("CHIRPS v2.0 Download Script")
    log.info("=" * 65)
    log.info("  Years      : %d – %d (%d files)", args.start_year, args.end_year, len(years))
    log.info("  Product    : %s", "Global" if args.global_product else "Africa only")
    log.info("  Destination: %s", rp(paths["raw"]["chirps"]))
    log.info(
        "  Est. size  : ~%.0f MB",
        len(years) * (200 if args.global_product else 130),
    )
    log.info("  No authentication required.")
    log.info("-" * 65)

    chirps_files = download_chirps_monthly(
        dest_dir=rp(paths["raw"]["chirps"]),
        years=years,
        africa_only=not args.global_product,
    )
    log.info(
        "CHIRPS complete: %d files in %s",
        len(chirps_files), rp(paths["raw"]["chirps"]),
    )

    # Optional TerraClimate
    if args.terraclimate:
        log.info("-" * 65)
        log.info("Downloading TerraClimate variables: %s", args.tc_vars)
        log.info(
            "  NOTE: TerraClimate files are ~30–80 MB each. "
            "Downloading %d variable(s) × %d year(s) = %d files.",
            len(args.tc_vars), len(years), len(args.tc_vars) * len(years),
        )
        tc_files = download_terraclimate(
            dest_dir=rp(paths["raw"]["terraclimate"]),
            variables=args.tc_vars,
            years=years,
        )
        log.info(
            "TerraClimate complete: %d files in %s",
            len(tc_files), rp(paths["raw"]["terraclimate"]),
        )

    log.info("=" * 65)
    log.info("Download complete.")
    log.info("Next step: python scripts/download_boundaries.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
