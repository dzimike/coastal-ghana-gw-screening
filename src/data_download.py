"""
src/data_download.py
====================
Core download functions for all Earth observation datasets used in the
Ghana groundwater-subsidence project.

This module wraps the earthaccess library (for NASA Earthdata datasets)
and direct HTTP downloads (for open datasets such as CHIRPS and GADM).

NASA Earthdata products covered
--------------------------------
  - GRACE / GRACE-FO RL06v3 Mascon (CSR)
  - GLDAS NOAH 0.25° Monthly v2.1

Open products covered
---------------------
  - CHIRPS v2.0 monthly rainfall
  - GADM v4.1 Ghana administrative boundaries
  - Copernicus GLO-30 DEM tiles
  - Google Open Buildings temporal dataset (Africa subset)
  - WorldPop 100m population (Ghana)

Usage
-----
Import functions directly or run the dedicated download scripts in scripts/.

Credentials
-----------
NASA Earthdata: store credentials in ~/.netrc
    machine urs.earthdata.nasa.gov
    login   YOUR_USERNAME
    password YOUR_PASSWORD

Run  python scripts/setup_earthdata_credentials.py  to set this up interactively.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from src.utils import (
    already_downloaded,
    download_file,
    ensure_dir,
    log_download_summary,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NASA Earthdata product identifiers
# ---------------------------------------------------------------------------
# JPL GRACE + GRACE-FO combined mascon (RL06.3, CRI-filtered, global grid).
# This is distributed as a single NetCDF covering 2002-04 to present.
GRACE_MASCON_SHORT_NAME = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"

GLDAS_SHORT_NAME  = "GLDAS_NOAH025_M"
GLDAS_VERSION     = "2.1"

# CHIRPS v2.0 — global monthly all-in-one NetCDF (~7.7 GB).
# The Africa-only subset is not distributed as NetCDF; per-month tifs are
# available separately but require ~540 downloads.  The global file is
# preferred for simplicity: clip to Ghana extent in preprocessing.
CHIRPS_GLOBAL_NC_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    "global_monthly/netcdf/chirps-v2.0.monthly.nc"
)
CHIRPS_GLOBAL_NC_FNAME = "chirps-v2.0.monthly.nc"

# Africa monthly tif.gz (one file per month) — alternative to global NC
CHIRPS_AFRICA_TIFS_BASE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    "africa_monthly/tifs/"
)
CHIRPS_AFRICA_TIF_PATTERN = "chirps-v2.0.{year}.{month:02d}.tif.gz"

# Copernicus GLO-30 DEM on AWS (requester-pays bucket — free for public)
# Alternative: Copernicus Data Space (requires registration)
COP_DEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

# Google Open Buildings — Africa temporal dataset (GCS public bucket)
GOB_BASE_URL = (
    "https://storage.googleapis.com/open-buildings-public/"
    "v3/temporal_countries/GH/"
)

# WorldPop — Ghana 100m population (total, unconstrained)
WORLDPOP_BASE = "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/"


# ---------------------------------------------------------------------------
# 1. NASA Earthdata: GRACE + GLDAS via earthaccess
# ---------------------------------------------------------------------------
def authenticate_earthdata(strategy: str = "netrc") -> None:
    """
    Authenticate with NASA Earthdata using earthaccess.

    Parameters
    ----------
    strategy : str
        Authentication strategy passed to ``earthaccess.login()``.
        Options: ``"netrc"`` (default), ``"environment"``, ``"interactive"``.
    """
    try:
        import earthaccess
    except ImportError:
        raise ImportError(
            "earthaccess is required for NASA Earthdata downloads.\n"
            "Install with: pip install earthaccess"
        )
    auth = earthaccess.login(strategy=strategy)
    if auth.authenticated:
        log.info("NASA Earthdata authentication successful.")
    else:
        raise RuntimeError(
            "NASA Earthdata authentication failed.\n"
            "Check your ~/.netrc credentials at urs.earthdata.nasa.gov"
        )


def download_grace_mascon(
    dest_dir: str | Path,
    temporal: tuple[str, str] = ("2002-04", "2023-12"),
    bounding_box: tuple[float, float, float, float] | None = None,
    auth_strategy: str = "netrc",
) -> list[Path]:
    """
    Download GRACE and GRACE-FO CSR RL06 mascon monthly solutions.

    Both the GRACE (pre-2017) and GRACE-FO (post-2018) products are
    downloaded automatically to cover the full record.

    Parameters
    ----------
    dest_dir : str or Path
        Destination directory (``data/raw/grace/``).
    temporal : tuple of str
        Start and end dates as "YYYY-MM" strings.
    bounding_box : (west, south, east, north) or None
        Optional spatial bounding box in decimal degrees (WGS 84).
        If None, the global product is downloaded (recommended for
        GRACE mascon which is already a global gridded product).
    auth_strategy : str
        earthaccess authentication strategy.

    Returns
    -------
    list of Path
        Paths to all downloaded files.
    """
    import earthaccess

    authenticate_earthdata(auth_strategy)
    dest_dir = ensure_dir(dest_dir)

    downloaded, skipped, failed = [], [], []
    start, end = temporal

    for short_name in [GRACE_SHORT_NAME, GRACE_FO_SHORT_NAME]:
        log.info("Searching for %s  (%s to %s) ...", short_name, start, end)
        try:
            kwargs = dict(
                short_name=short_name,
                temporal=(start, end),
                count=-1,
            )
            if bounding_box:
                kwargs["bounding_box"] = bounding_box

            results = earthaccess.search_data(**kwargs)
            log.info("Found %d granules for %s.", len(results), short_name)

            if not results:
                log.warning("No results for %s. Check product name or temporal range.", short_name)
                continue

            for granule in results:
                for link in earthaccess.results.DataGranule.data_links(granule, access="external"):
                    fname = Path(link).name
                    dest = dest_dir / fname
                    if already_downloaded(dest):
                        skipped.append(dest)
                        continue
                    try:
                        # earthaccess handles authentication automatically
                        paths = earthaccess.download([granule], local_path=str(dest_dir))
                        downloaded.extend([Path(p) for p in paths])
                        break   # one download per granule
                    except Exception as exc:
                        log.error("Failed to download granule %s: %s", fname, exc)
                        failed.append(link)

        except Exception as exc:
            log.error("Search failed for %s: %s", short_name, exc)
            failed.append(short_name)

    log_download_summary(downloaded, skipped, failed)
    return downloaded + skipped


def download_grace_mascon_bulk(
    dest_dir: str | Path,
    temporal: tuple[str, str] = ("2002-04", "2023-12"),
    auth_strategy: str = "netrc",
) -> list[Path]:
    """
    Download the JPL GRACE/GRACE-FO combined mascon product.

    The product ``TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4`` is distributed
    as a single global NetCDF file covering the full mission record
    (2002-04 to present).  The ``temporal`` parameter is accepted for
    API compatibility but the full file is always downloaded.

    Parameters
    ----------
    dest_dir : str or Path
        Destination directory (``data/raw/grace/``).
    temporal : tuple of str
        Unused for this product (full time series in one file).
    auth_strategy : str
        earthaccess auth strategy.

    Returns
    -------
    list of Path
        List containing the single downloaded file.
    """
    import earthaccess

    authenticate_earthdata(auth_strategy)
    dest_dir = ensure_dir(dest_dir)

    log.info("Searching %s ...", GRACE_MASCON_SHORT_NAME)
    try:
        results = earthaccess.search_data(
            short_name=GRACE_MASCON_SHORT_NAME,
            count=-1,
        )
        if not results:
            log.warning("No granules found for %s.", GRACE_MASCON_SHORT_NAME)
            return []

        log.info(
            "Downloading %d granule(s) for %s (~200–400 MB) ...",
            len(results), GRACE_MASCON_SHORT_NAME,
        )
        paths = earthaccess.download(results, local_path=str(dest_dir))
        completed = [Path(p) for p in paths if p]
        log.info("GRACE complete: %d file(s) in %s", len(completed), dest_dir)
        return completed

    except Exception as exc:
        log.error("Error downloading %s: %s", GRACE_MASCON_SHORT_NAME, exc)
        return []


def download_gldas_noah(
    dest_dir: str | Path,
    temporal: tuple[str, str] = ("2002-01", "2023-12"),
    auth_strategy: str = "netrc",
) -> list[Path]:
    """
    Download GLDAS NOAH 0.25° monthly v2.1 from NASA GES DISC.

    Parameters
    ----------
    dest_dir : str or Path
        Destination directory (``data/raw/gldas/``).
    temporal : tuple of str
        (start, end) as "YYYY-MM" strings.
    auth_strategy : str
        earthaccess auth strategy.

    Returns
    -------
    list of Path
    """
    import earthaccess

    authenticate_earthdata(auth_strategy)
    dest_dir = ensure_dir(dest_dir)

    start, end = temporal
    log.info("Searching GLDAS NOAH025_M v%s  (%s to %s) ...",
             GLDAS_VERSION, start, end)

    try:
        results = earthaccess.search_data(
            short_name=GLDAS_SHORT_NAME,
            version=GLDAS_VERSION,
            temporal=(start, end),
            count=-1,
        )
        if not results:
            log.warning("No GLDAS granules found. Check product name and temporal range.")
            return []

        log.info("Downloading %d GLDAS granules ...", len(results))
        paths = earthaccess.download(results, local_path=str(dest_dir))
        completed = [Path(p) for p in paths if p]
        log.info("GLDAS download complete: %d files.", len(completed))
        return completed

    except Exception as exc:
        log.error("GLDAS download failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# 2. CHIRPS v2.0 monthly (open, no authentication)
# ---------------------------------------------------------------------------
def download_chirps_monthly(
    dest_dir: str | Path,
    years: list[int] | None = None,
    africa_only: bool = True,
) -> list[Path]:
    """
    Download CHIRPS v2.0 monthly precipitation data.

    The UCSB server distributes the Africa subset as per-month ``.tif.gz``
    files only — not as per-year NetCDF.  This function therefore downloads
    the single global monthly NetCDF (``chirps-v2.0.monthly.nc``, ~7.7 GB)
    which covers 1981 to present and all regions.  Clip to Ghana extent
    during preprocessing.

    The ``years`` and ``africa_only`` parameters are accepted for API
    compatibility but are not used: the full global file is always
    downloaded in one request.

    Parameters
    ----------
    dest_dir : str or Path
        Destination directory (``data/raw/chirps/``).
    years : list of int or None
        Not used for the global NetCDF download.
    africa_only : bool
        Not used; the global file is always downloaded.

    Returns
    -------
    list of Path
        List containing the single downloaded file.
    """
    dest_dir = ensure_dir(dest_dir)
    dest = dest_dir / CHIRPS_GLOBAL_NC_FNAME

    if already_downloaded(dest):
        log.info("CHIRPS global monthly NC already present: %s", dest)
        return [dest]

    log.info(
        "Downloading CHIRPS v2.0 global monthly NetCDF (~7.7 GB) ..."
    )
    log.info("  URL: %s", CHIRPS_GLOBAL_NC_URL)
    log.info("  Destination: %s", dest)
    log.info("  This is a large file — expect 20–60 min on typical connections.")

    try:
        download_file(CHIRPS_GLOBAL_NC_URL, dest)
        log_download_summary([dest], [], [])
        return [dest]
    except RuntimeError as exc:
        log.error("CHIRPS global download failed: %s", exc)
        log_download_summary([], [], [CHIRPS_GLOBAL_NC_URL])
        return []


def download_chirps_africa_tifs(
    dest_dir: str | Path,
    years: list[int] | None = None,
    months: list[int] | None = None,
) -> list[Path]:
    """
    Download CHIRPS v2.0 Africa monthly tif.gz files (one per month).

    Use this as an alternative to the global NetCDF when only a subset
    of years is needed, or when bandwidth is limited.

    Parameters
    ----------
    dest_dir : str or Path
        Destination directory (``data/raw/chirps/africa_tifs/``).
    years : list of int or None
        Years to download.  Default: 2002–2025 (aligned with GRACE).
    months : list of int or None
        Months (1–12) to download.  Default: all 12 months.

    Returns
    -------
    list of Path
    """
    import datetime

    dest_dir = ensure_dir(dest_dir)

    if years is None:
        years = list(range(2002, datetime.date.today().year + 1))
    if months is None:
        months = list(range(1, 13))

    downloaded, skipped, failed = [], [], []

    for year in years:
        for month in months:
            fname = CHIRPS_AFRICA_TIF_PATTERN.format(year=year, month=month)
            url   = CHIRPS_AFRICA_TIFS_BASE + fname
            dest  = dest_dir / fname

            if already_downloaded(dest):
                skipped.append(dest)
                continue
            try:
                download_file(url, dest)
                downloaded.append(dest)
            except RuntimeError as exc:
                log.error("CHIRPS tif %d-%02d failed: %s", year, month, exc)
                failed.append(url)

    log_download_summary(downloaded, skipped, failed)
    return downloaded + skipped


# ---------------------------------------------------------------------------
# 3. GADM administrative boundaries
# ---------------------------------------------------------------------------
def download_gadm_ghana(
    dest_dir: str | Path,
    version: str = "4.1",
) -> Path:
    """
    Download GADM Ghana GeoPackage (all administrative levels, 0–3).

    Parameters
    ----------
    dest_dir : str or Path
        Destination directory (``data/raw/boundaries/``).
    version : str
        GADM version string (default "4.1").

    Returns
    -------
    Path to the downloaded GeoPackage.
    """
    dest_dir = ensure_dir(dest_dir)

    # GADM direct download URL
    version_str = version.replace(".", "_")
    url  = f"https://geodata.ucdavis.edu/gadm/gadm{version}/gpkg/gadm{version_str}_GHA.gpkg"
    dest = dest_dir / f"gadm{version_str}_GHA.gpkg"

    if already_downloaded(dest):
        log.info("GADM Ghana already downloaded: %s", dest)
        return dest

    log.info("Downloading GADM Ghana v%s ...", version)
    try:
        download_file(url, dest)
    except RuntimeError:
        # Try alternative URL format
        alt_url = f"https://geodata.ucdavis.edu/gadm/gadm{version}/gpkg/gadm{version_str}_GHA.gpkg"
        log.warning("Primary URL failed, trying alternative: %s", alt_url)
        download_file(alt_url, dest)

    return dest


# ---------------------------------------------------------------------------
# 4. Copernicus GLO-30 DEM tiles for Ghana
# ---------------------------------------------------------------------------
def get_ghana_dem_tiles() -> list[dict]:
    """
    Return the list of Copernicus GLO-30 DEM tile names covering Ghana.

    Ghana bounding box (approx): W -3.25, E 1.20, S 4.50, N 11.20
    GLO-30 tiles are named: Copernicus_DSM_COG_10_N{lat}_{lon}_00_DEM.tif

    Returns
    -------
    list of dicts with keys: tile_name, lat, lon, url
    """
    tiles = []
    # Latitudes 4–11 N, Longitudes 4W–2E (1° tiles)
    # Correct Copernicus GLO-30 naming: N{lat}_00_{lon}_00
    for lat in range(4, 12):        # N04 to N11
        for lon in range(-4, 2):    # W004 to W001 + E000, E001
            lat_str = f"N{lat:02d}"
            lon_str = f"W{abs(lon):03d}" if lon < 0 else f"E{lon:03d}"
            tile_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00"
            url = (
                f"https://copernicus-dem-30m.s3.amazonaws.com/"
                f"{tile_name}_DEM/{tile_name}_DEM.tif"
            )
            tiles.append({
                "tile_name": tile_name,
                "lat": lat,
                "lon": lon,
                "url": url,
                "filename": f"{tile_name}_DEM.tif",
            })
    return tiles


def download_copernicus_dem(
    dest_dir: str | Path,
    tiles: list[dict] | None = None,
) -> list[Path]:
    """
    Download Copernicus GLO-30 DEM tiles covering Ghana.

    The DEM tiles are hosted as public GeoTIFFs on AWS S3 (no authentication
    required for the public endpoint).

    Parameters
    ----------
    dest_dir : str or Path
        Destination directory (``data/raw/boundaries/dem/``).
    tiles : list of dict or None
        List of tile dicts from ``get_ghana_dem_tiles()``.
        If None, all Ghana tiles are downloaded.

    Returns
    -------
    list of Path
    """
    dest_dir = ensure_dir(dest_dir)

    if tiles is None:
        tiles = get_ghana_dem_tiles()

    log.info("Downloading %d Copernicus GLO-30 DEM tiles ...", len(tiles))
    downloaded, skipped, failed = [], [], []

    for tile in tiles:
        dest = dest_dir / tile["filename"]
        if already_downloaded(dest):
            skipped.append(dest)
            continue
        try:
            download_file(tile["url"], dest)
            downloaded.append(dest)
        except RuntimeError as exc:
            log.warning("DEM tile %s not available: %s", tile["tile_name"], exc)
            failed.append(tile["url"])

    log_download_summary(downloaded, skipped, failed)
    return downloaded + skipped


# ---------------------------------------------------------------------------
# 5. WorldPop 100m Ghana population
# ---------------------------------------------------------------------------
def download_worldpop_ghana(
    dest_dir: str | Path,
    years: list[int] | None = None,
    resolution: str = "100m",
) -> list[Path]:
    """
    Download WorldPop unconstrained total population for Ghana at 100m.

    Parameters
    ----------
    dest_dir : str or Path
        Destination directory (``data/raw/population/``).
    years : list of int or None
        Years to download (default: [2000, 2005, 2010, 2015, 2020]).
    resolution : str
        Grid resolution string (default "100m"; only 100m available globally).

    Returns
    -------
    list of Path
    """
    dest_dir = ensure_dir(dest_dir)
    if years is None:
        years = [2000, 2005, 2010, 2015, 2020]

    # WorldPop direct download URL pattern
    base = "https://data.worldpop.org/GIS/Population/Global_2000_2020/"
    downloaded, skipped, failed = [], [], []

    for year in years:
        fname = f"gha_ppp_{year}.tif"
        url   = f"{base}{year}/GHA/{fname}"
        dest  = dest_dir / fname
        if already_downloaded(dest):
            skipped.append(dest)
            continue
        try:
            download_file(url, dest)
            downloaded.append(dest)
        except RuntimeError as exc:
            log.error("WorldPop %d download failed: %s", year, exc)
            failed.append(url)

    log_download_summary(downloaded, skipped, failed)
    return downloaded + skipped


# ---------------------------------------------------------------------------
# 6. Google Open Buildings — Ghana temporal subset
# ---------------------------------------------------------------------------
def download_google_open_buildings_ghana(
    dest_dir: str | Path,
) -> list[Path]:
    """
    Download the Google Open Buildings temporal dataset for Ghana.

    The dataset is hosted as GCS-public Parquet files partitioned by
    country. Ghana files are downloaded directly.

    Parameters
    ----------
    dest_dir : str or Path
        Destination directory (``data/raw/buildings/``).

    Returns
    -------
    list of Path

    Notes
    -----
    Alternative: use the Google Open Buildings API or GEE to extract
    Ghana-specific data with a confidence threshold filter.
    """
    dest_dir = ensure_dir(dest_dir)

    # Ghana temporal building files (v3)
    urls = [
        "https://storage.googleapis.com/open-buildings-public/v3/temporal_countries/GH/GH.csv.gz",
        # Parquet version (preferred for large datasets)
        "https://storage.googleapis.com/open-buildings-public/v3/temporal_countries/GH/GH.parquet",
    ]

    downloaded, skipped, failed = [], [], []
    for url in urls:
        fname = Path(url).name
        dest  = dest_dir / fname
        if already_downloaded(dest):
            skipped.append(dest)
            continue
        try:
            download_file(url, dest)
            downloaded.append(dest)
        except RuntimeError as exc:
            log.warning("GOB download failed (%s): %s", fname, exc)
            failed.append(url)

    log_download_summary(downloaded, skipped, failed)
    return downloaded + skipped


# ---------------------------------------------------------------------------
# 7. TerraClimate (NetCDF per variable per year)
# ---------------------------------------------------------------------------
TERRACLIMATE_VARS = ["PDSI", "ppt", "pet", "aet", "soil", "tmax", "tmin"]
TERRACLIMATE_BASE = "https://climate.northwestknowledge.net/TERRACLIMATE-DATA/"


def download_terraclimate(
    dest_dir: str | Path,
    variables: list[str] | None = None,
    years: list[int] | None = None,
) -> list[Path]:
    """
    Download TerraClimate NetCDF files (one per variable per year).

    Parameters
    ----------
    dest_dir : str or Path
        Destination directory (``data/raw/terraclimate/``).
    variables : list of str or None
        TerraClimate variable names. Default: PDSI, ppt, pet, aet, soil.
    years : list of int or None
        Years to download. Default: 1981 to 2023.

    Returns
    -------
    list of Path
    """
    import datetime

    dest_dir = ensure_dir(dest_dir)
    if variables is None:
        variables = ["PDSI", "ppt", "pet", "aet", "soil"]
    if years is None:
        years = list(range(1981, datetime.date.today().year))

    downloaded, skipped, failed = [], [], []

    for var in variables:
        for year in years:
            fname = f"TerraClimate_{var}_{year}.nc"
            url   = TERRACLIMATE_BASE + fname
            dest  = dest_dir / fname
            if already_downloaded(dest):
                skipped.append(dest)
                continue
            try:
                download_file(url, dest)
                downloaded.append(dest)
            except RuntimeError as exc:
                log.error("TerraClimate %s %d failed: %s", var, year, exc)
                failed.append(url)

    log_download_summary(downloaded, skipped, failed)
    return downloaded + skipped
