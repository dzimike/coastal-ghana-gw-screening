#!/usr/bin/env python3
"""
ERA5 Tropospheric Correction for MintPy SBAS Time-Series.

Downloads ERA5 GRIB files for all 196 acquisition dates (2016–2024,
Sentinel-1 track 147, ascending), runs MintPy correct_troposphere step,
recomputes velocity, and re-exports the corrected GeoTIFFs.

Usage
-----
  # 1. First, set up CDS API credentials (see below).
  # 2. Then run:
  conda activate accra-risk
  python scripts/run_era5_correction.py

CDS API Credentials
-------------------
  Register at https://cds.climate.copernicus.eu/ (free account).
  After logging in, go to: https://cds.climate.copernicus.eu/how-to-api
  Copy your Personal Access Token and create ~/.cdsapirc:

    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>

  Save the file, then re-run this script.
  You must also accept the ERA5 licence:
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels

Notes
-----
- Download hour: 18 UTC (Sentinel-1 track 147 ascending overpass ≈ 18:17 UTC).
- ERA5 GRIB files are downloaded once and cached in WEATHER_DIR/ERA5/.
- ERA5 files for 2024 acquisitions near Dec 2024 may not yet be available
  (ERA5 has a ~5-day latency); those dates are skipped automatically by MintPy.
- ERA5 corrects stratified tropospheric delay but does NOT recover decorrelated
  pixels. Coastal study zone coherence issues remain; the correction improves
  velocity estimates in the coherent inland savanna (7.1–7.4°N).
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MINTPY_DIR   = PROJECT_ROOT / 'data/processed/subsidence/mintpy'
WEATHER_DIR  = MINTPY_DIR / 'WEATHER'
ERA5_DIR     = WEATHER_DIR / 'ERA5'
SUBDIR       = PROJECT_ROOT / 'data/processed/subsidence'
CFG_FILE     = MINTPY_DIR / 'smallbaselineApp.cfg'
CONDA_ENV    = 'accra-risk'

# GeoTIFF outputs
VEL_UTM_TMP  = SUBDIR / '_mintpy_velocity_utm_tmp.tif'
VEL_LOS      = SUBDIR / 'mintpy_velocity_los.tif'
VEL_VERT     = SUBDIR / 'mintpy_velocity_vert.tif'
VEL_STD      = SUBDIR / 'mintpy_velocity_std.tif'


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def run(cmd: list[str], env: dict | None = None, cwd: Path | None = None) -> None:
    """Run a command, printing it first; raise on failure."""
    print(f'\n>>> {" ".join(str(c) for c in cmd)}')
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(cmd, env=merged_env, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f'Command failed with exit code {result.returncode}.')


def conda_python() -> str:
    """Return path to the accra-risk Python interpreter."""
    base = Path(os.environ.get('CONDA_PREFIX', '/opt/miniconda3'))
    # Walk up to base if we're inside an env
    envs_candidate = base / 'envs' / CONDA_ENV / 'bin' / 'python3'
    if envs_candidate.exists():
        return str(envs_candidate)
    # Try common install paths
    for prefix in [Path('/opt/miniconda3'), Path.home() / 'miniconda3',
                   Path.home() / 'anaconda3', Path('/opt/anaconda3')]:
        p = prefix / 'envs' / CONDA_ENV / 'bin' / 'python3'
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f'Cannot find Python for conda env "{CONDA_ENV}". '
        'Activate the environment and re-run this script.'
    )


def conda_bin(tool: str) -> str:
    """Return path to a binary in the accra-risk env bin directory."""
    py = Path(conda_python())
    candidate = py.parent / tool
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f'Cannot find {tool} in {py.parent}')


# ---------------------------------------------------------------------------
# Step 0: Pre-flight checks
# ---------------------------------------------------------------------------
def check_cds_credentials() -> None:
    """Verify CDS API credentials exist and contain a non-placeholder key."""
    rc = Path.home() / '.cdsapirc'
    if not rc.exists():
        sys.exit(
            '\n[ERROR] ~/.cdsapirc not found.\n'
            'Create it with:\n\n'
            '  url: https://cds.climate.copernicus.eu/api\n'
            '  key: <your-personal-access-token>\n\n'
            'Get your token at: https://cds.climate.copernicus.eu/how-to-api\n'
            'You must also accept the ERA5 licence:\n'
            '  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels\n'
        )
    content = rc.read_text()
    if 'your-personal-access-token' in content or 'key:' not in content:
        sys.exit(
            '\n[ERROR] ~/.cdsapirc contains a placeholder key.\n'
            'Replace <your-personal-access-token> with your actual CDS token.\n'
            'Get it at: https://cds.climate.copernicus.eu/how-to-api\n'
        )
    print('[OK] ~/.cdsapirc found and appears configured.')


def check_era5_licence() -> None:
    """Remind user to accept ERA5 licence (cannot check programmatically)."""
    print(
        '\n[INFO] Make sure you have accepted the ERA5 licence:\n'
        '       https://cds.climate.copernicus.eu/datasets/'
        'reanalysis-era5-pressure-levels\n'
        '       (Required once; log in and click "Accept Terms").\n'
    )


def check_dependencies() -> None:
    """Verify pyaps3, cdsapi, and pygrib are importable."""
    py = conda_python()
    for pkg in ['pyaps3', 'cdsapi', 'pygrib', 'mintpy', 'h5py', 'rasterio']:
        result = subprocess.run(
            [py, '-c', f'import {pkg}; print("{pkg}:", {pkg}.__version__'
             f' if hasattr({pkg}, "__version__") else "ok")'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f'[WARN] {pkg} import failed: {result.stderr.strip()}')
        else:
            print(f'[OK] {result.stdout.strip()}')


# ---------------------------------------------------------------------------
# Step 1: Pre-download ERA5 GRIB files via pyaps3
# ---------------------------------------------------------------------------
ERA5_SNWE = (0, 10, -10, 10)  # MintPy-computed SNWE (rounded to 10° multiples with 2° buffer)
# NB: MintPy's get_snwe() rounds the scene extent (4.7–7.4°N, -1.3–0.1°E)
#     outward by 2° then to the nearest 10° → S=0, N=10, W=-10, E=10.
# Files will be named: ERA5_N0_N10_W10_E10_YYYYMMDD_18.grb
ERA5_HOUR = '18'              # Nearest 6h slot to CENTER_LINE_UTC 18:17 UTC

DOWNLOAD_SCRIPT = '''\
import sys
import h5py
import pyaps3.autoget as ag
from pathlib import Path

mintpy_dir = sys.argv[1]
era5_dir   = sys.argv[2]
hour       = sys.argv[3]
snwe       = tuple(int(x) for x in sys.argv[4].split(','))

Path(era5_dir).mkdir(parents=True, exist_ok=True)

with h5py.File(f"{mintpy_dir}/timeseries_demErr.h5", "r") as f:
    dates = [d.decode() for d in f["date"][:]]

print(f"Total acquisition dates: {len(dates)}")

# Filter to dates not already downloaded
existing = {p.stem.split("_")[-2] for p in Path(era5_dir).glob("ERA5*_*.grb")}
todo = [d for d in dates if d not in existing]
print(f"Already downloaded: {len(dates) - len(todo)}")
print(f"To download: {len(todo)}")

if not todo:
    print("All ERA5 files already downloaded.")
    sys.exit(0)

# Download in batches of 20 to avoid CDS queue timeouts
BATCH = 20
for start in range(0, len(todo), BATCH):
    batch = todo[start:start + BATCH]
    print(f"\\nDownloading batch {start//BATCH + 1}: dates {batch[0]} to {batch[-1]}")
    try:
        ag.ECMWFdload(
            bdate=batch,
            hr=hour,
            filedir=era5_dir,
            model="ERA5",
            snwe=snwe,
        )
    except Exception as exc:
        print(f"[WARN] Batch failed: {exc}")
        print("       Continuing with remaining batches...")

# Final count
downloaded = list(Path(era5_dir).glob("ERA5*_*.grb"))
print(f"\\nERA5 GRIB files in {era5_dir}: {len(downloaded)}")
'''


def download_era5() -> int:
    """Download ERA5 GRIB files for all acquisition dates. Returns count of files."""
    py = conda_python()
    era5_existing = list(ERA5_DIR.glob('ERA5*_*.grb'))
    print(f'\n[ERA5 DOWNLOAD] {len(era5_existing)} files already in {ERA5_DIR}')

    # Write temp download script
    dl_script = MINTPY_DIR / '_era5_download_tmp.py'
    dl_script.write_text(DOWNLOAD_SCRIPT)

    snwe_str = ','.join(str(x) for x in ERA5_SNWE)
    result = subprocess.run(
        [py, str(dl_script),
         str(MINTPY_DIR), str(ERA5_DIR), ERA5_HOUR, snwe_str],
        env={**os.environ, 'WEATHER_DIR': str(WEATHER_DIR)}
    )
    dl_script.unlink(missing_ok=True)

    # Report
    downloaded = sorted(ERA5_DIR.glob('ERA5*_*.grb'))
    n = len(downloaded)
    print(f'\n[ERA5 DOWNLOAD] Done — {n} GRIB files in {ERA5_DIR}')
    return n


# ---------------------------------------------------------------------------
# Step 2: Run MintPy from correct_troposphere → velocity
# ---------------------------------------------------------------------------
def run_mintpy_correction() -> None:
    """Re-run MintPy pipeline from correct_troposphere to velocity."""
    smallbaseline = conda_bin('smallbaselineApp.py')

    env = {
        'WEATHER_DIR': str(WEATHER_DIR),
        'PATH': os.environ.get('PATH', ''),
    }

    print('\n[MINTPY] Running correct_troposphere step...')
    run(
        [smallbaseline, str(CFG_FILE),
         '--start', 'correct_troposphere',
         '--end',   'correct_troposphere'],
        env=env, cwd=MINTPY_DIR
    )

    print('\n[MINTPY] Running correct_topography (DEM error correction)...')
    run(
        [smallbaseline, str(CFG_FILE),
         '--start', 'correct_topography',
         '--end',   'correct_topography'],
        env=env, cwd=MINTPY_DIR
    )

    print('\n[MINTPY] Running velocity step...')
    run(
        [smallbaseline, str(CFG_FILE),
         '--start', 'velocity',
         '--end',   'velocity'],
        env=env, cwd=MINTPY_DIR
    )


# ---------------------------------------------------------------------------
# Step 3: Re-export corrected GeoTIFFs
# ---------------------------------------------------------------------------
EXPORT_SCRIPT = r'''
import h5py
import numpy as np
import rasterio
from rasterio.crs import CRS
import rasterio.transform
import subprocess, sys
from pathlib import Path

mintpy_dir = Path(sys.argv[1])
subdir     = Path(sys.argv[2])

vel_h5  = mintpy_dir / "velocity.h5"
std_h5  = mintpy_dir / "velocity.h5"   # std stored in same file
tmp_utm = subdir / "_mintpy_velocity_utm_tmp.tif"
los_out = subdir / "mintpy_velocity_los.tif"
vert_out= subdir / "mintpy_velocity_vert.tif"
std_out = subdir / "mintpy_velocity_std.tif"

print(f"Reading {vel_h5}")
with h5py.File(vel_h5, "r") as f:
    vel = f["velocity"][:]
    std = f["velocityStd"][:]
    attrs = dict(f.attrs)

x_first   = float(attrs["X_FIRST"])
y_first   = float(attrs["Y_FIRST"])
x_step    = float(attrs["X_STEP"])
y_step    = float(attrs["Y_STEP"])
width     = int(attrs["WIDTH"])
length    = int(attrs["FILE_LENGTH"])
epsg_code = int(attrs.get("EPSG", 32630))
incidence = float(attrs.get("INCIDENCE_ANGLE", 38.0))

print(f"Grid: {length} x {width}, EPSG:{epsg_code}")
print(f"Velocity range: {np.nanmin(vel)*1000:.1f} to {np.nanmax(vel)*1000:.1f} mm/yr")

# ---- Convert m/yr → mm/yr ----
vel_mmyr = vel * 1000.0
std_mmyr = std * 1000.0
vert_mmyr = vel_mmyr / np.cos(np.radians(incidence))

# ---- Write UTM GeoTIFF ----
ulx = x_first - 0.5 * x_step
uly = y_first - 0.5 * y_step
transform = rasterio.transform.from_origin(ulx, uly, abs(x_step), abs(y_step))
crs_utm = CRS.from_epsg(epsg_code)

NODATA = -9999.0
vel_mmyr[np.isnan(vel_mmyr)] = NODATA
std_mmyr[np.isnan(std_mmyr)] = NODATA
vert_mmyr[np.isnan(vert_mmyr)] = NODATA

def write_utm(arr, path):
    with rasterio.open(str(path), "w",
                       driver="GTiff", height=length, width=width,
                       count=1, dtype="float32",
                       crs=crs_utm, transform=transform,
                       nodata=NODATA) as dst:
        dst.write(arr.astype("float32"), 1)

write_utm(vel_mmyr, tmp_utm)

# ---- Warp UTM → WGS84 ----
def warp_to_wgs84(src, dst):
    gdalwarp = "gdalwarp"
    cmd = [gdalwarp, "-s_srs", f"EPSG:{epsg_code}", "-t_srs", "EPSG:4326",
           "-r", "bilinear", "-overwrite", str(src), str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] gdalwarp stderr: {result.stderr}")

warp_to_wgs84(tmp_utm, los_out)

# ---- Write vertical and std in UTM then warp ----
tmp_vert = subdir / "_mintpy_vert_tmp.tif"
tmp_std  = subdir / "_mintpy_std_tmp.tif"
write_utm(vert_mmyr, tmp_vert)
write_utm(std_mmyr,  tmp_std)
warp_to_wgs84(tmp_vert, vert_out)
warp_to_wgs84(tmp_std,  std_out)

# Cleanup temps
for t in [tmp_utm, tmp_vert, tmp_std]:
    t.unlink(missing_ok=True)

# ---- Verify ----
import rasterio as rio
with rio.open(los_out) as src:
    data = src.read(1)
    data[data == NODATA] = np.nan
    n_valid = int(np.sum(~np.isnan(data)))
    print(f"\n[LOS] Valid pixels: {n_valid:,}")
    print(f"[LOS] Range: {np.nanmin(data):.1f} to {np.nanmax(data):.1f} mm/yr")
    print(f"[LOS] Bounds: {src.bounds}")
    print(f"[LOS] CRS: {src.crs}")

print("\n[OK] ERA5-corrected GeoTIFFs exported:")
print(f"  LOS:  {los_out}")
print(f"  Vert: {vert_out}")
print(f"  Std:  {std_out}")
'''


def export_geotiffs() -> None:
    """Re-export corrected velocity to WGS84 GeoTIFFs."""
    py = conda_python()
    export_script = MINTPY_DIR / '_era5_export_tmp.py'
    export_script.write_text(EXPORT_SCRIPT)

    run(
        [py, str(export_script), str(MINTPY_DIR), str(SUBDIR)],
        env={'WEATHER_DIR': str(WEATHER_DIR)}
    )
    export_script.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Step 4: Print a post-correction report
# ---------------------------------------------------------------------------
REPORT_SCRIPT = r'''
import h5py, numpy as np, sys
from pathlib import Path

mintpy_dir = Path(sys.argv[1])

# Compare velocity stats before/after
for fname in ["velocity.h5"]:
    f_path = mintpy_dir / fname
    if not f_path.exists():
        continue
    with h5py.File(f_path, "r") as f:
        vel = f["velocity"][:] * 1000  # mm/yr
        std = f["velocityStd"][:] * 1000

    mask = np.abs(vel) <= 300
    v = vel[mask & ~np.isnan(vel)]
    s = std[mask & ~np.isnan(std)]
    print(f"\n=== {fname} (ERA5-corrected) ===")
    print(f"  Valid pixels : {len(v):,}")
    print(f"  Velocity     : {np.nanmin(v):.1f} to {np.nanmax(v):.1f} mm/yr")
    print(f"  Mean ± StdDev: {np.nanmean(v):.2f} ± {np.nanstd(v):.2f} mm/yr")
    print(f"  Median       : {np.nanmedian(v):.2f} mm/yr")
    print(f"  Vel Std mean : {np.nanmean(s):.2f} mm/yr")
    print(f"  Vel Std median: {np.nanmedian(s):.2f} mm/yr")

# Check for troposphericDelay file
td_path = mintpy_dir / "timeseries_tropHgt.h5"
td_path2 = mintpy_dir / "ERA5.h5"
for p in [td_path, td_path2, *mintpy_dir.glob("*ERA5*")]:
    if p.exists():
        print(f"\n[Tropo file] {p.name}")
        with h5py.File(p, "r") as f:
            print(f"  Keys: {list(f.keys())}")
'''


def print_report() -> None:
    """Print velocity statistics after ERA5 correction."""
    py = conda_python()
    rpt = MINTPY_DIR / '_era5_report_tmp.py'
    rpt.write_text(REPORT_SCRIPT)
    run([py, str(rpt), str(MINTPY_DIR)])
    rpt.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print('=' * 70)
    print('ERA5 Tropospheric Correction — Coastal Ghana InSAR (Track 147)')
    print('=' * 70)
    print(f'MINTPY_DIR : {MINTPY_DIR}')
    print(f'WEATHER_DIR: {WEATHER_DIR}')
    print(f'ERA5_DIR   : {ERA5_DIR}')
    print(f'ERA5 SNWE  : {ERA5_SNWE}  (integer bbox)')
    print(f'ERA5 hour  : {ERA5_HOUR}:00 UTC')

    # Ensure WEATHER_DIR exists
    ERA5_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-flight
    check_cds_credentials()
    check_era5_licence()
    check_dependencies()

    # Run pipeline
    n_downloaded = download_era5()

    if n_downloaded == 0:
        print('\n' + '!' * 70)
        print('ERA5 DOWNLOAD FAILED — No GRIB files were downloaded.')
        print('!' * 70)
        print('\nMost likely cause: ERA5 licence not yet accepted.')
        print('\nFix:')
        print('  1. Log in at https://cds.climate.copernicus.eu/')
        print('  2. Go to: https://cds.climate.copernicus.eu/datasets/'
              'reanalysis-era5-pressure-levels?tab=download#manage-licences')
        print('  3. Click "Accept Terms" (one-time step)')
        print('  4. Wait a few minutes for the licence to activate')
        print('  5. Re-run: python scripts/run_era5_correction.py')
        sys.exit(1)

    print(f'\n[OK] {n_downloaded} ERA5 GRIB files ready — proceeding with MintPy correction.')
    run_mintpy_correction()
    export_geotiffs()
    print_report()

    print('\n' + '=' * 70)
    print('ERA5 CORRECTION COMPLETE')
    print('=' * 70)
    print(f'  Corrected velocity GeoTIFFs:')
    print(f'    LOS  : data/processed/subsidence/mintpy_velocity_los.tif')
    print(f'    Vert : data/processed/subsidence/mintpy_velocity_vert.tif')
    print(f'    Std  : data/processed/subsidence/mintpy_velocity_std.tif')
    print(f'\n  Next steps:')
    print(f'    1. Re-run NB10 (risk index) to use updated velocity data.')
    print(f'    2. Re-run NB12 Fig 07 to show ERA5-corrected velocity map.')
    print(f'\n  Note: ERA5 corrects stratified tropospheric delay for the')
    print(f'  coherent inland zone (7.1–7.4°N). It does NOT recover the')
    print(f'  decorrelated coastal pixels (4.5–6.5°N).')


if __name__ == '__main__':
    main()
