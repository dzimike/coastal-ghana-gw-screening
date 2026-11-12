"""
run_mintpy_sbas.py
─────────────────────────────────────────────────────────────────────────────
Run MintPy SBAS time-series inversion on HyP3 ISCE burst products.

WHAT IS SBAS / MintPy?
-----------------------
SBAS (Small Baseline Subset) is the correct method for deriving reliable
surface velocity maps from a stack of InSAR interferograms.

  - Each interferogram contains: displacement + atmosphere + noise + orbit
  - Stacking averages only N pairs → atmospheric noise does NOT average out
  - SBAS inverts the full network: for each pixel, it solves the least-squares
    system Δφ = G × v across all pairs simultaneously. Atmospheric signals
    (temporally uncorrelated) cancel. Cumulative deformation is preserved.

IMPORTANT NOTES ON PRODUCT TYPE
---------------------------------
HyP3 products are ALWAYS in GeoTIFF (.tif) + metadata (.txt) format regardless
of which algorithm ran inside HyP3 (GAMMA, ISCE, ISCE_MULTI_BURST etc.).
Therefore mintpy.load.processor MUST be 'hyp3' for all HyP3 products.
Do NOT use processor = 'isce' — that reads raw ISCE-native .xml files.

RESAMPLING REQUIREMENT
-----------------------
HyP3 ISCE burst products have minor pixel-count variations (±2–3 pixels across
acquisitions) even for the same burst. MintPy rejects any file that differs in
size from the most common size. To load ALL products:
  1. Resample all GeoTIFFs to a common reference grid (done by this script)
  2. Run prep_hyp3.py on the resampled files (positional args, no --mintpy-dir)
  3. Run smallbaselineApp.py with processor = hyp3

OUTPUTS
-------
  data/processed/subsidence/mintpy/
    velocity.h5              — mean LOS velocity map (m/yr)
    timeseries.h5            — cumulative displacement time series
    temporalCoherence.h5     — pixel quality metric (0–1)
  data/processed/subsidence/
    mintpy_velocity_los.tif  — GeoTIFF LOS velocity (mm/yr)
    mintpy_velocity_vert.tif — GeoTIFF vertical velocity (approximate, mm/yr)

USAGE
-----
  # Full pipeline (resample + prep + SBAS + export):
  python scripts/run_mintpy_sbas.py

  # Skip resampling if already done:
  python scripts/run_mintpy_sbas.py --skip-resample

  # Skip prep + resample if stack already loaded:
  python scripts/run_mintpy_sbas.py --skip-prep

  # Export velocity.h5 to GeoTIFF only:
  python scripts/run_mintpy_sbas.py --export-only

REQUIREMENTS
------------
  conda install -c conda-forge mintpy gdal
"""

import os
import sys
import json
import shutil
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
ISCE_DIR      = ROOT / 'data' / 'raw' / 'sentinel1' / 'hyp3_isce_products'
RESAMPLED_DIR = ROOT / 'data' / 'raw' / 'sentinel1' / 'hyp3_isce_resampled'
MINTPY_DIR    = ROOT / 'data' / 'processed' / 'subsidence' / 'mintpy'
OUT_DIR       = ROOT / 'data' / 'processed' / 'subsidence'

MINTPY_DIR.mkdir(parents=True, exist_ok=True)
CFG_FILE = MINTPY_DIR / 'smallbaselineApp.cfg'

# ── conda environment paths ────────────────────────────────────────────────────
CONDA_ENV = Path(sys.executable).parents[1]   # e.g. /opt/miniconda3/envs/accra-risk
GDALINFO  = CONDA_ENV / 'bin' / 'gdalinfo'
GDALWARP  = CONDA_ENV / 'bin' / 'gdalwarp'


# ══════════════════════════════════════════════════════════════════════════════
# 1. Check MintPy
# ══════════════════════════════════════════════════════════════════════════════
def check_mintpy() -> bool:
    try:
        import mintpy
        log.info(f'MintPy {mintpy.__version__} found ✓')
        return True
    except ImportError:
        log.error('MintPy not installed.  conda install -c conda-forge mintpy')
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 2. Find reference grid from the most-common-sized product
# ══════════════════════════════════════════════════════════════════════════════
def find_reference_grid(products_dir: Path) -> dict:
    """
    Return gdalwarp parameters for a common reference grid.
    Selects the product with the most common pixel count (excluding outliers).
    """
    log.info('Detecting reference grid from product sizes …')
    unw_files = sorted(products_dir.glob('*/*_unw_phase.tif'))
    if not unw_files:
        raise FileNotFoundError(f'No *_unw_phase.tif files found in {products_dir}')

    size_map: dict[tuple, list] = {}
    for f in unw_files:
        r = subprocess.run([str(GDALINFO), '-json', str(f)], capture_output=True, text=True)
        info = json.loads(r.stdout)
        sz = tuple(info['size'])        # (width, height)
        if sz not in size_map:
            size_map[sz] = []
        size_map[sz].append(f)

    # Most common size (excluding tiny outliers)
    best_sz = max(
        (sz for sz in size_map if sz[0] > 1000 and sz[1] > 1000),
        key=lambda s: len(size_map[s])
    )
    ref_file = size_map[best_sz][0]

    r = subprocess.run([str(GDALINFO), '-json', str(ref_file)], capture_output=True, text=True)
    ref_info = json.loads(r.stdout)
    gt = ref_info['geoTransform']   # [xOrigin, xPixelSize, 0, yOrigin, 0, yPixelSize]
    w, h = ref_info['size']

    xmin = gt[0]
    xmax = gt[0] + w * gt[1]
    ymax = gt[3]
    ymin = gt[3] + h * gt[5]

    grid = {
        'te':   f'{xmin} {ymin} {xmax} {ymax}',
        'tr':   f'{gt[1]} {abs(gt[5])}',
        'ts':   f'{w} {h}',
        't_srs': ref_info.get('coordinateSystem', {}).get('epsg', 'EPSG:32630'),
        'ref_file': ref_file,
        'size': best_sz,
    }
    log.info(f'  Reference grid: {w}×{h} pixels  (from {ref_file.parent.name})')
    log.info(f'  Extent: {xmin:.0f},{ymin:.0f} → {xmax:.0f},{ymax:.0f}  res={gt[1]:.0f}m')
    return grid


# ══════════════════════════════════════════════════════════════════════════════
# 3. Resample all products to common grid
# ══════════════════════════════════════════════════════════════════════════════
def resample_products(products_dir: Path, out_dir: Path, grid: dict, workers: int = 4):
    """Warp all GeoTIFF products to the reference grid; copy .txt metadata."""
    RESAMPLE_METHOD = {
        '_unw_phase.tif':  'bilinear',
        '_corr.tif':       'bilinear',
        '_conncomp.tif':   'near',
        '_dem.tif':        'bilinear',
        '_lv_theta.tif':   'bilinear',
        '_lv_phi.tif':     'bilinear',
        '_water_mask.tif': 'near',
    }

    def warp_one(src: Path, dst: Path, method: str) -> str:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return f'SKIP {dst.name}'
        cmd = [
            str(GDALWARP),
            '-t_srs', 'EPSG:32630',
            '-te',   *grid['te'].split(),
            '-tr',   *grid['tr'].split(),
            '-ts',   *grid['ts'].split(),
            '-r',    method,
            '-q',
            str(src), str(dst),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return f'OK {dst.name}' if r.returncode == 0 else f'ERROR {dst.name}: {r.stderr[:60]}'

    tasks = []
    for prod_dir in sorted(products_dir.iterdir()):
        if not prod_dir.is_dir():
            continue
        for suffix, method in RESAMPLE_METHOD.items():
            for src in prod_dir.glob(f'*{suffix}'):
                dst = out_dir / prod_dir.name / src.name
                tasks.append((src, dst, method))

    log.info(f'Resampling {len(tasks)} files to common grid …')
    ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(warp_one, *t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            if result.startswith('OK'):    ok   += 1
            elif result.startswith('SKIP'): skip += 1
            else: err += 1; log.warning(result)
            if i % 200 == 0:
                log.info(f'  {i}/{len(tasks)} ({ok} ok, {skip} skip, {err} err)')

    # Copy metadata .txt files
    for prod_dir in sorted(products_dir.iterdir()):
        if not prod_dir.is_dir():
            continue
        for txt in prod_dir.glob('*.txt'):
            dst = out_dir / prod_dir.name / txt.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(txt, dst)

    log.info(f'Resampling done  ({ok} warped, {skip} already done, {err} errors)')


# ══════════════════════════════════════════════════════════════════════════════
# 4. Run prep_hyp3.py on resampled products
# ══════════════════════════════════════════════════════════════════════════════
def run_prep_hyp3(resampled_dir: Path, ref_product: Path):
    """
    Run MintPy prep_hyp3.py on all resampled product files.

    IMPORTANT: prep_hyp3.py accepts ONLY positional file arguments (individual
    .tif paths). It does NOT accept --mintpy-dir or a directory as argument.
    Signature: prep_hyp3.py file [file ...]
    """
    log.info('Running prep_hyp3.py on resampled products …')
    PY = sys.executable

    file_groups = {
        'unw_phase': sorted(resampled_dir.glob('*/*_unw_phase.tif')),
        'corr':      sorted(resampled_dir.glob('*/*_corr.tif')),
        'conncomp':  sorted(resampled_dir.glob('*/*_conncomp.tif')),
        'dem':       sorted(ref_product.glob('*_dem.tif')),
        'lv_theta':  sorted(ref_product.glob('*_lv_theta.tif')),
        'lv_phi':    sorted(ref_product.glob('*_lv_phi.tif')),
    }

    for group_name, files in file_groups.items():
        if not files:
            log.warning(f'  No files for group: {group_name}')
            continue
        cmd = [PY, '-m', 'mintpy.cli.prep_hyp3'] + [str(f) for f in files]
        log.info(f'  prep_hyp3.py {group_name}: {len(files)} files …')
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f'  prep_hyp3.py failed for {group_name}:\n{result.stderr[:200]}')
            sys.exit(1)

    n_rsc = len(list(resampled_dir.rglob('*.rsc')))
    log.info(f'prep_hyp3.py done — {n_rsc} .rsc files created ✓')


# ══════════════════════════════════════════════════════════════════════════════
# 5. Compute WGS84 subset from resampled products
# ══════════════════════════════════════════════════════════════════════════════
def compute_safe_subset(resampled_dir: Path) -> str:
    """Return the tightest WGS84 extent common to all resampled products."""
    files = sorted(resampled_dir.glob('*/*_unw_phase.tif'))
    lons_min, lats_min, lons_max, lats_max = [], [], [], []
    for f in files:
        r = subprocess.run([str(GDALINFO), '-json', str(f)], capture_output=True, text=True)
        info = json.loads(r.stdout)
        w, h = info['size']
        if w < 1000 or h < 1000:
            continue
        coords = info['wgs84Extent']['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        lons_min.append(min(lons)); lons_max.append(max(lons))
        lats_min.append(min(lats)); lats_max.append(max(lats))

    W = max(lons_min); E = min(lons_max)
    S = max(lats_min); N = min(lats_max)
    margin = 0.005
    subset = f'{S+margin:.3f}:{N-margin:.3f},{W+margin:.3f}:{E-margin:.3f}'
    log.info(f'Safe WGS84 subset: {subset}')
    return subset


# ══════════════════════════════════════════════════════════════════════════════
# 6. Write corrected smallbaselineApp.cfg
# ══════════════════════════════════════════════════════════════════════════════
def write_mintpy_config(resampled_dir: Path, ref_product: Path, subset: str):
    """
    Write smallbaselineApp.cfg for HyP3 ISCE burst products.

    KEY CORRECTNESS NOTES:
      processor = hyp3   (NOT isce — HyP3 always outputs GeoTIFF/.txt format)
      reference.lalo = lat,lon  (comma-separated, NOT colon-separated)
    """
    ref_dem   = sorted(ref_product.glob('*_dem.tif'))[0]
    ref_theta = sorted(ref_product.glob('*_lv_theta.tif'))[0]
    ref_phi   = sorted(ref_product.glob('*_lv_phi.tif'))[0]

    cfg = f"""# vim: set filetype=cfg:
##──────────────────────────────────────────────────────────────────────────────
## MintPy SBAS Configuration — Ghana Groundwater-Subsidence Project
## Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
##
## NOTE on processor = hyp3:
##   HyP3 ALWAYS packages outputs as GeoTIFF + .txt metadata regardless of which
##   algorithm (GAMMA, ISCE, INSAR_ISCE_MULTI_BURST) ran internally.
##   processor = hyp3 handles ALL HyP3 GeoTIFF products.
##   processor = isce is for raw ISCE-native .xml files (NOT produced by HyP3).
##──────────────────────────────────────────────────────────────────────────────

########## computing resource configuration
mintpy.compute.cluster   = no
mintpy.compute.numWorker = 4

########## 1. load_data
mintpy.load.processor       = hyp3
mintpy.load.unwFile         = {resampled_dir}/*/*_unw_phase.tif
mintpy.load.corFile         = {resampled_dir}/*/*_corr.tif
mintpy.load.connCompFile    = {resampled_dir}/*/*_conncomp.tif
mintpy.load.demFile         = {ref_dem}
mintpy.load.incAngleFile    = {ref_theta}
mintpy.load.azAngleFile     = {ref_phi}

########## subset — tight intersection of all resampled products
mintpy.subset.lalo          = {subset}

########## reference point — auto selects highest-coherence pixel
## NOTE: must use lat,lon (comma) NOT lat:lon (colon)
mintpy.reference.lalo       = auto

########## unwrap error correction
## bridging only: 12-day consecutive pairs have no closed phase triangles
mintpy.unwrapError.method   = no

########## network inversion (classical SBAS)
mintpy.networkInversion.weightFunc      = no
mintpy.networkInversion.maskDataset     = coherence
mintpy.networkInversion.maskThreshold   = 0.3
mintpy.networkInversion.minRedundancy   = 0
mintpy.networkInversion.minTempCoh      = 0.5

########## tropospheric delay
## ERA5 is best for Ghana; set to no for first-pass run
mintpy.troposphericDelay.method = no

########## topographic residual correction
mintpy.topographicResidual  = yes

########## velocity estimation
mintpy.velocity.startDate   = auto
mintpy.velocity.endDate     = auto
"""

    CFG_FILE.write_text(cfg)
    log.info(f'Config written → {CFG_FILE}')


# ══════════════════════════════════════════════════════════════════════════════
# 7. Run smallbaselineApp.py
# ══════════════════════════════════════════════════════════════════════════════
def run_sbas():
    log.info('Running MintPy SBAS pipeline …  (expect 10–30 min)')
    cmd = [sys.executable, '-m', 'mintpy.cli.smallbaselineApp', str(CFG_FILE)]
    result = subprocess.run(cmd, cwd=str(MINTPY_DIR), capture_output=False, text=True)
    if result.returncode != 0:
        log.error('smallbaselineApp.py failed — check output above')
        sys.exit(1)
    log.info('MintPy SBAS complete ✓')


# ══════════════════════════════════════════════════════════════════════════════
# 8. Export velocity GeoTIFF
# ══════════════════════════════════════════════════════════════════════════════
def export_velocity_geotiff():
    """Convert MintPy velocity.h5 to GeoTIFF for use in NB12 / CGSRI."""
    vel_h5 = MINTPY_DIR / 'velocity.h5'
    if not vel_h5.exists():
        log.error(f'velocity.h5 not found at {vel_h5}')
        return

    try:
        import h5py, rasterio, rasterio.transform, math

        with h5py.File(vel_h5, 'r') as f:
            vel = f['velocity'][:]          # m/yr
            atr = dict(f['velocity'].attrs)
            lat0    = float(atr.get('Y_FIRST', 4.0))
            lon0    = float(atr.get('X_FIRST', -1.5))
            dlat    = float(atr.get('Y_STEP',  -0.0007))
            dlon    = float(atr.get('X_STEP',   0.0007))
            incl_deg = float(atr.get('incidenceAngle', 38.0))

        vel_mmyr = vel * 1000.0
        nodata   = -9999.0

        transform = rasterio.transform.from_origin(lon0, lat0, abs(dlon), abs(dlat))
        profile = {
            'driver': 'GTiff', 'dtype': 'float32',
            'width': vel.shape[1], 'height': vel.shape[0],
            'count': 1, 'crs': 'EPSG:4326',
            'transform': transform, 'nodata': nodata,
            'compress': 'lzw', 'tiled': True,
        }

        # LOS velocity
        los_out = OUT_DIR / 'mintpy_velocity_los.tif'
        with rasterio.open(los_out, 'w', **profile) as dst:
            data = np.where(np.isfinite(vel_mmyr), vel_mmyr, nodata).astype('float32')
            dst.write(data, 1)
        log.info(f'LOS velocity  → {los_out}')

        # Approximate vertical (divide LOS by cos(incidence angle))
        vert_mmyr = vel_mmyr / math.cos(math.radians(incl_deg))
        vert_out = OUT_DIR / 'mintpy_velocity_vert.tif'
        with rasterio.open(vert_out, 'w', **profile) as dst:
            data = np.where(np.isfinite(vert_mmyr), vert_mmyr, nodata).astype('float32')
            dst.write(data, 1)
        log.info(f'Vertical velocity → {vert_out}')

        valid = vel_mmyr[np.isfinite(vel_mmyr)]
        if valid.size > 0:
            log.info(f'Velocity stats (mm/yr): '
                     f'p5={np.percentile(valid,5):.1f}  '
                     f'median={np.median(valid):.1f}  '
                     f'p95={np.percentile(valid,95):.1f}')
            pct_sub = float(np.mean(valid < -2)) * 100
            log.info(f'Pixels < −2 mm/yr (subsiding): {pct_sub:.1f}%')

    except Exception as e:
        log.error(f'Export failed: {e}')
        import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Run MintPy SBAS on HyP3 ISCE burst products')
    parser.add_argument('--products-dir', type=Path, default=ISCE_DIR,
                        help=f'Source product directory (default: {ISCE_DIR})')
    parser.add_argument('--resampled-dir', type=Path, default=RESAMPLED_DIR,
                        help=f'Resampled output directory (default: {RESAMPLED_DIR})')
    parser.add_argument('--skip-resample', action='store_true',
                        help='Skip resampling step (resampled files already exist)')
    parser.add_argument('--skip-prep',    action='store_true',
                        help='Skip prep_hyp3.py (stack HDF5 already exists)')
    parser.add_argument('--export-only',  action='store_true',
                        help='Only export velocity.h5 to GeoTIFF')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel workers for gdalwarp (default: 4)')
    args = parser.parse_args()

    if not check_mintpy():
        sys.exit(1)

    if args.export_only:
        export_velocity_geotiff()
        return

    products_dir  = args.products_dir
    resampled_dir = args.resampled_dir

    # Find reference grid (from source products, before resampling)
    grid = find_reference_grid(products_dir)
    ref_product = resampled_dir / grid['ref_file'].parent.name

    # Step 1: resample
    if not args.skip_resample:
        resample_products(products_dir, resampled_dir, grid, workers=args.workers)
    else:
        log.info('Skipping resample (--skip-resample)')

    # Step 2: prep_hyp3
    if not args.skip_prep:
        run_prep_hyp3(resampled_dir, ref_product)
    else:
        log.info('Skipping prep_hyp3 (--skip-prep)')

    # Step 3: compute subset
    subset = compute_safe_subset(resampled_dir)

    # Step 4: write config
    write_mintpy_config(resampled_dir, ref_product, subset)

    # Step 5: run SBAS
    run_sbas()

    # Step 6: export
    export_velocity_geotiff()

    log.info('')
    log.info('══════════════════════════════════════════════')
    log.info(' MintPy SBAS complete.')
    log.info(f' Velocity GeoTIFFs: {OUT_DIR}')
    log.info(' Next: re-run notebooks/12_maps_and_figures.ipynb')
    log.info('══════════════════════════════════════════════')


if __name__ == '__main__':
    main()
