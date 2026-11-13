#!/usr/bin/env python3
"""MintPy SBAS rerun (v2) on the gap-filled Track 147 network (159 original + 230 fill pairs).

Keeps the published v1 settings (data/processed/subsidence/mintpy/smallbaselineApp.cfg) except:
  - interferograms: v1 pairs plus the 230 HyP3 fill pairs (outputs/tables/hyp3_isce_fill_jobs.csv)
  - reference point fixed to the v1 pixel (REF_Y 2612, REF_X 1265 in the subset grid)
  - minTempCoh 0.7, matching the temporal-coherence threshold stated in the manuscript
  - plotting off
Absolute paths are used throughout; v1 inputs and outputs are not modified.

Steps (each skippable): link v1 resampled pairs, resample the fill pairs to the v1 grid,
prep_hyp3 metadata, write config, run smallbaselineApp (ERA5 delays copied from v1).
The velocity export is a separate step (export_velocity_v2.py) that must use
timeseries_ERA5_demErr.h5, not velocityERA5.h5.
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_mintpy_sbas as v1  # noqa: E402  (reuses resample_products and logging)

PRODUCTS = ROOT / 'data/raw/sentinel1/hyp3_isce_products'
RESAMPLED_V1 = ROOT / 'data/raw/sentinel1/hyp3_isce_resampled'
RESAMPLED_V2 = ROOT / 'data/raw/sentinel1/hyp3_isce_resampled_v2'
MINTPY_V1 = ROOT / 'data/processed/subsidence/mintpy'
MINTPY_V2 = ROOT / 'data/processed/subsidence/mintpy_v2'
FILL_LOG = ROOT / 'outputs/tables/hyp3_isce_fill_jobs.csv'
GEOM_PRODUCT = RESAMPLED_V1 / 'S1_147_000000s1n00-313622s2n15-000000s3n00_IW_20170520_20170601_VV_INT80_2FAF'

# v1 resampled grid (all 159 v1 products share it)
GRID = {'te': '680880 541200 838560 841040', 'tr': '80 80', 'ts': '1971 3748'}
REF_YX = '2612,1265'
log = v1.log


def fill_pairs():
    return {(r['date1'], r['date2']) for r in csv.DictReader(open(FILL_LOG))}


def pair_of(d):
    return tuple(d.name.split('_')[4:6])


def link_v1():
    RESAMPLED_V2.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted(RESAMPLED_V1.iterdir()):
        dst = RESAMPLED_V2 / d.name
        if d.is_dir() and not dst.exists():
            dst.symlink_to(d, target_is_directory=True)
            n += 1
    log.info(f'linked {n} v1 resampled pairs into {RESAMPLED_V2}')


def resample_fill(workers):
    fills = fill_pairs()
    stage = MINTPY_V2 / '_fill_src'
    stage.mkdir(parents=True, exist_ok=True)
    for d in PRODUCTS.iterdir():
        if d.is_dir() and pair_of(d) in fills and not (stage / d.name).exists():
            (stage / d.name).symlink_to(d, target_is_directory=True)
    log.info(f'{len(list(stage.iterdir()))} fill products staged for resampling')
    v1.resample_products(stage, RESAMPLED_V2, GRID, workers=workers)


def prep_fill():
    fills = fill_pairs()
    dirs = [d for d in RESAMPLED_V2.iterdir() if not d.is_symlink() and pair_of(d) in fills]
    for kind in ('unw_phase', 'corr', 'conncomp'):
        files = [str(f) for d in dirs for f in d.glob(f'*_{kind}.tif') if not Path(str(f) + '.rsc').exists()]
        if files:
            log.info(f'prep_hyp3 {kind}: {len(files)} files')
            subprocess.run([sys.executable, '-m', 'mintpy.cli.prep_hyp3', *files], check=True,
                           capture_output=True, text=True)
    n_unw = len(list(RESAMPLED_V2.glob('*/*_unw_phase.tif.rsc')))
    log.info(f'{n_unw} unwrapped interferograms with metadata in {RESAMPLED_V2}')
    if n_unw != 389:
        raise SystemExit(f'expected 389 interferograms, found {n_unw}')


def write_config():
    src = (MINTPY_V1 / 'smallbaselineApp.cfg').read_text()
    geom = lambda s: str(next(GEOM_PRODUCT.glob(f'*_{s}.tif')))
    repl = {
        'mintpy.load.unwFile': f'{RESAMPLED_V2}/*/*_unw_phase.tif',
        'mintpy.load.corFile': f'{RESAMPLED_V2}/*/*_corr.tif',
        'mintpy.load.connCompFile': f'{RESAMPLED_V2}/*/*_conncomp.tif',
        'mintpy.load.demFile': geom('dem'),
        'mintpy.load.incAngleFile': geom('lv_theta'),
        'mintpy.load.azAngleFile': geom('lv_phi'),
        'mintpy.reference.yx': REF_YX,
        'mintpy.reference.lalo': 'auto',
        # no pixel stays in a non-zero connected component in all 389 pairs (maskConnComp is empty),
        # so the v1 reference pixel is kept without that mask
        'mintpy.reference.maskFile': 'no',
        # 20220915 covers 18% of the scene and is empty at the reference pixel; the network stays connected without it
        'mintpy.network.excludeDate': '20220915',
        'mintpy.networkInversion.minTempCoh': '0.7',
        'mintpy.troposphericDelay.weatherDir': str(MINTPY_V2 / 'WEATHER'),
        'mintpy.plot': 'no',
    }
    out = []
    for line in src.splitlines():
        key = line.split('=')[0].strip()
        if key in repl and '=' in line:
            line = f'{key} = {repl.pop(key)}'
        out.append(line)
    if repl:
        raise SystemExit(f'config keys not found: {sorted(repl)}')
    MINTPY_V2.mkdir(parents=True, exist_ok=True)
    (MINTPY_V2 / 'smallbaselineApp.cfg').write_text('\n'.join(out) + '\n')
    weather = MINTPY_V2 / 'WEATHER'
    if not weather.exists():
        weather.symlink_to(MINTPY_V1 / 'WEATHER', target_is_directory=True)
    log.info(f'config written: {MINTPY_V2 / "smallbaselineApp.cfg"}')


def sba(*args):
    cmd = [sys.executable, '-m', 'mintpy.cli.smallbaselineApp', str(MINTPY_V2 / 'smallbaselineApp.cfg'), *args]
    log.info('smallbaselineApp ' + ' '.join(args))
    subprocess.run(cmd, cwd=MINTPY_V2, check=True)


def run():
    sba('--end', 'load_data')
    era5 = MINTPY_V2 / 'inputs/ERA5.h5'
    if not era5.exists():
        # ERA5 delays depend only on dates and grid, both unchanged from v1
        shutil.copy(MINTPY_V1 / 'inputs/ERA5.h5', era5)
        log.info('copied v1 inputs/ERA5.h5')
    sba('--start', 'modify_network')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--skip-resample', action='store_true')
    ap.add_argument('--skip-prep', action='store_true')
    ap.add_argument('--prepare-only', action='store_true', help='stop after writing the config')
    ap.add_argument('--start', help='resume smallbaselineApp from this step (inputs already loaded)')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    link_v1()
    if not a.skip_resample:
        resample_fill(a.workers)
    if not a.skip_prep:
        prep_fill()
    write_config()
    if a.start:
        sba('--start', a.start)
    elif not a.prepare_only:
        run()
    log.info('MintPy v2 finished')


if __name__ == '__main__':
    main()
