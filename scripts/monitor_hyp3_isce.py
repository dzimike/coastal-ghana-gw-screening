"""
monitor_hyp3_isce.py
─────────────────────────────────────────────────────────────────────────────
Monitor INSAR_ISCE_MULTI_BURST job progress, download completed products,
and trigger MintPy SBAS when sufficient products are available.

USAGE
-----
  # Check status only:
  python scripts/monitor_hyp3_isce.py --status

  # Download all completed products:
  python scripts/monitor_hyp3_isce.py --download

  # Check + download + run MintPy if ≥50 products ready:
  python scripts/monitor_hyp3_isce.py --download --auto-mintpy --min-products 50

  # Full pipeline (continuous polling):
  python scripts/monitor_hyp3_isce.py --download --auto-mintpy --poll 600

OUTPUTS
-------
  data/raw/sentinel1/hyp3_isce_products/   — downloaded zip + extracted dirs
  outputs/tables/hyp3_isce_burst_jobs.csv  — updated with final statuses
  data/processed/subsidence/mintpy/        — MintPy outputs (when auto-mintpy runs)
"""

import os
import sys
import time
import zipfile
import argparse
import logging
from pathlib import Path

import hyp3_sdk
import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
ISCE_DIR = ROOT / 'data' / 'raw' / 'sentinel1' / 'hyp3_isce_products'
TABS     = ROOT / 'outputs' / 'tables'
LOG_FILE = TABS / 'hyp3_isce_burst_jobs.csv'
ISCE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Fetch live statuses
# ══════════════════════════════════════════════════════════════════════════════

def fetch_status(hyp3_client) -> pd.DataFrame:
    """Return updated DataFrame with current job statuses."""
    log.info('Fetching live job statuses from HyP3…')
    df = pd.read_csv(LOG_FILE)
    isce_submitted = df[~df['job_id'].isin(['ERROR', 'SUBMISSION_ERROR', 'DRY_RUN'])]

    id_map = {row['job_id']: i for i, row in isce_submitted.iterrows()}
    all_jobs = hyp3_client.find_jobs()

    updated = 0
    for j in all_jobs:
        if j.job_id in id_map:
            idx = id_map[j.job_id]
            old = df.at[idx, 'status']
            new = j.status_code
            if old != new:
                df.at[idx, 'status'] = new
                # Store download URL when succeeded
                if new == 'SUCCEEDED' and j.files:
                    df.at[idx, 'download_url'] = j.files[0]['url']
                updated += 1

    df.to_csv(LOG_FILE, index=False)
    if updated:
        log.info(f'  Updated {updated} job statuses')

    counts = df['status'].value_counts()
    return df, counts


# ══════════════════════════════════════════════════════════════════════════════
# 2. Download completed products
# ══════════════════════════════════════════════════════════════════════════════

def download_products(hyp3_client, df: pd.DataFrame) -> int:
    """Download SUCCEEDED products not yet on disk. Returns count downloaded."""
    succeeded = df[df['status'] == 'SUCCEEDED'].copy()

    downloaded = 0
    for _, row in succeeded.iterrows():
        job_name = row['job_name']
        # Expected zip name follows HyP3 naming: job_name.zip or product dir
        # Check if already downloaded (any zip or dir with job_id prefix)
        existing = list(ISCE_DIR.glob(f'*{row["job_id"][:8]}*.zip'))
        if existing:
            continue

        # Re-fetch job to get download URL
        try:
            job = hyp3_client.get_job_by_id(row['job_id'])
        except Exception as e:
            log.warning(f'Could not fetch job {row["job_id"]}: {e}')
            continue

        if not job.files:
            continue

        url = job.files[0]['url']
        filename = url.split('/')[-1].split('?')[0]
        dest = ISCE_DIR / filename

        if dest.exists():
            log.info(f'  Already on disk: {filename}')
        else:
            log.info(f'  Downloading {filename}…')
            try:
                import urllib.request
                urllib.request.urlretrieve(url, dest)
                log.info(f'  ✓ {filename} ({dest.stat().st_size / 1e6:.0f} MB)')
            except Exception as e:
                log.warning(f'  Download failed for {filename}: {e}')
                continue

        # Extract zip
        extract_dir = ISCE_DIR / dest.stem
        if not extract_dir.exists():
            log.info(f'  Extracting {filename}…')
            try:
                with zipfile.ZipFile(dest, 'r') as zf:
                    zf.extractall(ISCE_DIR)
            except Exception as e:
                log.warning(f'  Extraction failed: {e}')

        downloaded += 1

    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# 3. Count downloaded products
# ══════════════════════════════════════════════════════════════════════════════

def count_products() -> int:
    """Count extracted ISCE product directories (each has *_unw_phase.tif)."""
    return len(list(ISCE_DIR.rglob('*_unw_phase.tif')))


# ══════════════════════════════════════════════════════════════════════════════
# 4. Run MintPy SBAS
# ══════════════════════════════════════════════════════════════════════════════

def trigger_mintpy():
    """Launch run_mintpy_sbas.py with the ISCE product directory."""
    import subprocess
    log.info('Triggering MintPy SBAS on ISCE products…')
    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'run_mintpy_sbas.py'),
        '--products-dir', str(ISCE_DIR),
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode == 0:
        log.info('MintPy SBAS completed ✓')
    else:
        log.error('MintPy SBAS failed — check output above')
    return result.returncode == 0


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Monitor HyP3 ISCE burst jobs and download results')
    parser.add_argument('--status',       action='store_true',
                        help='Print current status summary and exit')
    parser.add_argument('--download',     action='store_true',
                        help='Download completed products')
    parser.add_argument('--auto-mintpy',  action='store_true',
                        help='Run MintPy SBAS when ≥ min-products are on disk')
    parser.add_argument('--min-products', type=int, default=50,
                        help='Minimum products before triggering MintPy (default 50)')
    parser.add_argument('--poll',         type=int, default=0,
                        help='Poll interval in seconds (0 = run once and exit)')
    args = parser.parse_args()

    hyp3 = hyp3_sdk.HyP3()

    while True:
        df, counts = fetch_status(hyp3)

        print('\n' + '═' * 50)
        print(' HyP3 ISCE Burst Job Status')
        print('═' * 50)
        for status, n in counts.items():
            print(f'  {status:<20s}: {n:4d}')
        print('─' * 50)

        n_on_disk = count_products()
        print(f'  Products on disk         : {n_on_disk:4d}')
        print('═' * 50 + '\n')

        if args.status:
            break

        if args.download:
            n = download_products(hyp3, df)
            if n:
                log.info(f'Downloaded {n} new products. Total on disk: {count_products()}')

        if args.auto_mintpy:
            n_ready = count_products()
            if n_ready >= args.min_products:
                log.info(f'{n_ready} products ready — triggering MintPy SBAS…')
                trigger_mintpy()
                break

            remaining = counts.get('PENDING', 0) + counts.get('RUNNING', 0)
            if remaining == 0:
                log.info('All jobs finished. Running MintPy with available products…')
                trigger_mintpy()
                break

        if args.poll <= 0:
            break

        log.info(f'Sleeping {args.poll}s before next check…')
        time.sleep(args.poll)


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
