"""
submit_hyp3_isce_burst.py
─────────────────────────────────────────────────────────────────────────────
Resubmit the 221 failed INSAR_GAMMA jobs as INSAR_ISCE_MULTI_BURST jobs.

WHY THIS IS NEEDED
------------------
The original INSAR_GAMMA workflow uses GAMMA's amplitude cross-correlation
to coregister image pairs. In tropical coastal Ghana, dense vegetation and
high atmospheric moisture cause this to fail with:
  "offset_pwr: ERROR: no points available for determining average intensity"

INSAR_ISCE_MULTI_BURST uses ISCE2 with Enhanced Spectral Diversity (ESD)
coregistration, which relies on orbital geometry rather than amplitude
matching. This is far more robust in vegetated tropical areas.

WHAT THIS SCRIPT DOES
---------------------
1. Loads all previously submitted HyP3 jobs.
2. Identifies the 221 FAILED jobs and extracts their date pairs.
3. Searches ASF for path-147 burst granules on those dates (cached).
4. Builds reference/secondary burst lists (IW2 subswath, all burst indices).
5. Submits INSAR_ISCE_MULTI_BURST jobs for each matchable pair.
6. Saves a submission log to outputs/tables/hyp3_isce_burst_jobs.csv.

USAGE
-----
  # Dry run (prints what would be submitted, no API calls):
  python scripts/submit_hyp3_isce_burst.py --dry-run

  # Submit up to 50 jobs at a time (default):
  python scripts/submit_hyp3_isce_burst.py --max-jobs 50

  # Submit all 209 matchable pairs:
  python scripts/submit_hyp3_isce_burst.py --max-jobs 999

NOTES
-----
- Each INSAR_ISCE_MULTI_BURST job covers the full IW2 strip over coastal
  Ghana (~13 bursts, ~400 km along-track).
- After jobs complete, run: python scripts/monitor_hyp3_jobs.py --isce
- Then run: python scripts/run_mintpy_sbas.py for time-series inversion.

CRS: EPSG:32630. Output products: same as GAMMA (vert_disp, corr, etc.)
"""

import os
import sys
import time
import argparse
import logging
import pandas as pd
import asf_search as asf
import hyp3_sdk
from pathlib import Path
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from datetime import date, datetime

# ── project root ──────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[1]
TABS  = ROOT / 'outputs' / 'tables'
TABS.mkdir(parents=True, exist_ok=True)

CACHE_FILE = TABS / 'path147_burst_catalogue.csv'
LOG_FILE   = TABS / 'hyp3_isce_burst_jobs.csv'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
BBOX       = 'POLYGON((-3.5 4.0,1.5 4.0,1.5 7.5,-3.5 7.5,-3.5 4.0))'
REL_ORBIT  = 147
SUBSWATH   = 'IW2'          # central subswath — best coastal Ghana coverage
LOOKS      = '20x4'         # consistent with original GAMMA jobs


# ══════════════════════════════════════════════════════════════════════════════
# 1. Fetch burst catalogue (cached)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_burst_catalogue(force_refresh: bool = False) -> pd.DataFrame:
    """Search ASF for all path-147 IW VV burst granules 2015–2024.

    Results are cached in outputs/tables/path147_burst_catalogue.csv.
    Monthly batches avoid the 500-result/query limit.
    """
    if CACHE_FILE.exists() and not force_refresh:
        log.info(f'Loading burst catalogue from cache: {CACHE_FILE}')
        df = pd.read_csv(CACHE_FILE, parse_dates=['date'])
        log.info(f'  {len(df)} granules, {df["date"].nunique()} unique dates')
        return df

    log.info('Searching ASF for path-147 VV burst granules (monthly batches)…')
    all_results = []
    start = date(2015, 1, 1)
    end   = date(2024, 12, 31)
    cur   = start
    while cur <= end:
        nxt = min(cur + relativedelta(months=1), end)
        r = asf.search(
            platform=asf.PLATFORM.SENTINEL1,
            processingLevel='BURST',
            intersectsWith=BBOX,
            start=cur.isoformat(), end=nxt.isoformat(),
            beamMode='IW',
            flightDirection='ASCENDING',
            relativeOrbit=REL_ORBIT,
            polarization='VV',
            maxResults=500,
        )
        all_results.extend(r)
        cur = nxt + relativedelta(days=1)

    rows = []
    for res in all_results:
        p  = res.properties
        bi = p.get('burst') or {}
        rows.append({
            'granule':     p.get('sceneName', ''),
            'date':        p.get('startTime', '')[:10],
            'subswath':    bi.get('subswath', ''),
            'burst_index': bi.get('burstIndex', ''),
            'burst_id':    bi.get('fullBurstID', ''),
        })

    df = pd.DataFrame(rows).drop_duplicates('granule')
    df['date'] = pd.to_datetime(df['date'])
    df.to_csv(CACHE_FILE, index=False)
    log.info(f'Catalogue saved: {len(df)} granules, {df["date"].nunique()} dates → {CACHE_FILE}')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Build date→burst-list lookup
# ══════════════════════════════════════════════════════════════════════════════

def build_date_lookup(catalogue: pd.DataFrame) -> dict:
    """Return {date: [granule_name, ...]} for the target subswath, sorted by
    burst_index so reference and secondary lists are consistently ordered."""
    sub = catalogue[catalogue['subswath'] == SUBSWATH].copy()
    sub = sub.sort_values(['date', 'burst_index'])
    lookup = defaultdict(list)
    for _, row in sub.iterrows():
        lookup[row['date'].date()].append(row['granule'])
    return dict(lookup)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Get failed date pairs from existing HyP3 account
# ══════════════════════════════════════════════════════════════════════════════

def get_failed_pairs(hyp3_client) -> pd.DataFrame:
    """Return a DataFrame of failed INSAR_GAMMA jobs with parsed date pairs."""
    log.info('Fetching job history from HyP3…')
    jobs = hyp3_client.find_jobs()
    rows = []
    for j in jobs:
        if j.status_code != 'FAILED':
            continue
        grans = j.job_parameters.get('granules', [])
        if len(grans) != 2:
            continue
        # SLC granule names: S1A_IW_SLC__1SDV_YYYYMMDDTHHMMSS_...
        d1 = grans[0][17:25]
        d2 = grans[1][17:25]
        rows.append({'date1': d1, 'date2': d2,
                     'orig_g1': grans[0], 'orig_g2': grans[1],
                     'orig_name': j.name})
    df = pd.DataFrame(rows)
    df['date1'] = pd.to_datetime(df['date1'], format='%Y%m%d').dt.date
    df['date2'] = pd.to_datetime(df['date2'], format='%Y%m%d').dt.date
    # deduplicate: keep first occurrence for each date pair
    df = df.drop_duplicates(['date1', 'date2'])
    log.info(f'  {len(df)} unique failed date pairs')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Match burst granules to date pairs
# ══════════════════════════════════════════════════════════════════════════════

def match_burst_pairs(failed_df: pd.DataFrame,
                      date_lookup: dict) -> list[dict]:
    """
    For each failed date pair, find matching burst granule lists.

    Matching rule: for each burst_index present in BOTH dates (using the
    burst_id prefix), include both granules in the reference/secondary lists.
    This ensures spatial consistency required by ISCE2 multi-burst processing.
    """
    pairs = []
    skipped = 0
    for _, row in failed_df.iterrows():
        d1, d2 = row['date1'], row['date2']
        bursts1 = date_lookup.get(d1, [])
        bursts2 = date_lookup.get(d2, [])

        if not bursts1 or not bursts2:
            skipped += 1
            continue

        # Extract the numeric burst ID (S1_NNNNNN_IW2_...) to match across dates
        def burst_number(g):
            parts = g.split('_')
            return parts[1] if len(parts) > 1 else ''

        ids1 = {burst_number(g): g for g in bursts1}
        ids2 = {burst_number(g): g for g in bursts2}
        common_ids = sorted(set(ids1) & set(ids2))

        if not common_ids:
            skipped += 1
            continue

        ref = [ids1[i] for i in common_ids]
        sec = [ids2[i] for i in common_ids]

        # HyP3 INSAR_ISCE_MULTI_BURST hard-limits: max 15 burst pairs per job.
        # When >15 IW2 bursts span the full swath, trim to the central 15 so
        # we keep the bursts over coastal Ghana (4–7.5 °N, southern portion
        # of path-147).  Burst IDs are sorted numerically, so the lowest
        # indices correspond to the most-southerly bursts.
        MAX_BURSTS = 15
        if len(ref) > MAX_BURSTS:
            ref = ref[:MAX_BURSTS]
            sec = sec[:MAX_BURSTS]

        pairs.append({
            'date1':     d1,
            'date2':     d2,
            'n_bursts':  len(ref),
            'reference': ref,
            'secondary': sec,
            'job_name':  f'ghana_isce_burst_{d1}_{d2}',
        })

    log.info(f'Matchable pairs: {len(pairs)}, skipped (no burst coverage): {skipped}')
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# 5. Submit jobs
# ══════════════════════════════════════════════════════════════════════════════

def submit_pairs(pairs: list[dict], hyp3_client,
                 max_jobs: int = 50, dry_run: bool = False) -> pd.DataFrame:
    """Submit INSAR_ISCE_MULTI_BURST jobs. Returns a log DataFrame."""

    # Avoid resubmitting pairs that already SUCCEEDED in a previous run.
    # Pairs that previously errored (job_id == 'ERROR' / 'SUBMISSION_ERROR')
    # are NOT skipped so that the fixed burst-cap can retry them.
    already_ok = set()
    if LOG_FILE.exists():
        prev = pd.read_csv(LOG_FILE)
        ok_rows = prev[~prev['job_id'].isin(['ERROR', 'SUBMISSION_ERROR', 'DRY_RUN'])]
        already_ok = set(zip(ok_rows['date1'].astype(str), ok_rows['date2'].astype(str)))
        log.info(f'  Skipping {len(already_ok)} pairs already successfully submitted')

    to_submit = [p for p in pairs
                 if (str(p['date1']), str(p['date2'])) not in already_ok]

    if max_jobs < len(to_submit):
        log.info(f'Capping at {max_jobs} of {len(to_submit)} available pairs')
        to_submit = to_submit[:max_jobs]

    log.info(f'{"DRY RUN — " if dry_run else ""}Submitting {len(to_submit)} '
             f'INSAR_ISCE_MULTI_BURST jobs…')

    log_rows = []
    for i, p in enumerate(to_submit, 1):
        if dry_run:
            log.info(f'  [{i:3d}/{len(to_submit)}] DRY RUN  '
                     f'{p["date1"]} → {p["date2"]}  '
                     f'({p["n_bursts"]} bursts)')
            log_rows.append({'date1': p['date1'], 'date2': p['date2'],
                              'n_bursts': p['n_bursts'],
                              'job_id': 'DRY_RUN', 'status': 'DRY_RUN',
                              'job_name': p['job_name']})
        else:
            try:
                batch = hyp3_client.submit_insar_isce_multi_burst_job(
                    reference=p['reference'],
                    secondary=p['secondary'],
                    name=p['job_name'],
                    apply_water_mask=True,
                    looks=LOOKS,
                )
                job = list(batch)[0]
                log.info(f'  [{i:3d}/{len(to_submit)}] SUBMITTED  '
                         f'{p["date1"]} → {p["date2"]}  '
                         f'id={job.job_id}  n_bursts={p["n_bursts"]}')
                log_rows.append({
                    'date1':    p['date1'],
                    'date2':    p['date2'],
                    'n_bursts': p['n_bursts'],
                    'job_id':   job.job_id,
                    'status':   job.status_code,
                    'job_name': p['job_name'],
                })
                # Brief pause to avoid rate-limiting
                if i % 10 == 0:
                    time.sleep(2)
            except Exception as e:
                log.warning(f'  [{i:3d}] FAILED to submit {p["date1"]}→{p["date2"]}: {e}')
                log_rows.append({
                    'date1': p['date1'], 'date2': p['date2'],
                    'n_bursts': p['n_bursts'],
                    'job_id': 'ERROR', 'status': 'SUBMISSION_ERROR',
                    'job_name': p['job_name'],
                })

    df_log = pd.DataFrame(log_rows)

    if not dry_run and len(df_log) > 0:
        # Merge with existing log: keep previously-successful rows; replace
        # errored rows that were just retried (and may now have succeeded).
        if LOG_FILE.exists():
            prev = pd.read_csv(LOG_FILE)
            retried_keys = set(zip(df_log['date1'].astype(str),
                                   df_log['date2'].astype(str)))
            # Drop old rows for pairs we just re-attempted
            prev = prev[~prev.apply(
                lambda r: (str(r['date1']), str(r['date2'])) in retried_keys,
                axis=1
            )]
            df_log = pd.concat([prev, df_log], ignore_index=True)
        df_log.to_csv(LOG_FILE, index=False)
        log.info(f'Log saved → {LOG_FILE}')

    return df_log


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Resubmit failed INSAR_GAMMA jobs as INSAR_ISCE_MULTI_BURST')
    parser.add_argument('--dry-run',   action='store_true',
                        help='Print what would be submitted without calling HyP3')
    parser.add_argument('--max-jobs',  type=int, default=50,
                        help='Max jobs to submit per run (default 50)')
    parser.add_argument('--refresh-cache', action='store_true',
                        help='Force re-download of burst catalogue from ASF')
    args = parser.parse_args()

    # Connect to HyP3
    hyp3 = hyp3_sdk.HyP3()

    # Step 1: burst catalogue
    catalogue = fetch_burst_catalogue(force_refresh=args.refresh_cache)

    # Step 2: date lookup
    date_lookup = build_date_lookup(catalogue)
    log.info(f'Date lookup built: {len(date_lookup)} dates with IW2 bursts')

    # Step 3: failed pairs
    failed_df = get_failed_pairs(hyp3)

    # Step 4: match bursts
    pairs = match_burst_pairs(failed_df, date_lookup)

    # Step 5: submit
    log_df = submit_pairs(pairs, hyp3,
                          max_jobs=args.max_jobs,
                          dry_run=args.dry_run)

    # Summary
    if not log_df.empty:
        print('\n' + '=' * 60)
        print('SUBMISSION SUMMARY')
        print('=' * 60)
        print(f"  Pairs attempted:  {len(log_df)}")
        submitted = log_df[log_df['status'] != 'SUBMISSION_ERROR']
        print(f"  Successfully sent: {len(submitted)}")
        print(f"  Errors:            {len(log_df) - len(submitted)}")
        if not args.dry_run:
            print(f"\n  Monitor progress:  python scripts/monitor_hyp3_jobs.py")
            print(f"  After completion:  python scripts/run_mintpy_sbas.py")


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parents[1])
    main()
