"""Submit HyP3 INSAR_ISCE_MULTI_BURST jobs that connect the fragmented SBAS network.

Adds (1) one bridging pair per gap between the disconnected temporal subsets of the existing
159-pair network and (2) skip-one pairs (date_i, date_i+2) that create closed loops.
Pairs that already succeeded are skipped. Bursts: the standard set 313622-313636 (IW2) when
both dates have it, otherwise the common bursts capped at 15 (as in submit_hyp3_isce_burst.py).

Usage:
  python scripts/submit_hyp3_network_fill.py --dry-run
  python scripts/submit_hyp3_network_fill.py
"""
import argparse
import time
from pathlib import Path

import hyp3_sdk
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABS = ROOT / 'outputs/tables'
CATALOGUE = TABS / 'path147_burst_catalogue.csv'
PAIRS_DONE = TABS / 'revision_checks/sbas_pair_list.csv'
COMPONENTS = TABS / 'revision_checks/sbas_temporal_components.csv'
LOG = TABS / 'hyp3_isce_fill_jobs.csv'
STANDARD = [str(313622 + k) for k in range(15)]
LOOKS = '20x4'


def cost(n):
    return 10 if n >= 13 else 5 if n >= 5 else 1


def burst_lookup():
    c = pd.read_csv(CATALOGUE, dtype={'burst_index': str})
    c = c[c.subswath == 'IW2'].copy()
    c['bid'] = c.granule.str.split('_').str[1]
    c['d'] = pd.to_datetime(c.date).dt.strftime('%Y%m%d')
    return {d: dict(zip(g.bid, g.granule)) for d, g in c.groupby('d')}


def candidate_pairs():
    done = pd.read_csv(PAIRS_DONE, dtype=str)
    dates = sorted(set(done.date1) | set(done.date2))
    have = set(zip(done.date1, done.date2))
    comps = pd.read_csv(COMPONENTS, dtype=str).sort_values('first_date')
    cand = []
    for (_, a), (_, b) in zip(comps.iloc[:-1].iterrows(), comps.iloc[1:].iterrows()):
        cand.append((a.last_date, b.first_date, 'bridge'))
    for i in range(len(dates) - 2):
        cand.append((dates[i], dates[i + 2], 'skip1'))
    seen, out = set(), []
    for d1, d2, kind in cand:
        if (d1, d2) in have or (d1, d2) in seen:
            continue
        seen.add((d1, d2))
        out.append((d1, d2, kind))
    return out


def build_jobs():
    lk = burst_lookup()
    jobs, skipped = [], []
    for d1, d2, kind in candidate_pairs():
        b1, b2 = lk.get(d1, {}), lk.get(d2, {})
        ids = STANDARD if all(i in b1 and i in b2 for i in STANDARD) else sorted(set(b1) & set(b2))[:15]
        if not ids:
            skipped.append((d1, d2, kind))
            continue
        jobs.append(dict(date1=d1, date2=d2, kind=kind, n_bursts=len(ids),
                         reference=[b1[i] for i in ids], secondary=[b2[i] for i in ids],
                         job_name=f'ghana_isce_fill_{d1}_{d2}'))
    return jobs, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    jobs, skipped = build_jobs()
    credits = sum(cost(j['n_bursts']) for j in jobs)
    kinds = pd.Series([j['kind'] for j in jobs]).value_counts().to_dict()
    nb = pd.Series([j['n_bursts'] for j in jobs]).value_counts().to_dict()
    print(f'jobs {len(jobs)} {kinds}; bursts {nb}; credits {credits}; skipped (no common bursts) {len(skipped)}')
    if args.dry_run:
        pd.DataFrame(jobs).drop(columns=['reference', 'secondary']).to_csv(TABS / 'hyp3_isce_fill_plan.csv', index=False)
        return
    hyp3 = hyp3_sdk.HyP3()
    rows = []
    for k, j in enumerate(jobs, 1):
        try:
            job = list(hyp3.submit_insar_isce_multi_burst_job(
                reference=j['reference'], secondary=j['secondary'], name=j['job_name'],
                apply_water_mask=True, looks=LOOKS))[0]
            rows.append(dict(date1=j['date1'], date2=j['date2'], kind=j['kind'], n_bursts=j['n_bursts'],
                             job_id=job.job_id, status=job.status_code, job_name=j['job_name']))
        except Exception as e:
            rows.append(dict(date1=j['date1'], date2=j['date2'], kind=j['kind'], n_bursts=j['n_bursts'],
                             job_id='ERROR', status=f'SUBMISSION_ERROR: {e}'[:200], job_name=j['job_name']))
        if k % 10 == 0:
            pd.DataFrame(rows).to_csv(LOG, index=False)
            time.sleep(2)
    df = pd.DataFrame(rows)
    df.to_csv(LOG, index=False)
    print(df.status.str.slice(0, 16).value_counts().to_dict())


if __name__ == '__main__':
    main()
