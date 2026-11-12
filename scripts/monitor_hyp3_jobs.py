#!/usr/bin/env python
"""
Monitor HyP3 InSAR jobs and download products when complete.
Run with: python scripts/monitor_hyp3_jobs.py
Can also be run periodically (e.g. every few hours) to check progress.
"""
import time
import pandas as pd
import hyp3_sdk
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
TABS      = ROOT / 'outputs' / 'tables'
PROD      = ROOT / 'data' / 'raw' / 'sentinel1' / 'hyp3_products'
jobs_file = TABS / 'hyp3_jobs_submitted.csv'
PROD.mkdir(parents=True, exist_ok=True)

if not jobs_file.exists():
    print('No jobs file found. Run notebook 06 first.')
    exit(1)

df = pd.read_csv(jobs_file)
print(f'Jobs in log: {len(df)}')
print(df['status'].value_counts().to_string())
print()

hyp3 = hyp3_sdk.HyP3()

# Update all pending/running statuses
pending = df[df['status'].isin(['PENDING','RUNNING'])]['job_id'].tolist()
print(f'Checking {len(pending)} pending/running jobs...')
for jid in pending:
    try:
        job = hyp3.get_job_by_id(jid)
        df.loc[df['job_id']==jid, 'status'] = job.status_code
        if job.status_code not in ('PENDING','RUNNING'):
            print(f'  {job.status_code}: {str(jid)[:8]}...')
    except Exception as e:
        print(f'  WARN {str(jid)[:8]}: {e}')

df.to_csv(jobs_file, index=False)
print()
print('Updated status:')
print(df['status'].value_counts().to_string())

# Download completed products
succeeded = df[df['status']=='SUCCEEDED']['job_id'].tolist()
already = len(list(PROD.glob('*.zip')))
print(f'\nSucceeded: {len(succeeded)}  |  ZIPs already downloaded: {already}')

for jid in succeeded:
    if list(PROD.glob('*' + str(jid)[:8] + '*.zip')):
        continue
    try:
        job = hyp3.get_job_by_id(jid)
        job.download_files(PROD)
        print(f'  Downloaded: {str(jid)[:8]}')
    except Exception as e:
        print(f'  FAILED {str(jid)[:8]}: {e}')

total_zips = len(list(PROD.glob('*.zip')))
print(f'\nTotal ZIPs: {total_zips}  |  Run again in ~1 hour to check for more completions.')
print('Once >= 5 products downloaded, run notebook 06 cells 6-7 to compute velocity map.')
