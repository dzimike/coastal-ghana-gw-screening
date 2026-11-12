"""Poll the network-fill HyP3 jobs, download and extract finished products, exit when all are done."""
import time
import zipfile
from pathlib import Path

import hyp3_sdk
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/raw/sentinel1/hyp3_isce_products'
LOG = ROOT / 'outputs/tables/hyp3_isce_fill_jobs.csv'
POLL = 600

while True:
    try:
        hyp3 = hyp3_sdk.HyP3()
        jobs = [j for j in hyp3.find_jobs() if (j.name or '').startswith('ghana_isce_fill_')]
    except Exception as e:
        print('poll error:', e, flush=True)
        time.sleep(POLL)
        continue
    status = {j.job_id: j.status_code for j in jobs}
    for j in jobs:
        if j.status_code != 'SUCCEEDED':
            continue
        name = Path(j.files[0]['filename']).stem
        if (DEST / name).exists():
            continue
        try:
            zp = j.download_files(DEST)[0]
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(DEST)
            print('downloaded', name, flush=True)
        except Exception as e:
            print('download error', name, e, flush=True)
    log = pd.read_csv(LOG)
    log['status'] = log.job_id.map(status).fillna(log.status)
    log.to_csv(LOG, index=False)
    counts = pd.Series(status).value_counts().to_dict()
    print(time.strftime('%H:%M'), counts, flush=True)
    if not any(s in ('PENDING', 'RUNNING') for s in status.values()):
        missing = [j for j in jobs if j.status_code == 'SUCCEEDED'
                   and not (DEST / Path(j.files[0]['filename']).stem).exists()]
        if not missing:
            print('ALL DONE', counts, flush=True)
            break
    time.sleep(POLL)
