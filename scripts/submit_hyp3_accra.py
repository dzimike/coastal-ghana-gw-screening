#!/usr/bin/env python
"""Submit S1 InSAR jobs to ASF HyP3 for Accra hotspot.
pip install hyp3_sdk  then  python scripts/submit_hyp3_accra.py
"""
import hyp3_sdk as hyp3
import pandas as pd
from pathlib import Path

pairs = pd.read_csv('outputs/tables/s1_insar_recommended_pairs_path147.csv')
pairs = pairs.head(20)

api = hyp3.HyP3()
jobs = []
for _, row in pairs.iterrows():
    job = api.submit_insar_job(
        granules=[row['master'], row['slave']],
        name=f"accra_{row['master_date']}_{row['slave_date']}",
        include_dem=True, include_displacement_maps=True, looks='20x4',
    )
    jobs.append(job)
    print(f"Submitted: {row['master_date']} to {row['slave_date']}")
print(f"Total jobs: {len(jobs)}")
print("Check: https://hyp3.asf.alaska.edu")