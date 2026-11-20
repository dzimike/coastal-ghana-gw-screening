"""Download the inputs added for Paper 3 (v2).

- CSR GRACE/GRACE-FO RL06.3 mascons (all corrections) and the CSR land mask (UT Austin, public)
- GSFC GRACE/GRACE-FO RL06v2.0 mascons, half-degree grid (NASA GSFC, public)
- GLDAS-2.1 monthly 1-degree Noah, CLSM and VIC, January 2002 - December 2025 (GES DISC, Earthdata login in ~/.netrc)
- G-REALM 10-day Lake Volta levels (target 000393; NASA GSFC Global Water Monitor, public)

Files already present are skipped. Usage: python scripts/paper3/download_paper3_inputs.py
"""
from pathlib import Path
import urllib.request

import earthaccess

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw'
UA = {'User-Agent': 'Mozilla/5.0 (research data download)'}

URLS = {
    RAW / 'grace_ensemble/csr': [
        'https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc',
        'https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc',
    ],
    RAW / 'grace_ensemble/gsfc': [
        'https://earth.gsfc.nasa.gov/sites/default/files/geo/GSFC.glb_.200204_202606_RL06v2.0_OBP-ICE6GD_HALFDEGREE.nc',
    ],
    RAW / 'lake_volta/gwm': [
        'https://earth.gsfc.nasa.gov/gwm/timeseries/lake000393.10d.2.txt',
    ],
}
GLDAS = {'noah': 'GLDAS_NOAH10_M', 'clsm': 'GLDAS_CLSM10_M', 'vic': 'GLDAS_VIC10_M'}


def fetch(url, dest):
    out = dest / url.rsplit('/', 1)[1]
    if out.exists() and out.stat().st_size > 0:
        print('present', out.name)
        return
    dest.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.part')
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(tmp, 'wb') as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(out)
    print('downloaded', out.name, out.stat().st_size // 1_000_000, 'MB')


for dest, urls in URLS.items():
    for u in urls:
        fetch(u, dest)

earthaccess.login(strategy='netrc')
for key, sn in GLDAS.items():
    dest = RAW / 'gldas_1deg' / key
    dest.mkdir(parents=True, exist_ok=True)
    granules = earthaccess.search_data(short_name=sn, version='2.1', temporal=('2002-01-01', '2025-12-31'))
    todo = [g for g in granules if not (dest / g.data_links()[0].rsplit('/', 1)[1]).exists()]
    print(sn, len(granules), 'granules,', len(todo), 'to download')
    if todo:
        earthaccess.download(todo, str(dest), threads=4)
