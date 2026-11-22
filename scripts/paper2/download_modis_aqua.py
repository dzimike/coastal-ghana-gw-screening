"""Download Aqua MODIS MYD11A2 v6.1 (tiles h17v08, h18v08) for the December-February dry seasons.

Aqua's first complete dry season is 2003 (December 2002 - February 2003). The files are used by
build_paper2_erc_checks.py as an independent test of the Terra LST trends. Earthdata login in ~/.netrc.
"""
from pathlib import Path

import earthaccess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/raw/modis_lst_aqua'
OUT.mkdir(parents=True, exist_ok=True)
TILES = ('h17v08', 'h18v08')
SEASONS = range(2003, 2024)

earthaccess.login(strategy='netrc')
urls = []
for y in SEASONS:
    res = earthaccess.search_data(short_name='MYD11A2', version='061', temporal=(f'{y - 1}-12-01', f'{y}-02-28'),
                                  bounding_box=(-3.4, 4.4, 1.4, 7.2))
    found = [u for r in res for u in r.data_links() if u.endswith('.hdf') and any(t in u for t in TILES)]
    urls += found
    print(f'season {y}: {len(found)} granules', flush=True)
urls = sorted(set(urls))
have = {f.name for f in OUT.glob('MYD11A2*.hdf') if f.stat().st_size > 0}
todo = [u for u in urls if u.rsplit('/', 1)[-1] not in have]
print(f'granules {len(urls)}; already present {len(urls) - len(todo)}; to download {len(todo)}', flush=True)
if todo:
    earthaccess.download(todo, str(OUT), threads=6)
final = sorted(OUT.glob('MYD11A2*.hdf'))
print(f'files on disk {len(final)}; {sum(f.stat().st_size for f in final) / 1e9:.2f} GB', flush=True)
missing = [u.rsplit('/', 1)[-1] for u in urls if not (OUT / u.rsplit('/', 1)[-1]).exists()]
print(f'missing after download: {len(missing)}', missing[:5], flush=True)
