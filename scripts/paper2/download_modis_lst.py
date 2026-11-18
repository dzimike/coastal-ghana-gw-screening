"""
Download MODIS MOD11A2 v6.1 (8-day LST) for Ghana coastal belt.
Targets tile h17v08 only (covers all of Ghana).
Downloads 2000-2023 to data/raw/modis_lst/.
"""
import earthaccess
import os
from pathlib import Path
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
OUT  = BASE / "data/raw/modis_lst"
OUT.mkdir(parents=True, exist_ok=True)

# ── auth ───────────────────────────────────────────────────────────────────
earthaccess.login(strategy="netrc")

# ── search h17v08 only ─────────────────────────────────────────────────────
# Ghana coastal belt: lon -3.8 to 1.6, lat 4.4 to 7.2
# MOD11A2 covers whole tile; we filter by tile ID post-search
results = earthaccess.search_data(
    short_name="MOD11A2",
    version="061",
    temporal=("2000-01-01", "2023-12-31"),
    bounding_box=(-3.8, 4.4, 1.6, 7.2),
)

# Filter to h17v08 only
h17_results = [r for r in results if "h17v08" in r["umm"]["GranuleUR"]]
print(f"Total results: {len(results)}, h17v08 only: {len(h17_results)}")

# Skip already downloaded
already = {f.name for f in OUT.glob("*.hdf")}
to_download = [r for r in h17_results if Path(r["umm"]["GranuleUR"]).stem + ".hdf" not in already]
print(f"Already downloaded: {len(already)}, remaining: {len(to_download)}")

if to_download:
    print(f"Starting download of {len(to_download)} granules...")
    downloaded = earthaccess.download(to_download, str(OUT))
    print(f"Downloaded: {len(downloaded)} files")
else:
    print("All granules already present.")

print("Done.")
