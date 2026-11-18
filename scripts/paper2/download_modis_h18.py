"""Download MODIS MOD11A2 h18v08 for eastern Ghana coverage."""
import earthaccess
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT  = BASE / "data/raw/modis_lst"
OUT.mkdir(exist_ok=True)

earthaccess.login(strategy="netrc")

results = earthaccess.search_data(
    short_name="MOD11A2", version="061",
    temporal=("2000-01-01", "2023-12-31"),
    bounding_box=(0.0, 4.4, 1.8, 7.2),
)
h18 = [r for r in results if "h18v08" in r["umm"]["GranuleUR"]]
print(f"h18v08 granules: {len(h18)}")

already = {f.name for f in OUT.glob("*.hdf")}
todo = [r for r in h18 if r["umm"]["GranuleUR"].split("/")[-1] + ".hdf" not in already]
# Some granule IDs don't end in .hdf — also check by stem
stems = {f.stem for f in OUT.glob("*.hdf")}
todo = [r for r in h18 if r["umm"]["GranuleUR"] not in stems]
print(f"To download: {len(todo)}")

if todo:
    downloaded = earthaccess.download(todo, str(OUT))
    print(f"Downloaded: {len(downloaded)} h18v08 files")
print("Done.")
