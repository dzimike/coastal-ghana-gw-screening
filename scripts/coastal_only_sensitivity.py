"""Sensitivity of the harmonised CGSRI and the UGPI-SH co-location to the study-area definition.

The 453-cell grid was buffered from the national boundary south of ~6.5°N, so it includes inland arms
along the Ivorian and Togolese borders. This script measures each cell centroid's distance to the sea
(Natural Earth 10 m coastline), re-ranks the indices within the 302 cells <= 50 km from the sea, and
repeats the Fisher co-location test. Outputs: outputs/tables/revision_checks_v2/coastal_only_sensitivity.csv
and cell_distance_to_sea.csv.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
from scipy import stats
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'data/processed/risk_index/cgsri_grid10km_v2.gpkg'
NE = Path.home() / '.local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_coastline.shp'
OUT = ROOT / 'outputs/tables/revision_checks_v2'

g = gpd.read_file(V2)
coast = gpd.read_file(NE).clip(box(-3.4, 4.4, 1.4, 6.4)).to_crs(g.crs).union_all()
g['dist_sea_km'] = g.geometry.centroid.distance(coast) / 1000
g['within_50km_of_sea'] = g.dist_sea_km <= 50
c = g.within_50km_of_sea
rows = [dict(item='cells > 50 km from sea', value=int((~c).sum())),
        dict(item='population 2020 in cells > 50 km (M)', value=round(g.population_2020[~c].sum() / 1e6, 2)),
        dict(item='built-up 2020 share in cells > 50 km', value=round(g.builtup_m2_2020[~c].sum() / g.builtup_m2_2020.sum(), 3))]
for col in ['cgsri', 'cgsri5']:
    vh = g[col].rank(method='first') > len(g) - 91
    s = g[c]
    k = round(len(s) / 5)
    vh2 = s[col].rank(method='first') > len(s) - k
    rows += [dict(item=f'{col}: Very High cells > 50 km from sea', value=int((vh & ~c).sum())),
             dict(item=f'{col}: coastal-only top-quintile cells', value=int(vh2.sum())),
             dict(item=f'{col}: of which in original Very High', value=int((vh2 & vh[c]).sum())),
             dict(item=f'{col}: coastal-only top-quintile population (M)', value=round(s.population_2020[vh2].sum() / 1e6, 2))]
s = g[c]
for q in (0.75, 0.90):
    a = s.ugpi_urban_only >= s.ugpi_urban_only.quantile(q)
    b = s.sub_hazard >= s.sub_hazard.quantile(q)
    orr, p = stats.fisher_exact(pd.crosstab(a, b).values)
    rows += [dict(item=f'Q{int(q * 100)} co-location, coastal-only: observed', value=int((a & b).sum())),
             dict(item=f'Q{int(q * 100)} co-location, coastal-only: expected', value=round(a.sum() * b.sum() / len(s), 1)),
             dict(item=f'Q{int(q * 100)} co-location, coastal-only: Fisher p', value=round(p, 3))]
res = pd.DataFrame(rows)
res.to_csv(OUT / 'coastal_only_sensitivity.csv', index=False)
g[['cell_id', 'dist_sea_km', 'within_50km_of_sea']].round(2).to_csv(OUT / 'cell_distance_to_sea.csv', index=False)
print(res.to_string(index=False))
