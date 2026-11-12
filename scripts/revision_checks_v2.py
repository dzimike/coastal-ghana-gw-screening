"""Diagnostics for the harmonised CGSRI (v2). Writes outputs/tables/revision_checks_v2/."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/tables/revision_checks_v2'
OUT.mkdir(parents=True, exist_ok=True)
g = gpd.read_file(ROOT / 'data/processed/risk_index/cgsri_grid10km_v2.gpkg')
g1 = gpd.read_file(ROOT / 'data/processed/risk_index/cgsri_grid10km.gpkg')
N = len(g)
lines = []
rng = np.random.default_rng(20261112)


def log(s=''):
    print(s)
    lines.append(str(s))


def top(s, n=91):
    r = s.rank(method='first')
    return set(r[r > len(s) - n].index)


def rho(a, b):
    return spearmanr(a, b)[0]


P, P5 = g['cgsri'], g['cgsri5']
VH, VH5 = top(P), top(P5)
pop = g['population_2020']

log('== 1. v1 -> v2 ==')
v1_4 = (g1.gw_hazard + g1.climate_stress + g1.exposure + g1.vulnerability) / 4
log(f'primary: rho={rho(v1_4, P):.3f}, VH shared {len(top(v1_4) & VH)}/91; '
    f'5comp: rho={rho(g1.cgsri, P5):.3f}, VH shared {len(top(g1.cgsri) & VH5)}/91')

log('\n== 2. Results ==')
for name, s, vh in [('primary', P, VH), ('5comp', P5, VH5)]:
    hv = set(s.index[s.rank(method="first") > N - 182])
    log(f'{name}: range {s.min():.3f}-{s.max():.3f}, mean {s.mean():.3f}; VH pop {pop[list(vh)].sum()/1e6:.3f}M '
        f'(±25%: {0.75*pop[list(vh)].sum()/1e6:.1f}-{1.25*pop[list(vh)].sum()/1e6:.1f}); '
        f'H+VH pop {pop[list(hv)].sum()/1e6:.2f}M; VH lat {g.loc[list(vh),"centroid_lat"].min():.2f}-'
        f'{g.loc[list(vh),"centroid_lat"].max():.2f}; VH >=6.3N: {(g.loc[list(vh),"centroid_lat"]>=6.3).sum()}')
    bins = [-4, -2.2, -1.6, -1.0, -0.3, 0.3, 1.3]
    lab = ['W of 2.2W (Axim–Ivorian border)', '2.2–1.6W (Dixcove–Sekondi-Takoradi)', '1.6–1.0W (Elmina–Cape Coast–Saltpond)',
           '1.0–0.3W (Apam–Winneba–Kasoa)', '0.3W–0.3E (Accra–Tema)', 'E of 0.3E (Ada–Keta)']
    z = pd.cut(g.loc[list(vh), 'centroid_lon'], bins, labels=lab).value_counts().reindex(lab)
    log('   ' + '; '.join(f'{k}: {v}' for k, v in z.items()))
log(f'primary vs 5comp: rho={rho(P, P5):.3f}, VH shared {len(VH & VH5)}/91')
C = ['climate_stress', 'exposure', 'vulnerability', 'sub_hazard']
for nm, vh in [('primary VH', VH), ('5comp VH', VH5)]:
    log(f'{nm} component means: ' + ', '.join(f'{c} {g.loc[list(vh), c].mean():.2f}' for c in C)
        + ' | all: ' + ', '.join(f'{c} {g[c].mean():.2f}' for c in C))

log('\n== 3. Component importance (Table 4) ==')
rows = []
for c, lbl in [('climate_stress', 'Climate Stress (CS)'), ('exposure', 'Exposure (E)'), ('vulnerability', 'Vulnerability (V)')]:
    others = [x for x in ['climate_stress', 'exposure', 'vulnerability'] if x != c]
    rows.append(dict(component=lbl, rho_primary=round(rho(g[c], P), 2),
                     rho_primary_without=round(rho(g[others].mean(axis=1), P), 2),
                     rho_5comp=round(rho(g[c], P5), 2)))
rows.append(dict(component='Subsidence Hazard (SH)', rho_primary=round(rho(g.sub_hazard, P), 2),
                 rho_primary_without=np.nan, rho_5comp=round(rho(g.sub_hazard, P5), 2)))
T4 = pd.DataFrame(rows)
T4.to_csv(OUT / 'table4_component_importance.csv', index=False)
log(T4.to_string(index=False))
log(f'rho(CS,E)={rho(g.climate_stress, g.exposure):.2f} rho(CS,V)={rho(g.climate_stress, g.vulnerability):.2f} '
    f'rho(E,V)={rho(g.exposure, g.vulnerability):.2f}')

log('\n== 4. Weighting sensitivity (primary) ==')
X = g[['climate_stress', 'exposure', 'vulnerability']].values
schemes = {'equal': [1/3, 1/3, 1/3], 'climate_dominant': [.5, .25, .25], 'exposure_dominant': [.25, .5, .25],
           'vulnerability_dominant': [.25, .25, .5]}
srows = []
s2 = g[['cell_id', 'centroid_lat', 'centroid_lon', 'population_2020']].copy()
for k, w in schemes.items():
    s = pd.Series(X @ np.array(w), index=g.index)
    s2[f'cgsri_primary_{k}'] = s.round(4)
    srows.append(dict(scheme=f'primary {k}', spearman_vs_primary=round(rho(s, P), 3), vh_retained=len(top(s) & VH),
                      vh_pop_m=round(pop[list(top(s))].sum() / 1e6, 2)))
for k, w in {'5comp equal': [.2] * 5, '5comp SH-sceptic': [.2375, .05, .2375, .2375, .2375]}.items():
    s = sum(wi * g[c] for wi, c in zip(w, ['gw_hazard', 'sub_hazard', 'climate_stress', 'exposure', 'vulnerability']))
    s2[f'cgsri_{k.replace(" ", "_")}'] = s.round(4)
    srows.append(dict(scheme=k, spearman_vs_primary=round(rho(s, P), 3), vh_retained=len(top(s) & VH),
                      vh_pop_m=round(pop[list(top(s))].sum() / 1e6, 2)))
SW = pd.DataFrame(srows)
SW.to_csv(OUT / 'weighting_schemes.csv', index=False)
log(SW.to_string(index=False))
draws = []
while len(draws) < 1000:
    w = rng.dirichlet([1, 1, 1])
    if w.min() >= 0.05 and w.max() <= 0.5:
        draws.append(w)
draws = np.array(draws)
hits = np.zeros(N)
for w in draws:
    s = pd.Series(X @ w, index=g.index)
    hits[list(top(s))] += 1
stab = hits / len(draws)
s2['mc_vh_probability_primary'] = stab.round(3)
s2['class_primary'] = g['cgsri_class']
s2['class_5comp'] = g['cgsri5_class']
s2.to_csv(ROOT / 'outputs/tables/supplementary_table_S2_sensitivity_analysis_v2.csv', index=False)
np.save(OUT / 'mc_weights.npy', draws)
robust = int((stab[list(VH)] >= 0.75).sum())
log(f'MC (1000 draws, uniform 3-simplex, 0.05-0.50, seed 20261112): {robust}/91 VH cells with p>=0.75; '
    f'median p in VH {np.median(stab[list(VH)]):.2f}')

log('\n== 5. E+V comparison ==')
ev = g.exposure + g.vulnerability
only = list(VH - top(ev))
log(f'E+V top quintile shared with primary VH {len(top(ev) & VH)}; with 5comp VH {len(top(ev) & VH5)}; '
    f'primary-only cells: n={len(only)}, mean CS {g.climate_stress[only].mean():.2f} vs {g.climate_stress.mean():.2f}')

log('\n== 6. Co-location (Fisher) ==')
u, sh = g['ugpi_urban_only'], g['sub_hazard']
crow = []
for q in (0.75, 0.90):
    a, b = u >= u.quantile(q), sh >= sh.quantile(q)
    tab = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    orr, p = fisher_exact(tab)
    crow.append(dict(threshold=f'Q{int(q*100)}', ugpi_thr=round(u.quantile(q), 3), sh_thr=round(sh.quantile(q), 3),
                     hh=tab[0][0], expected=round(a.sum() * b.sum() / N, 1), odds_ratio=round(orr, 2), p=round(p, 3)))
CL = pd.DataFrame(crow)
CL.to_csv(OUT / 'colocation_fisher.csv', index=False)
log(CL.to_string(index=False))
hh = g[(u >= u.quantile(.9)) & (sh >= sh.quantile(.9))].copy()


def area(lon, lat):
    if lon < -1.6:
        return 'Takoradi'
    if lon < -1.0:
        return 'Cape Coast'
    if lon < -0.3:
        return 'Winneba–Kasoa'
    if lon < 0.3:
        return 'Accra–Tema'
    return 'Ada–Keta'


hh['area'] = [area(x, y) for x, y in zip(hh.centroid_lon, hh.centroid_lat)]
hh[['cell_id', 'area', 'centroid_lat', 'centroid_lon', 'insar_n_valid', 'ugpi_urban_only', 'sub_hazard']].round(3).to_csv(
    OUT / 'q90_colocation_cells.csv', index=False)
log(hh[['cell_id', 'area', 'centroid_lat', 'centroid_lon', 'insar_n_valid']].round(3).to_string(index=False))
log(f'Q90 cells with >=10 valid InSAR px: {(hh.insar_n_valid >= 10).sum()}; area counts {hh["area"].value_counts().to_dict()}')

log('\n== 7. Ramp sensitivity (two-point plane) ==')
REF, KM = (5.55, -0.42), 110.6


def dxy(lat, lon):
    return (lon - REF[1]) * KM * np.cos(np.deg2rad(REF[0])), (lat - REF[0]) * KM


kx, ky = dxy(5.60, -0.19)
ux, uy = dxy(5.75, -0.31)
ax_, ay_ = dxy(5.56, -0.20)
a_, b_ = np.linalg.solve([[kx, ky], [ux - ax_, uy - ay_]], [9.0, -7.4 * np.cos(np.deg2rad(38))])
has = g.insar_n_valid >= 10
cx, cy = dxy(g.centroid_lat, g.centroid_lon)
v0 = g.insar_vel_mmyr
v1 = v0 - (a_ * cx + b_ * cy)


def haz(v):
    vc = v[has].clip(-60, 20)
    return 1 - (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)


sh1 = g.sub_proxy.copy()
w = g.insar_weight
sh1 = (1 - w) * g.sub_proxy + w * haz(v1).reindex(g.index).fillna(g.sub_proxy)
P5r = (g.gw_hazard + sh1 + g.climate_stress + g.exposure + g.vulnerability) / 5
hh_r = set(g.index[(u >= u.quantile(.9)) & (sh1 >= sh1.quantile(.9))])
log(f'plane {a_:.3f} E, {b_:.3f} N; n cells {int(has.sum())}; east-offset vs velocity rho {rho(cx[has], v0[has]):.2f}; '
    f'InSAR_haz rho {rho(haz(v0), haz(v1)):.3f}; 5comp rho {rho(P5, P5r):.3f}, VH retained {len(top(P5r) & VH5)}/91; '
    f'Q90 HH {len(hh_r)}, shared with original {len(hh_r & set(hh.index))}/{len(hh)}')
(OUT / 'summary.txt').write_text('\n'.join(lines))
