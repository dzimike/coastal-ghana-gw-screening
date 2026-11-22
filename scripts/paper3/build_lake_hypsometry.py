"""Level-dependent Lake Volta storage from JRC Global Surface Water and G-REALM altimetry, and its effect on Papers 3 and 2.

Run extract_gsw_lake_monthly.py and build_paper3_v2.py first (and build_paper2_erc_checks.py for the Paper 2 test).
1. Fits A(h) to months with good Landsat coverage; compares polynomial degrees by leave-one-year-out cross-validation.
2. Checks A(h) against the independent G-REALM MODIS lake extents (2022-2026).
3. Integrates A(h) to storage: dS = integral of A dh, as an anomaly from the 2004-2009 calendar-month climatology.
4. Recomputes the Paper 3 diagnostics for the level-dependent correction alongside the earlier constant-area bounds.
5. Repeats the Paper 2 storage-LST tests with the new correction.
Outputs: data/processed/paper3/lake_volta_hypsometric_*.csv, gwsa_members_hypsometric.csv, outputs/tables/paper3_v2/hypsometry_*.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
PRO = ROOT / 'data/processed/paper3'
TAB = ROOT / 'outputs/tables/paper3_v2'
GSW = ROOT / 'data/interim/paper3/gsw'
BASE = ('2004-01-01', '2009-12-31')
A_DOMAIN = 376774.0
MIN_COVER = 0.7
N_BOOT = 2000
GRID = np.arange(-8.0, 9.0001, 0.01)
rng = np.random.default_rng(20261122)
lines = []


def log(s=''):
    print(s, flush=True)
    lines.append(str(s))


def dec_year(idx):
    return idx.year + (idx.dayofyear - 0.5) / 365.25


def trend(s):
    s = s.dropna()
    with np.errstate(invalid='ignore'):
        return stats.theilslopes(s.values, dec_year(s.index))[0], mk.hamed_rao_modification_test(s.values).p


def detrend(s):
    s = s.dropna()
    x = dec_year(s.index)
    return s - np.polyval(np.polyfit(x, s.values, 1), x)


def zscore(s):
    return (s - s.mean()) / s.std()


def corr_eff(a, b):
    m = a.notna() & b.notna()
    x, y = a[m].values, b[m].values
    r = np.corrcoef(x, y)[0, 1]
    r1, r2 = pd.Series(x).autocorr(1), pd.Series(y).autocorr(1)
    ne = max(len(x) * (1 - r1 * r2) / (1 + r1 * r2), 3.0)
    return r, 2 * stats.t.sf(abs(r * np.sqrt((ne - 2) / (1 - r * r))), ne - 2), ne


def anomaly(s):
    b = s[BASE[0]:BASE[1]]
    return s - s.index.month.map(b.groupby(b.index.month).mean()).values


def huber_fit(x, y, deg, it=12):
    w = np.ones(len(x))
    p = np.polyfit(x, y, deg)
    for _ in range(it):
        res = y - np.polyval(p, x)
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        u = np.abs(res) / (1.345 * s)
        w = np.where(u <= 1, 1.0, 1 / u)
        p = np.polyfit(x, y, deg, w=np.sqrt(w))
    return p


def curve(p, lo, hi):
    """A(h) on GRID: the polynomial within the fitted level range, linear continuation outside it."""
    a = np.polyval(p, GRID)
    d = np.polyder(p)
    a = np.where(GRID < lo, np.polyval(p, lo) + np.polyval(d, lo) * (GRID - lo), a)
    a = np.where(GRID > hi, np.polyval(p, hi) + np.polyval(d, hi) * (GRID - hi), a)
    return a


def storage_mm(area, level):
    """Lake storage anomaly (mm over the domain) from A(h) on GRID and a monthly level series."""
    v = np.concatenate([[0], np.cumsum(.5 * (area[1:] + area[:-1]) * np.diff(GRID))])   # km2 m
    return anomaly(pd.Series(np.interp(level.values, GRID, v), index=level.index)) / A_DOMAIN * 1000


# data
d = pd.read_csv(GSW / 'monthly_area.csv')
d['date'] = pd.to_datetime(dict(year=d.year, month=d.month, day=1))
g = pd.read_csv(PRO / 'lake_volta_grealm_10day.csv', parse_dates=['date']).set_index('date').h
lvl = g.resample('MS').mean()
j = d.set_index('date').join(lvl.rename('h')).dropna(subset=['h', 'area_km2'])
occ = pd.read_csv(GSW / 'occurrence_area.csv')
log(f'JRC GSW v1.4 months processed: {len(d)} ({d.date.min():%Y-%m} to {d.date.max():%Y-%m}); with area estimate {d.area_km2.notna().sum()}; paired with altimetry {len(j)}')
log(f'maximum water extent of the lake mask (occurrence > 0): {occ.area_km2[1:].sum():,.0f} km2; permanent-like (occurrence >= 90%): {occ.area_km2[90:].sum():,.0f} km2')
k = j[j.edge_valid_frac >= MIN_COVER].copy()
log(f'months used (shoreline-zone coverage >= {MIN_COVER:.0%}): {len(k)}, in {k.year.nunique()} years; level {k.h.min():.2f} to {k.h.max():.2f} m; '
    f'area {k.area_km2.min():,.0f} to {k.area_km2.max():,.0f} km2; Spearman rho {stats.spearmanr(k.h, k.area_km2)[0]:.3f}')

# model selection: leave-one-year-out
log('\nA(h) MODEL SELECTION (leave-one-year-out)')
cv = []
for deg in (1, 2, 3):
    pred = pd.Series(index=k.index, dtype=float)
    for y in k.year.unique():
        tr_ = k[k.year != y]
        pred[k.year == y] = np.polyval(huber_fit(tr_.h.values, tr_.area_km2.values, deg), k.h[k.year == y].values)
    e = k.area_km2 - pred
    cv.append(dict(degree=deg, rmse_km2=np.sqrt((e ** 2).mean()), mae_km2=e.abs().mean(), bias_km2=e.mean(), r2=1 - (e ** 2).sum() / ((k.area_km2 - k.area_km2.mean()) ** 2).sum()))
cvd = pd.DataFrame(cv)
log(cvd.round(3).to_string(index=False))
DEG = int(cvd.loc[cvd.rmse_km2.idxmin(), 'degree'])
if DEG > 1 and cvd.rmse_km2[DEG - 2] - cvd.rmse_km2[DEG - 1] < 0.01 * cvd.rmse_km2[DEG - 1]:
    DEG -= 1
cvd.to_csv(TAB / 'hypsometry_cross_validation.csv', index=False)
p0 = huber_fit(k.h.values, k.area_km2.values, DEG)
LO, HI = k.h.min(), k.h.max()
A0 = curve(p0, LO, HI)
assert np.all(np.diff(A0) > 0), 'A(h) is not monotonic'
res = k.area_km2 - np.polyval(p0, k.h)
log(f'selected degree {DEG}; coefficients {np.round(p0, 3).tolist()}; residual SD {res.std():.0f} km2; A at h = {LO:.2f}: {np.interp(LO, GRID, A0):,.0f}; '
    f'h = 0: {np.interp(0, GRID, A0):,.0f}; h = {HI:.2f}: {np.interp(HI, GRID, A0):,.0f} km2')

# independent check with MODIS extents
mo = pd.read_csv(PRO / 'lake_volta_modis_extent.csv', parse_dates=['date']).set_index('date')
mo = mo[mo.q == 0].area.resample('MS').median().dropna()
jm = pd.concat([mo.rename('modis'), lvl.rename('h')], axis=1).dropna()
jm['pred'] = np.interp(jm.h, GRID, A0)
e = jm.modis - jm.pred
log(f'\nMODIS check, 2022-2026 (independent of the fit): n {len(jm)}; MODIS median {jm.modis.median():,.0f}; predicted median {jm.pred.median():,.0f} km2; '
    f'bias {e.mean():+.0f} km2 ({100 * e.mean() / jm.pred.mean():+.1f}%); RMSE {np.sqrt((e ** 2).mean()):.0f} km2; r {np.corrcoef(jm.modis, jm.pred)[0, 1]:.2f}; level range {jm.h.min():.2f} to {jm.h.max():.2f} m')

# storage
cal = pd.date_range('2002-01-01', '2025-12-01', freq='MS')
hs = lvl.reindex(cal).interpolate(limit=3, limit_area='inside')
hs_v = hs.dropna()
gw = pd.read_csv(PRO / 'gwsa_members_uncorrected.csv', index_col=0, parse_dates=True)
lk = pd.read_csv(PRO / 'lake_volta_storage_ewh.csv', index_col=0, parse_dates=True)
months = gw.index
chk = storage_mm(np.full(GRID.size, 8502.0), hs_v).reindex(months)
log(f'\npipeline check: constant 8,502 km2 through the new integration vs original lake term: max abs difference {np.abs(chk - lk.A8502).max():.3f} mm')
S = storage_mm(A0, hs_v).reindex(months)
years = k.year.unique()
boots, areas = [], []
for _ in range(N_BOOT):
    ys = rng.choice(years, len(years))
    b = pd.concat([k[k.year == y] for y in ys])
    pb = huber_fit(b.h.values, b.area_km2.values, DEG, it=5)
    a = curve(pb, b.h.min(), b.h.max())
    a = np.maximum.accumulate(a)
    areas.append(np.interp([-4, 0, 4], GRID, a))
    boots.append(storage_mm(a, hs_v).reindex(months).values)
B = np.array(boots)
lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
log(f'bootstrap (resampling years, {N_BOOT} draws): median monthly 95% half-width of lake storage {np.median((hi - lo) / 2):.1f} mm; A(0) 95% interval '
    f'{np.percentile(np.array(areas)[:, 1], 2.5):,.0f} to {np.percentile(np.array(areas)[:, 1], 97.5):,.0f} km2')
cov = {}
for c in (0.5, 0.9):
    q = j[j.edge_valid_frac >= c]
    cov[c] = storage_mm(np.maximum.accumulate(curve(huber_fit(q.h.values, q.area_km2.values, DEG), q.h.min(), q.h.max())), hs_v).reindex(months)
    log(f'coverage threshold {c:.0%}: {len(q)} months; RMS difference from the selected series {np.sqrt(((cov[c] - S) ** 2).mean()):.1f} mm')
MARGIN = 8502.0 / occ.area_km2[1:].sum()
Sm = storage_mm(A0 * MARGIN, hs_v).reindex(months)
log(f'margin case: A(h) scaled by {MARGIN:.3f} so that the JRC maximum extent equals the 8,502 km2 full-supply area')
eff = np.polyfit(lk.A8502, S, 1)[0] * 8502
log(f'effective constant area implied by the level-dependent storage: {eff:,.0f} km2 (regression on the constant-area series); SD of lake storage {S.std():.1f} mm (6,000 km2: {lk.A6000.std():.1f}; 8,502 km2: {lk.A8502.std():.1f})')
out = pd.DataFrame({'hypsometric': S, 'p2.5': lo, 'p97.5': hi, 'margin_scaled': Sm, 'cover50': cov[0.5], 'cover90': cov[0.9]})
out.round(3).to_csv(PRO / 'lake_volta_hypsometric_storage.csv')
pd.DataFrame({'h_m': GRID, 'area_km2': A0}).iloc[::10].round(2).to_csv(PRO / 'lake_volta_hypsometric_curve.csv', index=False)
k[['year', 'month', 'h', 'area_km2', 'valid_frac', 'edge_valid_frac']].round(3).to_csv(PRO / 'lake_volta_level_area_pairs.csv')
gh = pd.DataFrame({c: gw[c] - S for c in gw})
gh.to_csv(PRO / 'gwsa_members_hypsometric.csv')

# Paper 3 diagnostics
CASES = {'none': 0 * S, 'hypsometric': S, 'margin': Sm, 'A6000': lk.A6000, 'A8502': lk.A8502, 'half': .5 * lk.A8502}
log('\nPAPER 3 DIAGNOSTICS BY CORRECTION (median [range] across nine members)')
dg = []
for name, L in CASES.items():
    if name == 'none':
        continue
    ve = [1 - (gw[c] - L).var() / gw[c].var() for c in gw]
    r = [np.corrcoef(gw[c], L)[0, 1] for c in gw]
    sl = [np.polyfit(L, gw[c], 1)[0] for c in gw]
    sd = [np.polyfit(detrend(L), detrend(gw[c]), 1)[0] for c in gw]
    sdc = [(gw[c] - L).std() for c in gw]
    dg.append(dict(case=name, var_explained=np.median(ve), ve_min=min(ve), ve_max=max(ve), r=np.median(r), slope=np.median(sl), slope_detr=np.median(sd),
                   slope_detr_min=min(sd), slope_detr_max=max(sd), sd_corrected=np.median(sdc), sd_min=min(sdc), sd_max=max(sdc)))
dgd = pd.DataFrame(dg)
log(dgd.round(3).to_string(index=False))
dgd.to_csv(TAB / 'hypsometry_lake_contribution.csv', index=False)
tr_rows = []
for name, L in CASES.items():
    for pn, (a, b) in {'2002-04..2025-12': ('2002-04', '2025-12'), '2010-01..2025-12': ('2010-01', '2025-12')}.items():
        v = [trend((gw[c] - L)[a:b]) for c in gw]
        s_ = np.array([x[0] for x in v])
        tr_rows.append(dict(case=name, period=pn, slope=np.median(s_), min=s_.min(), max=s_.max(), n_sig=int(sum(x[1] < .05 for x in v))))
trd = pd.DataFrame(tr_rows)
log('\nTRENDS (mm/yr)\n' + trd.round(2).to_string(index=False))
trd.to_csv(TAB / 'hypsometry_trends.csv', index=False)
log(f'lake storage trend 2002-2025: hypsometric {trend(S)[0]:.2f}; 6,000 km2 {trend(lk.A6000)[0]:.2f}; 8,502 km2 {trend(lk.A8502)[0]:.2f} mm/yr')
st = []
for y in range(2002, 2013):
    s_, p_ = trend((gw.mean(axis=1) - S)[f'{y}-01':'2025-12'])
    st.append(dict(start=y, slope=s_, p=p_))
std = pd.DataFrame(st)
log('ensemble-mean hypsometric trend by start year: ' + '; '.join(f'{int(r.start)}: {r.slope:.1f}{"*" if r.p < .05 else ""}' for r in std.itertuples()))
std.to_csv(TAB / 'hypsometry_trend_start_year.csv', index=False)
WINDOWS = {'2002-04..2003-12': ('2002-04', '2003-12'), '2004-05..2005-09': ('2004-05', '2005-09'), '2006-05..2007-09': ('2006-05', '2007-09'),
           '2015-01..2016-12': ('2015-01', '2016-12'), '2020-06..2022-06': ('2020-06', '2022-06'), '2025-01..2025-12': ('2025-01', '2025-12')}
wr = []
for wn, (a, b) in WINDOWS.items():
    for name, L in CASES.items():
        z = [zscore(gw[c] - L)[a:b].mean() for c in gw]
        zd = [zscore(detrend(gw[c] - L))[a:b].mean() for c in gw]
        wr.append(dict(window=wn, case=name, z=np.median(z), z_min=min(z), z_max=max(z), n_z=int(sum(x < -.5 for x in z)),
                       zdetr=np.median(zd), zdetr_min=min(zd), zdetr_max=max(zd), n_zdetr=int(sum(x < -.5 for x in zd))))
wrd = pd.DataFrame(wr)
log('\nDROUGHT WINDOWS (median z, detrended z; members below -0.5)')
for wn, q in wrd.groupby('window', sort=False):
    log(wn + ': ' + ' | '.join(f'{r.case} {r.z:+.2f}/{r.zdetr:+.2f} ({r.n_z},{r.n_zdetr})' for r in q.itertuples()))
wrd.to_csv(TAB / 'hypsometry_windows.csv', index=False)
log('\nPLAUSIBILITY: change in annual-mean storage, 2012 to 2015 (mm; rainfall below normal throughout 2013-2017)')
ann = lambda s: s.groupby(s.index.year).mean()
pl = {}
for name, L in CASES.items():
    q = ann((gw - pd.DataFrame({c: L for c in gw})))
    dq = q.loc[2015] - q.loc[2012]
    pl[name] = (dq.mean(), dq.min(), dq.max(), int((dq > 0).sum()))
    log(f'  {name:12s} ensemble {dq.mean():+7.1f}; members {dq.min():+.1f} to {dq.max():+.1f}; rising in {int((dq > 0).sum())}/9')
log(f'  lake term alone: hypsometric {ann(S)[2015] - ann(S)[2012]:+.1f}; 6,000 {ann(lk.A6000)[2015] - ann(lk.A6000)[2012]:+.1f}; 8,502 {ann(lk.A8502)[2015] - ann(lk.A8502)[2012]:+.1f}')
dr = pd.read_csv(PRO / 'climate_drivers_domain.csv', index_col=0, parse_dates=True).reindex(cal)
fam = pd.read_csv(TAB / 'climate_linkage_fdr.csv')
cl = []
for name in ('hypsometric', 'margin'):
    x = (gw.mean(axis=1) - CASES[name]).reindex(cal)
    for dn in dr.columns:
        r, p, ne = corr_eff(x, dr[dn])
        rd, pd_, _ = corr_eff(detrend(x.dropna()).reindex(cal), detrend(dr[dn].dropna()).reindex(cal))
        cl += [dict(case=name, driver=dn, form='monthly', r=r, p=p, n_eff=ne), dict(case=name, driver=dn, form='detrended', r=rd, p=pd_, n_eff=np.nan)]
cld = pd.DataFrame(cl)
hy = cld.case == 'hypsometric'
all_p = np.r_[fam.p.values, cld.loc[hy, 'p'].values]
cld.loc[hy, 'q'] = multipletests(all_p, method='fdr_bh')[1][len(fam):]
cld.to_csv(TAB / 'hypsometry_climate_linkage.csv', index=False)
log(f'\nCLIMATE LINKAGE, hypsometric ensemble mean (q over {len(all_p)} tests):')
for r in cld[hy].itertuples():
    log(f'  {r.driver:7s} {r.form:9s} r {r.r:+.2f}  p {r.p:.3f}  q {r.q:.3f}')

# Paper 2 test
log('\nPAPER 2: REGIONAL STORAGE AGAINST DRY-SEASON LST')
dom = pd.read_csv(ROOT / 'data/processed/lst/v2/domain_dry_season_lst.csv', index_col=0).iloc[:, 0]
lst = dom - dom.mean()


def a_z(s):
    z = zscore(s)
    n = z.groupby(z.index.year).count()
    return z.groupby(z.index.year).mean()[n >= 6]


def d_z(s):
    z = zscore(s)
    z = z[z.index.month.isin([12, 1, 2])]
    lab = np.where(z.index.month == 12, z.index.year + 1, z.index.year)
    n = z.groupby(lab).count()
    return z.groupby(lab).mean()[n >= 2]


def dt(s):
    t = s.index.values.astype(float)
    return s - np.polyval(np.polyfit(t, s.values, 1), t)


def lag1(s):
    f = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
    return f.corr(f.shift(1))


def p2_stats(series, member):
    out = []
    for pa, fn in (('calendar year', a_z), ('Dec-Feb', d_z)):
        q = pd.concat([fn(series.dropna()).rename('g'), lst.rename('l')], axis=1).dropna().loc[2002:2023]
        for fo in ('raw', 'detrended'):
            x, y = (q.g, q.l) if fo == 'raw' else (zscore(dt(q.g)), dt(q.l))
            r = np.corrcoef(x, y)[0, 1]
            r1, r2 = lag1(x), lag1(y)
            ne = float(np.clip(len(x) * (1 - r1 * r2) / (1 + r1 * r2), 3, len(x)))
            p = 2 * stats.t.sf(abs(r * np.sqrt((ne - 2) / (1 - r * r))), ne - 2)
            dfl = x < -.5
            kk = int(dfl.sum())
            v = y.values
            sel = np.array([rng.permutation(len(v))[:kk] for _ in range(4999)])
            m1 = v[sel].mean(axis=1)
            m0 = (v.sum() - m1 * kk) / (len(v) - kk)
            obs = y[dfl].mean() - y[~dfl].mean()
            out.append(dict(member=member, pairing=pa, form=fo, n=len(x), r=r, n_eff=ne, p=p, deficit_n=kk, lst_deficit=y[dfl].mean(), lst_other=y[~dfl].mean(),
                            p_hotter=(1 + ((m1 - m0) >= obs).sum()) / 5000))
    return out


rows = []
for c in gw:
    rows += p2_stats(gw[c] - S, c)
rows += p2_stats(gw.mean(axis=1) - S, 'ensemble mean')
t2 = pd.DataFrame(rows)
old = pd.read_csv(ROOT / 'outputs/tables/paper2_erc/temporal_gwsa_lst_ensemble.csv')
old_m = old[(old.member != 'ensemble mean')]
mem = t2.member != 'ensemble mean'
pc = np.r_[old_m.p.values, t2.loc[mem, 'p'].values]
ph = np.r_[old_m.p_hotter.values, t2.loc[mem, 'p_hotter'].values]
t2.loc[mem, 'q'] = multipletests(pc, method='fdr_bh')[1][len(old_m):]
t2.loc[mem, 'q_hotter'] = multipletests(ph, method='fdr_bh')[1][len(old_m):]
t2.round(4).to_csv(TAB / 'hypsometry_paper2_storage_lst.csv', index=False)
log(f'member-level tests with the new correction: {int(mem.sum())}; p<0.05: {int((t2[mem].p < .05).sum())}; q<0.05 (with the original 108): {int((t2[mem].q < .05).sum())}; '
    f'deficit-hotter p<0.05: {int((t2[mem].p_hotter < .05).sum())}; q<0.05: {int((t2[mem].q_hotter < .05).sum())}')
for (pa, fo), q in t2[mem].groupby(['pairing', 'form']):
    e = t2[(~mem) & (t2.pairing == pa) & (t2.form == fo)].iloc[0]
    log(f'{pa} | {fo}: ensemble r {e.r:+.2f} (n_eff {e.n_eff:.1f}, p {e.p:.2f}); members {q.r.min():+.2f} to {q.r.max():+.2f}; p<0.05 {int((q.p < .05).sum())}/9; '
        f'LST deficit/other {e.lst_deficit:+.2f}/{e.lst_other:+.2f} (p hotter {e.p_hotter:.2f})')
(TAB / 'hypsometry_build_log.txt').write_text('\n'.join(lines) + '\n')
print('done')
