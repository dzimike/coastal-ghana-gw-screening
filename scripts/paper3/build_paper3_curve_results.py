"""Paper 3 results with the level-dependent Lake Volta correction as the main case.

Run build_paper3_v2.py, extract_gsw_lake_monthly.py and build_lake_hypsometry.py first. Data loading and statistics are the
functions of build_paper3_v2.py, so that every series and test is computed as before. Correction cases:
  JRC          level-dependent storage from JRC Global Surface Water and G-REALM (main case)
  JRC_scaled   JRC area scaled so that its maximum extent equals the 8,502 km2 full-supply area
  half         half of the JRC term (incomplete recovery of the lake mass by GRACE)
  A6000, A8502 constant lake areas
  JRC+Bui_*    JRC term plus the Bui filling scenarios
Outputs: outputs/tables/paper3_jhrs/ and data/processed/paper3_jhrs/. results_log.txt holds every number used in the text.
"""
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
_src = (HERE / 'build_paper3_v2.py').read_text().split('# ================================================================== build')[0]
_ns = {'__file__': str(HERE / 'build_paper3_v2.py'), '__name__': 'p3base'}
exec(compile(_src, str(HERE / 'build_paper3_v2.py'), 'exec'), _ns)
globals().update({k: v for k, v in _ns.items() if not k.startswith('__')})

OT = ROOT / 'outputs/tables/paper3_jhrs'
OP = ROOT / 'data/processed/paper3_jhrs'
OT.mkdir(parents=True, exist_ok=True)
OP.mkdir(parents=True, exist_ok=True)
L_ = []


def say(s=''):
    print(s, flush=True)
    L_.append(str(s))


cal = pd.date_range('2002-01-01', END, freq='MS')
d10 = pd.read_csv(OUTD / 'lake_volta_grealm_10day.csv', parse_dates=['date']).set_index('date')
lvl = d10.h.resample('MS').mean()
lvl_c = lvl.reindex(cal).interpolate(limit=3, limit_area='inside')
tws = {'JPL': grace_jpl(), 'CSR': grace_csr(), 'GSFC': grace_gsfc()}
months = pd.DatetimeIndex(sorted(set.intersection(*[set(s[START:END].dropna().index) for s in tws.values()])))
twsa = pd.DataFrame({k: anomaly(s.reindex(cal.union(s.index))) for k, s in tws.items()}).reindex(months)
lsm = {k: gldas(k, *v) for k, v in LSM.items()}
smsa = pd.DataFrame({k: anomaly(v.sm.reindex(cal)) for k, v in lsm.items()}).reindex(months)
cwsa = pd.DataFrame({k: anomaly(v.cw.reindex(cal)) for k, v in lsm.items()}).reindex(months)
gw = pd.DataFrame({f'{g}-{m}': twsa[g] - smsa[m] - cwsa[m] for g, m in product(tws, LSM)})
saved = pd.read_csv(OUTD / 'gwsa_members_uncorrected.csv', index_col=0, parse_dates=True)
assert np.nanmax(np.abs(gw.values - saved[gw.columns].values)) < 1e-6, 'GWSA members differ from the saved series'
lk0 = pd.read_csv(OUTD / 'lake_volta_storage_ewh.csv', index_col=0, parse_dates=True).reindex(months)
H = pd.read_csv(OUTD / 'lake_volta_hypsometric_storage.csv', index_col=0, parse_dates=True).reindex(months)
S = H.hypsometric
bd, bg = bui_storage(4.85).reindex(months), bui_storage(12.57).reindex(months)
TERMS = {'none': 0 * S, 'JRC': S, 'JRC_scaled': H.margin_scaled, 'half': .5 * S, 'A6000': lk0.A6000, 'A8502': lk0.A8502,
         'JRC+Bui_dead': S + bd, 'JRC+Bui_gross': S + bg}
MAIN = ['none', 'JRC', 'JRC_scaled', 'half', 'A6000', 'A8502']
cor = lambda c, t: gw[c] - TERMS[t]
ens = lambda t: gw.mean(axis=1) - TERMS[t]
say(f'domain {A_DOMAIN:,.0f} km2; months {len(months)}; members {gw.shape[1]}')
pd.DataFrame({t: TERMS[t] for t in ('JRC', 'JRC_scaled', 'A6000', 'A8502')}).to_csv(OP / 'lake_storage_terms.csv')

# 1. lake contribution
say('\n1. LAKE CONTRIBUTION')
rows = []
for c, t in product(gw.columns, ('JRC', 'JRC_scaled', 'A6000', 'A8502', 'half')):
    x, l = gw[c], TERMS[t]
    r, p, ne, n = corr_eff(x, l)
    rows.append(dict(member=c, case=t, r_lake_gwsa=r, p_eff=p, n_eff=ne, var_explained=1 - (x - l).var() / x.var(), sd_raw=x.std(), sd_corr=(x - l).std(),
                     r_lake_twsa=np.corrcoef(twsa[c.split('-')[0]], l)[0, 1]))
lc = pd.DataFrame(rows)
lc.to_csv(OT / 'lake_contribution_by_member.csv', index=False)
for t, g in lc.groupby('case', sort=False):
    say(f'{t:11s} r {g.r_lake_gwsa.median():.3f} [{g.r_lake_gwsa.min():.3f}-{g.r_lake_gwsa.max():.3f}]; var {g.var_explained.median():.3f} [{g.var_explained.min():.3f}-{g.var_explained.max():.3f}]; '
        f'SD raw {g.sd_raw.median():.1f} [{g.sd_raw.min():.1f}-{g.sd_raw.max():.1f}]; SD corr {g.sd_corr.median():.1f} [{g.sd_corr.min():.1f}-{g.sd_corr.max():.1f}]; p_eff median {g.p_eff.median():.3f}; r_TWSA {g.r_lake_twsa.median():.3f}')
nv = [c for c in gw.columns if not c.endswith('CLSM')]
g = lc[(lc.case == 'JRC') & lc.member.isin(nv)]
say(f'Noah/VIC JRC var explained {g.var_explained.median():.3f} [{g.var_explained.min():.3f}-{g.var_explained.max():.3f}]')
say(f'lake JRC SD {S.std():.1f}, range {S.min():.1f} to {S.max():.1f} mm; |lake| median {S.abs().median():.1f}; scaled SD {TERMS["JRC_scaled"].std():.1f}')
rows = []
for c in gw:
    for nm, y, x in (('GWSA', gw[c], S), ('GWSA_detr', detrend(gw[c]), detrend(S)), ('TWSA', twsa[c.split('-')[0]], S), ('TWSA_detr', detrend(twsa[c.split('-')[0]]), detrend(S))):
        rows.append(dict(member=c, y=nm, beta=np.polyfit(x.values, y.values, 1)[0]))
bt = pd.DataFrame(rows)
bt.to_csv(OT / 'lake_regression_slopes.csv', index=False)
for nm, g in bt.groupby('y', sort=False):
    say(f'slope of {nm} on JRC lake: median {g.beta.median():.2f} [{g.beta.min():.2f}-{g.beta.max():.2f}]')
ann = lambda s: s.groupby(s.index.year).mean()
for nm, s in (('GWSA ensemble', gw.mean(axis=1)), ('lake JRC', S), ('lake scaled', TERMS['JRC_scaled']), ('TWSA', twsa.mean(axis=1))):
    a = ann(s)
    say(f'{nm}: annual 2011 {a[2011]:.1f}, 2012 {a[2012]:.1f}, 2015 {a[2015]:.1f}, 2016 {a[2016]:.1f}; fall 2012-2016 {a[2012] - a[2016]:.1f}; 2012-2015 {a[2012] - a[2015]:.1f}')
say('plausibility, change in annual-mean corrected storage 2012 to 2015:')
for t in MAIN:
    dq = pd.Series({c: ann(cor(c, t))[2015] - ann(cor(c, t))[2012] for c in gw})
    say(f'  {t:11s} ensemble {dq.mean():+.1f}; members {dq.min():+.1f} to {dq.max():+.1f}; rising {int((dq > 0).sum())}/9')
u = pd.read_csv(OUTD / 'jpl_domain_uncertainty.csv', index_col=0, parse_dates=True).reindex(months)
hr = (gw.max(axis=1) - gw.min(axis=1)) / 2
sdiff = (TERMS['JRC_scaled'] - S).abs()
say(f'uncertainty (median mm): JPL formal {u.independent.median():.1f} to {u.correlated.median():.1f}; member half-range {hr.median():.1f} (IQR {hr.quantile(.25):.1f}-{hr.quantile(.75):.1f}); '
    f'JRC vs scaled difference {sdiff.median():.1f}; JRC bootstrap 95% half-width {((H["p97.5"] - H["p2.5"]) / 2).median():.1f}; |lake| {S.abs().median():.1f}')
say(f'detrended SD of ensemble-mean GWSA: uncorrected {detrend(gw.mean(axis=1)).std():.1f}; JRC {detrend(ens("JRC")).std():.1f}; scaled {detrend(ens("JRC_scaled")).std():.1f}')

# 2. trends
say('\n2. TRENDS')
PER = {'2002-04..2025-12': (START, END), '2010-01..2025-12': ('2010-01-01', END)}
rows = []
for pn, (a, b) in PER.items():
    for c, t in product(gw.columns, TERMS):
        sl, p = trend(cor(c, t)[a:b])
        rows.append(dict(period=pn, member=c, lake=t, slope=sl, p=p))
tr = pd.DataFrame(rows)
tr.to_csv(OT / 'trends_by_member.csv', index=False)
for (pn, t), g in tr.groupby(['period', 'lake'], sort=False):
    say(f'{pn} {t:14s} {g.slope.median():6.2f} [{g.slope.min():6.2f}, {g.slope.max():6.2f}] sig {int((g.p < .05).sum())}/9')
say(f'lake term trend 2002-2025: JRC {trend(S)[0]:.2f}; scaled {trend(TERMS["JRC_scaled"])[0]:.2f}; A6000 {trend(lk0.A6000)[0]:.2f}; A8502 {trend(lk0.A8502)[0]:.2f}; share of the median uncorrected trend {100 * trend(S)[0] / tr[(tr.period == "2002-04..2025-12") & (tr.lake == "none")].slope.median():.0f}%')
q = tr[(tr.period == '2002-04..2025-12') & (tr.lake == 'JRC')].copy()
q['grace'], q['lsm'] = q.member.str.split('-').str[0], q.member.str.split('-').str[1]
say('JRC trend by factor: ' + str(q.groupby('grace').slope.mean().round(2).to_dict()) + ' ' + str(q.groupby('lsm').slope.mean().round(2).to_dict()))
g = tr[tr.member.isin(nv) & (tr.lake == 'JRC')]
for pn, x in g.groupby('period'):
    say(f'Noah/VIC JRC {pn}: {x.slope.median():.2f} [{x.slope.min():.2f}, {x.slope.max():.2f}] sig {int((x.p < .05).sum())}/6')
x = tr[(tr.period == '2010-01..2025-12') & (tr.lake == 'JRC') & (tr.p < .05)]
say('members with significant 2010-2025 JRC trend: ' + ', '.join(x.member))
rows = []
for nm in ('none', 'JRC', 'JRC_scaled', 'A8502'):
    s = ens(nm)
    for y0 in range(2002, 2013):
        sl, p = trend(s[f'{y0}-01-01':END])
        rows.append(dict(series=nm, vary='start', year=y0, slope=sl, p=p))
    for y1 in range(2014, 2026):
        sl, p = trend(s[START:f'{y1}-12-01'])
        rows.append(dict(series=nm, vary='end', year=y1, slope=sl, p=p))
ts = pd.DataFrame(rows)
ts.to_csv(OT / 'trend_start_end_sensitivity.csv', index=False)
for (nm, v), g in ts.groupby(['series', 'vary'], sort=False):
    sg = g[g.p < .05]
    say(f'{nm:11s} vary {v:5s}: first {g.slope.iloc[0]:.2f}, last {g.slope.iloc[-1]:.2f}, range {g.slope.min():.2f}..{g.slope.max():.2f}; significant years {list(sg.year)}')

# 3. drought index
say('\n3. DROUGHT INDEX')
Z = pd.DataFrame({'raw_z': zscore(gw.mean(axis=1)), 'raw_zdetr': zscore(detrend(gw.mean(axis=1))), 'corrected_z': zscore(ens('JRC')), 'corrected_zdetr': zscore(detrend(ens('JRC')))})
Z.to_csv(OP / 'gdsi_ensemble_mean.csv')
ep = []
for k in Z:
    z = Z[k].dropna()
    say(f'{k}: months < -0.5 {int((z < -.5).sum())}/{len(z)}; < -1 {int((z < -1).sum())}; min {z.min():.2f} ({z.idxmin():%Y-%m})')
    for e in episodes(z):
        ep.append(dict(series=k, **e))
        say(f"   {e['start']}..{e['end']} months {e['months']} mean {e['mean']} min {e['min']} ({e['min_month']}) left-censored {e['left_censored']}")
pd.DataFrame(ep).to_csv(OT / 'episodes_ensemble_mean.csv', index=False)
pd.DataFrame({'raw': (gw.apply(zscore) < -.5).mean(axis=1), 'corrected': (pd.DataFrame({c: cor(c, 'JRC') for c in gw}).apply(zscore) < -.5).mean(axis=1)}).to_csv(OP / 'drought_member_agreement.csv')
WINDOWS = {'2002-04..2003-12': ('2002-04', '2003-12'), '2004-05..2005-09': ('2004-05', '2005-09'), '2006-05..2007-09': ('2006-05', '2007-09'),
           '2015-01..2016-12': ('2015-01', '2016-12'), '2020-06..2022-06': ('2020-06', '2022-06'), '2025-01..2025-12': ('2025-01', '2025-12')}
rows = []
for wn, (a, b) in WINDOWS.items():
    for t, c in product(MAIN, gw.columns):
        s = cor(c, t)
        rows.append(dict(window=wn, lake=t, member=c, z=zscore(s)[a:b].mean(), zdetr=zscore(detrend(s))[a:b].mean()))
wr = pd.DataFrame(rows)
wr.to_csv(OT / 'episode_windows_by_member.csv', index=False)
for (wn, t), g in wr.groupby(['window', 'lake'], sort=False):
    say(f'{wn} {t:11s} z {g.z.median():+.2f} [{g.z.min():+.2f},{g.z.max():+.2f}] n<-0.5 {int((g.z < -.5).sum())} | detr {g.zdetr.median():+.2f} [{g.zdetr.min():+.2f},{g.zdetr.max():+.2f}] n<-0.5 {int((g.zdetr < -.5).sum())}')
g = wr[wr.member.isin(nv) & (wr.window == '2015-01..2016-12') & wr.lake.isin(['none', 'JRC'])]
say('Noah/VIC 2015-16: ' + '; '.join(f'{k} {v.z.median():+.2f} [{v.z.min():+.2f},{v.z.max():+.2f}]' for k, v in g.groupby('lake', sort=False)))
rows = []
for thr in (-.5, -.8, -1.0):
    for k in Z:
        z = Z[k].dropna()
        e = episodes(z, thr=thr)
        rows.append(dict(series=k, threshold=thr, months_below=int((z < thr).sum()), n_episodes=len(e), episodes='; '.join(f"{x['start']}..{x['end']} ({x['min']:.2f})" for x in e)))
        say(f'  {k:16s} thr {thr:+.1f}: {rows[-1]["months_below"]} months: {rows[-1]["episodes"]}')
pd.DataFrame(rows).to_csv(OT / 'threshold_sensitivity.csv', index=False)


def full_base(s):
    return s - s.index.month.map(s.groupby(s.index.month).mean()).values


rows = []
for nm, a_ in (('raw', gw.mean(axis=1)), ('corrected', ens('JRC'))):
    b_ = full_base(a_)
    za, zb = zscore(a_), zscore(b_)
    row = dict(series=nm, r_z=np.corrcoef(za, zb)[0, 1], trend_base0409=trend(a_)[0], trend_fullclim=trend(b_)[0])
    for w, (a, b) in WINDOWS.items():
        row[f'zdetr_{w}_0409'], row[f'zdetr_{w}_full'] = zscore(detrend(a_))[a:b].mean(), zscore(detrend(b_))[a:b].mean()
    rows.append(row)
    say(f'baseline {nm}: r(z) {row["r_z"]:.3f}; trend {row["trend_base0409"]:.2f} vs {row["trend_fullclim"]:.2f}; max window change {max(abs(row[f"zdetr_{w}_0409"] - row[f"zdetr_{w}_full"]) for w in WINDOWS):.3f}')
pd.DataFrame(rows).to_csv(OT / 'baseline_sensitivity.csv', index=False)

# 4. climate linkage
say('\n4. CLIMATE LINKAGE')
drv = pd.read_csv(OUTD / 'climate_drivers_domain.csv', index_col=0, parse_dates=True)
DR = ['PDSI', 'SPI-1', 'SPI-3', 'SPI-6', 'SPI-12', 'SPI-24']
TG = {'TWSA': twsa.mean(axis=1), 'Lake storage': S, 'GWSA raw': gw.mean(axis=1), 'GWSA JRC': ens('JRC'), 'GWSA JRC_scaled': ens('JRC_scaled'),
      'GWSA half': ens('half'), 'GWSA A6000': ens('A6000'), 'GWSA A8502': ens('A8502')}
rows = []
for t, s in TG.items():
    for dn in DR:
        for lag in (range(7) if t in ('TWSA', 'Lake storage', 'GWSA raw', 'GWSA JRC') else [0]):
            x, y = s.reindex(cal), drv[dn].reindex(cal).shift(lag)
            r, p, ne, n = corr_eff(x, y)
            rd, pdt, _, _ = corr_eff(detrend(x.dropna()).reindex(cal), detrend(y.dropna()).reindex(cal))
            rows.append(dict(target=t, driver=dn, lag=lag, n=n, r=r, p_eff=p, neff=ne, r_detr=rd, p_detr=pdt))
cl = pd.DataFrame(rows)
cl.to_csv(OT / 'climate_linkage.csv', index=False)
c0 = cl[cl.lag == 0]
pv = np.r_[c0.p_eff.values, c0.p_detr.values]
o = np.argsort(pv)
qv = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
qv = np.minimum.accumulate(qv[::-1])[::-1]
q_ = np.empty(len(pv))
q_[o] = np.minimum(qv, 1)
lab = [f'{a}|{b}|monthly' for a, b in zip(c0.target, c0.driver)] + [f'{a}|{b}|detrended' for a, b in zip(c0.target, c0.driver)]
fd = pd.DataFrame({'test': lab, 'p': pv, 'q_bh': q_}).sort_values('p')
fd.to_csv(OT / 'climate_linkage_fdr.csv', index=False)
say(f'{len(pv)} lag-0 tests; p<0.05 {int((pv < .05).sum())}; q<0.05 {int((q_ < .05).sum())}')
for r in fd.head(16).itertuples():
    say(f'  {r.test}: p {r.p:.4f}, q {r.q_bh:.3f}')
for t in TG:
    g = c0[c0.target == t].set_index('driver').reindex(DR)
    say(f'{t:16s} r: ' + ' '.join(f'{d}={g.r[d]:+.2f}' for d in DR) + ' | detrended: ' + ' '.join(f'{d}={g.r_detr[d]:+.2f}{"*" if g.p_detr[d] < .05 else ""}' for d in DR))
    say(f'{"":16s} n_eff PDSI {g.neff["PDSI"]:.1f}, SPI-12 {g.neff["SPI-12"]:.1f}, SPI-24 {g.neff["SPI-24"]:.1f}')
g = cl[(cl.target == 'GWSA raw') & (cl.driver == 'PDSI')]
say(f'GWSA raw vs PDSI over lags 0-6: r {g.r.min():+.2f} to {g.r.max():+.2f}')
g = cl[(cl.target == 'GWSA JRC') & (cl.driver == 'PDSI')]
say(f'GWSA JRC vs PDSI over lags 0-6: r {g.r.min():+.2f} to {g.r.max():+.2f}')

# 5. seasonal cycle of the lake term
say('\n5. SEASONAL CYCLE')
cv = pd.read_csv(OUTD / 'lake_volta_hypsometric_curve.csv')
gh, ga = cv.h_m.values, cv.area_km2.values
vol = np.r_[0, np.cumsum(.5 * (ga[1:] + ga[:-1]) * np.diff(gh))]
V = pd.Series(np.interp(lvl_c.dropna().values, gh, vol), index=lvl_c.dropna().index) / A_DOMAIN * 1000


def clim(s):
    s = s.reindex(months).dropna()
    c = s.groupby(s.index.month).mean()
    return c - c.mean()


sc0 = pd.read_csv(OUTD / 'seasonal_climatology.csv', index_col=0)
sc = sc0.drop(columns=[c for c in sc0 if c.startswith('Lake')]).assign(**{'Lake (level-dependent)': clim(V)})
sc.to_csv(OP / 'seasonal_climatology.csv')
tb = pd.DataFrame({k: dict(amplitude_mm=sc[k].max() - sc[k].min(), month_max=int(sc[k].idxmax()), month_min=int(sc[k].idxmin())) for k in sc}).T
tb.to_csv(OT / 'seasonal_cycle_components.csv')
say(tb.round(1).to_string())
(OT / 'results_log.txt').write_text('\n'.join(L_) + '\n')
print('done')
