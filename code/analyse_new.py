"""Assemble every likelihood-dependent number of the paper from the Laplace-grid robust runs.

python3 analyse_new.py <model> [prefix]   (prefix 'new': remeasured w1, runs new*; 'sg': plus the
                                           physical-warp calibration S2, runs sg*)
Reads <lap dir>/<model>/*.json (lap_run.py) and results/descriptive_sg.json; writes
  results/warp_darksector_bounds_<prefix>[_masked].csv   bound vs lambda_C, fiducial and conservative
  results/summary_<prefix>[_masked].json                  everything below, machine-readable
and prints the report (redirect it to results/report_sg.txt). The paper uses prefix 'sg'; set DFW_MASKED=1 for
the point-source-masked analysis.
"""
import sys
import glob
import json
import os
import numpy as np
from scipy import stats
import paths

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1]
OUT = os.path.join(paths.LAPDIR, MODEL)
SUF = '_masked' if paths.MASKED else ''
P = sys.argv[2] if len(sys.argv) > 2 else 'sg'
_all = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, '*.json')))
        if os.path.basename(f).startswith(P)}
EX = {n: r for n, r in _all.items() if n.startswith(P + 'excal_')}        # excision calibration units
R = {n: r for n, r in _all.items() if not n.startswith(P + 'excal_')}
DESC = json.load(open(paths.results('descriptive_sg.json'))) if os.path.exists(paths.results('descriptive_sg.json')) else {}
LAMS = np.round(0.4 * (np.arange(20) + 1), 1)
VARS = [('iband', 'i band in place of r band'), ('top1', 'excise top 1% by predicted signal'),
        ('top5', 'excise top 5% by predicted signal'), ('alt', 'alternative noise model (gt)'),
        ('cut3', '|w1| < 3 sigma_MAD kept, selection-normalised'), ('cut5', '|w1| < 5 sigma_MAD kept, selection-normalised'),
        ('hz01', 'disk scale height 0.1 R in S'),
        ('hz02', 'disk scale height 0.2 R in S'), ('frac03', '30% forward-model uncertainty')]
if P == 'sg':
    VARS += [('g1', 'inner halo slope 1 in the warp shape'), ('reff', 'R_eff = 1.678 R_d (face-on) in the warp shape'),
             ('Mp', 'stellar mass +0.2 dex'), ('Mm', 'stellar mass -0.2 dex'),
             ('gas', 'gas self-gravity (xGASS median, exponential 2 R_d)')]
summ = {}


def g(name, key='q95'):
    return R[name][key] if name in R else np.nan


# ---------------------------------------------------------------- 0. grid diagnostics
bad = [(n, r['n_bad_in_mass'], r['tail_top5pct']) for n, r in R.items() if r['n_bad_in_mass'] > 0 or r['tail_top5pct'] > 1e-4]
print('model %s: %d runs; flagged (Laplace failure carrying weight, or mass near grid top): %d' % (MODEL, len(R), len(bad)))
for x in bad[:30]:
    print('   ', x)
summ['flagged'] = bad

# ---------------------------------------------------------------- 1. bound vs range + variations
rows, table = [], []
for ii, lam in enumerate(LAMS):
    f68, f95 = g(P + 'pub%d_fid' % ii, 'q68'), g(P + 'pub%d_fid' % ii)
    rat95 = {v: g(P + 'pub%d_%s' % (ii, v)) / f95 for v, _ in VARS}
    rat68 = {v: g(P + 'pub%d_%s' % (ii, v), 'q68') / f68 for v, _ in VARS}
    w95 = max(rat95, key=lambda k: np.nan_to_num(rat95[k], nan=-1))
    w68 = max(rat68, key=lambda k: np.nan_to_num(rat68[k], nan=-1))
    table.append(dict(lam=float(lam), fid68=f68, fid95=f95, cons68=f68 * rat68[w68], cons95=f95 * rat95[w95],
                      worst95=w95, factor95=rat95[w95], factor68=rat68[w68], ratios95=rat95))
    rows.append((lam, f68, f95, rat68[w68], rat95[w95], w95))
summ['by_range'] = table
head = ['# Bound on Delta G/G_N for a fifth force on dark matter but not stars, from galaxy warps.',
        '# Outlier-robust likelihood with no quality cut on the observed warp;',
        '# noise model %s (robust_lap.py); posterior on a fine dG grid with the noise parameters' % MODEL,
        '# integrated out by Laplace at each grid point.',
        ('# Warps remeasured from the NSA atlas images (matched filter, r band); predicted warps from the forward model'
         ' of DF20 x per-galaxy suppression S2 (seeing, pixels, window, edge-on projection, disk self-gravity).'
         if P == 'sg' else '# Sample: all galaxies except the seven catalogue artefacts (4338). Predicted signals: DF20.'),
        '# Sample: %d galaxies (measurable, consistent Sersic/Petrosian sizes, catalogue artefacts removed).'
        % R.get(P + 'pub19_fid', {}).get('n', -1),
        '# Conservative = fiducial x largest ratio over the variations: ' + ', '.join(v for v, _ in VARS),
        'lambda_C_Mpc,dG_over_GN_68pc,dG_over_GN_95pc,dG_over_GN_68pc_inflated,dG_over_GN_95pc_inflated,'
        'worst_factor_68pc,worst_factor_95pc,worst_variation_95pc']
ROWS_CSV = rows

print('\n=== 95%% bound vs range: fiducial, ratio of each variation, conservative ===')
print('%5s %9s %9s ' % ('lam', 'fid68', 'fid95') + ' '.join('%7s' % v for v, _ in VARS) + ' %9s %7s %s' % ('cons95', 'factor', 'worst'))
for t in table:
    print('%5.1f %9.3g %9.3g ' % (t['lam'], t['fid68'], t['fid95']) + ' '.join('%7.2f' % t['ratios95'][v] for v, _ in VARS)
          + ' %9.3g %7.2f %s' % (t['cons95'], t['factor95'], t['worst95']))
fac = np.array([t['factor95'] for t in table])
print('largest factor across ranges: %.2f - %.2f; worst variation counts: %s'
      % (np.nanmin(fac), np.nanmax(fac), {v: sum(t['worst95'] == v for t in table) for v, _ in VARS}))

# power law and range growth of the fiducial
f95 = np.array([t['fid95'] for t in table])
ok = np.isfinite(f95)
slope = np.polyfit(np.log(LAMS[ok]), np.log(f95[ok]), 1)[0]
slope8 = np.sum(np.log(LAMS[ok] / 8) * np.log(f95[ok] / f95[-1])) / np.sum(np.log(LAMS[ok] / 8) ** 2)
two = np.log(f95[-1] / f95[0]) / np.log(LAMS[-1] / LAMS[0])
summ['powerlaw'] = dict(free=slope, pinned8=slope8, twopoint=two, improve_0p4_to_8=f95[0] / f95[-1])
print('fiducial power law: free fit %.3f, pinned at 8 Mpc %.3f, two-point %.3f; 0.4->8 Mpc improvement x%.2f'
      % (slope, slope8, two, f95[0] / f95[-1]))

# MC noise floor from the range-to-range scatter of the fiducial 95% limit about a smooth curve
okf = np.isfinite(f95)
resid = np.log(f95[okf]) - np.polyval(np.polyfit(np.log(LAMS[okf]), np.log(f95[okf]), 2), np.log(LAMS[okf]))
SIG_MC = float(np.sqrt(np.sum(resid ** 2) / max(okf.sum() - 3, 1)))
summ['mc_floor'] = SIG_MC
print('MC noise floor (rms scatter of fiducial q95 about a quadratic in log lambda): %.3f' % SIG_MC)
for t in table:
    t['cons95_mc'] = t['cons95'] * (1. + SIG_MC)
    t['cons68_mc'] = t['cons68'] * (1. + SIG_MC)

# excision calibration (HD 2026-10-09): an excision inflates the envelope only by its excess over the median
# excision ratio in signal-free mocks (floor 1); when calibration units exist this defines the conservative bound
CAL = bool(EX)
if CAL:
    for ii, t in enumerate(table):
        mk = [r for n, r in EX.items() if r['ii'] == ii and r['kind'] != 'data']
        f68, f95 = t['fid68'], t['fid95']
        r68 = {v: g(P + 'pub%d_%s' % (ii, v), 'q68') / f68 for v, _ in VARS}
        for q, rr, key in (('q95', t['ratios95'], '95'), ('q68', r68, '68')):
            cal = dict(rr)
            for ex in ('top1', 'top5'):
                med = np.median([r[ex][q] / r['fid'][q] for r in mk])
                cal[ex] = max(rr[ex] / med, 1.)
            w = max(cal, key=lambda k: np.nan_to_num(cal[k], nan=-1))
            t['uncal_cons%s_mc' % key] = t['cons%s_mc' % key]
            t['factor%s_cal' % key], t['worst%s_cal' % key] = cal[w], w
            t['cons%s_mc' % key] = (f95 if key == '95' else f68) * cal[w] * (1. + SIG_MC)
    print('excision-calibrated envelope: factor at 8 Mpc %.2f (%s); range %.2f-%.2f; worst counts %s'
          % (table[-1]['factor95_cal'], table[-1]['worst95_cal'], min(t['factor95_cal'] for t in table),
             max(t['factor95_cal'] for t in table),
             {v: sum(t['worst95_cal'] == v for t in table) for v in sorted(set(t['worst95_cal'] for t in table))}))

# ---------------------------------------------------------------- 2. headline numbers
t8 = table[-1]
cons = t8['cons95_mc']
summ['headline'] = dict(fid95=t8['fid95'], fid68=t8['fid68'], cons95=cons, cons68=t8['cons68_mc'], cons95_nofloor=t8['cons95'],
                        factor=t8.get('factor95_cal', t8['factor95']), worst=t8.get('worst95_cal', t8['worst95']),
                        cons95_uncal=t8.get('uncal_cons95_mc'), calibrated=CAL, beta_fid=np.sqrt(t8['fid95'] / 2), beta_cons=np.sqrt(cons / 2),
                        theta_mode=R[P + 'pub19_fid']['theta_at_mode'], logZ=R[P + 'pub19_fid']['logZ'],
                        logZ_alt=R.get(P + 'pub19_alt', {}).get('logZ'))
h = summ['headline']
print('\n=== headline (8 Mpc) ===')
print('fiducial 95%% < %.3g, 68%% < %.3g ; conservative 95%% < %.3g (x%.2f %s, x%.3f MC floor), 68%% < %.3g'
      % (h['fid95'], h['fid68'], cons, h['factor'], h['worst'], 1 + SIG_MC, h['cons68']))
print('|beta| < %.4f (fiducial %.4f) ; noise at mode %s ; logZ %.2f vs alt %s'
      % (h['beta_cons'], h['beta_fid'], h['theta_mode'], h['logZ'], h['logZ_alt']))
for ii in (0, 4, 19):
    t = table[ii]
    print('Table I(a) lam %.1f: fiducial %.2g conservative %.2g (68%% %.2g)%s' % (t['lam'], t['fid95'], t['cons95_mc'], t['cons68_mc'],
          ' [uncalibrated %.2g]' % t['uncal_cons95_mc'] if CAL else ''))
print('Table I(b) at 8 Mpc:')
print('   fiducial (N=%d) %.3g 1.00' % (R[P + 'pub19_fid']['n'], t8['fid95']))
for v, lab in VARS:
    if P + 'pub19_' + v in R:
        print('   %-42s N=%4d  %.3g  %.2f' % (lab, R[P + 'pub19_' + v]['n'], R[P + 'pub19_' + v]['q95'], t8['ratios95'][v]))

# ---------------------------------------------------------------- 3. tests at 8 Mpc
print('\n=== tests at 8 Mpc (ratio to fiducial) ===')
summ['tests'] = {}
for tname in ('screen', 'all', 'signfree', 'signfree_i'):
    n = P + 'test_' + tname
    if n in R:
        summ['tests'][tname] = dict(R[n], ratio=R[n]['q95'] / t8['fid95'])
        extra = (' sign-free mean %.3g sd %.3g P(<0) %.3f' % (R[n]['signfree_mean'], R[n]['signfree_sd'], R[n]['P_neg'])
                 if 'signfree_mean' in R[n] else '')
        print('%-10s N=%d q95 %.4g q68 %.4g ratio %.3f%s' % (tname, R[n]['n'], R[n]['q95'], R[n]['q68'], R[n]['q95'] / t8['fid95'], extra))

# ---------------------------------------------------------------- 4. comparisons (Table I(c), Sec. VIII)
# Bottaro+25 Fig. 1 (digitise_bottaro2025.py), log-interpolated to 8 Mpc; compared with our 8 Mpc conservative bound
_b25 = np.genfromtxt(os.path.join(HERE, 'Bottaro2025_fig1_digitised.csv'), delimiter=',', skip_header=3, names=True)
b25_8 = 10 ** np.interp(np.log10(8.), np.log10(_b25['lambda_Mpc'][::-1]), np.log10(_b25['beta95'][::-1]))
lims = [('Bottaro+25 (8 Mpc)', b25_8), ('Bottaro+24 Planck+BOSS', 4.8e-3), ('Archidiacono+22 CMB+BAO', 5.4e-3),
        ('Bean+08 (10 Mpc)', 0.05), ('Bonvin+25', 0.07), ('Kesden+06', 0.1), ('Zamlung+26', 0.2)]
claims = [('Clini DR2 BAO', 0.0037, 0.0015), ('Clini DR1 FS', 0.0069, 0.0021), ('Clini FS+BAO', 0.0043, 0.0014),
          ('Bottaro+25', 0.0040, 0.0020), ('Bedroya+25', 0.0050, 0.0020), ('coupled DE |b|=0.03', 2 * 0.03 ** 2, None),
          ('Agrawal c=0.30', 2 * 0.30 ** 2, None)]
s_us = cons / 1.96
print('\n=== comparisons (conservative %.3g) ===' % cons)
summ['limits'] = {n: v / cons for n, v in lims}
print('  '.join('%s %.3g (%sx)' % (n, v, ('%.2g' if v < 10 * cons else '%.0f') % (v / cons)) for n, v in lims))
summ['claims'] = {}
for n, v, e in claims:
    sig = v / np.sqrt((e or 0.) ** 2 + s_us ** 2)
    sig_fid = v / np.sqrt((e or 0.) ** 2 + (h['fid95'] / 1.96) ** 2)
    summ['claims'][n] = dict(sigma=sig, sigma_fid=sig_fid, ratio_central=v / cons, ratio_lower=(v - e) / cons if e else None)
    print('%-22s %.4g  excluded at %.1f sigma (fiducial posterior %.1f); central %.1fx, 68%% lower edge %s'
          % (n, v, sig, sig_fid, v / cons, ('%.1fx' % ((v - e) / cons)) if e else '-'))
cb = np.sqrt(cons / 2)
for div in (1.96, 1.645):
    print('Bedroya in ctilde (divisor %.3f): %.2f sigma' % (div, 0.05 / np.sqrt(0.01 ** 2 + (cb / div) ** 2)))

# ---------------------------------------------------------------- 5. App. A: FULL runs
print('\n=== Appendix A (FULL forward-model runs) ===')
summ['full'] = {}
for sub in ('all', 'union'):
    e8 = np.array([g(P + 'full_lam8_' + sub)] + [g(P + 'full_s%d_lam8_%s' % (s, sub)) for s in (1, 2, 3, 4)])
    e20 = np.array([g(P + 'full_lam20_' + sub)] + [g(P + 'full_s%d_lam20_%s' % (s, sub)) for s in (1, 2, 3, 4)])
    for L in (12, 16, 20):                                       # seed-matched pairs at every range with seeds
        eL = np.array([g(P + 'full_s%d_lam%d_%s' % (s, L, sub)) for s in (1, 2, 3, 4)])
        if np.all(np.isfinite(eL)):
            prL = eL / e8[1:]
            ttL = stats.ttest_1samp(np.log(prL), 0.)
            pone = float(ttL.pvalue / 2 if ttL.statistic < 0 else 1 - ttL.pvalue / 2)
            summ.setdefault('full_pairs', {}).setdefault(sub, {})[L] = dict(
                pairs=prL.tolist(), mean=prL.mean(), sd=prL.std(ddof=1), t=float(ttL.statistic), p_one=pone,
                mean_ratio=np.nanmean(np.append(eL, g(P + 'full_lam%d_%s' % (L, sub)))) / np.nanmean(e8))
            print('[%s] %d Mpc seed-matched pairs %s -> %.3f +- %.3f (sd); t %.2f, p one-sided %.2g; ratio of 5-run means %.3f'
                  % (sub, L, np.round(prL, 3), prL.mean(), prL.std(ddof=1), ttL.statistic, pone,
                     summ['full_pairs'][sub][L]['mean_ratio']))
    base = {L: g(P + 'full_lam%d_%s' % (L, sub)) for L in (8, 12, 16, 20)}
    pr = e20[1:] / e8[1:]
    lr = np.log(pr)
    tt = stats.ttest_1samp(lr, 0.)
    d = dict(N=R.get(P + 'full_lam8_' + sub, {}).get('n'),
             ratio_to_base8={L: base[L] / base[8] for L in (12, 16, 20)},
             ratio_to_mean8={L: base[L] / e8.mean() for L in (12, 16, 20)},
             mean20_over_mean8=e20.mean() / e8.mean(),
             sd8=e8.std(ddof=1) / e8.mean(), sd20=e20.std(ddof=1) / e20.mean(), all20_below_all8=bool(e20.max() < e8.min()),
             pairs=pr.tolist(), pair_mean=pr.mean(), pair_sd=pr.std(ddof=1), pair_sem=pr.std(ddof=1) / np.sqrt(len(pr)),
             t=float(tt.statistic), p_one=float(tt.pvalue / 2 if tt.statistic < 0 else 1 - tt.pvalue / 2), p_two=float(tt.pvalue),
             abs8=e8.tolist(), abs20=e20.tolist())
    summ['full'][sub] = d
    print('[%s, N=%s] ratios to base 8 Mpc: %s ; to mean of five 8 Mpc: %s'
          % (sub, d['N'], ' '.join('%.3f' % d['ratio_to_base8'][L] for L in (12, 16, 20)),
             ' '.join('%.3f' % d['ratio_to_mean8'][L] for L in (12, 16, 20))))
    print('   five 8 Mpc: sd/mean %.3f ; five 20 Mpc: %.3f ; every 20 < every 8: %s'
          % (d['sd8'], d['sd20'], d['all20_below_all8']))
    print('   seed-matched pairs %s -> %.3f +- %.3f (sd), sem %.3f ; paired t on log ratio t=%.2f, p one-sided %.2g, two-sided %.2g'
          % (np.round(pr, 3), d['pair_mean'], d['pair_sd'], d['pair_sem'], d['t'], d['p_one'], d['p_two']))
for sub in ('near', 'far'):
    r = {L: g(P + 'full_lam%d_%s' % (L, sub)) / g(P + 'full_lam8_' + sub) for L in (12, 16, 20)}
    summ['full'][sub] = dict(N=R.get(P + 'full_lam8_' + sub, {}).get('n'), ratio_to_base8=r)
    print('[%s, N=%s] ratios to own 8 Mpc: %s' % (sub, summ['full'][sub]['N'], ' '.join('%.3f' % r[L] for L in (12, 16, 20))))

# ---------------------------------------------------------------- 6. injections
print('\n=== injection-recovery (8 Mpc fiducial sample, 20 mocks per level) ===')
summ['inj'] = {}
for kind in ('gj', 'emp', 'tsig', 'empsig'):
    levels = sorted({R[n]['level'] for n in R if n.startswith(P + 'inj_%s_' % kind)})
    for L in levels:
        rs = [R[n] for n in R if n.startswith(P + 'inj_%s_%g_' % (kind, L))]
        q50 = np.array([r['q50'] for r in rs]); q95 = np.array([r['q95'] for r in rs])
        d = dict(n=len(rs), med_q50=float(np.median(q50)), sd_q50=float(np.std(q50, ddof=1)) if len(rs) > 1 else np.nan,
                 med_q95=float(np.median(q95)), coverage95=float(np.mean(L <= q95)) if L > 0 else np.nan,
                 min_q50=float(q50.min()))
        summ['inj']['%s_%g' % (kind, L)] = d
        print('%-6s inj %-7g n=%2d  median of medians %.3g (sd %.2g, min %.2g)  median 95%% UL %.3g  coverage %s'
              % (kind, L, d['n'], d['med_q50'], d['sd_q50'], d['min_q50'], d['med_q95'],
                 '-' if L == 0 else '%.2f' % d['coverage95']))

# ---------------------------------------------------------------- 7. descriptive numbers at the new bounds
import lap_run as _L
from robust_like import load_signals as _ls, select as _sel
_d = _sel(_ls('Signals_warp_NoScreen_19.dat'), **_L.new_data(sg=(P == 'sg')))
_amp = (_d['W'] * np.abs(_d['M'])).sum(1) / _d['W'].sum(1)
P99 = float(np.quantile(_amp, 0.99))
print('S-scaled predicted amplitude on the new sample: median %.3f, P99 %.3f (N=%d)' % (np.median(_amp), P99, len(_amp)))
summ['P99_amp_scaled'] = P99
for lab, b in (('fiducial', h['fid95']), ('conservative', cons)):
    w1 = P99 * b; z = w1 * 15.0 * (1.5 * 2.5 / 0.5)
    print('P99 galaxy at the %s bound: w1 = %.2g, z(L_x=15 kpc) = %.0f pc, z/L = %.2g' % (lab, w1, 1e3 * z, z / 15.0))
summ['descriptive'] = DESC

# ---------------------------------------------------------------- 8. excision calibration (signal-free mocks)
if EX:
    print('\n=== excision calibration: top-1%/top-5% excision ratio on signal-free mocks vs data ===')
    summ['excal'] = []
    for ii, t in enumerate(table):
        mk = [r for n, r in EX.items() if r['ii'] == ii and r['kind'] != 'data']
        if not mk:
            continue
        r1 = np.array([r['r1'] for r in mk]); r5 = np.array([r['r5'] for r in mk])
        d1, d5 = t['ratios95']['top1'], t['ratios95']['top5']
        e1, e5 = max(d1 / np.median(r1), 1.), max(d5 / np.median(r5), 1.)
        others = {v: t['ratios95'][v] for v, _ in VARS if v not in ('top1', 'top5')}
        cal = dict(others, top1=e1, top5=e5)
        w = max(cal, key=lambda k: np.nan_to_num(cal[k], nan=-1))
        rec = dict(lam=t['lam'], n=len(mk), r1_med=float(np.median(r1)), r5_med=float(np.median(r5)),
                   r5_16_84=np.percentile(r5, [16, 84]).tolist(), data_r1=d1, data_r5=d5,
                   pct_r1=float(np.mean(r1 < d1)), pct_r5=float(np.mean(r5 < d5)), excess1=e1, excess5=e5,
                   factor_cal=cal[w], worst_cal=w, cons95_cal=t['fid95'] * cal[w] * (1 + SIG_MC))
        summ['excal'].append(rec)
        print('%5.1f  mocks %2d  r5 median %.2f [%.2f-%.2f]  data %.2f (pct %.2f) excess %.2f | r1 median %.2f data %.2f excess %.2f'
              ' | calibrated factor %.2f (%s) cons95 %.3g'
              % (t['lam'], len(mk), rec['r5_med'], *rec['r5_16_84'], d5, rec['pct_r5'], e5, rec['r1_med'], d1, e1,
                 cal[w], w, rec['cons95_cal']))
csv = ['%g,%.4g,%.4g,%.4g,%.4g,%.3f,%.3f,%s' % (t['lam'], t['fid68'], t['fid95'], t['cons68_mc'], t['cons95_mc'],
       t.get('factor68_cal', t['factor68']), t.get('factor95_cal', t['factor95']), t.get('worst95_cal', t['worst95']))
       for t in table]
if CAL:
    head[-2] = head[-2] + '. Excisions enter by their excess over the median excision ratio in signal-free mocks (floor 1).'

head[-2] = head[-2] + ' Conservative columns also include the MC noise floor x(1 + %.3f).' % SIG_MC
open(paths.results('warp_darksector_bounds_%s%s.csv' % (P, SUF)), 'w').write('\n'.join(head + csv) + '\n')
json.dump(summ, open(paths.results('summary_%s%s.json' % (P, SUF)), 'w'), indent=1, default=float)
print('\nwrote results/summary_%s%s.json and warp_darksector_bounds_%s%s.csv' % (P, SUF, P, SUF))
