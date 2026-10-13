"""Production runs of the robust likelihood on the deterministic Laplace grid (robust_lap.py).

The paper uses the sg* configurations only (groups sgpub, sgtest, sgexcalA, sgexcalB, sginj, sgfull); the others
are earlier configurations of the same likelihood, kept because the sg names are built from them.

  python3 lap_run.py list <group> <model>      groups: pub test full inj
  python3 lap_run.py <model> <name> [...]      writes lap/<model>/<name>.json

Names (as in robust_run.py):
  pub<ii>_<var>   var in fid, iband, top1, top5, cut002, cut0035, df20, alt (the other noise model)
  test_<what>     8 Mpc: screen, all, union, signfree, signfree_i
  full_<tag>_<sub>
  inj_<kind>_<level>_<rep>   mock outliers: gj (signal-free Gaussian), emp (signal-free, resampled from
                  the real |w1|>0.0025), tsig (signal-carrying Student-t noise, as fitted by 't'),
                  empsig (real-outlier noise added to the signal)
"""
import json
import os
import sys
import time
import numpy as np
from robust_like import load_signals, select, union_rows
from robust_lap import posterior, quantiles
import paths as P

HERE = os.path.dirname(os.path.abspath(__file__))
NEWF = P.NEWF                                    # new_w1.npz, or the masked file if DFW_MASKED=1
LAP = P.LAPDIR


def new_data(band='r', hz=0.15, est='mf', psf=True, sg=False, cal=None):
    """Remeasured w1 (measurement/new_w1.npz): observed values, errors, per-galaxy S and the measurable mask,
    as keyword arguments for robust_like.select (arrays over the 4345 signals rows).
    sg=True: S2 from calib_sg.py, the physical warp (edge-on projection of the self-gravitating
    disk) through the same seeing/pixel/window/estimator simulation; cal picks a calib_sg.RUNS entry
    (default from band/hz/psf: fid, hz0.1, hz0.2, ifid, nopsf)."""
    N = np.load(NEWF)
    wk, ek = ('w1_mf_%s' % band, 'w1_mf_err_%s' % band) if est == 'mf' else ('w1_%s' % band, 'w1_err_%s' % band)
    if sg:
        if cal is None:
            cal = ('ifid' if band == 'i' else 'nopsf' if not psf else {0.1: 'hz0.1', 0.2: 'hz0.2'}.get(hz, 'fid'))
        S = N['S2_%s_%s' % (est, cal)]
    else:
        S = N['S_%s_%s_hz%g' % (est, band, hz)] if psf else np.ones(len(N[wk]))
    keep = N['ok_%s' % band] & (N['npix_med_%s' % band] >= 3) & np.isfinite(N[wk]) & np.isfinite(S)
    keep &= N['size_ok']                     # HD 2026-10-09: drop failed Sersic fits, SERSIC/ELPETRO TH50 outside [1/3, 3]
    return dict(w_new=np.where(keep, N[wk], np.nan), e_new=N[ek], scale=np.nan_to_num(S), keep=keep)


def new_cuts():
    """Quality-cut variants for the new statistic: 3 and 5 x the MAD scatter of the fiducial sample."""
    d = new_data()
    w = d['w_new'][np.isfinite(d['w_new'])]
    sd = 1.4826 * np.median(np.abs(w - np.median(w)))
    return 3. * sd, 5. * sd
PUBVAR = {'fid': {}, 'iband': dict(band='i'), 'top1': dict(excise=0.01), 'top5': dict(excise=0.05),
          'cut002': dict(cut=0.002), 'cut0035': dict(cut=0.0035), 'df20': dict(sample='df20'), 'alt': {}}
NEWVAR = {'fid': {}, 'iband': dict(band='i'), 'top1': dict(excise=0.01), 'top5': dict(excise=0.05),
          'cut3': dict(cutk=3), 'cut5': dict(cutk=5), 'alt': {},
          'hz01': dict(hz=0.1), 'hz02': dict(hz=0.2), 'frac03': dict(fracerr=0.3)}
SGVAR = dict(NEWVAR, g1=dict(cal='g1'), reff=dict(cal='reff'), Mp=dict(cal='Mp'), Mm=dict(cal='Mm'), gas=dict(cal='gas'))
SGTESTS = {'signfree': dict(signfree=True), 'signfree_i': dict(signfree=True, band='i'), 'screen': dict(screen=4.22e-5),
           'all': dict(sample='all')}
SGFULLTAGS = ['lam8', 'lam12', 'lam16', 'lam20'] + ['s%d_lam%d' % (s, l) for s in (1, 2, 3, 4) for l in (8, 12, 16, 20)]
NEWTESTS = {'noS': dict(psf=False), 'eq17': dict(est='eq17'), 'signfree': dict(signfree=True),
            'signfree_i': dict(signfree=True, band='i'), 'screen': dict(screen=4.22e-5), 'all': dict(sample='all')}
TESTS = {'screen': dict(screen=4.22e-5), 'all': dict(sample='all'), 'union': dict(sample='union'),
         'signfree': dict(signfree=True), 'signfree_i': dict(signfree=True, band='i')}
FULLTAGS = ['lam8', 'lam12', 'lam16', 'lam20'] + ['s%d_lam%d' % (s, l) for s in (1, 2, 3, 4) for l in (8, 20)]
FULLSUB = {'all': {}, 'near': dict(dist=('lt', 150.)), 'far': dict(dist=('gt', 200.)), 'union': dict(sample='union')}
INJ = {'gj': [0., 1e-4, 2e-4, 4e-4, 1e-3], 'emp': [0., 1e-4, 2e-4, 4e-4, 1e-3],
       'tsig': [0., 1e-4, 2e-4, 4e-4, 1e-3], 'empsig': [0., 1e-4, 2e-4, 4e-4, 1e-3]}
NREP = 20
EXCAL = ('tsig', 'emp')       # signal-free mock families for the excision calibration (HD 2026-10-09)
ALT = {'gj': 't', 't': 'gt', 'tj': 't', 'gt': 't'}   # gj is pathological (spurious mode at 1.4e-3)
# noise truths for the mocks, from the 8 Mpc fiducial fits
TRUTH = dict(sigma=7.40e-4, eps=0.044, so=5.24e-3)   # gj fit, r band, 8 Mpc (job 937384)
SELKW = ('band', 'excise', 'cut', 'sample', 'dist', 'screen', 'fracerr')


def grid(signfree=False, wide=False):
    g = np.concatenate([np.arange(0., 6e-4, 2e-6), np.arange(6e-4, 3.0e-3 + 1e-12, 1e-5)])
    if wide:
        g = np.concatenate([g, np.arange(3.1e-3, 2e-2 + 1e-12, 1e-4)])
        gmax = float(os.environ.get('LAP_GRIDMAX', 2e-2))       # short ranges (sg runs) need more room
        if gmax > 2e-2:
            g = np.concatenate([g, np.arange(2.05e-2, gmax + 1e-12, 5e-4)])
    return np.concatenate([-g[:0:-1], g]) if signfree else g


def names(group):
    if group == 'pub':
        return ['pub%d_%s' % (ii, v) for v in PUBVAR for ii in range(20)]
    if group == 'pubcut':
        return ['pub%d_%s' % (ii, v) for v in ('cut002', 'cut0035') for ii in range(20)]
    if group == 'pubalt':
        return ['pub%d_alt' % ii for ii in range(20)]
    if group == 'test':
        return ['test_' + t for t in TESTS]
    if group == 'full':
        return ['full_%s_%s' % (t, s) for t in FULLTAGS for s in FULLSUB]
    if group == 'newpub':
        return ['newpub%d_%s' % (ii, v) for v in NEWVAR for ii in range(20)]
    if group == 'newpubmain':
        return ['newpub%d_%s' % (ii, v) for v in NEWVAR if not v.startswith('cut') for ii in range(20)]
    if group == 'newpubcut':
        return ['newpub%d_%s' % (ii, v) for v in ('cut3', 'cut5') for ii in range(20)]
    if group == 'newtest':
        return ['newtest_' + t for t in NEWTESTS]
    if group == 'newfull':
        return ['newfull_%s_%s' % (t, s) for t in FULLTAGS for s in FULLSUB]
    if group == 'sgpub':
        return ['sgpub%d_%s' % (ii, v) for v in SGVAR for ii in range(20)]
    if group == 'sggas':
        return ['sgpub%d_gas' % ii for ii in range(20)]
    if group == 'sgtest':
        return ['sgtest_' + t for t in SGTESTS]
    if group == 'sgfull':
        return ['sgfull_%s_%s' % (t, sb) for t in SGFULLTAGS for sb in ('all', 'near', 'far')]
    if group == 'sginj':
        return ['sginj_%s_%g_%d' % (k, L, r) for k in INJ for L in INJ[k] for r in range(NREP)]
    if group in ('sgexcalA', 'sgexcalB'):
        return ['sg' + n for n in names(group[2:])]
    if group == 'excal':
        return names('excalA') + names('excalB')
    if group in ('excalA', 'excalB'):                           # two halves, one node each
        rng_ii = range(0, 10) if group == 'excalA' else range(10, 20)
        return (['excal_19_data_0'] if group == 'excalB' else []) + \
            ['excal_%d_%s_%d' % (ii, k, r) for ii in rng_ii for k in EXCAL for r in range(NREP)]
    if group == 'newinj':
        return ['newinj_%s_%g_%d' % (k, L, r) for k in INJ for L in INJ[k] for r in range(NREP)]
    if group == 'inj':
        return ['inj_%s_%g_%d' % (k, L, r) for k in INJ for L in INJ[k] for r in range(NREP)]
    raise ValueError(group)


def config(name):
    if name.startswith('sgpub'):
        ii, var = name[5:].split('_', 1)
        return 'Signals_warp_NoScreen_%s.dat' % ii, dict(SGVAR[var], new=True, **({} if var == 'oldS' else dict(sg=True))), None, var == 'alt'
    if name.startswith('sgtest_'):
        kw = dict(SGTESTS[name[7:]], new=True)
        kw.setdefault('sg', True)
        return 'Signals_warp_NoScreen_19.dat', kw, None, False
    if name.startswith(('sgfull_', 'sginj_')):
        f, kw, inj, alt = config(name[2:])
        return f, dict(kw, new=True, sg=True), inj, alt
    if name.startswith('newpub'):
        ii, var = name[6:].split('_', 1)
        return 'Signals_warp_NoScreen_%s.dat' % ii, dict(NEWVAR[var], new=True), None, var == 'alt'
    if name.startswith('newtest_'):
        return 'Signals_warp_NoScreen_19.dat', dict(NEWTESTS[name[8:]], new=True), None, False
    if name.startswith(('newfull_', 'newinj_')):
        f, kw, inj, alt = config(name[3:])
        return f, dict(kw, new=True), inj, alt
    if name.startswith('pub'):
        ii, var = name[3:].split('_')
        return 'Signals_warp_NoScreen_%s.dat' % ii, dict(PUBVAR[var]), None, var == 'alt'
    if name.startswith('test_'):
        return 'Signals_warp_NoScreen_19.dat', dict(TESTS[name[5:]]), None, False
    if name.startswith('full_'):
        tag, sub = name[5:].rsplit('_', 1)
        f = ('Signals_warp_FULL_lam%.1f.dat' % float(tag[3:]) if tag.startswith('lam')
             else 'Signals_warp_FULL_%s_lam%.1f.dat' % (tag.split('_')[0], float(tag.split('_')[1][3:])))
        return f, dict(FULLSUB[sub]), None, False
    if name.startswith('inj_'):
        _, kind, L, rep = name.split('_')
        return 'Signals_warp_NoScreen_19.dat', {}, dict(kind=kind, level=float(L), rep=int(rep)), False
    raise ValueError(name)


def mock(d, S, kind, level, rep, tfit, truth=None, pool=None):
    rng = np.random.default_rng(1000 * rep + 7)
    W, M, C = d['W'], d['M'], d['C']
    n = len(W)
    p = W / W.sum(1, keepdims=True)
    k = np.minimum((p.cumsum(1) < rng.random(n)[:, None]).sum(1), (W > 0).sum(1) - 1)
    Mk, Ck = M[np.arange(n), k], C[np.arange(n), k]
    sig = rng.normal(level * Mk, level * np.sqrt(Ck))          # the galaxy's true fifth-force warp
    T = TRUTH if truth is None else truth
    if pool is None:
        pool = np.abs(S[:, 49][np.abs(S[:, 49]) > 0.0025])
    bad = rng.random(n) < T['eps']
    if kind == 'tsig':                                         # signal + Student-t noise
        y = sig + tfit['sigma'] * rng.standard_t(tfit['nu'], n)
        bad[:] = False
    else:
        y = sig + rng.normal(0., T['sigma'], n)
        if kind == 'gj':
            y[bad] = rng.normal(0., T['so'], bad.sum())
        elif kind == 'emp':
            y[bad] = rng.choice(pool, bad.sum()) * rng.choice([-1., 1.], bad.sum())
        elif kind == 'empsig':
            y[bad] = sig[bad] + rng.choice(pool, bad.sum()) * rng.choice([-1., 1.], bad.sum())
    if d.get('e') is not None and np.any(d['e'] > 0):            # per-galaxy measurement noise (remeasured data)
        y = y + rng.normal(0., 1., n) * d['e']
    return y, int(bad.sum())


def excal(model, name):
    """Excision calibration: the fiducial, top-1% and top-5%-excised bounds on the SAME signal-free mock
    (kind 'data': the real warps, which must reproduce newpub<ii>_{fid,top1,top5})."""
    sg = name.startswith('sg')
    _, ii, kind, rep = name.split('_')
    S = load_signals('Signals_warp_NoScreen_%s.dat' % ii)
    d = select(S, **new_data(sg=sg))
    out = dict(name=name, kind=kind, ii=int(ii), rep=int(rep))
    if kind != 'data':
        tfit = json.load(open(P.lap('t_truth_new.json')))
        gj = json.load(open(P.lap('gj_truth_new.json')))
        wf = d['w']
        sd = 1.4826 * np.median(np.abs(wf - np.median(wf)))
        pool = np.abs(wf[np.abs(wf) > 3.5 * sd])
        d['w'], out['n_bad'] = mock(d, S, kind, 0., int(rep), tfit, truth=gj, pool=pool)
    n = len(d['w'])
    amp = (d['W'] * np.abs(d['M'])).sum(1) / np.maximum(d['W'].sum(1), 1e-30)   # as in select(excise=)
    for tag, ex in (('fid', 0.), ('top1', 0.01), ('top5', 0.05)):
        k = amp < np.quantile(amp, 1. - ex) if ex > 0 else np.ones(n, bool)
        dd = {key: (v[k] if isinstance(v, np.ndarray) and v.shape[0] == n else v) for key, v in d.items()}
        q = quantiles(posterior(dd, grid(wide=True), model))
        out[tag] = dict(n=int(k.sum()), q68=q['q68'], q95=q['q95'], mode=q['mode'], n_bad_in_mass=q['n_bad_in_mass'],
                        tail_top5pct=q['tail_top5pct'])
    out['r1'] = out['top1']['q95'] / out['fid']['q95']
    out['r5'] = out['top5']['q95'] / out['fid']['q95']
    outdir = os.path.join(LAP, model)
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, name + '.json'), 'w'), indent=1)
    print('%s fid %.3g top1 %.3g top5 %.3g r1 %.2f r5 %.2f' % (name, out['fid']['q95'], out['top1']['q95'],
                                                              out['top5']['q95'], out['r1'], out['r5']), flush=True)


def main(model, name):
    outdir = os.path.join(LAP, model)
    os.makedirs(outdir, exist_ok=True)
    fname, kw, inj, alt = config(name)
    S = load_signals(fname)
    sel = {k: kw[k] for k in SELKW if k in kw}
    if sel.get('sample') == 'union':
        sel['union'] = union_rows()
    if kw.get('new'):
        nd = new_data(band=kw.get('band', 'r'), hz=kw.get('hz', 0.15), est=kw.get('est', 'mf'), psf=kw.get('psf', True),
                      sg=kw.get('sg', False), cal=kw.get('cal'))
        sel.update(nd)
        sel.pop('band', None)
        if 'cutk' in kw:
            sel['cut'] = new_cuts()[0 if kw['cutk'] == 3 else 1]
            kw['cut'] = sel['cut']
    d = select(S, **sel)
    extra = {}
    if inj is not None:
        tfit = json.load(open(P.lap('t_truth_new.json' if kw.get('new') else 't_truth.json')))
        if kw.get('new'):
            gj = json.load(open(P.lap('gj_truth_new.json')))
            wf = d['w']
            sd = 1.4826 * np.median(np.abs(wf - np.median(wf)))
            pool = np.abs(wf[np.abs(wf) > 3.5 * sd])
            d['w'], extra['n_bad'] = mock(d, S, inj['kind'], inj['level'], inj['rep'], tfit, truth=gj, pool=pool)
        else:
            d['w'], extra['n_bad'] = mock(d, S, inj['kind'], inj['level'], inj['rep'], tfit)
        extra.update(inj)
    m = ALT[model] if alt else model
    t = time.time()
    res = posterior(d, grid(kw.get('signfree', False), wide=kw.get('new', False)), m, cut=kw.get('cut'))
    q = quantiles(res)
    q.update(name=name, model=m, file=fname, n=int(len(d['w'])), seconds=round(time.time() - t, 1),
             kw={k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}, **extra)
    json.dump(q, open(os.path.join(outdir, name + '.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(outdir, name + '_post.npz'), DG=res['DG'], lp=res['lp'])
    print('%s %s N=%d q68 %.4g q95 %.4g logZ %.2f bad %d (in mass %d) tail %.1e %.0fs'
          % (m, name, q['n'], q['q68'], q['q95'], q['logZ'], q['n_bad'], q['n_bad_in_mass'], q['tail_top5pct'],
             q['seconds']), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'list':
        print('\n'.join(names(sys.argv[2])))
    else:
        for nm in sys.argv[2:]:
            (excal if nm.startswith(('excal_', 'sgexcal_')) else main)(sys.argv[1], nm)
