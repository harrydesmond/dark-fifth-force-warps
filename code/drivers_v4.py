"""Which galaxies change the masked posterior (Gaia stars, residual-amplitude criterion) relative to the unmasked one?
Per-galaxy log-likelihood at the masked fit's noise parameters theta, dG = +-2e-3 vs 0, masked minus unmasked; then
fixed-theta posteriors with the top drivers removed or restored to their unmasked w.
python3 drivers_v4.py [ii=19] [var=fid|iband] [signed=0|1]"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
import paths as P
import lap_run as L
from robust_like import load_signals, select
from robust_lap import build
ii = int(sys.argv[1]) if len(sys.argv) > 1 else 19
var = sys.argv[2] if len(sys.argv) > 2 else 'fid'
signed = len(sys.argv) > 3 and sys.argv[3] == '1'
band = 'i' if var == 'iband' else 'r'
S = load_signals('Signals_warp_NoScreen_%d.dat' % ii)
D = {}
for tag, npz in (('u', P.meas('new_w1.npz')), ('m', P.mask('new_w1_masked_gaia5.npz'))):
    L.NEWF = npz
    D[tag] = select(S, **L.new_data(sg=True, band=band))
jf = ('sgtest_signfree%s.json' % ('_i' if band == 'i' else '')) if signed else 'sgpub%d_%s.json' % (ii, var)
j = json.load(open(P.lap('masked', 't', jf)))
th = np.array([j['theta_at_mode'][k] for k in ('sigma', 'lnu')])
DG = np.linspace(-5e-3 if signed else 0, 5e-3, 251 if signed else 126)
def ll(d):
    pg = build(d, 't')[-1]
    return np.array([np.asarray(pg(x, th)) for x in DG])
LL = {t: ll(D[t]) for t in D}
ru, rm = D['u']['rows'], D['m']['rows']
com = np.intersect1d(ru, rm); iu = np.searchsorted(ru, com); im = np.searchsorted(rm, com)
k0 = np.argmin(np.abs(DG)); kp = np.argmin(np.abs(DG - 2e-3)); kn = np.argmin(np.abs(DG + 2e-3))
gu = LL['u'][kp, iu] - LL['u'][k0, iu]; gm = LL['m'][kp, im] - LL['m'][k0, im]
dd = gm - gu
if signed:
    hu = LL['u'][kn, iu] - LL['u'][k0, iu]; hm = LL['m'][kn, im] - LL['m'][k0, im]
    dd = np.maximum(np.abs(gm - gu), np.abs(hm - hu)) * np.sign(gm - gu + hm - hu)
print('ii %d %s band %s signed %s theta %s; galaxies: unmasked %d masked %d common %d' % (ii, var, band, signed, np.round(th, 4), len(ru), len(rm), len(com)))
print('sum gain(+2e-3 vs 0): unmasked %.2f masked %.2f' % ((LL['u'][kp] - LL['u'][k0]).sum(), (LL['m'][kp] - LL['m'][k0]).sum()))
if signed:
    print('sum gain(-2e-3 vs 0): unmasked %.2f masked %.2f' % ((LL['u'][kn] - LL['u'][k0]).sum(), (LL['m'][kn] - LL['m'][k0]).sum()))
N = np.load(P.mask('new_w1_masked_gaia5.npz')); O = np.load(P.meas('new_w1.npz'))
o = np.argsort(-np.abs(dd))
cal = 'ifid' if band == 'i' else 'fid'
print('|change|>0.5: %d galaxies; sum of |change| top 1/3/10: %s' % ((np.abs(dd) > 0.5).sum(), np.round(np.cumsum(np.abs(dd[o]))[[0, 2, 9]], 2)))
print('  row  wm_row  gain_m  gain_u   w_mask    w_unmask   e_m      e_u    chi2 m/u    S2 m/u   nstrip mfrac contam')
for i in o[:15]:
    r = com[i]
    print('%5d %6d %7.2f %7.2f %+.2e %+.2e %.1e %.1e %5.0f/%-5.0f %.2f/%.2f %3d %.2f %.1f' % (
        r, N['warp_match_row'][r], gm[i], gu[i], N['w1_mf_' + band][r], O['w1_mf_' + band][r], N['w1_mf_err_' + band][r], O['w1_mf_err_' + band][r],
        N['mf_chi2_dof_' + band][r], O['mf_chi2_dof_' + band][r], N['S2_mf_' + cal][r], O['S2_mf_' + cal][r], N['nstrip_' + band][r], N['mask_frac_' + band][r], N['contam_max_' + band][r]))
def qs(lp):
    p = np.exp(lp - lp.max()); c = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(DG))]); c /= c[-1]
    if signed:
        m = np.trapz(p * DG, DG) / np.trapz(p, DG); sd = np.sqrt(np.trapz(p * (DG - m) ** 2, DG) / np.trapz(p, DG))
        return m, sd
    return np.interp(0.68, c, DG), np.interp(0.95, c, DG)
lab = 'mean/sd' if signed else 'q68/q95'
print('fixed-theta %s: unmasked %.2e/%.2e  masked %.2e/%.2e' % (lab, *qs(LL['u'].sum(1)), *qs(LL['m'].sum(1))))
for k in (1, 3, 5, 10, 20):
    drop = np.zeros(len(rm), bool); drop[im[o[:k]]] = True
    swap = LL['m'].sum(1) - LL['m'][:, im[o[:k]]].sum(1) + LL['u'][:, iu[o[:k]]].sum(1)
    print('  top %2d drivers removed: %.2e/%.2e   restored to unmasked w: %.2e/%.2e' % (k, *qs(LL['m'][:, ~drop].sum(1)), *qs(swap)))
np.save(P.results('drivers_v4_%d_%s%s.npy' % (ii, var, '_signed' if signed else '')), np.array([com[o], dd[o]]))
