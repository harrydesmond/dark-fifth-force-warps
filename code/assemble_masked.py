"""new_w1_masked.npz: a copy of measurement/new_w1.npz with every measurement and suppression key replaced by its
point-source-masked value (meas_mask<SUFFIX>/, meas_sg_mask<SUFFIX>/), plus nstar_mask_<b> and mask_frac_<b>.
Catalogue-level keys (size_ok, sersic_elpetro_ratio, seeing, th50, sg_mu*) are kept; S2_*_gas (gas variation) is
not recomputed and is kept from the unmasked calibration (flag gas_S2_unmasked). python3 assemble_masked.py  -> new_w1_masked<SUFFIX>.npz"""
import glob, json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
import paths as P
SUF = os.environ.get('SUFFIX', '_gaia5')
SRC = P.meas('new_w1.npz')
N = dict(np.load(SRC))
rows = N['warp_match_row']
def recs(d):
    out = {}
    for f in glob.glob(P.mask(d, 'rows_*.json')):
        for r in json.load(open(f)):
            out[r['row']] = r
    return out
M, G = recs('meas_mask' + SUF), recs('meas_sg_mask' + SUF)
keys = ['w1_mf', 'w1_mf_err', 'w1_mf_noise', 'w1', 'w1_err', 'w1_noise', 'w1_odd', 'tilt', 'mf_chi2_dof', 'npix_med',
        'nbins', 'flux', 'nstar_mask', 'mask_frac', 'nstrip', 'ngaia', 'contam_max', 'w1_mf_starsub', 'nresid']
for b in ('r', 'i'):
    for k in keys:
        N['%s_%s' % (k, b)] = np.array([M.get(r, {}).get('%s_%s' % (k, b), np.nan) for r in rows], float)
    N['ok_%s' % b] = np.array([M.get(r, {}).get('flag_%s' % b) == 'ok' for r in rows])
    for hz in (0.1, 0.15, 0.2):
        for est in ('mf', 'eq17'):
            k = 'S_%s_%s_hz%g' % (est, b, hz)
            N[k] = np.array([M.get(r, {}).get(k, np.nan) for r in rows], float)
for k in [k for k in N if k.startswith('S2_') and not k.endswith('_gas')]:
    N[k] = np.array([G.get(r, {}).get(k, np.nan) for r in rows], float)
N['gas_S2_unmasked'] = np.array(True)                      # S2_*_gas kept from the unmasked calibration
out = P.mask('new_w1_masked%s.npz' % SUF)
np.savez(out, **N)
O = np.load(SRC)
same = O['ok_r'] & N['ok_r']
d = N['w1_mf_r'][same] - O['w1_mf_r'][same]
print('%s: records %d/%d; ok r %d (was %d); galaxies with a star masked in the r window %d; median mask_frac (masked) %.3f;'
      ' w1_mf_r changed by > 1e-6 for %d; median S2_mf_fid %.3f (was %.3f)'
      % (out, sum(r in M for r in rows), len(rows), N['ok_r'].sum(), O['ok_r'].sum(), (N['mask_frac_r'] > 0).sum(),
         np.nanmedian(N['mask_frac_r'][N['mask_frac_r'] > 0]), (np.abs(d) > 1e-6).sum(), np.nanmedian(N['S2_mf_fid']),
         np.nanmedian(O['S2_mf_fid'])))
