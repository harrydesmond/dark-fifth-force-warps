"""Assemble meas/rows_*.json into per-signals-row arrays (the 4345 rows of Signals_warp_*.dat).
Signals rows are the warp_match.fits rows with finite DF20 w1_r and w1_i (prepare_warp_cat.py mask), in order;
verified against warp_halos.npy RA/Dec. Also sets the size-consistency flag size_ok (SERSIC_TH50 / ELPETRO_TH50[r]
within a factor 3; outside it the Sersic fit has failed). Writes <data>/measurement/new_w1.npz."""
import glob, json, numpy as np
from astropy.io import fits
import paths as P
c = fits.open(P.cat('warp_match.fits'))[1].data
sig_rows = np.where(np.isfinite(c['w1_r']) & np.isfinite(c['w1_i']))[0]
H = np.load(P.cat('warp_halos.npy'))
assert len(sig_rows) == len(H) and np.allclose(c['RA'][sig_rows], H[:, 0], atol=1e-4) and np.allclose(c['DEC'][sig_rows], H[:, 1], atol=1e-4)
recs = {}
for f in glob.glob(P.meas('meas', 'rows_*.json')):
    for r in json.load(open(f)):
        recs[r['row']] = r
keys = ['w1_mf', 'w1_mf_err', 'w1_mf_noise', 'w1', 'w1_err', 'w1_noise', 'w1_odd', 'tilt', 'mf_chi2_dof', 'npix_med', 'nbins', 'flux']
out = {'warp_match_row': sig_rows, 'th50': c['SERSIC_TH50'][sig_rows]}
for b in ('r', 'i'):
    for k in keys:
        out['%s_%s' % (k, b)] = np.array([recs.get(r, {}).get('%s_%s' % (k, b), np.nan) for r in sig_rows], float)
    out['ok_%s' % b] = np.array([recs.get(r, {}).get('flag_%s' % b) == 'ok' for r in sig_rows])
    out['seeing_%s' % b] = np.array([recs.get(r, {}).get('seeing_%s' % b, np.nan) for r in sig_rows], float)
    for hz in (0.1, 0.15, 0.2):
        for est in ('mf', 'eq17'):
            out['S_%s_%s_hz%g' % (est, b, hz)] = np.array([recs.get(r, {}).get('S_%s_%s_hz%g' % (est, b, hz), np.nan) for r in sig_rows], float)
out['sersic_elpetro_ratio'] = c['SERSIC_TH50'][sig_rows] / c['ELPETRO_TH50'][sig_rows, 4]
out['size_ok'] = (out['sersic_elpetro_ratio'] >= 1. / 3.) & (out['sersic_elpetro_ratio'] <= 3.)
np.savez(P.meas('new_w1.npz'), **out)
print('rows with a record: %d / %d; ok r %d, i %d' % (sum(r in recs for r in sig_rows), len(sig_rows), out['ok_r'].sum(), out['ok_i'].sum()))
