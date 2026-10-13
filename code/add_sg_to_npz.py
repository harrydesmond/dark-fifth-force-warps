"""Add the calib_sg.py suppressions (S2_<est>_<run>, per signals row) to new_w1.npz; the previous file is kept
as new_w1_preSG.npz."""
import glob, json, os, shutil, numpy as np
from calib_sg import RUNS
import paths as P
N = dict(np.load(P.meas('new_w1.npz')))
if not os.path.exists(P.meas('new_w1_preSG.npz')):
    shutil.copy(P.meas('new_w1.npz'), P.meas('new_w1_preSG.npz'))
recs = {}
for f in glob.glob(P.sg('meas_sg', 'rows_*.json')):
    for r in json.load(open(f)):
        recs[r['row']] = r
rows = N['warp_match_row']
for name in RUNS:
    for est in ('mf', 'eq17'):
        k = 'S2_%s_%s' % (est, name)
        N[k] = np.array([recs.get(r, {}).get(k, np.nan) for r in rows], float)
for k in ('mu', 'mu1'):
    N['sg_' + k] = np.array([recs.get(r, {}).get(k, np.nan) for r in rows], float)
gas = {}                                                   # gas self-gravity variation (calib_gas.py)
for f in glob.glob(P.sg('meas_sg_gas', 'rows_*.json')):
    for r in json.load(open(f)):
        gas[r['row']] = r
if gas:
    for k in ('S2_mf_gas', 'S2_eq17_gas', 'fg'):
        N[k if k.startswith('S2') else 'sg_fg'] = np.array([gas.get(r, {}).get(k, np.nan) for r in rows], float)
    print('gas variation: finite S2_mf_gas %d, median f_g %.3f' % (np.isfinite(N['S2_mf_gas']).sum(), np.nanmedian(N['sg_fg'])))
np.savez(P.meas('new_w1.npz'), **N)
print('records %d / %d; finite S2_mf_fid %d; median S2_mf_fid %.3f (old S_mf_r_hz0.15 %.3f)'
      % (sum(r in recs for r in rows), len(rows), np.isfinite(N['S2_mf_fid']).sum(), np.nanmedian(N['S2_mf_fid']),
         np.nanmedian(N['S_mf_r_hz0.15'])))
