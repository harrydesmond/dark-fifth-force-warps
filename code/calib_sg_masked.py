"""calib_sg.py (S2: seeing + pixels + window + edge-on projection + disk self-gravity) with each galaxy's point-source
mask (from measure_masked.py, per band) applied to every synthetic image, and the masked measurement's window flux.
python3 calib_sg_masked.py <start> <stop>   ->  meas_sg_mask<SUFFIX>/rows_<start>_<stop>.json
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import paths as P
import calib_sg as cs                                       # noqa: E402
import sg_profiles as sgp                                   # noqa: E402
import measure_masked as mm                                 # noqa: E402  (installs nothing in calib_sg by itself)

SUF = os.environ.get('SUFFIX', '_gaia5')
cs.w1_from_image = mm.synth_w1                              # synthetic images get the current galaxy's mask


def main(start, stop):
    tabs = dict(np.load(P.sg('sg_tables.npz')))
    od = P.mask('meas_sg_mask' + SUF)
    os.makedirs(od, exist_ok=True)
    prev = {}
    md = P.mask('meas_mask' + SUF)
    for f in os.listdir(md):
        if f.startswith('rows_') and f.endswith('.json'):
            for r in json.load(open(os.path.join(md, f))):
                prev[r['row']] = r
    out = []
    for row in range(start, stop):
        rec = dict(row=row, mu=float(cs.MU[row, 0]), mu1=float(cs.MU[row, 1]))
        p = prev.get(row, {})
        f = os.path.join(cs.ATLAS, '%s-%d-atlas-%d.fits.gz' % (cs.CAT['IAUNAME'][row].strip(), cs.CAT['PID'][row], cs.CAT['AID'][row]))
        if not np.isfinite(rec['mu']) or not os.path.exists(f):
            out.append(rec); continue
        mus = {'m': rec['mu'], 'm1': rec['mu1'], 'mp': rec['mu'] / cs.DEX, 'mm': rec['mu'] * cs.DEX}
        hdr = {}
        from astropy.io import fits
        with fits.open(f) as h:
            for b, hdu in (('r', 2), ('i', 3)):
                hdr[b] = (h[hdu].header.get('GAIN', np.nan), h[hdu].header.get('RNOISE', 0.))
        R, phi = float(cs.CAT['SERSIC_TH50'][row]), float(cs.CAT['SERSIC_PHI'][row])
        for name, (tab, hz, mn, mdn, b, fs) in cs.RUNS.items():
            if p.get('flag_%s' % b) != 'ok' or 'flux_%s' % b not in p:
                continue
            mm.CIRC = [tuple(c) for c in p.get('circ_%s' % b, [])]
            mm.STRIPS = [tuple(c) for c in p.get('strip_%s' % b, [])]
            zf = cs.zfun_for(tabs, tab, mus[mn], mus[mdn], R)
            smf, s17 = cs.synth_S2(R, phi, float(cs.SEE[row, 0 if b == 'r' else 1]) * fs, hz, p['flux_%s' % b],
                                   hdr[b][0], hdr[b][1], row, zf, sgp.CONFIGS[tab][0])
            mm.CIRC, mm.STRIPS = [], []
            rec['S2_mf_' + name], rec['S2_eq17_' + name] = smf, s17
        out.append(rec)
    json.dump(out, open(os.path.join(od, 'rows_%05d_%05d.json' % (start, stop)), 'w'), default=float)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
