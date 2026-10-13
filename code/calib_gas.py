"""Gas self-gravity variation of calib_sg.py: S2 with the gas disk of sg_profiles.table_gas added to the disk's
self-gravity (r band, hz = 0.15 R, fiducial geometry). python3 calib_gas.py <start> <stop> -> meas_sg_gas/rows_*.json"""
import json, os, sys
import numpy as np
from astropy.io import fits
import sg_profiles as sgp
from calib_sg import CAT, SEE, MU, ATLAS, synth_S2
import paths as P

H2 = 0.7 ** 2


def main(start, stop):
    tg = dict(np.load(P.sg('sg_tables_gas.npz')))
    reff = sgp.CONFIGS['fid'][0]
    os.makedirs(P.sg('meas_sg_gas'), exist_ok=True)
    prev = {}
    for f in os.listdir(P.meas('meas')):
        if f.startswith('rows_') and f.endswith('.json'):
            for r in json.load(open(P.meas('meas', f))):
                prev[r['row']] = r
    out = []
    for row in range(start, stop):
        ms = CAT['SERSIC_MASS'][row] / H2
        rec = dict(row=row, mu=float(MU[row, 0]), fg=float(sgp.gas_fraction(np.log10(ms))) if ms > 0 else np.nan)
        p = prev.get(row, {})
        f = os.path.join(ATLAS, '%s-%d-atlas-%d.fits.gz' % (CAT['IAUNAME'][row].strip(), CAT['PID'][row], CAT['AID'][row]))
        if not (np.isfinite(rec['mu']) and np.isfinite(rec['fg'])) or not os.path.exists(f) \
                or p.get('flag_r') != 'ok' or 'flux_r' not in p:
            out.append(rec); continue
        with fits.open(f) as h:
            gain, rn = h[2].header.get('GAIN', np.nan), h[2].header.get('RNOISE', 0.)
        R, phi = float(CAT['SERSIC_TH50'][row]), float(CAT['SERSIC_PHI'][row])
        Rd = R / reff
        s = 0.05 * R / (sgp.denominator_L('fid', rec['mu']) * Rd)
        zb = -sgp.profile_gas(tg, rec['mu'], rec['fg'])
        zf = lambda ax: s * Rd * np.interp(ax / Rd, sgp.U, zb)
        smf, s17 = synth_S2(R, phi, float(SEE[row, 0]), 0.15, p['flux_r'], gain, rn, row, zf, reff)
        rec['S2_mf_gas'], rec['S2_eq17_gas'] = smf, s17
        out.append(rec)
    json.dump(out, open(P.sg('meas_sg_gas', 'rows_%05d_%05d.json' % (start, stop)), 'w'), default=float)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
