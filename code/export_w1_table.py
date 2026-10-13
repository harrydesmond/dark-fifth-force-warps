"""Readable per-galaxy table of the warp measurements, measurement/w1_table.csv, one row per galaxy of the forward
model's signals tables (the same order), with the flags that define the fiducial sample of 3968.
python3 export_w1_table.py
"""
import numpy as np
from astropy.io import fits
import paths as P
from robust_like import ARTEFACTS

N = np.load(P.meas('new_w1.npz'))
c = fits.open(P.cat('warp_match.fits'))[1].data[N['warp_match_row']]
fid = N['ok_r'] & (N['npix_med_r'] >= 3) & N['size_ok'] & ~np.isin(np.arange(len(N['ok_r'])), ARTEFACTS)
cols = [('signals_row', np.arange(len(fid)), '%d'), ('nsaid', c['NSAID'], '%d'), ('iauname', [s.strip() for s in c['IAUNAME']], '%s'),
        ('ra_deg', c['RA'], '%.6f'), ('dec_deg', c['DEC'], '%.6f'), ('sersic_th50_arcsec', c['SERSIC_TH50'], '%.3f'),
        ('sersic_phi_deg', c['SERSIC_PHI'], '%.2f'), ('w1_r', N['w1_mf_r'], '%.6e'), ('w1_err_r', N['w1_mf_err_r'], '%.3e'),
        ('w1_i', N['w1_mf_i'], '%.6e'), ('w1_err_i', N['w1_mf_err_i'], '%.3e'), ('seeing_r_arcsec', N['seeing_r'], '%.3f'),
        ('S_i', N['S2_mf_fid'], '%.4f'), ('mu_halo_over_star', N['sg_mu'], '%.4f'), ('measured_r', N['ok_r'], '%d'),
        ('resolved_r', N['npix_med_r'] >= 3, '%d'), ('size_ok', N['size_ok'], '%d'), ('fiducial', fid, '%d')]
with open(P.meas('w1_table.csv'), 'w') as f:
    f.write('# Warp measurements of Desmond & Ferreira (2026): matched-filter w1 (r and i band) and its statistical\n'
            '# uncertainty, seeing, the suppression factor S_i (self-gravity, projection, seeing, pixels, window) and\n'
            '# M_halo(<R_eff)/M_star. fiducial = measured, resolved, size_ok and not a catalogue artefact.\n')
    f.write(','.join(n for n, _, _ in cols) + '\n')
    for i in range(len(fid)):
        f.write(','.join(fmt % (int(v[i]) if fmt == '%d' else v[i]) if not (fmt != '%s' and fmt != '%d' and np.isnan(v[i])) else ''
                         for _, v, fmt in cols) + '\n')
print('measurement/w1_table.csv: %d galaxies, %d in the fiducial sample' % (len(fid), fid.sum()))
