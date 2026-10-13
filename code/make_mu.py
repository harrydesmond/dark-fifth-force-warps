"""Per warp_match row: mu = M_halo(<R_eff)/M* for the gamma = 0.5 forward-model halo and for gamma = 1, with
R_eff = SERSIC_TH50 x D (as in the forward model), the halo's median r_s and rho(r_s) from warp_halos.npy
(rho = rho_s (r/r_s)^-gamma inside r_s, Calc_Proxies_warp_NoScreen.py:381-391) and M* = NSA SERSIC_MASS/h^2
(NSA masses are h = 1; h = 0.7). Writes mu_sg.npy (4357 x 2; NaN where the row has no signals entry)."""
import numpy as np
from astropy.io import fits
import paths as P
c = fits.open(P.cat('warp_match.fits'))[1].data
sig_rows = np.where(np.isfinite(c['w1_r']) & np.isfinite(c['w1_i']))[0]          # as build_new_w1.py
H = np.load(P.cat('warp_halos.npy'))
assert len(sig_rows) == len(H) and np.allclose(c['RA'][sig_rows], H[:, 0], atol=1e-4)
Reff, rs, rho = H[:, 4], 10 ** H[:, 11], 10 ** H[:, 13]
Ms = c['SERSIC_MASS'][sig_rows] / 0.7 ** 2
mu = np.full((len(c), 2), np.nan)
for j, g in enumerate((0.5, 1.)):
    mu[sig_rows, j] = 4 * np.pi * rho * rs ** g * Reff ** (3 - g) / (3 - g) / Ms
mu[~np.isfinite(mu) | (mu <= 0)] = np.nan
np.save(P.sg('mu_sg.npy'), mu)
print('rows with mu: %d; bad M*: %d; median mu(0.5) %.3f, mu(1) %.3f' % (np.isfinite(mu[:, 0]).sum(),
      np.sum(~(Ms > 0)), np.nanmedian(mu[:, 0]), np.nanmedian(mu[:, 1])))
