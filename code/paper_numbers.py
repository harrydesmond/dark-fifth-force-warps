"""Numbers quoted in the paper that are not printed by analyse_new.py or descriptive_sg.py: sample counts and noise of
the warp measurement (Sec. III), the self-gravity suppression S_i (Sec. IV B), the galaxies that set the bound
(Sec. V), the Vainshtein radii (Sec. VI), the shared-density-field and coupled-quintessence estimates (Sec. VII) and
the coupling conversions (App. B). python3 paper_numbers.py
"""
import json
import os
import numpy as np
from scipy import stats
from astropy.io import fits
import paths as P
import lap_run
from robust_like import ARTEFACTS

N = np.load(P.meas('new_w1.npz'))
c = fits.open(P.cat('warp_match.fits'))[1].data[N['warp_match_row']]
H = np.load(P.cat('warp_halos.npy'))
D = json.load(open(P.results('descriptive_sg.json')))
CONS = 1.645e-3                                                # conservative 95% bound at 8 Mpc (analyse_new.py)

print('=== Sec. III: sample and measurement noise')
art = np.isin(np.arange(len(N['ok_r'])), ARTEFACTS)
ok, res = N['ok_r'], N['ok_r'] & (N['npix_med_r'] >= 3)
fid = res & N['size_ok'] & ~art
print('catalogue %d; usable images %d; resolved (window spans >= 3 pixel rows) %d; failed Sersic fits removed %d;'
      ' artefacts removed %d; fiducial sample %d' % (len(ok), ok.sum(), res.sum(), (res & ~N['size_ok']).sum(),
                                                     (res & N['size_ok'] & art).sum(), fid.sum()))
w, e = N['w1_mf_r'][fid], N['w1_mf_err_r'][fid]
print('median uncertainty %.2g; scatter 1.4826 MAD %.2g; r-i Spearman %.2f'
      % (np.median(e), 1.4826 * np.median(np.abs(w - np.median(w))),
         stats.spearmanr(N['w1_mf_r'][fid & N['ok_i']], N['w1_mf_i'][fid & N['ok_i']])[0]))
q = res & N['size_ok']
ratio = N['w1_noise_r'][q] / N['w1_mf_noise_r'][q]
print('pixel-noise redraws: direct sum (Eq. 1) / matched filter, per galaxy: median %.0f (16-84%%: %.1f-%.0f);'
      ' direct-sum median noise %.2g' % (np.nanmedian(ratio), *np.nanpercentile(ratio, [16, 84]), np.nanmedian(N['w1_noise_r'][q])))

print('\n=== Secs. IV A-B: self-gravity')
S, mu = N['S2_mf_fid'][fid], N['sg_mu'][fid]
print('M*/M_halo(<R_eff) median %.1f; S_i median %.2f (16-84%%: %.2f-%.2f); S_i median %.2f at mu < 0.03 (N=%d), %.2f at'
      ' mu > 0.3 (N=%d); projection only %.2f; seeing factor %.3f'
      % (1 / np.median(mu), np.median(S), *np.percentile(S, [16, 84]), np.median(S[mu < 0.03]), (mu < 0.03).sum(),
         np.median(S[mu > 0.3]), (mu > 0.3).sum(), np.median(N['S2_mf_proj'][fid]), np.median(S / N['S2_mf_nopsf'][fid])))
print('P99 predicted warp %.3f per unit dG/G_N -> w1 = %.2g at the conservative bound, midline displacement %.0f pc'
      ' at L_x = 15 kpc' % (D['P99_amp'], D['P99_amp'] * CONS, 1e3 * D['P99_amp'] * CONS * 15. * 1.5 * 2.5 / 0.5))

print('\n=== Sec. V: the 5% of galaxies with the largest predicted signal')
S19 = np.genfromtxt(os.path.join(P.SIGNALS, 'Signals_warp_NoScreen_19.dat'))
nc = S19[:, 2].astype(int); v = np.arange(15)[None, :] < nc[:, None]
Wt, M = np.where(v, S19[:, 3:18], 0.), np.where(v, S19[:, 18:33], 0.)
pred = (Wt * np.abs(M)).sum(1) / Wt.sum(1) * lap_run.new_data(sg=True)['scale']
t = pred[fid] >= np.quantile(pred[fid], 0.95)
Ms, rho = c['SERSIC_MASS'][fid] / 0.7 ** 2, 10 ** H[fid, 13]
for lab, x in (('fifth-force field |a5|', 10 ** S19[fid, 55]), ('stellar mass', Ms), ('halo rho_s', rho),
               ('M_halo(<R_eff)/M*', mu), ('S_i', S), ('R_eff [kpc]', H[fid, 4]), ('distance [Mpc]', H[fid, 2])):
    print('%-24s top 5%% %.3g  rest %.3g  ratio %.2f' % (lab, np.median(x[t]), np.median(x[~t]), np.median(x[t]) / np.median(x[~t])))

print('\n=== Sec. VI: Vainshtein radius r_V = (16 G M / 9 m^2)^(1/3), m = H0/c')
G, c_kms, H0 = 4.301e-9, 299792.458, 67.4                     # Mpc (km/s)^2 / Msun
for Mh in (1e12, 1e14):
    print('M = %.0e Msun: r_V = %.1f Mpc' % (Mh, (16 * G * Mh / 9 / (H0 / c_kms) ** 2 / c_kms ** 2) ** (1 / 3)))
print('lambda_C at which (H0 lambda_C / c)^2 = 6e-5: %.0f Mpc' % (c_kms / H0 * np.sqrt(6e-5)))

print('\n=== Sec. VII: independence of the galaxies and the coupled-quintessence velocity term')
for fsky in (0.19, 0.25):                                      # SDSS spectroscopic footprint, bracketed
    V = fsky * 4 / 3 * np.pi * 250. ** 3
    print('f_sky %.2f: mean separation %.1f Mpc; (8 Mpc)^3 cells %.1e' % (fsky, (V / fid.sum()) ** (1 / 3), V / 8. ** 3))
Om, fb = 0.315, 0.0493 / 0.315
coef = 2 * Om ** 0.55 / (3 * Om * (1 - fb))                     # a_drag / a_5 = coef (kappa phidot / H) beta
for wphi in (-0.99, -0.95, -0.90):
    A = coef * np.sqrt(3 * (1 + wphi) * (1 - Om))               # dG/G_eff = 2 beta^2 + A beta
    b = (-A + np.sqrt(A * A + 8 * CONS)) / 4
    print('w = %.2f: velocity/static = %.2f/beta; cancellation at beta = %.2f (dG/G = %.2f, %.0fx the bound);'
          ' combined bound |beta| < %.1e' % (wphi, A / 2, A / 2, 2 * (A / 2) ** 2, 2 * (A / 2) ** 2 / CONS, b))
kp = 3 * Om * (1 - fb) / (3 - 1.5 * Om)                         # kappa phidot / H = kp beta, sourced by the coupling alone
print('w -> -1, phidot sourced by the coupling: velocity/static = %.2f' % (coef * kp / 2))

print('\n=== App. B: conventions')
print('beta_Clini = 0.0037 -> beta = %.3f; at the bound beta / beta_Clini = %.0f; |beta| < %.3f'
      % (np.sqrt(0.0037 / 2), np.sqrt(CONS / 2) / CONS, np.sqrt(CONS / 2)))
print('lambda_C = 1 Mpc <-> m_phi = %.2g eV; lambda_C = c/H0 = %.0f Mpc' % (1.97327e-7 / 3.0857e22, c_kms / H0))
