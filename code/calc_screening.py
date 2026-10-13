"""Phi_LSS, the rms potential of the linear density modes, which sets the screening threshold of Sec. VI
(Phi_LSS/c^2 = 4.2e-5 for a massless mediator, R = 50 Mpc top hat, infrared cut at the horizon).

Phi_LSS is computed as an rms over a linear power spectrum with a top-hat window (Eisenstein & Hu 1998 zero-baryon
transfer function, normalised to sigma_8). The thin-shell condition compares chi with the potential the scalar sees,
whose kernel is 1/(k^2 + m_phi^2): it reduces to the Newtonian 1/k^2 only as m_phi -> 0, where the variance is weakly
infrared divergent and is cut at the horizon, k_min = H0/c. With a finite range the mass regulates the integral itself.
The galaxy potentials compared with it (columns 51-53 of the signals tables) are summarised by descriptive_sg.py.
python3 calc_screening.py
"""
import numpy as np

Om, Ob, h, ns, sig8 = 0.315, 0.0493, 0.674, 0.965, 0.811
Tcmb = 2.7255
CoverH0 = 299792.458 / (100. * h)          # Mpc
HoverC = 1. / CoverH0                       # Mpc^-1


def T_EH98(k_hMpc):
    """Zero-baryon transfer function; k in h/Mpc."""
    om, ob = Om * h * h, Ob * h * h
    s = 44.5 * np.log(9.83 / om) / np.sqrt(1. + 10. * ob**0.75)          # Mpc
    aG = (1. - 0.328 * np.log(431. * om) * (Ob / Om)
          + 0.38 * np.log(22.3 * om) * (Ob / Om)**2)
    k = k_hMpc * h                                                       # Mpc^-1
    Geff = Om * h * (aG + (1. - aG) / (1. + (0.43 * k * s)**4))
    q = k_hMpc * (Tcmb / 2.7)**2 / Geff
    L0 = np.log(2. * np.e + 1.8 * q)
    C0 = 14.2 + 731. / (1. + 62.5 * q)
    return L0 / (L0 + C0 * q * q)


lnk = np.linspace(np.log(1e-5), np.log(1e3), 6000)      # k in h/Mpc
k = np.exp(lnk)
Pk = k**ns * T_EH98(k)**2


def W(x):
    return np.where(x < 1e-4, 1. - x*x/10., 3. * (np.sin(x) - x * np.cos(x)) / x**3)


def sigma(R_hMpc):
    """rms density contrast in top-hat spheres of radius R [Mpc/h], unnormalised P(k)."""
    return np.sqrt(np.trapz(k**3 * Pk * W(k * R_hMpc)**2 / (2. * np.pi**2), lnk))


norm = sig8**2 / sigma(8.)**2
Pk *= norm

# Poisson at a = 1 in comoving coordinates: Phi/c^2 = -(3/2) Om (H0/c)^2 delta / k^2, k in Mpc^-1
kMpc = k * h
pref = 1.5 * Om * HoverC**2 / kMpc**2

K_HORIZON = HoverC / h                      # H0/c in h/Mpc = 3.34e-4; the massless IR cut
LAM_MASSLESS = 1. / HoverC                  # c/H0 = 4448 Mpc, the largest physically distinct range


def phi_rms(R_Mpc, lamC_Mpc=None, kmin=None):
    """rms scalar potential fluctuation smoothed on radius R [Mpc].

    lamC_Mpc  Compton wavelength of the mediator; the kernel is 1/(k^2 + lamC^-2).  None gives
              the massless Newtonian 1/k^2 kernel, which needs an explicit IR cut.
    kmin      IR cut in h/Mpc.  Defaults to the horizon for the massless kernel and to no cut
              for a finite range, where the mass regulates the integral itself.
    """
    m = 0. if lamC_Mpc is None else 1. / lamC_Mpc                   # Mpc^-1
    if kmin is None:
        kmin = K_HORIZON if lamC_Mpc is None else 0.
    pref_m = 1.5 * Om * HoverC**2 / (kMpc**2 + m * m)
    integ = kMpc**3 * (Pk / h**3) * pref_m**2 * W(k * R_Mpc * h)**2 / (2. * np.pi**2)
    return np.sqrt(np.trapz(np.where(k >= kmin, integ, 0.), lnk))


def sigma_R(R_Mpc):
    return sigma(R_Mpc * h)


if __name__ == '__main__':
    print('sigma_8 check: %.4f' % sigma(8.))

    print('\nMASSLESS kernel 1/k^2: Phi_LSS against the infrared cut (R = 50 Mpc top hat):')
    print('%14s %14s %s' % ('k_min /(h/Mpc)', 'Phi_rms/c^2', 'note'))
    for km, note in ((1e-5, 'super-horizon; Newtonian Poisson invalid'),
                     (1e-4, ''),
                     (K_HORIZON, 'FIDUCIAL: k = H0/c'),
                     (1e-3, ''),
                     (2.*np.pi*K_HORIZON, 'k = 2 pi H0/c')):
        print('%14.2e %14.2e %s' % (km, phi_rms(50., None, km), note))

    print('\nFINITE-RANGE kernel 1/(k^2+lamC^-2), no IR cut needed:')
    print('%10s %14s %14s %s' % ('lamC/Mpc', 'Phi_LSS(R=50)', 'Phi_LSS(R=lamC)', 'note'))
    for lam, note in ((8., 'shortest range quoted in this paper'),
                      (12., ''), (20., ''), (50., ''), (100., ''), (200., ''),
                      (500., ''), (1000., ''), (2000., ''),
                      (LAM_MASSLESS, 'lamC = c/H0; cf. 4.22e-5 massless'),
                      (1e5, 'effectively massless')):
        print('%10.0f %14.2e %14.2e %s' % (lam, phi_rms(50., lam), phi_rms(lam, lam), note))

    print('\nPhi_LSS (massless, horizon cut) against top-hat radius:')
    print('%8s %12s %12s' % ('R/Mpc', 'sigma(R)', 'Phi_rms/c^2'))
    for R in (10., 25., 50., 100., 200.):
        print('%8.0f %12.3f %12.2e' % (R, sigma_R(R), phi_rms(R)))
