"""Projected midline of a fifth-force-warped, self-gravitating exponential disk seen edge-on, tabulated vs
mu = M_halo(<R_eff)/M* (physics and kernel validation in selfgrav.py, kernel_test.py).

Ring model: (nu_h^2 + sum_j K_ij) z_i - sum_j K_ij z_j = -g, nu_h^2 = G M_halo(<R)/R^3,
M_halo(<R) = mu M* (R/R_eff)^(3-gamma); without self-gravity this is the forward model's
z = -g R^3/(G M_halo(<R)). The edge-on midline at projected x is the surface-density-weighted mean of
z(R) along the line of sight, R = sqrt(x^2 + y^2). Units R_d = G = M* = g = 1.

CONFIGS: name -> (R_eff/R_d, disk scale height hz in units of R_eff (also the ring-force softening),
inner halo slope gamma, self-gravity on). R_eff is SERSIC_TH50: for an edge-on exponential disk the
major-axis half-light radius is 0.941 R_d (fiducial); 1.678 R_d is the face-on value (variation).
"""
import numpy as np
import paths as P
from scipy.special import ellipe

CONFIGS = {'fid': (0.941, 0.15, 0.5, True), 'hz0.1': (0.941, 0.10, 0.5, True), 'hz0.2': (0.941, 0.20, 0.5, True),
           'g1': (0.941, 0.15, 1.0, True), 'proj': (0.941, 0.15, 0.5, False), 'reff': (1.678, 0.15, 0.5, True)}
MUS = np.logspace(-3, 3, 49)
U = np.linspace(0., 30., 3001)                     # |x| / R_d
RMAX, NR = 30., 3000


def kernel(R, w, h):
    """K_ij = (w_j / 2 pi) int dphi (R_i^2 + R_j^2 - 2 R_i R_j cos phi + h^2)^-3/2 (elliptic closed form)."""
    Ri, Rj = R[:, None], R[None, :]
    a = Ri ** 2 + Rj ** 2 + h ** 2
    b = 2 * Ri * Rj
    K = w[None, :] / (2 * np.pi) * 4 * ellipe(2 * b / (a + b)) / ((a - b) * np.sqrt(a + b))
    np.fill_diagonal(K, 0.)
    return K


def table(cfg, mus=MUS):
    """zbar(U; mu) - zbar(0; mu), per unit g, R_d units; shape (len(mus), len(U))."""
    reff, hz, gam, sg = CONFIGS[cfg]
    R = np.linspace(0.004, RMAX, NR)
    dR = R[1] - R[0]
    S = 0.
    if sg:
        K = kernel(R, np.exp(-R) * R * dR, hz * reff)     # ring mass Sigma 2 pi R dR, Sigma = exp(-R)/2pi
        S = np.diag(K.sum(1)) - K
        del K
    y = np.linspace(0., RMAX, 3001)
    RR = np.sqrt(U[:, None] ** 2 + y[None, :] ** 2)
    Wt = np.exp(-RR)
    Wt /= Wt.sum(1, keepdims=True)
    out = np.empty((len(mus), len(U)))
    for k, mu in enumerate(mus):
        z = np.linalg.solve(np.diag(mu * (R / reff) ** (3 - gam) / R ** 3) + S, -np.ones(NR))
        zb = (Wt * np.interp(RR, R, z)).sum(1)
        out[k] = zb - zb[0]
    return out


# Gas variation: an uncoupled gas disk moves with the stars, so it adds to the disk's self-gravity but not to its
# light. Gas surface density exponential with scale length GAS_SCALE R_d and mass f_g M*; f_g per galaxy from the
# xGASS weighted-median total gas fractions, M_gas = 1.3(M_HI + M_H2) (Catinella et al. 2018, MNRAS 476, 875,
# Table 2), linear in log M* between bin centres and flat beyond them.
GAS_SCALE = 2.
FG = np.array([0., 0.03, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0])
XGASS_LOGMS = np.array([9.16, 9.44, 9.75, 10.05, 10.34, 10.65, 10.95, 11.21])
XGASS_LOGFG = np.array([0.148, 0.040, -0.511, -0.485, -0.785, -0.965, -1.238, -1.496])


def gas_fraction(logms):
    return 10 ** np.interp(logms, XGASS_LOGMS, XGASS_LOGFG)


def table_gas(fg, mus=MUS, reff=0.941, hz=0.15, gam=0.5):
    """As table('fid') with the gas disk added to the self-gravitating mass; light (projection) from stars only."""
    R = np.linspace(0.004, RMAX, NR)
    dR = R[1] - R[0]
    sig = np.exp(-R) + fg / GAS_SCALE ** 2 * np.exp(-R / GAS_SCALE)    # 2 pi Sigma, units M*/R_d^2
    K = kernel(R, sig * R * dR, hz * reff)
    S = np.diag(K.sum(1)) - K
    del K
    y = np.linspace(0., RMAX, 3001)
    RR = np.sqrt(U[:, None] ** 2 + y[None, :] ** 2)
    Wt = np.exp(-RR)
    Wt /= Wt.sum(1, keepdims=True)
    out = np.empty((len(mus), len(U)))
    for k, mu in enumerate(mus):
        z = np.linalg.solve(np.diag(mu * (R / reff) ** (3 - gam) / R ** 3) + S, -np.ones(NR))
        zb = (Wt * np.interp(RR, R, z)).sum(1)
        out[k] = zb - zb[0]
    return out


def build_gas(path=P.sg('sg_tables_gas.npz')):
    np.savez(path, MUS=MUS, U=U, FG=FG, Z=np.array([table_gas(f) for f in FG]))


def profile_gas(tg, mu, fg):
    """Midline for the gas variation, bilinear in (log mu, f_g)."""
    j = int(np.clip(np.searchsorted(FG, fg) - 1, 0, len(FG) - 2))
    t = (np.clip(fg, FG[0], FG[-1]) - FG[j]) / (FG[j + 1] - FG[j])
    return (1 - t) * profile({'g': tg['Z'][j]}, 'g', mu) + t * profile({'g': tg['Z'][j + 1]}, 'g', mu)


def build(path=P.sg('sg_tables.npz')):
    np.savez(path, MUS=MUS, U=U, **{c: table(c) for c in CONFIGS})


def profile(tabs, cfg, mu):
    """Midline (R_d units, per unit g, zero at the centre) at mu, interpolated in log mu."""
    lm = np.log(np.clip(mu, MUS[0], MUS[-1]))
    i = int(np.clip(np.searchsorted(np.log(MUS), lm) - 1, 0, len(MUS) - 2))
    t = (lm - np.log(MUS[i])) / (np.log(MUS[i + 1]) - np.log(MUS[i]))
    return (1 - t) * tabs[cfg][i] + t * tabs[cfg][i + 1]


def denominator_L(cfg, mu):
    """|z| at |x| = L = 3 R_eff of the forward model (in-plane, halo-only, gamma = 0.5), per unit g, R_d units."""
    reff = CONFIGS[cfg][0]
    return (3 * reff) ** 0.5 * reff ** 2.5 / mu


if __name__ == '__main__':
    import sys
    build_gas() if 'gas' in sys.argv[1:] else build()
