"""Disk self-gravity and edge-on projection of the fifth-force warp (linear ring model).

An exponential thin disk (scale length R_d, softening h for the vertical ring-ring force) sits in a
power-law halo, M_halo(<R) = mu M* (R/R_eff)^(3-gam), and feels a uniform acceleration deficit g relative
to it. Rings obey (nu_h^2 + sum_j K_ij) z_i - sum_j K_ij z_j = -g, with nu_h^2 = G M_halo(<R)/R^3 and K_ij the
vertical force per unit relative displacement between rings (elliptic-integral closed form; reproduces the
sheet bending result 2 pi G Sigma k to 2%, kernel_test.py). Without self-gravity this is the forward
model's z = -g R^3/(G M_halo(<R)) (Calc_Proxies_warp_NoScreen.py:391). Units G = R_d = M* = g = 1.

factor(): the matched-filter w1 (template |x|^GT) of the self-gravitating disk seen edge-on, divided by the
Eq. 17 w1 of the in-plane, halo-only gamma = 0.5 disk, i.e. by what the forward model predicts.
"""
import numpy as np
from scipy.special import ellipe

REFF = 1.678            # R_eff / R_d, exponential disk
L = 3. * REFF           # measurement half-window
GT = 0.5                # matched-filter template exponent = forward-model gamma


def kernel(R, w, h):
    """K_ij = (w_j / 2 pi) int_0^2pi dphi (R_i^2 + R_j^2 - 2 R_i R_j cos phi + h^2)^-3/2, w_j the ring mass."""
    Ri, Rj = R[:, None], R[None, :]
    a = Ri ** 2 + Rj ** 2 + h ** 2
    b = 2 * Ri * Rj
    K = w[None, :] / (2 * np.pi) * 4 * ellipe(2 * b / (a + b)) / ((a - b) * np.sqrt(a + b))
    np.fill_diagonal(K, 0.)
    return K


_KCACHE = {}


def solve(mu, h, gam=0.5, sg=True, n=1500, Rmax=12.):
    R = np.linspace(0.004, Rmax, n)
    dR = R[1] - R[0]
    nu2 = mu * (R / REFF) ** (3 - gam) / R ** 3
    A = np.diag(nu2)
    if sg:
        key = (h, n, Rmax)
        if key not in _KCACHE:
            w = np.exp(-R) * R * dR                      # Sigma 2 pi R dR with Sigma = exp(-R)/2pi
            K = kernel(R, w, h)
            _KCACHE[key] = np.diag(K.sum(1)) - K
        A = A + _KCACHE[key]
    return R, np.linalg.solve(A, -np.ones(n))


def stats(R, z, proj, weight='light'):
    """(Eq. 17 w1, matched-filter w1). proj: light-weighted mean height along the line of sight.
    weight: matched-filter column weights, 'light' (photon-limited, ~column flux), 'light2' (background-
    limited, ~flux^2) or 'uniform'."""
    x = np.linspace(-L, L, 801)
    if proj:
        y = np.linspace(0., 12., 2401)
        RR = np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
        Iw = np.exp(-RR)
        zx = (Iw * np.interp(RR, R, z)).sum(1) / Iw.sum(1)
        Ix = Iw.sum(1)
    else:
        zx = np.interp(np.abs(x), R, z)
        Ix = np.exp(-np.abs(x))
    zp = zx - zx.mean()
    w1_eq17 = np.trapz(np.abs(x) * zp, x) / L ** 3
    W = {'light': Ix, 'light2': Ix ** 2, 'uniform': np.ones_like(x)}[weight]
    sw = np.sqrt(W / W.max())
    X = np.vstack([np.ones_like(x), x, np.abs(x) ** GT]).T
    K = np.linalg.lstsq(X * sw[:, None], zx * sw, rcond=None)[0][2]
    return w1_eq17, GT * K * L ** (GT - 1) / ((GT + 1) * (GT + 2))


def numerator_table(h, gam=0.5, sg=True, proj=True, weight='light', mus=np.logspace(-3, 3, 37)):
    """MF w1 per unit g of the (self-gravitating, projected) disk vs mu."""
    return mus, np.array([stats(*solve(m, h, gam, sg=sg), proj, weight)[1] for m in mus])


def denominator_unit():
    """Eq. 17 w1 per unit g of the in-plane halo-only gamma = 0.5 disk at mu = 1 (it scales as 1/mu)."""
    return stats(*solve(1., 0.1, 0.5, sg=False), False)[0]


def factor(mu_num, mu_den, h, gam=0.5, sg=True, proj=True, weight='light'):
    """Per-galaxy factor: numerator at mu_num (halo slope gam) over the forward model at mu_den (gamma 0.5)."""
    mus, num = numerator_table(h, gam, sg, proj, weight)
    lm = np.log(np.clip(mu_num, mus[0], mus[-1]))
    return np.interp(lm, np.log(mus), num) * mu_den / denominator_unit()


if __name__ == '__main__':
    h = 0.15 * REFF
    mus = np.array([0.01, 0.03, 0.09, 0.3, 1., 3.])
    for wt in ('light', 'light2', 'uniform'):
        print(wt, 'F(gamma 0.5):', np.round(factor(mus, mus, h, weight=wt), 3),
              ' projection only:', np.round(factor(1., 1., h, sg=False, weight=wt), 3))
