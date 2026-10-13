"""Corner plot of the 8 Mpc fiducial posterior (dG/G_N, sigma_w1, nu) from the brute-force 3-D grid
results/corner_8mpc.npz (corner_8mpc.py). Flat priors in (dG, sigma, ln nu),
so the grid values are the posterior up to a constant. The 1-D dG panel and the dashed 95% line use the
paper's Laplace-grid posterior (inference/t/sgpub19_fid_post.npz, grid to 2e-2), which the 3-D grid reproduces
to 0.7% at 68% and 2% at 95% (the 3-D grid stops at 1.6e-3). Writes plots/Corner_8Mpc.{png,pdf}."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
import paths as P_

r = np.load(P_.results('corner_8mpc.npz'))
DG, SG, LNU, LL = r['DG'], r['SG'], r['LNU'], r['LL']
P = np.exp(LL - LL.max())
L = np.load(P_.lap('t', 'sgpub19_fid_post.npz'))
pl = np.exp(L['lp'] - L['lp'].max())
cl = np.concatenate([[0.], np.cumsum(0.5 * (pl[1:] + pl[:-1]) * np.diff(L['DG']))])
q95 = float(np.interp(0.95, cl / cl[-1], L['DG']))
lap_x, lap_p = L['DG'] / 1e-4, pl / (cl[-1] / 1e-4)

NU = np.exp(LNU)
X = [DG / 1e-4, SG / 1e-3, NU]                        # plotted coordinates (nu on a linear axis; grid uniform in ln nu)
JAC = [np.ones_like(DG), np.ones_like(SG), 1. / NU]   # density per plotted unit = density per grid unit x JAC
LAB = [r'$\Delta G/G_N\;[10^{-4}]$', r'$\sigma_{w_1}\;[10^{-3}]$', r'$\nu$']
C1, C2 = '#1f4e8c', '#9db7dc'                         # one hue: dark line / light 95% fill
PAD = 0.08                                            # zoomed panels: margin beyond the 95% contour, per side


def marg1(ax_keep):
    other = tuple(a for a in range(3) if a != ax_keep)
    m = P.sum(axis=other) * JAC[ax_keep]
    return m / np.trapz(m, X[ax_keep])


def marg2(i, j):
    """2-D marginal over (X[j], X[i]) as (density in plotted units, mass per grid cell), shape (len X[j], len X[i])."""
    other = tuple(a for a in range(3) if a not in (i, j))
    mass = P.sum(axis=other)                              # axes ordered (j, i) since j < i
    return mass * JAC[j][:, None] * JAC[i][None, :], mass


def levels(dens, mass):
    """Density thresholds whose super-level sets enclose 95% and 68% of the mass (highest-density regions)."""
    o = np.argsort(dens.ravel())[::-1]
    c = np.cumsum(mass.ravel()[o]) / mass.sum()
    return [dens.ravel()[o][np.searchsorted(c, 0.95)], dens.ravel()[o][np.searchsorted(c, 0.68)]]


# axis limits: Delta G over its plotted range; sigma and nu zoomed to the union of their 95% regions plus PAD
LIM = [(X[0][0], X[0][-1]), None, None]
for a in (1, 2):
    lo, hi = np.inf, -np.inf
    for b in (0, 1, 2):
        if b == a:
            continue
        i, j = max(a, b), min(a, b)
        dens, mass = marg2(i, j)
        inside = dens >= levels(dens, mass)[0]
        ax_vals = np.any(inside, axis=0 if a == i else 1)   # projection onto parameter a
        v = X[a][ax_vals]
        lo, hi = min(lo, v.min()), max(hi, v.max())
    w = hi - lo
    LIM[a] = (lo - PAD * w, hi + PAD * w)


plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8})
fig, axes = plt.subplots(3, 3, figsize=(3.4, 3.4))
for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        if j > i:
            ax.axis('off')
            continue
        if i == j:
            if i == 0:
                ax.plot(lap_x, lap_p, color=C1, lw=1.5)
            else:
                ax.plot(X[i], marg1(i), color=C1, lw=1.5)
            ax.set_ylim(0, None)
            ax.set_yticks([])
            if i == 0:
                ax.axvline(q95 / 1e-4, color=C1, ls='--', lw=1.0)
        else:
            m2, mass = marg2(i, j)
            lv = levels(m2, mass)
            ax.contourf(X[j], X[i], m2.T, levels=[lv[0], lv[1], m2.max() * 1.01], colors=[C2, C1], alpha=0.9)
            ax.contour(X[j], X[i], m2.T, levels=lv, colors=[C1], linewidths=0.8)
            if j == 0:
                ax.axvline(q95 / 1e-4, color=C1, ls='--', lw=1.0)
        ax.set_xlim(*LIM[j])
        if i != j:
            ax.set_ylim(*LIM[i])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        if i != j:
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        if i == 2:
            ax.set_xlabel(LAB[j])
        else:
            ax.set_xticklabels([])
        if j == 0 and i > 0:
            ax.set_ylabel(LAB[i])
        elif not (i == j):
            ax.set_yticklabels([])
        ax.tick_params(direction='in', length=3, labelsize=8)
fig.subplots_adjust(left=0.17, bottom=0.15, right=0.97, top=0.97, wspace=0.08, hspace=0.08)
PLOTS = os.path.join(P_.ROOT, 'plots')
os.makedirs(PLOTS, exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(PLOTS, 'Corner_8Mpc.%s' % ext), dpi=200)
print('q95 %.3g; sigma limits %.4g-%.4g e-3; nu limits %.2f-%.2f' % (q95, LIM[1][0], LIM[1][1], LIM[2][0], LIM[2][1]))
# correlation of the nuisance parameters with dG (posterior means conditional on dG)
w = P / P.sum(axis=(1, 2), keepdims=True)
ms = (w * SG[None, :, None]).sum(axis=(1, 2)); mn = (w * np.exp(LNU)[None, None, :]).sum(axis=(1, 2))
print('E[sigma|dG] at dG=0, q95: %.4g %.4g; E[nu|dG]: %.3f %.3f' % (ms[0], np.interp(q95, DG, ms), mn[0], np.interp(q95, DG, mn)))
