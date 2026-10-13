"""Joint posterior of (dG, sigma, nu) at lambda_C = 8 Mpc for the fiducial analysis (sgpub19_fid: remeasured
r-band w1, S2 self-gravity calibration, Student-t 't' model), on a brute-force 3-D grid, for the corner plot
requested by HD (2026-10-13). Flat priors as in robust_lap.py: dG >= 0, sigma U(2e-4, 3e-3), ln nu U(ln 0.5, ln 100).

The nuisance ranges are +-7 sd about the Laplace mode (Hessian at dG = 0, scaled up so the conditional modes at
large dG stay inside). Reports the dG quantiles, to check them against the Laplace grid (lap/t/sgpub19_fid.json:
q68 2.813e-4, q95 7.647e-4), and the largest posterior mass on any nuisance-grid edge.

python3 corner_8mpc.py   ->  <data>/results/corner_8mpc.npz
"""
import os
import time
import numpy as np
from robust_like import load_signals, select
from lap_run import new_data
from robust_lap import build, maximise
import paths

S = load_signals('Signals_warp_NoScreen_19.dat')
d = select(S, **new_data(sg=True))
names, loglike, vg, hs, lo, hi, per_galaxy = build(d, 't')
assert names == ['sigma', 'lnu']

th0, v0, H0, ok = maximise(0., [1.25e-3, np.log(5.6)], vg, hs, lo, hi)
sd0 = np.sqrt(np.diag(np.linalg.inv(-H0)))
print('mode at dG=0', th0, 'sd', sd0, ok, flush=True)
thm, _, _, _ = maximise(1.5e-3, th0, vg, hs, lo, hi)          # conditional mode far in the dG tail
print('conditional mode at dG=1.5e-3', thm, flush=True)

DG = np.linspace(0., 1.6e-3, 81)
SG = np.linspace(min(th0[0], thm[0]) - 7 * sd0[0], max(th0[0], thm[0]) + 7 * sd0[0], 61)
LNU = np.linspace(min(th0[1], thm[1]) - 7 * sd0[1], max(th0[1], thm[1]) + 7 * sd0[1], 61)

import jax
import jax.numpy as jnp
f = jax.jit(jax.vmap(jax.vmap(loglike, in_axes=(None, 0)), in_axes=(None, 0)))   # (nsg, nnu) per dG
TH = jnp.stack(jnp.meshgrid(jnp.asarray(SG), jnp.asarray(LNU), indexing='ij'), -1)
LL = np.empty((len(DG), len(SG), len(LNU)))
t = time.time()
for i, g in enumerate(DG):
    for j in range(len(SG)):                                   # one sigma row at a time keeps memory small
        LL[i, j] = np.asarray(f(float(g), TH[j:j + 1]))[0]
    if i % 10 == 0:
        print(i, g, LL[i].max(), '%.0fs' % (time.time() - t), flush=True)

P = np.exp(LL - LL.max())
pdg = np.trapz(np.trapz(P, LNU, axis=2), SG, axis=1)
c = np.concatenate([[0.], np.cumsum(0.5 * (pdg[1:] + pdg[:-1]) * np.diff(DG))]); c /= c[-1]
q = {x: float(np.interp(x, c, DG)) for x in (0.5, 0.68, 0.95, 0.99)}
print('dG quantiles', q, 'mass above top dG node bin', 1 - c[-2], flush=True)
edge = 0.
for ax in (1, 2):
    m = P.sum(axis=tuple(a for a in (0, 1, 2) if a != ax)); m = m / m.sum()
    edge = max(edge, m[0], m[-1])
print('largest marginal nuisance mass on a grid edge', edge, flush=True)
np.savez_compressed(paths.results('corner_8mpc.npz'), DG=DG, SG=SG, LNU=LNU, LL=LL, q=np.array([q[x] for x in (0.5, 0.68, 0.95, 0.99)]))
print('CORNER_DONE', flush=True)
