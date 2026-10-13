"""Deterministic posterior on dG for the robust (no-cut) warp likelihoods: a fine 1-D grid in dG with
the noise parameters integrated out by Laplace's method at every grid point.

NUTS proved unreliable here: the likelihood in dG is rugged (each galaxy's narrow mixture components
add bumps) and chains started together can stay in a local mode with R-hat ~ 1.  The noise parameters,
by contrast, are well constrained and smooth, so a Laplace integral over them is accurate (checked
against a brute-force 4-D grid in robust_grid.py).

Noise models for one galaxy with predicted-signal mixture (W_k, M_k, C_k):
  'gj'  (1-eps) sum_k W_k N(w | dG M_k, sigma^2 + dG^2 C_k) + eps N(w | 0, s^2)           signal-free Gaussian junk
  'tj'  (1-eps) sum_k W_k N(...)                             + eps t_nu(w | 0, s)          signal-free Student-t junk
  't'   sum_k W_k t_nu(w | dG M_k, sqrt(sigma^2 + dG^2 C_k))                              signal-carrying Student-t
  'gt'  (1-eps) sum_k W_k N(...) + eps sum_k W_k t_nu(w | dG M_k, sqrt(s^2 + dG^2 C_k))    Gaussian core + signal-carrying t
        (eps in [0, 1], so 'gt' nests both the single t (eps = 1) and the pure Gaussian (eps = 0))
(for the t terms the convolution with the Gaussian prediction is approximated by adding variances).
Options: cut=c (keep |w|<c, divide by the selection probability), screening via d['f']
(f_i L_i(dG) + (1 - f_i) L_i(0)), signfree (dG < 0 allowed).
Priors: dG flat on the grid; sigma U(2e-4, 3e-3); eps U(0, 0.5); log s U(log 2e-4, log 0.1);
log nu U(log 0.5, log 100).  The Laplace integral is taken in these flat-prior coordinates.
"""
import numpy as np

PARS = {'gj': ['sigma', 'eps', 'lso'], 'tj': ['sigma', 'eps', 'lso', 'lnu'], 't': ['sigma', 'lnu'],
        'gt': ['sigma', 'eps', 'lso', 'lnu']}
BOUNDS = {'sigma': (2e-4, 3e-3), 'eps': (0., 0.5), 'lso': (np.log(2e-4), np.log(0.1)), 'lnu': (np.log(0.5), np.log(100.))}
STARTS = {'gj': [[7.4e-4, 0.044, np.log(5.3e-3)], [7.4e-4, 0.017, np.log(0.027)]],
          'tj': [[6.5e-4, 0.16, np.log(1.6e-3), np.log(3.6)], [7.4e-4, 0.04, np.log(5e-3), np.log(2.)]],
          't': [[6.0e-4, np.log(4.)], [6.5e-4, np.log(10.)]],
          'gt': [[7.2e-4, 0.5, np.log(5.1e-4), np.log(1.9)], [7.0e-4, 0.85, np.log(5.5e-4), np.log(2.5)],
                 [7.4e-4, 0.04, np.log(5e-3), np.log(2.)]]}


def build(d, model='gj', cut=None):
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp, gammaln, ndtr

    W = jnp.asarray(d['W']); M = jnp.asarray(d['M']); C = jnp.asarray(d['C']); w = jnp.asarray(d['w'])
    E2 = jnp.asarray(d['e'] if d.get('e') is not None else np.zeros(len(d['w']))) ** 2   # measurement variance
    logW = jnp.where(W > 0, jnp.log(jnp.where(W > 0, W, 1.)), -1e30)
    f = None if d.get('f') is None else jnp.asarray(d['f'])
    names = PARS[model]

    def tlogpdf(x, nu, s):                      # x, s broadcastable
        z = x / s
        return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * jnp.log(nu * jnp.pi) - jnp.log(s)
                - (nu + 1) / 2 * jnp.log1p(z * z / nu))

    XG, WG = np.polynomial.legendre.leggauss(24)

    def tcdf(x, nu):
        # Student-t CDF as a Gauss-Legendre integral in theta = arctan(t / sqrt(nu)), where the density
        # is cos^(nu-1) theta / B(nu/2, 1/2): smooth, differentiable in x and nu, exact to 1e-15 for
        # nu >= 1 (1e-3 at nu = 0.7). jax's betainc is ~100x slower and has no derivative in nu.
        th = jnp.arctan(x / jnp.sqrt(nu))
        u = th[..., None] * (XG + 1.) / 2.
        norm = jnp.exp(gammaln((nu + 1.) / 2.) - gammaln(nu / 2.) - 0.5 * jnp.log(jnp.pi))
        return 0.5 + norm * (jnp.cos(u) ** (nu - 1.) * WG).sum(-1) * th / 2.

    def per_galaxy(dG, th):
        p = dict(zip(names, th))
        sig = p['sigma']
        var = sig ** 2 + E2[:, None] + dG ** 2 * C
        mu = dG * M
        out = []
        if model in ('gj', 'tj', 'gt'):
            core = logsumexp(logW - 0.5 * jnp.log(2 * jnp.pi * var) - (w[:, None] - mu) ** 2 / (2 * var), axis=1)
            eps = p['eps']; so = jnp.exp(p['lso'])
            if model == 'gj':
                bad = -0.5 * jnp.log(2 * jnp.pi * so ** 2) - w ** 2 / (2 * so ** 2)
            elif model == 'tj':
                bad = tlogpdf(w, jnp.exp(p['lnu']), so)
            else:
                bad = logsumexp(logW + tlogpdf(w[:, None] - mu, jnp.exp(p['lnu']), jnp.sqrt(so ** 2 + dG ** 2 * C)), axis=1)
            ll = jnp.logaddexp(jnp.log1p(-eps) + core, jnp.log(eps) + bad)
        else:
            ll = logsumexp(logW + tlogpdf(w[:, None] - mu, jnp.exp(p['lnu']), jnp.sqrt(var)), axis=1)
        if cut is not None:
            sd = jnp.sqrt(var)
            if model in ('gj', 'tj', 'gt'):
                pc = (W * (ndtr((cut - mu) / sd) - ndtr((-cut - mu) / sd))).sum(1)
                if model == 'gj':
                    pb = 2. * ndtr(cut / so) - 1.
                elif model == 'tj':
                    pb = 2. * tcdf(cut / so, jnp.exp(p['lnu'])) - 1.
                else:
                    st = jnp.sqrt(so ** 2 + dG ** 2 * C); nu = jnp.exp(p['lnu'])
                    pb = (W * (tcdf((cut - mu) / st, nu) - tcdf((-cut - mu) / st, nu))).sum(1)
                ll = ll - jnp.log((1 - eps) * pc + eps * pb)
            else:
                nu = jnp.exp(p['lnu'])
                ll = ll - jnp.log((W * (tcdf((cut - mu) / sd, nu) - tcdf((-cut - mu) / sd, nu))).sum(1))
        return ll

    if f is None:
        def loglike(dG, th):
            return per_galaxy(dG, th).sum()
    else:
        def loglike(dG, th):
            return jnp.logaddexp(jnp.log(jnp.clip(f, 1e-300)) + per_galaxy(dG, th),
                                 jnp.log(jnp.clip(1 - f, 1e-300)) + per_galaxy(0., th)).sum()

    bnd = dict(BOUNDS, eps=(0., 1.)) if model == 'gt' else BOUNDS      # gt nests the single t at eps = 1
    lo = jnp.array([bnd[n][0] for n in names]); hi = jnp.array([bnd[n][1] for n in names])
    vg = jax.jit(jax.value_and_grad(loglike, argnums=1))
    hs = jax.jit(jax.hessian(loglike, argnums=1))
    return names, loglike, vg, hs, np.asarray(lo), np.asarray(hi), jax.jit(per_galaxy)


def maximise(dG, th0, vg, hs, lo, hi, iters=60):
    """Damped Newton with backtracking inside the prior box. Returns (theta, logL, H, ok)."""
    th = np.clip(np.asarray(th0, float), lo + 1e-12 * (hi - lo), hi - 1e-12 * (hi - lo))
    v, g = vg(dG, th); v = float(v); g = np.asarray(g)
    lam = 0.
    for _ in range(iters):
        H = np.asarray(hs(dG, th))
        step = None
        for _ in range(30):
            A = -H + lam * np.diag(np.maximum(np.abs(np.diag(H)), 1e-30))
            try:
                np.linalg.cholesky(A)
                step = np.linalg.solve(A, g)
                break
            except np.linalg.LinAlgError:
                lam = max(1e-6, 10 * lam)
        if step is None:
            break
        t = 1.
        improved = False
        for _ in range(40):
            new = np.clip(th + t * step, lo + 1e-12 * (hi - lo), hi - 1e-12 * (hi - lo))
            v2, g2 = vg(dG, new); v2 = float(v2)
            if np.isfinite(v2) and v2 >= v - 1e-10:
                improved = True
                break
            t *= 0.5
        if not improved:
            break
        dv = v2 - v
        th, v, g = new, v2, np.asarray(g2)
        lam = lam / 10 if t == 1. else lam
        if abs(dv) < 1e-9 and np.max(np.abs(t * step) / (hi - lo)) < 1e-8:
            break
    H = np.asarray(hs(dG, th))
    edge = np.any((th - lo) / (hi - lo) < 1e-6) or np.any((hi - th) / (hi - lo) < 1e-6)
    ok = bool(np.all(np.linalg.eigvalsh(-H) > 0)) and not edge
    return th, v, H, ok


def posterior(d, DG, model='gj', cut=None, restart_every=10, verbose=False):
    """log p(dG) on the grid DG (Laplace over noise parameters, flat priors), plus diagnostics."""
    names, loglike, vg, hs, lo, hi, _ = build(d, model, cut)
    logvol = float(np.sum(np.log(hi - lo)))
    k = len(names)
    lp = np.full(len(DG), -np.inf)
    TH = np.zeros((len(DG), k))
    OK = np.ones(len(DG), bool)
    bad = 0
    warm = None
    for i, dG in enumerate(DG):
        cands = []
        starts = ([] if warm is None else [warm]) + (STARTS[model] if (warm is None or i % restart_every == 0) else [])
        for s0 in starts:
            th, v, H, ok = maximise(float(dG), s0, vg, hs, lo, hi)
            sign, logdet = np.linalg.slogdet(-H)
            lz = v + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdet - logvol if sign > 0 else -np.inf
            cands.append((lz, th, ok))
        lz, th, ok = max(cands, key=lambda c: c[0])
        lp[i], TH[i], OK[i] = lz, th, ok
        bad += (not ok)
        warm = th
        if verbose and i % 100 == 0:
            print(i, dG, lz, th, ok, flush=True)
    return dict(names=names, DG=np.asarray(DG), lp=lp, theta=TH, n_bad=bad, ok=OK)


def quantiles(res, qs=(0.5, 0.68, 0.95)):
    DG, lp = res['DG'], res['lp']
    p = np.exp(lp - lp.max())
    c = np.concatenate([[0.], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(DG))])
    Z = c[-1]
    c = c / Z
    out = {('q%g' % (100 * q)).replace('.', 'p'): float(np.interp(q, c, DG)) for q in qs}
    out['logZ'] = float(np.log(Z) + lp.max() - np.log(DG[-1] - DG[0]))       # flat dG prior over the grid
    out['tail_top5pct'] = float(1. - np.interp(DG[0] + 0.95 * (DG[-1] - DG[0]), DG, c))
    i95 = np.searchsorted(DG, out['q95'])
    out['theta_at_q95'] = dict(zip(res['names'], [float(x) for x in res['theta'][min(i95, len(DG) - 1)]]))
    out['theta_at_mode'] = dict(zip(res['names'], [float(x) for x in res['theta'][np.argmax(p)]]))
    out['mode'] = float(DG[np.argmax(p)])
    out['n_bad'] = int(res['n_bad'])
    out['n_bad_in_mass'] = int(np.sum(~res['ok'] & (lp > lp.max() - 15.)))   # flagged points that carry weight
    if DG[0] < 0:
        mean = np.trapz(p * DG, DG) / Z
        out['signfree_mean'] = float(mean)
        out['signfree_sd'] = float(np.sqrt(np.trapz(p * DG ** 2, DG) / Z - mean ** 2))
        out['P_neg'] = float(np.interp(0., DG, c))
    return out
