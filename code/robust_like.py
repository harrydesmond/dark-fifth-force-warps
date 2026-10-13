"""Outlier-robust warp likelihood with no quality cut on the observed warp.

Each galaxy's measured warp w is either a good measurement, drawn from DF20's per-galaxy mixture
for the predicted signal plus Gaussian astrophysical scatter,
    p_good(w) = sum_k W_k N(w | dG M_k, sigma^2 + dG^2 C_k),
or, with probability eps, a bad measurement that carries no fifth-force signal,
    p_bad(w)  = N(w | 0, sigma_out^2)                    (junk='gauss', fiducial)
              = Student-t(w | nu, 0, sigma_out)          (junk='t', nu free)
so each galaxy contributes (1 - eps) p_good + eps p_bad. Flat priors on dG in [0, dgmax], sigma in
[2e-4, 1.5e-3] and eps in [0, eps_max]; log-uniform sigma_out in [1.5e-3, so_max]. The two widths
cannot overlap, so the components cannot swap labels. Sampled with NUTS; bounds are posterior
quantiles of dG.

Options, all off in the fiducial:
  band='i'        i-band warps instead of r-band
  excise=f        drop the top fraction f of the sample by mixture-mean |M| (DF20's ranking)
  cut=c           keep only |w| < c and divide each term by its selection probability
  screen=chi      DF20's two-branch screening, f_i p_good(dG) + (1 - f_i) p_good(0), with f_i the
                  probability that the galaxy's potential lies below chi (log-normal over its
                  Monte Carlo interval, columns 51-53)
  sample          'fixed7': drop the seven catalogue artefacts at every range (fiducial)
                  'df20'  : drop |modal predicted signal| >= 100 in this file (DF20's cut)
                  'union' : drop every galaxy that fails the DF20 cut in any FULL run
                  'all'   : keep everything
  dist=('lt', D)  distance subsample (Mpc); ('gt', D) likewise
"""
import os
import numpy as np
import paths as P

NMAX = 15
ARTEFACTS = np.array([6, 1222, 1875, 3036, 4163, 4208, 4318])   # log.md 2026-10-07: catalogue artefacts
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = P.SIGNALS


def load_signals(name):
    return np.genfromtxt(os.path.join(DATA, name))


def union_rows():
    """Rows failing the DF20 modal-signal cut in any FULL run or in the published 8 Mpc file."""
    bad = set(ARTEFACTS.tolist())
    names = ['Signals_warp_FULL_lam%.1f.dat' % l for l in (8, 12, 16, 20)]
    names += ['Signals_warp_FULL_s%d_lam%.1f.dat' % (s, l) for s in (1, 2, 3, 4) for l in (8, 20)]
    for n in names:
        S = np.genfromtxt(os.path.join(DATA, n), usecols=(58,))
        bad |= set(np.where(np.abs(S) >= 100.)[0].tolist())
    return np.array(sorted(bad))


def select(S, band='r', excise=0., cut=None, sample='fixed7', dist=None, screen=None, union=None,
           w_new=None, e_new=None, scale=None, keep=None, fracerr=0.):
    """w_new/e_new: remeasured warps and their errors (one per row of S, NaN = unmeasured);
    scale: per-row PSF/pixel/window suppression S_i applied to the predicted signal (M*S, C*S^2);
    keep: extra per-row boolean mask."""
    n = len(S)
    keep_extra = keep
    keep = np.ones(n, bool)
    if sample == 'fixed7':
        keep[ARTEFACTS] = False
    elif sample == 'df20':
        keep &= np.abs(S[:, 58]) < 100.
    elif sample == 'union':
        keep[union] = False
    elif sample != 'all':
        raise ValueError(sample)
    col = 49 if band == 'r' else 50
    wcol = S[:, col] if w_new is None else np.asarray(w_new, float)
    if w_new is not None:
        keep &= np.isfinite(wcol)
    if keep_extra is not None:
        keep &= keep_extra
    if cut is not None:
        keep &= np.abs(wcol) < cut
    if dist is not None:
        keep &= (S[:, 1] < dist[1]) if dist[0] == 'lt' else (S[:, 1] > dist[1])
    s = S[keep]
    nc = s[:, 2].astype(int)
    v = np.arange(NMAX)[None, :] < nc[:, None]
    W = np.where(v, s[:, 3:3 + NMAX], 0.)
    M = np.where(v, s[:, 3 + NMAX:3 + 2 * NMAX], 0.)
    C = np.where(v, s[:, 3 + 2 * NMAX:3 + 3 * NMAX], 1.)
    rows = np.where(keep)[0]
    w = wcol[rows].copy()
    e = np.zeros(len(rows)) if e_new is None else np.nan_to_num(np.asarray(e_new, float)[rows])
    if fracerr > 0:                                            # fractional forward-model uncertainty
        C = np.where(v, C + (fracerr * M) ** 2, 1.)
    if scale is not None:
        sc = np.asarray(scale, float)[rows]
        M = M * sc[:, None]
        C = np.where(v, C * sc[:, None] ** 2, 1.)
    if excise > 0:
        amp = (W * np.abs(M)).sum(1) / np.maximum(W.sum(1), 1e-30)
        k = amp < np.quantile(amp, 1. - excise)
        W, M, C, w, s, rows, e = W[k], M[k], C[k], w[k], s[k], rows[k], e[k]
    f = None
    if screen is not None:
        from scipy.special import ndtr
        siglog = np.maximum((s[:, 53] - s[:, 51]) / 2., 1e-3)
        f = ndtr((np.log10(screen) - s[:, 52]) / siglog)
    return dict(W=W, M=M, C=C, w=w, f=f, rows=rows, D=s[:, 1], e=e)


def run_nuts(d, cut=None, junk='gauss', dgmax=1e-2, eps_max=0.5, so_max=0.1, seed=0, signfree=False,
             warmup=1000, samples=6000, chains=4, progress=False):
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp, ndtr, betainc
    from jax.scipy.stats import norm, t as student_t
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value

    W = jnp.asarray(d['W']); M = jnp.asarray(d['M']); C = jnp.asarray(d['C']); w = jnp.asarray(d['w'])
    logW = jnp.where(W > 0, jnp.log(jnp.where(W > 0, W, 1.)), -1e30)
    if d['f'] is not None:
        lf = jnp.log(jnp.clip(jnp.asarray(d['f']), 1e-300, 1.))
        l1f = jnp.log(jnp.clip(1. - jnp.asarray(d['f']), 1e-300, 1.))

    def model():
        dG = numpyro.sample('dG', dist.Uniform(-dgmax if signfree else 0., dgmax))
        sig = numpyro.sample('sigma', dist.Uniform(2e-4, 1.5e-3))
        eps = numpyro.sample('eps', dist.Uniform(0., eps_max))
        lso = numpyro.sample('log_sigma_out', dist.Uniform(np.log(1.5e-3), np.log(so_max)))
        so = jnp.exp(lso)
        var = sig ** 2 + dG ** 2 * C
        lg = logsumexp(logW - 0.5 * jnp.log(2 * jnp.pi * var) - (w[:, None] - dG * M) ** 2 / (2 * var), axis=1)
        if d['f'] is not None:
            lg = jnp.logaddexp(lf + lg, l1f + norm.logpdf(w, 0., sig))
        if junk == 'gauss':
            lb = norm.logpdf(w, 0., so)
        else:
            nu = jnp.exp(numpyro.sample('log_nu', dist.Uniform(0., np.log(100.))))
            lb = student_t.logpdf(w, nu, 0., so)
        ll = jnp.logaddexp(jnp.log1p(-eps) + lg, jnp.log(eps) + lb)
        if cut is not None:
            sd = jnp.sqrt(var)
            pg = (W * (ndtr((cut - dG * M) / sd) - ndtr((-cut - dG * M) / sd))).sum(1)
            if junk == 'gauss':
                pb = 2. * ndtr(cut / so) - 1.
            else:                                   # P(|t| < x) = 1 - I_{nu/(nu+x^2)}(nu/2, 1/2)
                x = cut / so
                pb = 1. - betainc(nu / 2., 0.5, nu / (nu + x * x))
            ll = ll - jnp.log((1. - eps) * pg + eps * pb)
        numpyro.factor('ll', ll.sum())

    init = dict(dG=min(3e-5, 0.1 * dgmax), sigma=7.2e-4, eps=0.05, log_sigma_out=np.log(5e-3))
    if junk == 't':
        init['log_nu'] = np.log(5.)
    kern = NUTS(model, init_strategy=init_to_value(values=init), target_accept_prob=0.9)
    mcmc = MCMC(kern, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method='sequential', progress_bar=progress)
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc


def summarise(mcmc, dgmax):
    from numpyro.diagnostics import summary
    sm = mcmc.get_samples(group_by_chain=True)
    stats = summary(sm)
    flat = {k: np.asarray(v).reshape(-1) for k, v in sm.items()}
    dG = flat['dG']
    out = dict(q68=float(np.quantile(dG, 0.68)), q95=float(np.quantile(dG, 0.95)),
               q50=float(np.quantile(dG, 0.50)), mean=float(dG.mean()),
               frac_top=float(np.mean(dG > 0.8 * dgmax)),
               n_div=int(np.asarray(mcmc.get_extra_fields()['diverging']).sum()))
    for k in flat:
        out['%s_med' % k] = float(np.median(flat[k]))
        out['%s_rhat' % k] = float(stats[k]['r_hat'])
        out['%s_ess' % k] = float(stats[k]['n_eff'])
    # Monte Carlo error on q95 from per-chain quantiles
    pc = [float(np.quantile(np.asarray(c), 0.95)) for c in sm['dG']]
    out['q95_chain_sd'] = float(np.std(pc, ddof=1) / np.sqrt(len(pc)))
    return out, dG
