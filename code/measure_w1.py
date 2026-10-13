"""Remeasure the U-shaped warp amplitude w1 (DF20 Eqs. 16-17) from NSA v1_0 atlas images.

DF20's published w1 came from a code whose w1 is the ODD moment sum (x - xbar)(zbar - z)/N^3 (log.md,
2026-10-08), blind to the U-shaped fifth-force warp. This replaces it with the documented statistic
and puts the measurement in exactly the frame of the forward model:

  * axes from the NSA Sersic position angle phi = SERSIC_PHI (deg E of N), as in
    Calc_Proxies_warp_NoScreen.py:387-389:
        u = N cos(phi) + E sin(phi)   (major axis),   v = N sin(phi) - E cos(phi)
    and the model's a_perp is the component of a5 along v, with w1_pred = -K a_perp;
  * pixel offsets (dE, dN) from the WCS (tangent plane, arcsec); x = dE sin(phi) + dN cos(phi),
    z = dN sin(phi) - dE cos(phi), so +z is the model's +v;
  * window |x| <= 3R, |z| <= 3 q R with R = SERSIC_TH50 (major-axis half-light radius, the model's R_eff)
    and q = 0.15 (DF20 Sec. III B);
  * warp curve: pixels binned in x (bin width one pixel, 0.396"), zbar_k = sum(I z)/sum(I) per bin
    (bins with sum(I) <= 0 dropped); z'_k = zbar_k - mean_k(zbar_k);
    w1 = sum_k |x_k| z'_k dx / (3R)^3   (dimensionless, Eq. 17);
  * centre: start at the NSA (RA, DEC), recentre twice on the light centroid inside the window;
  * no smoothing, no clipping of negative pixels (the atlas child images are sky-subtracted);
  * galaxies with R < 1 pixel are not measured (as in HD's 2019-20 notebooks).
Also returned: the odd moment (to check against DF20's published values' behaviour), the tilt slope,
the number of bins, and a noise estimate for w1 from 30 Monte Carlo re-draws of pixel noise with the
variance (I + sky)/GAIN + RNOISE^2 given in the header.
"""
import numpy as np

PIX = 0.396
Q = 0.15


def pixel_offsets(shape, header):
    """Tangent-plane (dE, dN) in arcsec of every pixel centre from (CRVAL1, CRVAL2), plus the
    pixel position of a given (ra, dec) via the same linear WCS (the atlases are simple TAN images)."""
    cd = np.array([[header['CD1_1'], header['CD1_2']], [header['CD2_1'], header['CD2_2']]]) * 3600.
    ny, nx = shape
    X, Y = np.meshgrid(np.arange(nx) + 1., np.arange(ny) + 1.)       # FITS 1-based pixel coordinates
    dx, dy = X - header['CRPIX1'], Y - header['CRPIX2']
    dRA = cd[0, 0] * dx + cd[0, 1] * dy                                # arcsec along RA (increasing = East)
    dDec = cd[1, 0] * dx + cd[1, 1] * dy
    return dRA, dDec, cd


def sky_offset(ra, dec, ra0, dec0):
    """Gnomonic (dE, dN) in arcsec of (ra, dec) about (ra0, dec0) [deg]."""
    r, d, r0, d0 = np.radians([ra, dec, ra0, dec0])
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(r - r0)
    xi = np.cos(d) * np.sin(r - r0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(r - r0)) / cosc
    return np.degrees(xi) * 3600., np.degrees(eta) * 3600.


def w1_from_image(img, dE, dN, phi, R, e0=0., n0=0., recentre=2, return_curve=False, var=None, gamma=0.5):
    """w1 (Eq. 17) for image img with pixel offsets dE, dN (arcsec), PA phi (deg E of N), R (arcsec)."""
    s, c = np.sin(np.radians(phi)), np.cos(np.radians(phi))
    L, Lz = 3. * R, 3. * Q * R
    for _ in range(recentre + 1):
        E, N = dE - e0, dN - n0
        x = E * s + N * c
        z = N * s - E * c
        win = (np.abs(x) <= L + 0.5 * PIX) & (np.abs(z) <= Lz)
        I = img[win]
        V = None if var is None else var[win]
        if I.sum() <= 0:
            return None
        xw, zw = x[win], z[win]
        # recentre on the light centroid inside the window
        dx_c, dz_c = np.sum(I * xw) / I.sum(), np.sum(I * zw) / I.sum()
        e0, n0 = e0 + dx_c * s - dz_c * c, n0 + dx_c * c + dz_c * s
    k = np.round(xw / PIX).astype(int)
    kmax = int(np.floor(L / PIX))
    keep = np.abs(k) <= kmax
    k, I, zw = k[keep], I[keep], zw[keep]
    V = None if V is None else V[keep]
    bins = np.arange(-kmax, kmax + 1)
    sI = np.bincount(k + kmax, weights=I, minlength=len(bins))
    sIz = np.bincount(k + kmax, weights=I * zw, minlength=len(bins))
    ok = sI > 0
    if ok.sum() < 3:
        return None
    xk = bins[ok] * PIX
    zbar = sIz[ok] / sI[ok]
    zp = zbar - zbar.mean()
    w1 = np.sum(np.abs(xk) * zp) * PIX / L ** 3
    odd = np.sum((xk - xk.mean()) * (-zp)) / len(xk) ** 3 / PIX       # DF20-code-like odd moment (pixels)
    tilt = np.polyfit(xk, zbar, 1)[0]
    npx = np.bincount(k + kmax, minlength=len(bins))[ok]
    out = dict(w1=w1, w1_odd=odd, tilt=tilt, nbins=int(ok.sum()), nbins_full=len(bins), e0=e0, n0=n0,
               flux=float(sI.sum()), npix_med=float(np.median(npx)))
    if V is not None:
        # per-bin centroid variance: var(zbar_k) = sum var_pix (z - zbar_k)^2 / (sum I)^2
        zb_full = np.zeros(len(bins)); zb_full[ok] = zbar
        sv = np.bincount(k + kmax, weights=V * (zw - zb_full[k + kmax]) ** 2, minlength=len(bins))
        vk = sv[ok] / sI[ok] ** 2
        # Eq. 17 statistic's error (mean subtraction ignored: second order)
        out['w1_err'] = float(np.sqrt(np.sum((np.abs(xk) - np.mean(np.abs(xk))) ** 2 * vk)) * PIX / L ** 3)
        # matched filter: zbar = a + b x + K |x|^gamma, weighted LS; w1_mf = gamma K L^(gamma-1)/((gamma+1)(gamma+2))
        u = xk / L                                           # dimensionless design: 1, x/L, |x/L|^gamma
        fin = np.isfinite(vk) & np.isfinite(zbar) & (vk > 0)
        u, zfit, vfit = u[fin], zbar[fin], vk[fin]
        A = np.vstack([np.ones_like(u), u, np.abs(u) ** gamma]).T
        sw = 1. / np.sqrt(np.maximum(vfit, 1e-12 * np.median(vfit))) if fin.sum() > 3 else np.zeros(0)
        try:
            if fin.sum() <= 3:
                raise np.linalg.LinAlgError('too few finite bins')
            beta, *_ = np.linalg.lstsq(A * sw[:, None], zfit * sw, rcond=None)
            cov = np.linalg.pinv((A * sw[:, None]).T @ (A * sw[:, None]))
            # z = K |x|^gamma = (K L^gamma) |u|^gamma, so K = beta2 / L^gamma and
            # w1 = gamma K L^(gamma-1)/((gamma+1)(gamma+2)) = gamma beta2 / (L (gamma+1)(gamma+2))
            fac = gamma / (L * (gamma + 1.) * (gamma + 2.))
            out['w1_mf'] = float(fac * beta[2])
            out['w1_mf_err'] = float(fac * np.sqrt(max(cov[2, 2], 0.)))
            out['mf_chi2_dof'] = float(np.sum((sw * (zfit - A @ beta)) ** 2) / max(len(zfit) - 3, 1))
        except Exception:                                    # LinAlgError (NumPy may wrap it as SystemError)
            pass
    if return_curve:
        out.update(xk=xk, zbar=zbar)
    return out


def measure_file(path, ra, dec, phi, th50, band='r', nmc=30, seed=0):
    from astropy.io import fits
    hdu = {'r': 2, 'i': 3}[band]
    with fits.open(path) as h:
        img = np.asarray(h[hdu].data, dtype=float)
        hd = h[hdu].header
    if th50 / PIX < 1.:
        return dict(flag='R<1pix')
    dRA, dDec, cd = pixel_offsets(img.shape, hd)
    e_c, n_c = sky_offset(ra, dec, hd['CRVAL1'], hd['CRVAL2'])
    gain, rn, skyv = hd.get('GAIN', np.nan), hd.get('RNOISE', 0.), hd.get('SKYVAL', 0.)
    var = np.clip(img + skyv, 0., None) / gain + rn ** 2 if (np.isfinite(gain) and gain > 0) else None
    # (dRA, dDec) are offsets from CRVAL in the linear WCS; dE = dRA (RA increases East)
    res = w1_from_image(img, dRA, dDec, phi, th50, e0=e_c, n0=n_c, var=var)
    if res is None:
        return dict(flag='empty window')
    # noise: re-draw pixel noise around the image (Poisson + read noise per the header)
    if var is not None:
        rng = np.random.default_rng(seed)
        sl = _window_slice(img.shape, dRA, dDec, res['e0'], res['n0'], th50)
        sub, sE, sN, sv = img[sl], dRA[sl], dDec[sl], var[sl]
        w, wm = [], []
        for _ in range(nmc):
            r = w1_from_image(sub + rng.normal(0., np.sqrt(sv)), sE, sN, phi, th50, e0=res['e0'], n0=res['n0'],
                              recentre=0, var=sv)
            if r is not None:
                w.append(r['w1']); wm.append(r.get('w1_mf', np.nan))
        res['w1_noise'] = float(np.std(w)) if len(w) > 2 else np.nan
        res['w1_mf_noise'] = float(np.nanstd(wm)) if len(wm) > 2 else np.nan
    res['flag'] = 'ok'
    res['band'] = band
    return res


def _window_slice(shape, dE, dN, e0, n0, R):
    rad = 3.5 * R + 2 * PIX
    m = (np.abs(dE - e0) <= rad) & (np.abs(dN - n0) <= rad)
    ys, xs = np.where(m)
    return slice(max(ys.min() - 1, 0), ys.max() + 2), slice(max(xs.min() - 1, 0), xs.max() + 2)
