"""Point-source masks for the warp measurement.

Sources: SDSS DR17 PhotoPrimary objects with type = 6 (STAR) within 3.2 R + 25" of each galaxy (stars_query.py).
A star is skipped if it lies within 1.5" of the galaxy's catalogue position (the galaxy itself or its nucleus).
Mask, per band: a circle centred on the star, out to the radius where the star's PSF surface brightness falls to
THRESH x the image's pixel noise, for a Gaussian PSF of the field seeing (FWHM) inflated by WING to cover the
non-Gaussian wings; radius clipped to [RMIN_FWHM x FWHM, RMAX]. Stars whose peak is below the threshold are not
masked. Masked pixels are removed from the measurement by zeroing both image and variance, which drops them from
every sum in measure_w1.w1_from_image (centroids, bin variances, recentring, noise re-draws).
"""
import json, os
import numpy as np

PIX = 0.396
THRESH = 0.5           # mask where the star's model surface brightness > THRESH x pixel noise
WING = 1.5             # radius inflation for the non-Gaussian PSF wings
RMIN_FWHM = 1.5        # minimum radius in units of the FWHM
RMAX = 25.             # arcsec
CENTRE_SKIP = 1.5      # arcsec: objects this close to the galaxy centre are not masked


def sky_offsets(ra, dec, ra0, dec0):
    """Tangent-plane offsets (arcsec, E and N) of (ra, dec) from (ra0, dec0)."""
    ra, dec, ra0, dec0 = map(np.radians, (ra, dec, ra0, dec0))
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc
    return np.degrees(xi) * 3600., np.degrees(eta) * 3600.


def pixel_noise(img, dE, dN, e0, n0, R):
    """Robust pixel noise from pixels outside 3.5 R of the galaxy (whole image if too few)."""
    far = np.hypot(dE - e0, dN - n0) > 3.5 * R
    v = img[far] if far.sum() > 200 else img.ravel()
    v = v[v != 0]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) > 10 else np.nan


def circles(stars, ra0, dec0, band, fwhm, sigpix, scale=1.):
    """[(dE, dN, radius)] in arcsec relative to the galaxy position (ra0, dec0), for one band.
    stars: list of [ra, dec, psfFlux_r, psfFlux_i] (nanomaggies). scale multiplies every radius (variation)."""
    out = []
    if not np.isfinite(sigpix) or sigpix <= 0 or not np.isfinite(fwhm):
        return out
    s = fwhm / 2.3548
    for ra, dec, fr, fi in stars:
        dE, dN = sky_offsets(ra, dec, ra0, dec0)
        if np.hypot(dE, dN) < CENTRE_SKIP:
            continue
        F = fr if band == 'r' else fi
        peak = F * PIX ** 2 / (2 * np.pi * s ** 2)            # nanomaggies per pixel at the PSF peak
        if not np.isfinite(peak) or peak <= THRESH * sigpix:
            continue
        r = WING * s * np.sqrt(2. * np.log(peak / (THRESH * sigpix)))
        out.append((float(dE), float(dN), float(scale * np.clip(r, RMIN_FWHM * fwhm, RMAX))))
    return out


def in_circles(dE, dN, circ):
    m = np.zeros(np.shape(dE), bool)
    for e, n, r in circ:
        m |= (dE - e) ** 2 + (dN - n) ** 2 <= r * r
    return m


def load_stars(path):
    st = {}
    for f in sorted(os.listdir(path)):
        if f.startswith('stars_') and f.endswith('.json'):
            st.update({int(k): v for k, v in json.load(open(os.path.join(path, f))).items()})
    return st


def strips(circ, phi, R, q=0.15, pix=PIX):
    """Column masks: for every circle that reaches the measurement window (|x| <= 3R, |z| <= 3qR), the band of x within
    one pixel of the circle's x-extent, at ALL z. Every surviving x-bin then keeps its full vertical extent, so its
    flux-weighted centroid responds to a disk displacement exactly as an unmasked bin does (masking pixels inside a
    bin truncates the disk profile and gives the centroid a large, sign-variable gain). Returns [(x_c, halfwidth)]."""
    s, c = np.sin(np.radians(phi)), np.cos(np.radians(phi))
    L, Lz = 3. * R, 3. * q * R
    out = []
    for e, n, r in circ:
        xc, zc = e * s + n * c, n * s - e * c
        if abs(zc) - r <= Lz and abs(xc) - r <= L:
            out.append((float(xc), float(r + pix)))
    return out


def in_strips(dE, dN, phi, st):
    s, c = np.sin(np.radians(phi)), np.cos(np.radians(phi))
    x = dE * s + dN * c
    m = np.zeros(np.shape(dE), bool)
    for xc, hw in st:
        m |= np.abs(x - xc) <= hw
    return m


# ---- Gaia DR3 stars and the contamination criterion (v4) ----
GAIA_SIG = 5.          # astrometric star: 5/6-parameter solution with proper motion or parallax significant at > GAIA_SIG
CONTAM = 0.2           # drop an x-bin where the star's modelled light shifts its centroid by > CONTAM x the centroid noise
EPOCH = 2004.5         # Gaia positions (2016.0) moved to the mean SDSS imaging epoch with the Gaia proper motion
WING_FRAC = 0.1        # PSF model: (1 - WING_FRAC) Moffat(beta 3, FWHM) + WING_FRAC Moffat(beta 1.5, alpha = 2 FWHM): the
                       # second term has an r^-3 surface-brightness tail, more far-wing light than a pure Moffat


def gaia_stars(path):
    """{warp_match row: [[ra, dec, G, bp_rp], ...]} for Gaia DR3 astrometric stars (positions moved to EPOCH), from the
    gaia_query.py outputs in directory path. Rows whose query failed are absent (no mask)."""
    out = {}
    for f in sorted(os.listdir(path)):
        if not (f.startswith('gaia_') and f.endswith('.json')):
            continue
        for r, v in json.load(open(os.path.join(path, f))).items():
            if isinstance(v, str):
                continue
            lst = []
            for ra, dec, ep, G, bprp, plx_snr, pmra, pmdec, epmra, epmdec, aps, aen, ruwe in v:
                if aps not in (31, 95):
                    continue
                pm, pme = np.hypot(pmra, pmdec), np.hypot(epmra, epmdec)
                if not ((pme > 0 and pm / pme > GAIA_SIG) or plx_snr > GAIA_SIG) or not np.isfinite(G):
                    continue
                d = dec + pmdec * (EPOCH - ep) / 3.6e6
                lst.append([float(ra + pmra * (EPOCH - ep) / 3.6e6 / np.cos(np.radians(d))), float(d), float(G), float(bprp)])
            out[int(r)] = lst
    return out


def gaia_flux(G, bprp, band):
    """SDSS r or i flux (nanomaggies) from Gaia G and BP-RP (Evans et al. 2018 polynomials; BP-RP = 1 if missing)."""
    x = np.clip(np.where(np.isfinite(bprp), bprp, 1.), 0., 3.)
    gm = (-0.12879 + 0.24662 * x - 0.027464 * x ** 2 - 0.049465 * x ** 3) if band == 'r' else (-0.29676 + 0.64728 * x - 0.10141 * x ** 2)
    return 10 ** (-0.4 * (G - gm - 22.5))


def psf_model(dE, dN, stars, fwhm, pix=PIX):
    """Pixel-integrated star light (nanomaggies per pixel) at offsets (dE, dN) for stars [(e, n, flux)]."""
    a1 = fwhm / (2. * np.sqrt(2. ** (1. / 3.) - 1.))
    a2 = 2. * fwhm
    out = np.zeros(np.shape(dE))
    for e, n, F in stars:
        r2 = (dE - e) ** 2 + (dN - n) ** 2
        out += F * pix ** 2 * ((1 - WING_FRAC) * 2. / (np.pi * a1 ** 2) * (1 + r2 / a1 ** 2) ** -3.
                               + WING_FRAC * 0.5 / (np.pi * a2 ** 2) * (1 + r2 / a2 ** 2) ** -1.5)
    return out


def contam_strips(img, var, dE, dN, e0, n0, phi, R, stars, fwhm, ecat, ncat, q=0.15, thresh=CONTAM, pix=PIX):
    """Column masks [(x_c, hw)] (x along the major axis from (ecat, ncat)) covering every x-bin, at the measurement
    centre (e0, n0), in which the modelled light of stars [(e, n, flux)] (offsets from (e0, n0)'s frame origin, i.e. the
    same frame as dE, dN) shifts the bin's flux-weighted centroid by more than thresh x its noise:
    |sum_pix model (z - zbar_k)| > thresh sqrt(sum_pix var (z - zbar_k)^2) over the bin's window pixels.
    Also returns the largest ratio."""
    if not stars:
        return [], 0.
    s, c = np.sin(np.radians(phi)), np.cos(np.radians(phi))
    L, Lz = 3. * R, 3. * q * R
    E, N = dE - e0, dN - n0
    x, z = E * s + N * c, N * s - E * c
    win = (np.abs(x) <= L + 0.5 * pix) & (np.abs(z) <= Lz)
    k = np.round(x[win] / pix).astype(int)
    kmax = int(np.floor(L / pix))
    keep = np.abs(k) <= kmax
    k, Z = k[keep] + kmax, z[win][keep]
    I, V = img[win][keep], var[win][keep]
    P = psf_model(dE[win][keep], dN[win][keep], stars, fwhm, pix)
    nb = 2 * kmax + 1
    sI = np.bincount(k, I, nb)
    zb = np.where(sI > 0, np.bincount(k, I * Z, nb) / np.where(sI > 0, sI, 1.), 0.)
    num = np.abs(np.bincount(k, P * (Z - zb[k]), nb))
    den = np.sqrt(np.bincount(k, V * (Z - zb[k]) ** 2, nb))
    ratio = np.where(den > 0, num / np.where(den > 0, den, 1.), 0.)
    bad = np.where(ratio > thresh)[0]
    dx0 = (e0 - ecat) * s + (n0 - ncat) * c
    out, i = [], 0
    while i < len(bad):
        j = i
        while j + 1 < len(bad) and bad[j + 1] == bad[j] + 1:
            j += 1
        lo, hi = (bad[i] - kmax - 0.5) * pix + dx0, (bad[j] - kmax + 0.5) * pix + dx0
        out.append((float(0.5 * (lo + hi)), float(0.5 * (hi - lo))))
        i = j + 1
    return out, float(ratio.max())


# ---- v5: the star's residual amplitude in the (NSA deblended) image ----
AMP_RAP = 2.           # fit aperture radius in FWHM
AMP_NSIG = 2.          # contamination model = (|A| + AMP_NSIG sigma_A) x the Gaia-flux PSF model: an over-subtracted star
                       # (A < 0) shifts centroids as much as an under-subtracted one; errs towards masking
AMP_MINPIX = 8         # fewer usable aperture pixels (star off the image): amplitude 0 (its light is not in this image)


def star_amp(img, var, dE, dN, e, n, F, fwhm, pix=PIX):
    """Amplitude A (and sigma_A) of the Gaia-flux PSF model of a star at (e, n) in img: inverse-variance weighted fit of
    A x model + plane (a + b dE + c dN) within AMP_RAP FWHM. A ~ 1: star present at its catalogue flux; A ~ 0: removed by
    the NSA deblender (the atlas images are deblended children)."""
    ap = (np.hypot(dE - e, dN - n) < AMP_RAP * fwhm) & (var > 0)
    if ap.sum() < AMP_MINPIX:
        return 0., 0.
    P = psf_model(dE[ap], dN[ap], [(e, n, F)], fwhm, pix)
    X = np.c_[P, np.ones(ap.sum()), dE[ap] - e, dN[ap] - n]
    w = 1. / var[ap]
    C = np.linalg.pinv(X.T @ (X * w[:, None]))
    beta = C @ (X.T @ (w * img[ap]))
    return float(beta[0]), float(np.sqrt(max(C[0, 0], 0.)))


def resid_stars(img, var, dE, dN, stars, fwhm):
    """stars [(e, n, F)] -> [(e, n, F_eff)] with F_eff = (|A| + AMP_NSIG sigma_A) F (zero-flux stars dropped), and
    the per-star [(F, A, sigma_A)]."""
    out, amps = [], []
    for e, n, F in stars:
        A, sA = star_amp(img, var, dE, dN, e, n, F, fwhm)
        amps.append((float(F), A, sA))
        Ae = abs(A) + AMP_NSIG * sA
        if Ae > 0:
            out.append((e, n, F * Ae))
    return out, amps
