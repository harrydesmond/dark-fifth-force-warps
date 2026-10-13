"""measure_all.py with SDSS point sources masked (mask_lib.py). Same estimator, same window, same noise re-draws, same
synthetic S calibration; the masked pixels are removed from both the real image and every synthetic image.
python3 measure_masked.py <start> <stop>   ->  meas_mask<SUFFIX>/rows_<start>_<stop>.json
env: MASK_MODE (default 'gaia5': Gaia DR3 stars, residual-amplitude contamination criterion; the paper's masking),
     SUFFIX (output-directory suffix, default '_gaia5'). Images from paths.ATLAS. The 'strip'/'circle' modes are
     earlier variants not used in the paper (they need an SDSS star list that is not distributed).
"""
import json, os, sys
import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
import paths as P
import measure_w1 as mw                                     # noqa: E402
import measure_all as ma                                    # noqa: E402
import mask_lib as ml                                       # noqa: E402

CAT = fits.open(P.cat('warp_match.fits'))[1].data
SEE = np.load(P.cat('gal_seeing.npy'))
ATLAS = P.ATLAS
SCALE = float(os.environ.get('MASK_SCALE', 1.))
SYMDROP = os.environ.get('MASK_SYMDROP', '0') == '1'   # also drop each dropped bin's mirror bin (-k): keeps the fit's support
                                                         # symmetric, so odd (tilt/S-warp) structure cannot leak into the even K term
MODE = os.environ.get('MASK_MODE', 'gaia5')               # 'gaia5' (Gaia stars, residual-amplitude criterion; the paper's),
                                                            # 'strip' (SDSS type-6 circles -> columns) or 'circle' (pixels; biased)
OUT = P.mask('meas_mask' + os.environ.get('SUFFIX', '_gaia5'))
ORIG = mw.w1_from_image
CIRC = []                                                   # circles for the synthetic images (galaxy at the origin)
STRIPS = []                                                 # column masks for the synthetic images (MODE strip)


def w1_masked(img, dE, dN, phi, R, e0=0., n0=0., recentre=2, return_curve=False, var=None, gamma=0.5, strips=(),
              eref=0., nref=0.):
    """mw.w1_from_image with column masks strips = [(x_c, hw)] (x along the major axis from (eref, nref)).
    Recentring takes the light centroid over columns unmasked on BOTH sides of the current centre (each strip and its
    mirror image are excluded), so a one-sided mask does not drag the centre. Every x-bin (at the final centre) that
    contains a masked pixel is then dropped whole: a bin cut by a strip edge keeps a sliver of pixels at a few z, whose
    centroid is meaningless and whose variance can be ~0 (one pixel), which gives it an arbitrary weight in the
    matched filter. Without strips this is mw.w1_from_image exactly."""
    if not strips:
        return ORIG(img, dE, dN, phi, R, e0=e0, n0=n0, recentre=recentre, return_curve=return_curve, var=var, gamma=gamma)
    s, c = np.sin(np.radians(phi)), np.cos(np.radians(phi))
    L, Lz = 3. * R, 3. * mw.Q * R
    xr = (dE - eref) * s + (dN - nref) * c
    M = np.zeros(np.shape(img), bool)
    for xc, hw in strips:
        M |= np.abs(xr - xc) <= hw
    img = np.where(M, 0., img)
    for _ in range(recentre):
        E, N = dE - e0, dN - n0
        x, z = E * s + N * c, N * s - E * c
        dxc = (e0 - eref) * s + (n0 - nref) * c
        sym = M.copy()
        for xc, hw in strips:
            sym |= np.abs(-x - (xc - dxc)) <= hw
        win = (np.abs(x) <= L + 0.5 * mw.PIX) & (np.abs(z) <= Lz) & ~sym
        I = img[win]
        if I.sum() <= 0:
            return None
        dx_c, dz_c = np.sum(I * x[win]) / I.sum(), np.sum(I * z[win]) / I.sum()
        e0, n0 = e0 + dx_c * s - dz_c * c, n0 + dx_c * c + dz_c * s
    k = np.round(((dE - e0) * s + (dN - n0) * c) / mw.PIX).astype(int)     # ORIG's bins at this centre (recentre=0)
    kd = np.unique(k[M])
    drop = np.isin(np.abs(k), np.abs(kd)) if SYMDROP else np.isin(k, kd)
    img = np.where(drop, 0., img)
    var = None if var is None else np.where(drop, 0., var)
    res = ORIG(img, dE, dN, phi, R, e0=e0, n0=n0, recentre=0, return_curve=return_curve, var=var, gamma=gamma)
    if res is not None:
        res['e0'], res['n0'] = e0, n0                                     # the centre the bins were built on
    return res


def synth_w1(img, dE, dN, phi, R, e0=0., n0=0., recentre=2, return_curve=False, var=None, gamma=0.5):
    """w1_from_image for the synthetic calibration images (galaxy at the origin), with the current galaxy's mask."""
    if MODE in ('strip', 'gaia', 'gaia5'):
        return w1_masked(img, dE, dN, phi, R, e0=e0, n0=n0, recentre=recentre, return_curve=return_curve, var=var,
                         gamma=gamma, strips=STRIPS)
    if CIRC:
        m = ml.in_circles(dE - e0, dN - n0, CIRC)
        img = np.where(m, 0., img)
        var = None if var is None else np.where(m, 0., var)
    return ORIG(img, dE, dN, phi, R, e0=e0, n0=n0, recentre=recentre, return_curve=return_curve, var=var, gamma=gamma)


ma.w1_from_image = synth_w1                                 # measure_all.synth_S uses this name


def measure_masked(path, ra, dec, phi, th50, band, fwhm, stars, nmc=20, seed=0):
    """measure_w1.measure_file with the point-source mask; also returns the circles and the masked window fraction."""
    hdu = {'r': 2, 'i': 3}[band]
    with fits.open(path) as h:
        img = np.asarray(h[hdu].data, dtype=float)
        hd = h[hdu].header
    if th50 / mw.PIX < 1.:
        return dict(flag='R<1pix'), ([], [])
    dRA, dDec, cd = mw.pixel_offsets(img.shape, hd)
    e_c, n_c = mw.sky_offset(ra, dec, hd['CRVAL1'], hd['CRVAL2'])
    gain, rn, skyv = hd.get('GAIN', np.nan), hd.get('RNOISE', 0.), hd.get('SKYVAL', 0.)
    var = np.clip(img + skyv, 0., None) / gain + rn ** 2 if (np.isfinite(gain) and gain > 0) else None
    sigpix = ml.pixel_noise(img, dRA, dDec, e_c, n_c, th50)
    extra = {}
    if MODE in ('gaia', 'gaia5'):
        circ = []
        sl = [(*mw.sky_offset(a, d, hd['CRVAL1'], hd['CRVAL2']), ml.gaia_flux(G, bprp, band)) for a, d, G, bprp in stars]
        sl = [(e, n, F) for e, n, F in sl if np.isfinite(F)]
        if MODE == 'gaia5' and sl and var is not None:     # v5: the star's residual light in this deblended image
            sl, amps = ml.resid_stars(img, var, dRA, dDec, sl, fwhm)
            extra_amp = dict(gaia_amp=[list(a) for a in amps], nresid=len(sl))
        else:
            extra_amp = {}
        r0 = mw.w1_from_image(img, dRA, dDec, phi, th50, e0=e_c, n0=n_c, var=var) if (sl and var is not None) else None
        st, cmax = (ml.contam_strips(img, var, dRA, dDec, r0['e0'], r0['n0'], phi, th50, sl, fwhm, e_c, n_c)
                    if r0 is not None else ([], 0.))
        extra = dict(ngaia=len(stars), contam_max=cmax, **extra_amp)
        model = ml.psf_model(dRA, dDec, sl, fwhm) if sl else 0.
    if MODE in ('gaia', 'gaia5'):
        m = ml.in_strips(dRA - e_c, dDec - n_c, phi, st)
        meas = lambda im, dE, dN, **kw: w1_masked(im, dE, dN, phi, th50, strips=st, eref=e_c, nref=n_c, **kw)
    elif MODE == 'strip':
        circ = ml.circles(stars, ra, dec, band, fwhm, sigpix, SCALE)
        st = ml.strips(circ, phi, th50)
        m = ml.in_strips(dRA - e_c, dDec - n_c, phi, st)
        meas = lambda im, dE, dN, **kw: w1_masked(im, dE, dN, phi, th50, strips=st, eref=e_c, nref=n_c, **kw)
    else:
        circ = ml.circles(stars, ra, dec, band, fwhm, sigpix, SCALE)
        st = []
        m = ml.in_circles(dRA - e_c, dDec - n_c, circ)
        meas = lambda im, dE, dN, **kw: mw.w1_from_image(im, dE, dN, phi, th50, **kw)
    img = np.where(m, 0., img)
    var = None if var is None else np.where(m, 0., var)
    res = meas(img, dRA, dDec, e0=e_c, n0=n_c, var=var)
    if res is None:
        return dict(flag='empty window', nstar_mask=len(circ), nstrip=len(st)), (circ, st)
    res.update(extra)
    if MODE in ('gaia', 'gaia5') and extra.get('ngaia') and np.ndim(model):
        # residual-star check: the same masked measurement on the image minus the modelled star light
        rs = meas(np.where(m, 0., img - model), dRA, dDec, e0=e_c, n0=n_c, var=var)
        res['w1_mf_starsub'] = rs.get('w1_mf', np.nan) if rs is not None else np.nan
    s, c = np.sin(np.radians(phi)), np.cos(np.radians(phi))
    E, N = dRA - res['e0'], dDec - res['n0']
    win = (np.abs(E * s + N * c) <= 3 * th50) & (np.abs(N * s - E * c) <= 3 * mw.Q * th50)
    res['mask_frac'] = float(m[win].mean()) if win.any() else 0.
    res['nstar_mask'] = len(circ)
    res['nstrip'] = len(st)
    res['sigpix'] = float(sigpix)
    if var is not None:
        rng = np.random.default_rng(seed)
        sl = mw._window_slice(img.shape, dRA, dDec, res['e0'], res['n0'], th50)
        sub, sE, sN, sv = img[sl], dRA[sl], dDec[sl], var[sl]
        w, wm = [], []
        for _ in range(nmc):
            r = meas(sub + rng.normal(0., np.sqrt(sv)) * (sv > 0), sE, sN, e0=res['e0'], n0=res['n0'], recentre=0, var=sv)
            if r is not None:
                w.append(r['w1']); wm.append(r.get('w1_mf', np.nan))
        res['w1_noise'] = float(np.std(w)) if len(w) > 2 else np.nan
        res['w1_mf_noise'] = float(np.nanstd(wm)) if len(wm) > 2 else np.nan
    res['flag'] = 'ok'
    return res, (circ, st)


def main(start, stop):
    global CIRC, STRIPS
    os.makedirs(OUT, exist_ok=True)
    stars = ml.gaia_stars(P.mask('gaia')) if MODE in ('gaia', 'gaia5') else ml.load_stars(os.path.join(HERE, 'stars'))
    out = []
    for row in range(start, stop):
        rec = dict(row=row, nsaid=int(CAT['NSAID'][row]), th50=float(CAT['SERSIC_TH50'][row]),
                   phi=float(CAT['SERSIC_PHI'][row]), seeing_r=float(SEE[row, 0]), seeing_i=float(SEE[row, 1]))
        st = stars.get(row, []) if MODE in ('gaia', 'gaia5') else stars.get(row)
        f = os.path.join(ATLAS, '%s-%d-atlas-%d.fits.gz' % (CAT['IAUNAME'][row].strip(), CAT['PID'][row], CAT['AID'][row]))
        if not os.path.exists(f):
            rec['flag'] = 'no image'; out.append(rec); continue
        if st is None or isinstance(st, str):
            rec['flag'] = 'no star list'; out.append(rec); continue
        for b, hdu in (('r', 2), ('i', 3)):
            try:
                m, (circ, sts) = measure_masked(f, float(CAT['RA'][row]), float(CAT['DEC'][row]), rec['phi'], rec['th50'], b,
                                                rec['seeing_' + b], st, nmc=20, seed=row)
            except Exception as e:
                m, circ, sts = dict(flag='error: %s' % e), [], []
            for k, v in m.items():
                if k not in ('e0', 'n0'):
                    rec['%s_%s' % (k, b)] = v
            rec['circ_%s' % b] = circ
            rec['strip_%s' % b] = sts
            if m.get('flag') == 'ok' and 'w1_mf' in m:
                with fits.open(f) as h:
                    hd = h[hdu].header
                gain, rn = hd.get('GAIN', np.nan), hd.get('RNOISE', 0.)
                CIRC, STRIPS = circ, sts
                for hz in ma.HZ:
                    smf, s17 = ma.synth_S(rec['th50'], rec['phi'], rec['seeing_' + b], hz, m['flux'], gain, rn, seed=row)
                    rec['S_mf_%s_hz%g' % (b, hz)] = smf
                    rec['S_eq17_%s_hz%g' % (b, hz)] = s17
                CIRC, STRIPS = [], []
        out.append(rec)
    json.dump(out, open(os.path.join(OUT, 'rows_%05d_%05d.json' % (start, stop)), 'w'), default=float)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
