"""Recalibrate each galaxy's suppression S with the physical warp shape: the edge-on projection of a
self-gravitating exponential disk (sg_profiles.py) instead of the in-plane halo-only K|x|^0.5 of
measure_all.synth_S. Everything else (R, PA, seeing, pixel grid, window, flux and noise weights, the
matched filter itself) is as in measure_all.synth_S. S2 = (estimator on warped - on unwarped) / the
forward model's Eq. 17 w1 (in-plane, halo-only, gamma = 0.5) at the same acceleration, so S2 carries the
seeing, the projection and the disk's self-gravity together.

python3 calib_sg.py <start> <stop>  -> meas_sg/rows_<start>_<stop>.json (warp_match rows)
"""
import json
import os
import sys
import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.special import k1
from measure_w1 import w1_from_image, PIX
import sg_profiles as sgp
import paths as P

CAT = fits.open(P.cat('warp_match.fits'))[1].data
SEE = np.load(P.cat('gal_seeing.npy'))
MU = np.load(P.sg('mu_sg.npy'))                                   # make_mu.py: mu for gamma 0.5 and 1
ATLAS = P.ATLAS
GAMMA = 0.5
DEX = 10 ** 0.2
# name -> (profile table, hz in units of R, mu_num, mu_den, band, seeing factor); mu_num/den given per galaxy
RUNS = {'fid': ('fid', 0.15, 'm', 'm', 'r', 1.), 'hz0.1': ('hz0.1', 0.10, 'm', 'm', 'r', 1.),
        'hz0.2': ('hz0.2', 0.20, 'm', 'm', 'r', 1.), 'g1': ('g1', 0.15, 'm1', 'm', 'r', 1.),
        'proj': ('proj', 0.15, 'm', 'm', 'r', 1.), 'reff': ('reff', 0.15, 'm', 'm', 'r', 1.),
        'Mp': ('fid', 0.15, 'mp', 'mp', 'r', 1.), 'Mm': ('fid', 0.15, 'mm', 'mm', 'r', 1.),
        'nopsf': ('fid', 0.15, 'm', 'm', 'r', 1e-3), 'ifid': ('fid', 0.15, 'm', 'm', 'i', 1.)}


def synth_S2(R, phi, fwhm, hz, flux, gain, rn, seed, zfun, reff, column='edgeon'):
    """As measure_all.synth_S, with midline zc(x) = zfun(|x|) (arcsec) and the edge-on exponential column
    profile u K1(u), u = |x|/R_d, R_d = R/reff ('inplane': exp(-1.678|x|/R), as synth_S)."""
    rng = np.random.default_rng(seed)
    sub = 4 if R / PIX < 15 else (2 if R / PIX < 60 else 1)
    half = 3. * R + 4. * fwhm + 4 * PIX
    n = int(np.ceil(2 * half / PIX)) | 1
    d = PIX / sub
    off = rng.random(2) * PIX - PIX / 2
    g = (np.arange(n * sub) - (n * sub - 1) / 2.) * d
    dE, dN = np.meshgrid(-g, g)
    dE, dN = dE - off[0], dN - off[1]
    s, c = np.sin(np.radians(phi)), np.cos(np.radians(phi))
    x = dE * s + dN * c
    z = dN * s - dE * c
    L = 3. * R
    K = 0.05 * R / L ** GAMMA                                # forward model: 0.05 R at |x| = L
    true = GAMMA * K * L ** (GAMMA - 1.) / ((GAMMA + 1.) * (GAMMA + 2.))
    if column == 'edgeon':
        u = np.abs(x) / (R / reff)
        col = np.where(u > 1e-6, np.maximum(u, 1e-6) * k1(np.maximum(u, 1e-6)), 1.)
    else:
        col = np.exp(-1.678 * np.abs(x) / R)
    sig = fwhm / 2.3548 / d
    hk = int(np.ceil(4 * sig))
    kx = np.arange(-hk, hk + 1)
    k1d = np.exp(-kx ** 2 / (2 * sig ** 2)); k1d /= k1d.sum()
    gp = (np.arange(n) - (n - 1) / 2.) * PIX
    E, N = np.meshgrid(-gp, gp)
    res = []
    for warp in (True, False):
        zc = zfun(np.abs(x)) if warp else 0.
        img = col / np.cosh((z - zc) / (hz * R)) ** 2
        img = fftconvolve(fftconvolve(img, k1d[None, :], mode='same'), k1d[:, None], mode='same')
        img = img.reshape(n, sub, n, sub).mean((1, 3))
        r0 = w1_from_image(img, E, N, phi, R, recentre=0)
        scale = flux / max(r0['flux'], 1e-30) if r0 is not None else 1.
        img = img * scale
        var = np.clip(img, 0., None) / gain + rn ** 2
        res.append(w1_from_image(img, E, N, phi, R, recentre=0, var=var))
    if res[0] is None or res[1] is None or 'w1_mf' not in res[0] or 'w1_mf' not in res[1]:
        return np.nan, np.nan
    return (res[0]['w1_mf'] - res[1]['w1_mf']) / true, (res[0]['w1'] - res[1]['w1']) / true


def zfun_for(tabs, table, mu_num, mu_den, R):
    """Midline in arcsec, normalised so that the forward model (mu_den) has 0.05 R at |x| = L."""
    reff = sgp.CONFIGS[table][0]
    Rd = R / reff
    s = 0.05 * R / (sgp.denominator_L(table, mu_den) * Rd)
    zb = -sgp.profile(tabs, table, mu_num)                   # U-shape along +z, zero at the centre
    return lambda ax: s * Rd * np.interp(ax / Rd, sgp.U, zb)


def main(start, stop):
    tabs = dict(np.load(P.sg('sg_tables.npz')))
    os.makedirs(P.sg('meas_sg'), exist_ok=True)
    prev = {}
    for f in os.listdir(P.meas('meas')):
        if f.startswith('rows_') and f.endswith('.json'):
            for r in json.load(open(P.meas('meas', f))):
                prev[r['row']] = r
    out = []
    for row in range(start, stop):
        rec = dict(row=row, mu=float(MU[row, 0]), mu1=float(MU[row, 1]))
        p = prev.get(row, {})
        f = os.path.join(ATLAS, '%s-%d-atlas-%d.fits.gz' % (CAT['IAUNAME'][row].strip(), CAT['PID'][row], CAT['AID'][row]))
        if not np.isfinite(rec['mu']) or not os.path.exists(f):
            out.append(rec); continue
        mus = {'m': rec['mu'], 'm1': rec['mu1'], 'mp': rec['mu'] / DEX, 'mm': rec['mu'] * DEX}   # M* +/- 0.2 dex
        hdr = {}
        with fits.open(f) as h:
            for b, hdu in (('r', 2), ('i', 3)):
                hdr[b] = (h[hdu].header.get('GAIN', np.nan), h[hdu].header.get('RNOISE', 0.))
        R, phi = float(CAT['SERSIC_TH50'][row]), float(CAT['SERSIC_PHI'][row])
        for name, (tab, hz, mn, md, b, fs) in RUNS.items():
            if p.get('flag_%s' % b) != 'ok' or 'flux_%s' % b not in p:
                continue
            zf = zfun_for(tabs, tab, mus[mn], mus[md], R)
            smf, s17 = synth_S2(R, phi, float(SEE[row, 0 if b == 'r' else 1]) * fs, hz, p['flux_%s' % b],
                                hdr[b][0], hdr[b][1], row, zf, sgp.CONFIGS[tab][0])
            rec['S2_mf_' + name], rec['S2_eq17_' + name] = smf, s17
        out.append(rec)
    json.dump(out, open(P.sg('meas_sg', 'rows_%05d_%05d.json' % (start, stop)), 'w'), default=float)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
