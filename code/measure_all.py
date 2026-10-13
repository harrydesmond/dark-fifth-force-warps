"""Measure w1 (matched filter + Eq. 17) for warp_match.fits rows [start, stop) in r and i, and calibrate each
galaxy's PSF/pixel/window suppression S by running the identical estimator on a synthetic warped disk
with the galaxy's own R (SERSIC_TH50), PA (SERSIC_PHI), seeing (SDSS Field.psfWidth), pixel grid,
window flux and noise model. Writes <data>/measurement/meas/rows_<start>_<stop>.json.
python3 measure_all.py <start> <stop>
"""
import json, os, sys
import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve
from measure_w1 import measure_file, w1_from_image, PIX
import paths as P

CAT = fits.open(P.cat('warp_match.fits'))[1].data
SEE = np.load(P.cat('gal_seeing.npy'))
ATLAS = P.ATLAS
HZ = (0.1, 0.15, 0.2)
GAMMA = 0.5


def synth_S(R, phi, fwhm, hz, flux, gain, rn, seed):
    """S = (estimator on warped synthetic - estimator on unwarped) / true Eq. 17 w1, for MF and Eq. 17."""
    rng = np.random.default_rng(seed)
    sub = 4 if R / PIX < 15 else (2 if R / PIX < 60 else 1)
    half = 3. * R + 4. * fwhm + 4 * PIX
    n = int(np.ceil(2 * half / PIX)) | 1
    d = PIX / sub
    off = rng.random(2) * PIX - PIX / 2                       # random sub-pixel position of the centre
    g = (np.arange(n * sub) - (n * sub - 1) / 2.) * d
    dE, dN = np.meshgrid(-g, g)
    dE, dN = dE - off[0], dN - off[1]
    s, c = np.sin(np.radians(phi)), np.cos(np.radians(phi))
    x = dE * s + dN * c
    z = dN * s - dE * c
    L = 3. * R
    K = 0.05 * R / L ** GAMMA                                 # midline displacement 0.05 R at |x| = L: linear regime
    true = GAMMA * K * L ** (GAMMA - 1.) / ((GAMMA + 1.) * (GAMMA + 2.))
    sig = fwhm / 2.3548 / d
    hk = int(np.ceil(4 * sig))
    kx = np.arange(-hk, hk + 1)
    k1 = np.exp(-kx ** 2 / (2 * sig ** 2)); k1 /= k1.sum()
    gp = (np.arange(n) - (n - 1) / 2.) * PIX
    E, N = np.meshgrid(-gp, gp)
    res = []
    for warp in (True, False):
        zc = K * np.abs(x) ** GAMMA if warp else 0.
        img = np.exp(-1.678 * np.abs(x) / R) / np.cosh((z - zc) / (hz * R)) ** 2
        img = fftconvolve(fftconvolve(img, k1[None, :], mode='same'), k1[:, None], mode='same')
        img = img.reshape(n, sub, n, sub).mean((1, 3))
        r0 = w1_from_image(img, E, N, phi, R, recentre=0)
        scale = flux / max(r0['flux'], 1e-30) if r0 is not None else 1.
        img = img * scale
        var = np.clip(img, 0., None) / gain + rn ** 2
        r = w1_from_image(img, E, N, phi, R, recentre=0, var=var)
        res.append(r)
    if res[0] is None or res[1] is None or 'w1_mf' not in res[0] or 'w1_mf' not in res[1]:
        return np.nan, np.nan
    return (res[0]['w1_mf'] - res[1]['w1_mf']) / true, (res[0]['w1'] - res[1]['w1']) / true


def main(start, stop):
    os.makedirs(P.meas('meas'), exist_ok=True)
    out = []
    for row in range(start, stop):
        rec = dict(row=row, nsaid=int(CAT['NSAID'][row]), th50=float(CAT['SERSIC_TH50'][row]),
                   phi=float(CAT['SERSIC_PHI'][row]), seeing_r=float(SEE[row, 0]), seeing_i=float(SEE[row, 1]))
        f = os.path.join(ATLAS, '%s-%d-atlas-%d.fits.gz' % (CAT['IAUNAME'][row].strip(), CAT['PID'][row], CAT['AID'][row]))
        if not os.path.exists(f):
            rec['flag'] = 'no image'; out.append(rec); continue
        for b, hdu in (('r', 2), ('i', 3)):
            try:
                m = measure_file(f, CAT['RA'][row], CAT['DEC'][row], rec['phi'], rec['th50'], band=b, nmc=20, seed=row)
            except Exception as e:
                m = dict(flag='error: %s' % e)
            for k, v in m.items():
                if k not in ('e0', 'n0'):
                    rec['%s_%s' % (k, b)] = v
            if m.get('flag') == 'ok' and 'w1_mf' in m:
                with fits.open(f) as h:
                    hd = h[hdu].header
                gain, rn = hd.get('GAIN', np.nan), hd.get('RNOISE', 0.)
                for hz in HZ:
                    smf, s17 = synth_S(rec['th50'], rec['phi'], rec['seeing_' + b], hz, m['flux'], gain, rn, seed=row)
                    rec['S_mf_%s_hz%g' % (b, hz)] = smf
                    rec['S_eq17_%s_hz%g' % (b, hz)] = s17
        out.append(rec)
    json.dump(out, open(P.meas('meas', 'rows_%05d_%05d.json' % (start, stop)), 'w'), default=float)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
