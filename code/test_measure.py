"""Validate measure_w1.w1_from_image on synthetic north-up/east-left images (no PSF)."""
import numpy as np
from measure_w1 import w1_from_image, PIX, Q

def synth(phi, R, warp, n=None, hz=0.1, sub=4):
    n = n or int(8 * R / PIX) | 1
    # sub-pixel sampled image of an edge-on disk whose midline is z_c(x) along v
    g = (np.arange(n * sub) - (n * sub - 1) / 2.) * PIX / sub
    dE, dN = np.meshgrid(-g, g)                       # east-left: column index increases to the West
    s, c = np.sin(np.radians(phi)), np.cos(np.radians(phi))
    x = dE * s + dN * c; z = dN * s - dE * c
    img = np.exp(-1.678 * np.abs(x) / R) / np.cosh((z - warp(x)) / (hz * R)) ** 2
    img = img.reshape(n, sub, n, sub).mean((1, 3))
    gp = (np.arange(n) - (n - 1) / 2.) * PIX
    E, N = np.meshgrid(-gp, gp)
    return img, E, N

def eq17(warp, R):
    L = 3 * R; x = np.linspace(-L, L, 20001); z = warp(x); z = z - z.mean()
    return np.trapz(np.abs(x) * z, x) / L ** 3

R = 8.0
for name, warp in [('U, arms to +v', lambda x: 0.3 * np.sqrt(np.abs(x) / (3 * R))),
                   ('U, arms to -v', lambda x: -0.3 * np.sqrt(np.abs(x) / (3 * R))),
                   ('tilt', lambda x: 0.3 * x / (3 * R)),
                   ('S-warp', lambda x: 0.3 * x * np.abs(x) / (3 * R) ** 2),
                   ('none', lambda x: 0. * x)]:
    vals = []
    for phi in (0., 37., 90., 143., 260.):
        img, E, N = synth(phi, R, warp)
        vals.append(w1_from_image(img, E, N, phi, R)['w1'])
    print('%-14s true Eq.17 %+.3e | measured at PA 0/37/90/143/260: %s' % (name, eq17(warp, R), ' '.join('%+.3e' % v for v in vals)))

print('--- matched filter, noiseless synthetic (var = flat) ---')
for name, warp in [('U +v', lambda x: 0.3 * np.sqrt(np.abs(x) / (3 * R))), ('tilt', lambda x: 0.3 * x / (3 * R)), ('none', lambda x: 0. * x)]:
    img, E, N = synth(37., R, warp)
    r = w1_from_image(img, E, N, 37., R, var=np.ones_like(img) * 1e-6)
    print('%-6s Eq17 %+.3e  matched filter %+.3e  (true %+.3e)' % (name, r['w1'], r['w1_mf'], eq17(warp, R)))

# Sec. III: recovery of the |x|^0.5 warp by the matched filter, with flat weights (sky-dominated noise) and with
# variance proportional to the image (source-dominated); S_i is calibrated with the same estimator, so it absorbs this.
print('--- matched filter / true for a |x|^0.5 warp at PA 0/37/90/143/260 ---')
U = lambda x: 0.3 * np.sqrt(np.abs(x) / (3 * R))
for wname in ('flat (sky)', 'image (source)'):
    rat = []
    for phi in (0., 37., 90., 143., 260.):
        img, E, N = synth(phi, R, U)
        v = np.ones_like(img) * 1e-6 if wname.startswith('flat') else np.maximum(img, 1e-12)
        rat.append(w1_from_image(img, E, N, phi, R, var=v)['w1_mf'] / eq17(U, R))
    print('var = %-14s %s   range %.3f-%.3f' % (wname, ' '.join('%.3f' % r for r in rat), min(rat), max(rat)))
