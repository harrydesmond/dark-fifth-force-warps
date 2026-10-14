"""Digitise the current 95% bound of Bottaro et al. 2025 (PRD 112, 023525; arXiv:2407.18252), Fig. 1.

The solid grey line (Planck + BOSS full shape + BAO) is traced from a 500 dpi render of page 2 of the
arXiv PDF. Axes are calibrated from the major tick marks found by a pixel scan (bottom: m_phi/H0 =
1e-2..1e5; left: beta = 1e-2..1e-4; top: lambda_phi = 1e5..1e-1 Mpc), and the line is followed
column by column from m_phi/H0 = 100, choosing the grey cluster nearest the previous point so that the
dashed Planck-only line, the DESI error bars and the axis labels are not picked up.
Precision is set by the line width (~4 px of 359 px per decade, i.e. ~2.5% in beta).

Usage: python3 digitise_bottaro2025.py /path/to/2407.18252.pdf
Writes Bottaro2025_fig1_digitised.csv next to this script (read by make_exclusion.py). Needs pdftoppm (poppler).
"""
import sys, subprocess, tempfile, os
import numpy as np
from PIL import Image

pdf = sys.argv[1]
DPI = 500
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(['pdftoppm', '-f', '2', '-l', '2', '-r', str(DPI), '-png', pdf,
                    os.path.join(tmp, 'p')], check=True)
    png = [f for f in os.listdir(tmp) if f.endswith('.png')][0]
    page = Image.open(os.path.join(tmp, png)).convert('RGB')
s = DPI / 200 * 1.10                                   # crop box located by eye on a 200 dpi render
im = np.asarray(page.crop((int(130*s), int(140*s), int(770*s), int(560*s)))).astype(int)
H, W, _ = im.shape
dark = im.sum(2) < 300

def groups(idx):
    g = []
    for v in idx:
        if g and v - g[-1][-1] <= 2: g[-1].append(v)
        else: g.append([v])
    return g

def major_ticks(cnt, lo, hi):
    """Centres of tick groups along one spine; major ticks are the long ones."""
    gg = groups([i for i in range(lo, hi) if cnt[i] >= 5])
    L = np.array([cnt[t].max() for t in gg]); c = np.array([np.mean(t) for t in gg])
    return c[L > 0.75*L.max()]

# spines: left = longest dark column, bottom/top = longest dark rows
xl = np.argmax(dark[:, :W//3].sum(0))
while dark[H//2, xl-1]: xl -= 1
rows = dark[:, W//4:3*W//4].sum(1)
yb = np.argmax(rows[H//2:]) + H//2
while dark[yb+1, W//2]: yb += 1
yt = np.argmax(rows[:H//2])
while dark[yt+1, W//2]: yt += 1
xr = np.argmax(dark[:, 2*W//3:].sum(0)) + 2*W//3
while dark[H//2, xr-1]: xr -= 1

xb = major_ticks(dark[yb+1:yb+26, :].sum(0), xl+5, xr-3)          # bottom: outward ticks
xt = major_ticks(dark[yt+1:yt+26, :].sum(0), xl+5, xr-3)          # top: inward ticks
yL = major_ticks(dark[:, xl-25:xl].sum(1), yt+5, yb-1)         # left: outward ticks
assert len(xb) == 8 and len(xt) == 7 and len(yL) == 3, (xb, xt, yL)
px, ax = np.polyfit(np.arange(-2, 6), xb, 1)                   # x = ax + px*log10(m/H0)
pl, al = np.polyfit(np.arange(5, -2, -1), xt, 1)               # x = al + pl*log10(lambda/Mpc)
py, ay = np.polyfit(np.array([-2, -3, -4]), yL, 1)             # y = ay + py*log10(beta)

grey = (np.abs(im - 169) <= 6).all(2)

def clusters(x):
    ys = np.where(grey[yt+3:yb-2, x])[0] + yt + 3
    return [np.mean(c) for c in groups(list(ys))] if len(ys) else []

x0 = int(round(ax + 2*px))                                     # m/H0 = 100: the solid line is the lowest grey
ytrace = {x0: max(clusters(x0))}
for step in (+1, -1):
    yprev, x = ytrace[x0], x0 + step
    while xb[0] <= x <= xb[-1]:
        c = clusters(x)
        if c:
            yc = min(c, key=lambda v: abs(v - yprev))
            if abs(yc - yprev) <= 8:
                ytrace[x] = yprev = yc
        x += step
xs = np.array(sorted(ytrace)); ys = np.array([ytrace[x] for x in xs])
logm = (xs - ax) / px
loglam = (xs - al) / pl
logb = (ys - ay) / py
keep = np.arange(0, len(xs), 4)
hdr = ('# Bottaro et al. 2025 (PRD 112, 023525), Fig. 1, solid grey: 95%% bound from Planck + BOSS FS + BAO.\n'
       '# Digitised by digitise_bottaro2025.py from a %d dpi render; ~2.5%% precision in beta.\n'
       '# lambda_phi from the figure top axis (c/H0 = %.0f Mpc). beta = G_s/(4 pi G_N) = Delta G/G_N.\n'
       'm_over_H0,lambda_Mpc,beta95' % (DPI, 10**((ax - al)/pl)))
np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Bottaro2025_fig1_digitised.csv'),
           np.c_[10**logm[keep], 10**loglam[keep], 10**logb[keep]],
           delimiter=',', header=hdr, comments='', fmt='%.5g')
print('calibration: %.2f px/dex (bottom), %.2f (top), %.2f (left); c/H0 = %.0f Mpc'
      % (px, -pl, -py, 10**((ax - al)/pl)))
print('traced %d of %d columns' % (len(xs), int(xb[-1] - xb[0]) + 1))
for lam in [0.4, 1, 2, 4, 8, 20, 45, 100, 500, 1000, 4420]:
    print('lambda %7.1f Mpc  beta95 %.2e' % (lam, 10**np.interp(np.log10(lam), loglam[::-1], logb[::-1])))
