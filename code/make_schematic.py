"""Fig. 1: the geometry of a fifth force on dark matter warping a stellar disk.

Follows Fig. 1 of Desmond, Ferreira, Lavaux & Jasche (2018), whose coordinates the derivation in
the paper uses: the origin is the halo centre, z is the disk normal, and the stellar centroid sits
at r_star, where the halo's restoring force balances the fifth-force acceleration.

Sign: that paper draws a_{5,z} < 0 and states this implies z > 0, so the disk bows AWAY from the
fifth-force direction -- the equilibrium z grows with x because the restoring force weakens
outward. The 2020 forward model carries the same sign (w1 proportional to -a_perp).

Earlier layouts (three-panel, two-panel, and a two-panel version carrying the warp curves) are in
Plots/Old/ but are superseded; the first three also drew the sign the wrong way round.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

HALO = '#4a6fa5'
DISK = '#c8442e'
ACC = '#1a7a3a'


def arrow(ax, p0, p1, color, lw=1.8, mut=13):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=mut,
                                 lw=lw, color=color, zorder=5, shrinkA=0, shrinkB=0))


def schematic(path):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))

    for f, al in zip((1.00, 0.72, 0.48, 0.28), (0.07, 0.09, 0.11, 0.13)):
        ax.add_patch(Circle((0, 0), 1.62*f, color=HALO, alpha=al, lw=0, zorder=0))

    ax.annotate('', xy=(1.90, 0), xytext=(-1.90, 0),
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.1))
    ax.annotate('', xy=(0, 1.36), xytext=(0, -1.62),
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.1))
    ax.text(1.94, -0.04, r'$x$', fontsize=13, va='top')
    ax.text(-0.09, 1.39, r'$z$', fontsize=13, ha='right')

    r0, xd = 0.34, np.linspace(-1.62, 1.62, 400)
    zd = r0 + 0.58*np.sqrt(np.abs(xd)/1.62)          # z = r_star + K|x|^n, fiducial n = 0.5
    ax.plot(xd, zd, '-', color=DISK, lw=3.8, solid_capstyle='round', zorder=4)
    ax.text(-1.64, r0 + 0.70, 'stellar disk', color=DISK, fontsize=10.5)

    xp = 1.02
    zp = r0 + 0.58*np.sqrt(xp/1.62)
    ax.plot([0, xp], [0, zp], '-', color='0.25', lw=1.1, zorder=3)
    ax.plot([xp, xp], [0, zp], ':', color='0.45', lw=1.0, zorder=3)
    ax.plot([0, xp], [zp, zp], ':', color='0.45', lw=1.0, zorder=3)
    ax.plot([xp], [zp], 'o', color='0.15', ms=4.8, zorder=5)
    ax.text(0.44, 0.38, r'$r$', fontsize=13, color='0.15')
    ax.text(xp/2., zp + 0.05, r'$x$', fontsize=12, color='0.35', ha='center')
    ax.text(xp + 0.07, zp/2., r'$z$', fontsize=12, color='0.35', va='center')

    ax.annotate('', xy=(-0.14, r0), xytext=(-0.14, 0),
                arrowprops=dict(arrowstyle='<->', color='0.15', lw=1.2))
    ax.text(-0.22, r0/2., r'$r_\star$', fontsize=13, color='0.15', ha='right', va='center')

    arrow(ax, (0.14, -0.12), (0.14, -1.10), ACC, lw=2.5, mut=16)
    ax.text(0.28, -0.64, 'fifth force\non halo', color=ACC, fontsize=11, ha='left', va='center')
    ax.plot([0], [0], 'o', color='k', ms=5.4, zorder=6)
    ax.text(-0.12, -0.12, 'halo centre', fontsize=10.5, ha='right', va='top')

    ax.set_xlim(-2.00, 2.10)
    ax.set_ylim(-1.78, 1.52)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(path.replace('.png', '.pdf'))
    fig.savefig(path, dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    import os
    import paths as P
    PLOTS = os.path.join(P.ROOT, 'plots')
    os.makedirs(PLOTS, exist_ok=True)
    schematic(os.path.join(PLOTS, 'Schematic_D_coords.png'))
    print('wrote plots/Schematic_D_coords.{png,pdf}')
