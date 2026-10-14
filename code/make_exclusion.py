"""Exclusion plot: dark-sector fifth force from galaxy warps.

Left panel  : the measured regime, 0.4 < lambda_C/Mpc < 8.
Right panel : the effectively-infinite-range regime (m_phi <~ H0) where coupled DE lives.
              Our bound there is the flat continuation of the 8 Mpc value: seed-matched forward-model
              replicates give 12/16/20 Mpc limits of 0.90/0.83/0.76 of the 8 Mpc one.
The left panel also shows the finite-range 95% bound of Bottaro+ 2025 (PRD 112, 023525), digitised from their Fig. 1
by digitise_bottaro2025.py into Bottaro2025_fig1_digitised.csv (in this directory).
Reads results/warp_darksector_bounds_sg.csv (analyse_new.py t sg); writes plots/Exclusion_coupledDE.{pdf,png}.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import paths as P

# The bounds table written by analyse_new.py is the single source for both the fiducial and the conservative limits.
_rows = [l for l in open(P.results('warp_darksector_bounds_sg.csv')) if not l.lstrip().startswith('#') and l.strip()]
_cols = _rows[0].strip().split(',')
_dat = np.genfromtxt(_rows[1:], delimiter=',')
lam = _dat[:, _cols.index('lambda_C_Mpc')]
dG2 = _dat[:, _cols.index('dG_over_GN_95pc')]
DEF = _dat[:, _cols.index('dG_over_GN_95pc_inflated')]

FID_INF, DEF_INF = dG2[-1], DEF[-1]

HBARC, MPC = 1.97327e-7, 3.0857e22
def lam2m(l):
    l = np.asarray(l, float); return HBARC/(np.where(l==0, np.nan, l)*MPC)
def m2lam(m):
    m = np.asarray(m, float); return HBARC/(np.where(m==0, np.nan, m)*MPC)
def dGb(x): return np.sqrt(np.asarray(x,float)/2.)
def bdG(b): return 2.*np.asarray(b,float)**2

YLO, YHI = 2e-4, 1.2
GREY, ALPH = '0.45', 0.5

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(9.6, 5.6), sharey=True, constrained_layout=True,
    gridspec_kw=dict(width_ratios=[2.15, 1.0]))

# ------------------------------------------------------------------ left: measured
axL.fill_between(lam, DEF, YHI, color=GREY, alpha=ALPH, lw=0)
axL.fill_between(lam, dG2, DEF, color=GREY, alpha=0.20, lw=0)
axL.plot(lam, dG2, color='crimson', lw=2.4, zorder=4)
axL.plot(lam, DEF, color='k', lw=1.7, zorder=4)
# Bottaro+ 2025 Fig. 1, solid grey (Planck + BOSS FS + BAO, 95%), digitised; beta there is our Delta G/G_N.
_b = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Bottaro2025_fig1_digitised.csv'),
                   delimiter=',', skip_header=3, names=True)
_s = (_b['lambda_Mpc'] > 0.3) & (_b['lambda_Mpc'] < 10)
axL.plot(_b['lambda_Mpc'][_s], _b['beta95'][_s], color='#b03a2e', lw=1.8, ls=(0,(6,1.5,1.5,1.5)), zorder=5)
axL.set_xscale('log'); axL.set_yscale('log')
axL.set_xlim(0.38, 8.4); axL.set_ylim(YLO, YHI)
axL.set_xlabel(r'$\lambda_C\ /\ \mathrm{Mpc}$', fontsize=13)
axL.set_ylabel(r'$\Delta G/G_N$  in the dark sector', fontsize=13)
axL.tick_params(labelsize=11)
axL.set_xticks([0.4,0.6,1,2,4,8]); axL.set_xticklabels(['0.4','0.6','1','2','4','8'])

secx = axL.secondary_xaxis('top', functions=(lam2m, m2lam))
secx.set_xlabel(r'$m_\phi\ /\ \mathrm{eV}$', fontsize=12); secx.tick_params(labelsize=10)

# ------------------------------------------------------------------ right: long range
axR.set_xlim(0, 1.02); axR.set_yscale('log')
axR.fill_between([0,1], [DEF_INF]*2, [YHI]*2, color=GREY, alpha=ALPH, lw=0)
axR.fill_between([0,1], [FID_INF]*2, [DEF_INF]*2, color=GREY, alpha=0.20, lw=0)
axR.plot([0,1], [FID_INF]*2, color='crimson', lw=2.4, ls=(0,(6,2)), zorder=4)
axR.plot([0,1], [DEF_INF]*2, color='k', lw=1.7, ls=(0,(6,2)), zorder=4)

pts = [(0.13, 0.0037, 0.0015, 'DESI DR2 BAO',          '#1f5fbf', 'o'),
       (0.30, 0.0069, 0.0021, 'DESI DR1 FS',           '#1f5fbf', 'o'),
       (0.47, 0.0043, 0.0014, 'DESI DR1 FS+DR2 BAO',   '#1f5fbf', 'o'),
       # Bottaro+ 2025 (DESI BAO, mild 2 sigma preference) and Bedroya/Vafa+ 2025, whose
       # ctilde = 0.05 +- 0.01 is the same convention as Agrawal+: Delta G/G_N = 2 ctilde^2.
       (0.60, 0.0040, 0.0020, 'Bottaro+ 2025',         '#7b4bbf', 'D'),
       (0.73, 2*0.05**2, 2*2*0.05*0.01, 'Bedroya+ 2025', '#7b4bbf', 'D'),
       (0.86, 2*0.03**2, None, r'coupled DE', '#0f9b8e', 's')]
for x, v, e, lab, c, mk in pts:
    if e: axR.errorbar(x, v, yerr=e, fmt=mk, ms=7.5, color=c, ecolor=c, capsize=3.5, lw=1.7, zorder=6)
    else: axR.plot(x, v, marker=mk, ms=8, color=c, zorder=6)

# Bottaro+ 2024 (PRL 132, 201002): Planck + BOSS full shape, beta = G_s/4 pi G_N < 4.8e-3 in
# exactly our convention: the tightest previous bound on this coupling, drawn across the panel.
BOTTARO = 4.8e-3
axR.axhline(BOTTARO, color='#b03a2e', lw=1.6, ls=(0,(4,2)), zorder=5)

lims = [(0.21, 0.07, 'Bonvin+ 2025'), (0.45, 0.10, 'Kesden & Kam. 2006'),
        (0.69, 0.20, 'Zamlung & Nusser 2026')]
for x, v, lab in lims:
    axR.plot(x, v, marker='v', ms=9, color='darkorange', zorder=6)
    axR.plot([x,x], [v, v*0.4], color='darkorange', lw=1.5, zorder=5)
    axR.annotate(lab, (x, v), textcoords='offset points', xytext=(0,12), ha='center',
                 fontsize=7.8, color='darkorange')
# Agrawal, Obied & Vafa: m_DM ~ exp(-ctilde kappa phi), best fit ctilde = 0.30, so
# Delta G/G_N = 2 ctilde^2 = 0.18.  Matches Table1.tex; do not hard-code a different value.
AGRAWAL = 2.*0.30**2
axR.plot(0.92, AGRAWAL, marker='*', ms=16, color='purple', zorder=7)
axR.annotate('Agrawal+', (0.92, AGRAWAL), textcoords='offset points', xytext=(0,-17),
             ha='center', fontsize=7.8, color='purple')
axR.annotate(r'$|\beta|\simeq0.03$', (0.86, 2*0.03**2), textcoords='offset points',
             xytext=(-4,-17), ha='center', fontsize=7.6, color='#0f9b8e')
axR.annotate('DESI-preferred\ncoupled dark sector', (0.36, 1.55e-2),
             ha='center', va='center', fontsize=8.4, color='#1f5fbf')

axR.set_xticks([])
axR.set_xlabel(r'$\lambda_C \gtrsim c/H_0$   $(m_\phi \lesssim H_0)$', fontsize=12)
axR.tick_params(labelsize=11)

secy = axR.secondary_yaxis('right', functions=(dGb, bdG))
secy.set_ylabel(r'$|\beta|=\sqrt{\Delta G/2G_N}$', fontsize=12)
secy.tick_params(labelsize=10)

handles = [Line2D([],[],color='crimson',lw=2.4,label='95% upper bound (warps, unscreened)'),
           Line2D([],[],color='k',lw=1.7,label=r'conservative systematics envelope'),
           Line2D([],[],color='#1f5fbf',marker='o',ls='none',ms=7,label='Clini+ 2026'),
           Line2D([],[],color='#7b4bbf',marker='D',ls='none',ms=6.5,label='Other DESI-related claims'),
           Line2D([],[],color='#b03a2e',lw=1.6,ls=(0,(4,2)),label='Bottaro+ 2024 (massless)'),
           Line2D([],[],color='#b03a2e',lw=1.8,ls=(0,(6,1.5,1.5,1.5)),label='Bottaro+ 2025 (finite range)'),
           Line2D([],[],color='darkorange',marker='v',ls='none',ms=8,label='Previous astrophysical limits'),
           Line2D([],[],color='#0f9b8e',marker='s',ls='none',ms=7,label='Coupled DE fit'),
           Line2D([],[],color='purple',marker='*',ls='none',ms=13,label='Proposed model')]
fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=8.8,
           framealpha=0.95, borderpad=0.7, columnspacing=1.8)

PLOTS = os.path.join(P.ROOT, 'plots')
os.makedirs(PLOTS, exist_ok=True)
plt.savefig(os.path.join(PLOTS, 'Exclusion_coupledDE.pdf'))
plt.savefig(os.path.join(PLOTS, 'Exclusion_coupledDE.png'), dpi=190)
print('saved. fiducial(inf)=%.3g  conservative(inf)=%.3g  |beta|<%.4f'%(FID_INF,DEF_INF,dGb(DEF_INF)))
print('improvement on Bottaro+ 2024 (4.8e-3): %.0fx conservative, %.0fx fiducial'
      %(4.8e-3/DEF_INF, 4.8e-3/FID_INF))
