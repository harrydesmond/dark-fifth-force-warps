"""Likelihood-independent numbers for the paper on the fiducial sample (remeasured w1, size cut, artefacts
removed; 3968 galaxies) with the physical-warp suppression S2 applied to the predictions. Ranking statistic:
the mixture-weighted mean |M| x S2 (the excision statistic). Writes <data>/results/descriptive_sg.json."""
import json, os
import numpy as np
from astropy.io import fits
from robust_like import load_signals, select, ARTEFACTS, DATA
import lap_run
import paths as P

CHI = 4.22e-5
H = np.load(P.cat('warp_halos.npy'))   # RA,Dec,D,Mr,Reff,...
N = np.load(lap_run.NEWF)
c = fits.open(P.cat('warp_match.fits'))[1].data
th50 = c['SERSIC_TH50'][N['warp_match_row']]
nd = lap_run.new_data(sg=True)
rows = np.where(nd['keep'])[0]
rows = rows[~np.isin(rows, ARTEFACTS)]


def amp(S, scale):
    nc = S[:, 2].astype(int); v = np.arange(15)[None, :] < nc[:, None]
    W = np.where(v, S[:, 3:18], 0.); M = np.where(v, S[:, 18:33], 0.)
    return (W * np.abs(M)).sum(1) / W.sum(1) * scale


S19, S0 = load_signals('Signals_warp_NoScreen_19.dat'), load_signals('Signals_warp_NoScreen_0.dat')
S2 = nd['scale']
a19, a0 = amp(S19, S2)[rows], amp(S0, S2)[rows]
raw19 = amp(S19, 1.)                                      # unsuppressed forward-model amplitude, all rows
s, h = S19[rows], H[rows]
out = dict(N=len(rows), S2_median=float(np.median(S2[rows])), S2_16_84=np.percentile(S2[rows], [16, 84]).tolist(),
           mu_median=float(np.nanmedian(N['sg_mu'][rows])), mu_16_84=np.nanpercentile(N['sg_mu'][rows], [16, 84]).tolist(),
           P99_amp=float(np.percentile(a19, 99)), median_amp=float(np.median(a19)),
           growth_median_amp=float(np.median(a19) / np.median(a0)), growth_P99_amp=float(np.percentile(a19, 99) / np.percentile(a0, 99)),
           raw_P99_amp=float(np.percentile(raw19[rows], 99)),
           artefacts_raw_amp=[float(raw19[i]) for i in ARTEFACTS], artefacts_mode=[float(abs(S19[i, 58])) for i in ARTEFACTS],
           artefacts_in_measured=[int(i) for i in ARTEFACTS if nd['keep'][i]],
           w1_absmax=float(np.nanmax(np.abs(nd['w_new'][rows]))),
           w1_mad=float(1.4826 * np.median(np.abs(nd['w_new'][rows] - np.median(nd['w_new'][rows])))),
           w1_err_median=float(np.median(nd['e_new'][rows])))
phi, phi_up = 10 ** s[:, 52], 10 ** s[:, 53]
a5 = 10 ** s[:, 55]
for lab, frac in (('top5', 0.05), ('top1', 0.01)):
    t = a19 >= np.quantile(a19, 1 - frac); r = ~t
    out[lab] = dict(n=int(t.sum()), a5_ratio=float(np.median(a5[t]) / np.median(a5[r])),
                    Reff_tail=float(np.median(h[t, 4])), Reff_rest=float(np.median(h[r, 4])),
                    th50_tail=float(np.median(th50[rows][t])), th50_rest=float(np.median(th50[rows][r])),
                    S2_tail=float(np.median(S2[rows][t])), mu_tail=float(np.nanmedian(N['sg_mu'][rows][t])),
                    mu_rest=float(np.nanmedian(N['sg_mu'][rows][r])),
                    phi_ratio=float(np.median(phi[t]) / np.median(phi[r])),
                    D_tail=float(np.median(h[t, 2])), D_rest=float(np.median(h[r, 2])),
                    Mr_tail=float(np.median(h[t, 3])), Mr_rest=float(np.median(h[r, 3])),
                    unscreened=float(np.mean(phi[t] < CHI)), unscreened_pess=float(np.mean(phi_up[t] < CHI)),
                    phi_median=float(np.median(phi[t])), phi_median_upper=float(np.median(phi_up[t])))
out['all'] = dict(unscreened=float(np.mean(phi < CHI)), unscreened_pess=float(np.mean(phi_up < CHI)),
                  phi_median=float(np.median(phi)), phi_max=float(phi.max()), D_median=float(np.median(h[:, 2])),
                  Reff_median=float(np.median(h[:, 4])))
json.dump(out, open(P.results('descriptive_sg.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
