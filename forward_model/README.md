# Forward model

The code that predicts, for every galaxy, the distribution of its U-shaped warp w₁ per unit ΔG/G_N (Secs. IV A–B of
the paper). Its outputs are the `signals/Signals_warp_*.dat` tables of the data record, whose 60 columns are
documented in the data record's README. The files are the code as it was run; they are not runnable from this
repository alone (see *Inputs*).

| File | Role |
|---|---|
| `Inverse_AM_avg_warp.py` | abundance matching: assigns each test galaxy its halo (scale radius r_s and density ρ(r_s)) over 200 mock catalogues; writes `warp_halos.npy` (`catalogue/warp_halos.npy` in the data record) |
| `Calc_Proxies_warp_NoScreen.py` | the forward model of Desmond & Ferreira (2020) at their 20 force ranges, λ_C = 0.4, 0.8, …, 8 Mpc; writes `Signals_warp_NoScreen_<ii>.dat`, ii = 0…19 |
| `Calc_Proxies_warp_FULL.py`, `run_full.sh` | the same model at arbitrary λ_C (8, 12, 16, 20 Mpc) for App. A; writes `Signals_warp_FULL_lam<λ>.dat` |
| `Calc_Proxies_warp_FULL_seed.py`, `run_full_seed.sh` | seeded replicates of the FULL runs (seeds 1–4, matched across ranges); writes `Signals_warp_FULL_s<seed>_lam<λ>.dat` |

## Inputs not distributed here

- The `BORG` reconstruction of the local density field (10 posterior samples on a 400³ grid, 290 Mpc radius) of
  Jasche & Lavaux (2019), as point masses (`LowFreq_fin1_<sample>_400.npy`).
- The source catalogue of NASA-Sloan Atlas galaxies with their abundance-matched halos (`Catalog_sources/`), the
  abundance-matching mock catalogues built on the DarkSky-400 simulation (`Catalog_AM/`), and the regressors that
  predict the contribution of galaxies too faint for the survey (`Corr/Correction_NoScreen_<ii>.npy`).

## How the runs differ

- The FULL runs use 300 Monte Carlo realisations and two BORG samples (rather than 1000 and 10), and the
  faint-galaxy regressors trained at λ_C = 8 Mpc (`ii = 19`) at every range. Ratios between ranges are therefore
  internally consistent, while the absolute limits differ from those of the NoScreen tables (App. A).
- Seeded replicates take their seed from the environment variable `REPLICATE_SEED`. The fourth command-line
  argument of the FULL producers instead limits the run to the first N galaxies, for testing.

## Known issue

The Gaussian softening of each density-field cell's potential is coded as `erf(d/sqrt(2)*sigma)` rather than
`erf(d/(sqrt(2)*sigma))`. This affects only the galaxy potentials used for the screening test (columns 51–53),
not the fifth-force field, and the difference is negligible on the 1.45 Mpc grid.
