# dark-fifth-force-warps

Code for **Desmond & Ferreira (2026), *Galactic warps constrain equivalence-principle violation in the dark
sector*** [[arXiv:2610.24618](https://arxiv.org/abs/2610.24618)]. The paper bounds a fifth force that acts on dark matter but not on stars, of strength
ΔG/G_N relative to gravity, using the U-shaped warps such a force induces in edge-on stellar disks: ΔG/G_N < 7.7×10⁻⁴
(95%, fiducial) or < 1.7×10⁻³ once systematic uncertainties are included, for force ranges λ_C ≥ 8 Mpc.

The data products (forward-model predictions, warp measurements, self-gravity calibration, likelihood runs and the
summary results) are on Zenodo: [doi:10.5281/zenodo.23343105](https://doi.org/10.5281/zenodo.23343105). `fetch_data.py` downloads them into `data/`.

## Quick start: the paper's numbers and figures from the released products

Minutes on a laptop.

```bash
pip install -r requirements.txt
python fetch_data.py                      # 74 MB download, 204 MB unpacked into data/
cd code
python analyse_new.py t sg > ../data/results/report_sg.txt                  # Table I, Secs. V and VI, App. A and C
DFW_MASKED=1 python analyse_new.py t sg > ../data/results/report_sg_masked.txt   # point-source-masking test
python descriptive_sg.py                   # sample statistics, galaxy potentials, unscreened fractions
python paper_numbers.py                    # remaining numbers of Secs. III-VII and App. B
python calc_screening.py                   # Phi_LSS, the screening threshold of Sec. VI
python make_schematic.py; python make_corner.py; python make_exclusion.py   # Figs. 1-3, into plots/
```

`analyse_new.py` reads the ~1600 likelihood runs in `data/inference/t/` (and `data/inference/masked/t/`), each a
JSON file of posterior quantiles written by `lap_run.py`, and assembles every likelihood-dependent number of the paper.

## The full pipeline

All scripts are in `code/` and find their inputs through `code/paths.py`:

| Variable | Meaning | Default |
|---|---|---|
| `DFW_DATA` | the unpacked data record | `data/` |
| `DFW_ATLAS` | the NSA atlas images | `atlas/` |
| `DFW_MASKED` | `1` runs the inference on the point-source-masked warps | `0` |

Re-running a step overwrites its products under `DFW_DATA`, so point `DFW_DATA` at a copy to keep the originals.
Steps 1–4 regenerate the per-galaxy inputs to the likelihood; each step's products are in the data record, so any
step can be re-run on its own.

**1. Inputs.** `make_inputs.py` writes the atlas URL list, the SDSS field list and the per-galaxy seeing from
`catalogue/warp_match.fits` (the NSA v1_0_1 entries of the warp sample of Desmond & Ferreira 2020) and
`catalogue/field_seeing.csv`. `get_seeing.py` queries SDSS DR17 SkyServer for that seeing (`Field.psfWidth`).
`download_atlas.sh` fetches the 4357 NSA atlas images from the public SDSS archive (7.2 GB).

**2. Warp measurement** (Sec. III). `measure_w1.py` holds the estimators: the matched filter z = a + bx + K|x|^γ fitted
to the column midlines with photon-noise weights, and the direct sum that defines w₁. `measure_all.py <start> <stop>`
measures a range of catalogue rows in the r and i bands (about 1 s per galaxy). `build_new_w1.py` collects the
results into `measurement/new_w1.npz` and sets the size-consistency flag. `test_measure.py` checks the estimators on
synthetic warped, tilted, S-warped and flat disks, and gives the matched filter's recovery of an |x|^0.5 warp quoted
in Sec. III. `export_w1_table.py` writes the readable `measurement/w1_table.csv`.

**3. Self-gravity and the suppression factor S_i** (Secs. IV A–B).
- Halo and profiles: `make_mu.py` computes M_halo(<R_eff)/M★ for every galaxy. `selfgrav.py` solves for the
  equilibrium midline of a self-gravitating exponential disk of rings, and `kernel_test.py` validates its ring kernel
  against the bending of a uniform sheet. `sg_profiles.py` (and `sg_profiles.py gas`) tabulates the projected midline
  against M_halo(<R_eff)/M★.
- Calibration: `calib_sg.py <start> <stop>` calibrates S_i on synthetic images of each galaxy, and `calib_gas.py` does
  the same for the gas-disk variation. `add_sg_to_npz.py` adds both to `new_w1.npz`.

**4. Point-source masking** (Sec. IV D, a systematics test). `gaia_query.py` fetches Gaia DR3 sources around each
galaxy. `measure_masked.py` fits each star's residual amplitude in the image and remeasures the warp with the
contaminated columns removed. `calib_sg_masked.py` recomputes S_i with the same mask, and `assemble_masked.py`
writes `masking/new_w1_masked_gaia5.npz`.

**5. Likelihood** (Secs. IV C–D, V, App. A, C). `robust_like.py` selects the sample and loads the forward-model
mixtures; `robust_lap.py` evaluates the Student-t likelihood of Sec. IV C on a fixed grid in ΔG/G_N, integrating out
σ_w1 and ν by Laplace's method. `lap_run.py t <name>` runs one configuration (about 1 min on one core), and
`run_lap.sh t <group>` runs a group in parallel on one node, skipping finished units. The paper's groups are:

| Group | What |
|---|---|
| `sgpub` | fiducial and every variation of Table I(b), at the 20 force ranges |
| `sgtest` | at 8 Mpc: sign-free fits, screening and the catalogue artefacts kept |
| `sgexcalA`, `sgexcalB` | excision calibration on signal-free mocks (App. C) |
| `sginj` | injection tests (App. C) |
| `sgfull` | 8–20 Mpc seed-matched forward-model runs (App. A) |

The masked analysis uses `DFW_MASKED=1` with `sgpub`, `sgtest`, `sgexcalA` and `sgexcalB`. At the six shortest
ranges the posterior extends beyond the default grid (ΔG/G_N ≤ 0.02), so `LAP_GRIDMAX=0.15` extends it to 0.15. The
released runs used it for `sgpub0`–`5`, every `sgpub*_gas`, the `sgexcalA` units at those ranges and every masked
run. Elsewhere the extension changes only `logZ`, by the flat-prior normalisation log(0.15/0.02), so it is safe to
set it for every run. Running all groups takes roughly 70 core-hours, most of it the excision calibration.
`corner_8mpc.py` evaluates the 8 Mpc posterior on a brute-force three-dimensional grid for Fig. 2 (5 min).

**6. Results.** `analyse_new.py`, `descriptive_sg.py`, `paper_numbers.py` and `calc_screening.py`, as in the quick
start. `drivers_v4.py 19 fid 1` identifies the galaxies that drive the difference between the masked and unmasked
sign-free fits (Sec. V). `Bottaro2025_fig1_digitised.csv` is the 95% bound of Bottaro et al. (2025, Phys. Rev. D 112,
023525) against force range, digitised from their Fig. 1 by `digitise_bottaro2025.py <arXiv:2407.18252 PDF>` (needs
`pdftoppm`); `make_exclusion.py` draws it in Fig. 3 and `analyse_new.py` interpolates it to 8 Mpc for Table I(c).

### Where each part of the paper comes from

| Paper | Script |
|---|---|
| Fig. 1 | `make_schematic.py` |
| Sec. III: measurement, sample, noise | `measure_all.py`, `test_measure.py`, `paper_numbers.py` |
| Sec. IV A–B: self-gravity, S_i, catalogue artefacts | `selfgrav.py`, `sg_profiles.py`, `calib_sg.py`, `paper_numbers.py`, `descriptive_sg.py` |
| Sec. IV C–D, Table I(b): likelihood, variations, conservative bound | `lap_run.py`, `analyse_new.py` |
| Sec. IV D: masking | `gaia_query.py`, `measure_masked.py`, `analyse_new.py` with `DFW_MASKED=1` |
| Sec. V, Table I(a, c), Fig. 3 | `analyse_new.py`, `make_exclusion.py`, `digitise_bottaro2025.py`, `paper_numbers.py`, `drivers_v4.py` |
| Fig. 2 | `corner_8mpc.py`, `make_corner.py` |
| Sec. VI: screening | `calc_screening.py`, `descriptive_sg.py`, `lap_run.py t sgtest_screen` |
| Sec. VII, App. B | `paper_numbers.py` |
| App. A: the massless limit | `forward_model/`, `lap_run.py` group `sgfull`, `analyse_new.py` |
| App. C: likelihood evaluation, injections, excision calibration | `robust_lap.py`, `lap_run.py`, `analyse_new.py` |

## The forward model

`forward_model/` holds the code that predicts each galaxy's warp per unit ΔG/G_N, which produced the
`signals/Signals_warp_*.dat` tables of the data record. It is the code as it was run, and it needs inputs that are not
redistributed here: the BORG reconstruction of the local density field, the DarkSky-400 halo catalogues and the
abundance-matching mocks. See `forward_model/README.md`.

## Environment

Python ≥ 3.9 with NumPy, SciPy, Astropy, Matplotlib and JAX (64-bit mode is switched on in `robust_lap.py`).
The likelihood runs used Python 3.9.7, NumPy 1.22.4, SciPy 1.11.3, Astropy 5.1 and JAX 0.4.13. The measurement,
calibration and figures were also run under Python 3.11.6, NumPy 1.26.4, SciPy 1.11.4, Astropy 7.0.1, JAX 0.5.3 and
Matplotlib 3.9.4. Every output checked under both environments agreed to floating-point precision.

## Citation and licence

Please cite Desmond & Ferreira (2026), arXiv:2610.24618, and, for the forward model and the galaxy sample, Desmond & Ferreira (2020),
Phys. Rev. D 102, 104060. Code: MIT licence (`LICENCE`). Data record: CC BY 4.0.
