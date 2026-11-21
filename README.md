# Horseshoe-Type Shrinkage from Ratio-Generated Priors

Companion code and numerical materials for the manuscript by Radhika Kulkarni,
Anirban DasGupta, and Brani Vidakovic. The experiments use synthetic
Donoho–Johnstone test signals and require no external dataset.

## Manuscript materials

| Item | Files |
|---|---|
| Table 1, representative ratio priors | [LaTeX table](tables/ratio_examples.tex) from version 13. This table summarizes analytical results. |
| Table 2, wavelet denoising | [Original Python code and simulation outputs](simulations/table2/), with the [LaTeX table](tables/dj_amse.tex) and [CSV](tables/dj_amse.csv) at manuscript precision. |
| Figures 1–4 | [Seven figure assets](figures/Figs/) for the manuscript. Figures 1–3 were recovered from the submitted PDF, and Figure 4 is the original PDF from the simulation archive. |
| MATLAB validation | [Original scripts and summaries](archive/matlab_validation/) for the alternative design used to check the Python ports. |
| Earlier wavelet study | [Original Python code, CSV summaries, and per-replication MSEs](archive/pilot/) from the 16-rule pilot. Its design and results predate Table 2. |
| Recovered plotting scripts | [MATLAB](figures/matlab/) for shrinkage geometry and [Wolfram Language](figures/wolfram/) for the bivariate prior and posterior summaries. These include developmental variants. |
| Independently rerun paired comparison | [Python](simulations/paired_posterior_means.py) and [saved results](simulations/reference_results/) for the Cauchy/Cauchy and horseshoe posterior means. |

The Table 2 package comes from the original `CaCa_DJ_python.zip` archive.
Its 16 base rules and marginal-maximum-likelihood Cauchy/Cauchy run give the
17 manuscript rows. It also includes the wider-grid horseshoe sensitivity
check. All 68 printed AMSEs and ranks agree with the saved replication losses.
The [paired summary](simulations/table2/out/paired_vs_horseshoe.csv) gives the
Cauchy/Cauchy minus horseshoe difference, Monte Carlo standard error and fraction
with lower MSE for each signal and both horseshoe grids. The original Doppler
difference on the standard grid is `-0.0036389519`, which rounds to `-0.0036`.
The MATLAB material comes from the `DJTestsSigmaPlugin` folder of
`Horseshoelike.zip`.

The independent four-specification Python comparison also remains available.
It uses the paper's sample size, root SNR, wavelet families and paired design
with independent seeds and numerical grids.

## Run the checks and simulations

From the repository root, in a Python 3.12 environment:

```bash
python -m pip install -r requirements.txt
python tables/check_results.py
python simulations/table2/verify_table2.py
python simulations/table2/make_outputs.py main
python simulations/table2/fig_amse.py
python simulations/paired_posterior_means.py --replications 1000
python figures/make_scalar_figures.py
```

To rerun the complete Table 2 simulation from its original seeds, execute
`sh simulations/table2/run_all.sh` in a working copy. It writes outputs to
its own `res/` and `out/` folders. The original README estimates about
30 minutes on two cores.

- The table check verifies the 17 ranks for each signal and all mean ranks,
  and reproduces the archived pilot's AMSEs and replication standard deviations
  from the saved losses.
- The paired simulation writes a CSV with mean MSE, Monte Carlo standard
  errors, and paired differences relative to the horseshoe on the standard
  scale grid. It also writes per-replication MSEs and median-ranked
  fixed-scale examples. Each signal in the MSE archive is a matrix with methods
  as rows and replications as columns. The default seed is 20261120 and the scale grids
  have 61 points. The bundled reference results use these defaults.
- The scalar figure script regenerates numerically comparable versions of
  Figures 1 and 2. The assets in `figures/Figs/` are the manuscript's artwork.
- The archived pilot script additionally requires Numba. To rerun it in an
  environment that has Numba, use
  `python archive/pilot/caucau_dj_snr5.py --M 1000 --output runs/pilot`.
- The MATLAB script uses `exportgraphics`. Run it from a separate working
  directory, with `figures/matlab/` on the MATLAB path. Wolfram scripts that
  use MaTeX require Mathematica and MaTeX. Run them from a separate working
  directory because the original scripts export to the current directory.

For citation metadata, see [CITATION.cff](CITATION.cff).
