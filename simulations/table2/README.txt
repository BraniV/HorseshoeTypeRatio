Cauchy/Cauchy posterior mean: Donoho-Johnstone wavelet illustration (Python)
===========================================================================

Reproduces Table "AMSE ... root SNR = 5" and the figure of the section
"Illustration: Wavelet Denoising of Standard Test Signals".

Requirements: Python >= 3.10, numpy >= 2.0, scipy, matplotlib.

Quick start
  ./run_all.sh                 full study, M = 1000 per signal (~30 min, 2 cores)
  python3 make_outputs.py main # rebuild table rows from the saved results in res/
  python3 fig_amse.py          # rebuild the figure from res/

Files
  djbench.py        signals, filters, periodic orthonormal DWT, and 16 rules:
                    Ca/Ca posterior mean (fixed tau_j), horseshoe posterior mean
                    (MMLE tau_j on [sigmaHat/n_j, sigmaHat]), BAMS, EBayesThresh
                    posterior median and mean, BayesThresh, raised-cosine,
                    Epanechnikov, NCPmean-2, LNWS, BayesShrink, BiShrink,
                    SureShrink, smooth SCAD-SURE, BlockJS, VisuShrink.
                    The rules shared with COMPARE_<SIGNAL>_CACA_EXPORT_ESTSIG.m
                    are line-by-line ports of that MATLAB code.
  run_mc.py         run_mc.py <Signal> <M> <SNR> <sd|msq> <seed> <out.npz>
                    'sd'  : root SNR = std(f)/sigma (population SD)  <- paper
                    'msq' : mean(f^2) = SNR (convention of the MATLAB scripts)
  run_caca_mmle.py  Ca/Ca posterior mean with tau_j by type-II (marginal) ML
                    over a log grid on [sigmaHat/n_j, 16 sigmaHat]
  run_hs_wide.py    horseshoe with the same wide grid (fairness check only)
  make_outputs.py   summary (out/main_summary.txt) and LaTeX rows (out/table_rows.tex)
  fig_amse.py       figure; titles = AMSE and MSE of the median-MSE replication
  res/*.npz         saved replication MSEs (main_, mmle_, hswide_) and sigmaHat
  out/DJ_SNR5_AMSE.csv   all AMSEs, SDs and Monte Carlo SEs

Design
  N = 1024; scaling coefficients at j0 = 3 kept; detail levels 3..9 shrunk.
  sigma = 1 generates the data only. sigmaHat = MAD(finest details)/0.6745 in
  every replication; BAMS, NCPmean, LNWS and Epanechnikov use their own
  internal noise estimates. No rule sees the true signal or the true sigma.
  Haar (Blocks), Daubechies-3 (Bumps), Symmlet-4 (Doppler, HeaviSine).
  Seeds (numpy default_rng): Blocks 20261104, Bumps 20261105,
  Doppler 20261106, HeaviSine 20261107. All rules, including the MMLE runs,
  see identical noise, so comparisons are paired.

Validation
  run_mc.py with SNR 7 and 'msq' (M = 300) reproduces the MATLAB AMSEs in
  DJTestsSigmaPlugin to within Monte Carlo error; the method ordering agrees.
