Cauchy/Cauchy Wavelet Denoising: 2-page Illustration (4 November 2026)
======================================================================

FILES
-----
CauCau_DJ_section.tex          LaTeX source section to insert immediately before Conclusions.
Horseshoelike10_with_DJ.tex   Original uploaded full LaTeX source with section input,
                              plus a revised concluding paragraph. Requires the original
                              project figures and references.bib in Overleaf.
CauCau_DJ_references.bib      Five verified additional bibliography records. If you choose
                              the inserted section rather than the integrated manuscript,
                              add this database to the existing bibliography call.
CauCau_DJ_preview.pdf         Clean 2-page standalone layout preview (citations expanded
                              to author/year; in Overleaf manuscript they use natbib).
CauCau_DJ_preview.tex         Source for standalone proof layout.
caucau_dj_snr5.py             Full reproducible Python simulation; no Wavelet Toolbox needed.
CauCau_DJ_M1000_AMSE.csv      Numerical results (average and replication-SD MSE).
CauCau_DJ_M1000_noise_estimates.csv  Diagnostics of the centered MAD estimate.
CauCau_DJ_M1000_replication_MSE.npz  Each replicated MSE for each method and signal.

EXPERIMENT (actually run, not a template)
-----------------------------------------
Signal: 4 Donoho--Johnstone test functions, t_i=i/n, n=1024.
Signal scale: multiply f0 by 5 / std_n(f0), with std_n using divisor n.
Noise: same independent N(0,1) realization for every method in a given
       signal/replication, M=1000, fixed RNG seed per signal.
SNR: std_n(signal)/population_std(noise) = 5 (SD ratio, not power ratio).
Transform: sparse periodic orthonormal Wavmat convention, 8 scaling coeffs
       unchanged; j=3,...,9 details shrink. Blocks=Haar, Bumps=db3 6 taps,
       Doppler and HeaviSine=symmlet4 8 taps.
Plug-in noise SD: centered finest-detail median absolute deviation,
       median(|d9 - median(d9)|)/0.6744897501960817.
Risk: mean squared error n^{-1}sum_i(fhat_i-f_i)^2 per replicate, averaged
       across M; identical to coefficient-domain MSE by Parseval.

FROZEN HYPERPARAMETERS / ALGORITHM DISCLOSURE
---------------------------------------------
Cauchy/Cauchy posterior mean: tau_j = 45*sigmahat*2^(-1.45*j).
Cauchy/Cauchy posterior median: tau_j=35*sigmahat*2^(-1.25*j).
Cauchy/Cauchy second mode: tau_j=100*sigmahat*2^(-1.1*j),
       threshold inflation sqrt(log(1024))/2.
Cauchy/Cauchy practical-null Bayes test: tau_j=20*sigmahat*2^(-j),
       epsilon_j=1.25*sigmahat*(j/3)^(2/3); retain d iff P(H1|d)>1/2.
BAMS: original wavelet-domain posterior mean, but its Laplace noise
       variance is matched to the same sigmahat^2 rather than its native
       Tukey IQR estimate. This changes the reported BAMS implementation;
       it should be labeled 'BAMS (common MAD plug-in)' in the manuscript.
EB Laplace: Johnstone--Silverman spike + Laplace(a=0.5) slab, levelwise MML
       mixing probability (universal median-threshold cap), mean & median.
EB Gaussian: zero spike + N(0,v) slab, levelwise EM fitting of v and w.
Horseshoe EB: independent half-Cauchy local scales, common tau chosen
       levelwise by MML over 28 predeclared values in [10^-1.5,10^1.5].
Raised cosine: 1D Gaussian posterior mean under cosine slab and
       levelwise atom weight 1-1/(j-2)^2, global beta=max detail magnitude.
Epanechnikov: 1D posterior mean under Epanechnikov slab and integrated-
       variance Laplace likelihood, with MAD-based variance matching.
BayesShrink: data-driven levelwise threshold sigmahat^2/sigmaX.
SureShrink: levelwise Stein soft-threshold with universal cap.
BiShrink: one-dimensional periodic parent-child extension, 7-tap local
       energy window; NOT the published image procedure unchanged.
BlockJS: fixed nonoverlapping blocks of floor(log N) wavelet coefficients.
Universal soft: sigmahat sqrt(2 log N) at each detail.

The code uses NumPy, SciPy and Numba. Run from any directory:
  python caucau_dj_snr5.py --M 1000 --output /path/to/results
The code produces all 16 columns using paired noise, and caches
completed signal conditions so interrupted runs can resume safely.

No clean signal value is used for parameter fitting; it is used only to
scale the fixed design SNR and to assess MSE. The parameters of the
Cauchy/Cauchy mean are carried over from the supplied earlier MATLAB
illustration and held fixed; they have not been optimized for this run.

DISCLOSURE FOR REFEREES
-----------------------
The 15 comparators include three alternative Cauchy/Cauchy decisions and
12 other empirical-Bayes/Bayesian/classical procedures. Several original
image algorithms and some prior methods are implemented as explicitly
labeled 1D adaptations, and this calculation is a transparent pilot rather
than an assertion that each line matches the original authors' software.
Published-method-specific validation should precede a claim of exact
external benchmark reproduction. The section makes no claim of dominance.

IN OVERLEAF
-----------
1. Upload CauCau_DJ_section.tex and CauCau_DJ_references.bib to the
   manuscript project (do not replace your existing references.bib).
2. Insert '\input{CauCau_DJ_section}' just before the Discussion and
   Conclusions section; replace the final 'no simulations' paragraph with
   the updated paragraph in Horseshoelike10_with_DJ.tex.
3. Update bibliography to '\bibliography{references,CauCau_DJ_references}'.
   Alternatively, upload the integrated manuscript as a new main .tex
   while retaining the existing project figures and references.bib.
