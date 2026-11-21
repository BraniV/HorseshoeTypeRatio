%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPARE_BLOCKS_CAUCAU_G_DE_17METHODS.m
%
% Blocks Monte Carlo comparison with 17 methods.
%
% DATA GENERATION
%   N = 1024, sigmaTrue = 1, target power SNR = 7.
%   Signal/filter: Donoho-Johnstone Blocks / Haar.
%   sigmaTrue is simulation truth and is used ONLY to generate noise.
%
% CAUCAU RULES (exactly seven)
%   1. Gaussian CauCau posterior mean
%   2. Gaussian CauCau posterior median
%   3. Gaussian CauCau SPoM
%   4. Gaussian CauCau PSP
%   5. Variance-matched DE CauCau posterior mean
%   6. Variance-matched DE CauCau posterior median
%   7. Variance-matched DE CauCau PSP
%
% COMPARISON METHODS (same ten as the previous Blocks benchmark)
%   8.  BAMS
%   9.  Johnstone-Silverman EBayes posterior median
%  10.  Raised-Cosine Bayes
%  11.  Epanechnikov Bayes
%  12.  Smooth SCAD-SURE
%  13.  NCPmean-2
%  14.  LNWS posterior mean
%  15.  SureShrink
%  16.  BayesShrink
%  17.  Sendur-Selesnick BiShrink
%
% ESTIMATION PRINCIPLE
%   One robust Gaussian noise estimate is computed in every replication:
%
%       sigmaHat = MAD(dFine)/Phi^{-1}(3/4).
%
%   Gaussian CauCau rules use sigmaHat.
%   The DE CauCau rules use variance matching:
%
%       muHat_VM = 1/sigmaHat^2,
%       sHat_DE  = sigmaHat,
%       aHat_DE  = sqrt(2)/sigmaHat.
%
%   PSP uses H0: |theta_j| <= one estimated noise SD and
%
%       thetaHat_j = d_j P(|theta_j| > noise SD | d_j).
%
% OUTPUT
%   Figs/Blocks_CauCau_17_MSE_Boxplot.pdf
%   Figs/Blocks_CauCau_17_MSE_Boxplot.png
%   AMSES/Blocks_CauCau_17_AMSE_SD_MSE.txt
%
% No Wavelet Toolbox or Statistics Toolbox is required.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;
clc;

%% ========================================================================
% OUTPUT DIRECTORIES
% =========================================================================
scriptFullPath = mfilename('fullpath');
if isempty(scriptFullPath)
    scriptDir = pwd;
else
    scriptDir = fileparts(scriptFullPath);
end

figDir  = fullfile(scriptDir,'Figs');
amseDir = fullfile(scriptDir,'AMSES');
if ~exist(figDir,'dir'),  mkdir(figDir);  end
if ~exist(amseDir,'dir'), mkdir(amseDir); end

signalTag = 'Blocks';

%% ========================================================================
% SIMULATION PARAMETERS
% =========================================================================
N         = 1024;
L         = 3;
sigmaTrue = 1.0;       % simulation truth: noise generation only
SNRtarget = 7;
M         = 100;      % set to 100 for a quick test run

J = round(log2(N));
if 2^J ~= N
    error('N must be a power of 2.');
end
if L >= J
    error('Need L < J.');
end

%% ========================================================================
% FIXED CAUCAU SCHEDULES
% =========================================================================
% Posterior mean (also the PSP prior-scale schedule).
C_mean      = 45;
alpha_mean  = 1.45;

% Posterior median.
C_med       = 35;
alpha_med   = 5/4;

% Gaussian SPoM.
C_spom_G    = 100;
alpha_spom  = 1.1;
gammaG      = sqrt(log(N))/2;

% PSP practical-null radius in standardized units: epsilon/noiseSD = 1.
epsilonPSP_std = 1.0;

% Numerical log-grid for CauCau posterior integration.
Ntgrid = 6001;
tgrid  = linspace(-24,12,Ntgrid).';

%% ========================================================================
% COMPARATOR SETTINGS -- unchanged from the Blocks benchmark
% =========================================================================
BAMS_Ctukey = 2;
BAMS_gam    = 2;

EB_a          = 0.5;
EB_estimate_a = false;
EB_a_grid     = logspace(log10(0.10),log10(3.0),31);

gammaRC = 2;

gammaEP = 2;
ellEP   = 1;
cEP     = 1;
tauEP   = 2;

Nquad = 64;
[xGL,wGL] = gauss_legendre_rule(Nquad);

aSCAD_grid    = [2.0 2.5 3.0 3.5];
nLambdaSCAD  = 121;
cSCAD_floor  = 2.0;

NCP_useKnownSigma = false;

LNWS_useKnownSigma = false;
LNWS_b_min   = 1e-6;
LNWS_b_max   = 10;
LNWS_b_ngrid = 61;

SS_window = 7;

rng('default');
rng(1,'twister');

%% ========================================================================
% BLOCKS SIGNAL
% =========================================================================
t = (1:N)/(N+1);
pos = [0.10 0.13 0.15 0.23 0.25 0.40 0.44 0.65 0.76 0.78 0.81];
ht  = [4.00 -5.00 3.00 -4.00 5.00 -4.20 2.10 4.30 -3.10 2.10 -4.20];

f0 = zeros(size(t));
for ib = 1:numel(pos)
    f0 = f0 + ht(ib)*(t >= pos(ib));
end

% Unit-noise power-SNR convention: mean(f.^2)=SNRtarget.
f = f0 * sqrt(SNRtarget/mean(f0.^2));

%% ========================================================================
% HAAR FILTERS
% =========================================================================
h = [1/sqrt(2)  1/sqrt(2)];
g = [1/sqrt(2) -1/sqrt(2)];

%% ========================================================================
% DIMENSIONLESS GAUSSIAN SPoM GEOMETRY
% =========================================================================
tauSPG_std   = NaN(1,J);
tBirthG_std  = NaN(1,J);
dcritG_std   = NaN(1,J);

for j = L:J-1
    tauSPG_std(j+1) = C_spom_G*2^(-alpha_spom*j);
    [tBirthG_std(j+1),dcritG_std(j+1)] = ...
        caucau_2mod_birth_standardized(tauSPG_std(j+1));
end

%% ========================================================================
% METHOD BOOKKEEPING -- 17 METHODS
% =========================================================================
methodNames = { ...
    'G_CauCauMean', ...
    'G_CauCauMedian', ...
    'G_CauCauSPoM', ...
    'G_CauCauPSP', ...
    'DEvm_CauCauMean', ...
    'DEvm_CauCauMedian', ...
    'DEvm_CauCauPSP', ...
    'BAMS', ...
    'EBayesMed', ...
    'RaisedCosineBayes', ...
    'EpanechnikovBayes', ...
    'SmoothSCAD', ...
    'NCPmean_2', ...
    'LNWSMean', ...
    'SureShrink', ...
    'BayesShrink', ...
    'SendurSelesnick'};

methodLabels = { ...
    '$G:\ {\rm Mean}$', ...
    '$G:\ {\rm Median}$', ...
    '$G:\ {\rm SPoM}$', ...
    '$G:\ {\rm PSP}$', ...
    '${\rm DE}_{VM}:\ {\rm Mean}$', ...
    '${\rm DE}_{VM}:\ {\rm Median}$', ...
    '${\rm DE}_{VM}:\ {\rm PSP}$', ...
    '$\mathrm{BAMS}$', ...
    '$\mathrm{EBayes}_{\mathrm{med}}$', ...
    '$\mathrm{RaisedCosine}$', ...
    '$\mathrm{Epanechnikov}$', ...
    '$\mathrm{SmoothSCAD}$', ...
    '$\mathrm{NCPmean}\! -\! 2$', ...
    '$\mathrm{LNWS}_{\mathrm{mean}}$', ...
    '$\mathrm{SureShrink}$', ...
    '$\mathrm{BayesShrink}$', ...
    '$\mathrm{BiShrink}_{\mathrm{SS}}$'};

K = numel(methodNames);
if K ~= 17
    error('Internal method-count error: expected 17 methods.');
end

MSE = NaN(M,K);
sigmaHatStore = NaN(M,1);
muVMStore     = NaN(M,1);

%% ========================================================================
% MONTE CARLO
% =========================================================================
for m = 1:M

    if mod(m,max(1,round(M/10))) == 0
        fprintf('Replication %d / %d\n',m,M);
    end

    % Same noisy realization for all 17 methods.
    y = f + sigmaTrue*randn(size(f));

    [aL,D] = my_periodic_dwt(y,h,g,L);

    % Common observed-data Gaussian noise scale.
    dFine = D{J}(:).';
    sigmaHat = robust_sigma_gaussian(dFine);
    sigmaHatStore(m) = sigmaHat;

    % Variance-matched DE nuisance parameter.
    muHatVM = 1/(sigmaHat^2);
    sHatDE  = 1/sqrt(muHatVM);       % algebraically sigmaHat
    aHatDE  = sqrt(2*muHatVM);       % algebraically sqrt(2)/sigmaHat
    muVMStore(m) = muHatVM;

    % ---------------------------------------------------------------------
    % 1. Gaussian CauCau posterior mean
    % ---------------------------------------------------------------------
    Dhat = D;
    for j = J-1:-1:L
        tau_j = C_mean*sigmaHat*2^(-alpha_mean*j);
        Dhat{j+1} = caucau_mean_vectorized(D{j+1},tau_j,sigmaHat,tgrid);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,1) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 2. Gaussian CauCau posterior median
    % ---------------------------------------------------------------------
    Dhat = D;
    for j = J-1:-1:L
        tau_j = C_med*sigmaHat*2^(-alpha_med*j);
        Dhat{j+1} = caucau_median_vectorized(D{j+1},tau_j,sigmaHat,tgrid);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,2) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 3. Gaussian CauCau SPoM
    % ---------------------------------------------------------------------
    Dhat = D;
    for j = J-1:-1:L
        tau_j   = sigmaHat*tauSPG_std(j+1);
        dcrit_j = sigmaHat*dcritG_std(j+1);
        Dhat{j+1} = caucau_2mod_vectorized( ...
            D{j+1},tau_j,sigmaHat,tBirthG_std(j+1),dcrit_j,gammaG);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,3) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 4. Gaussian CauCau PSP
    %    thetaHat = d P(|theta| > sigmaHat | d)
    % ---------------------------------------------------------------------
    Dhat = D;
    for j = J-1:-1:L
        tau_j = C_mean*sigmaHat*2^(-alpha_mean*j);
        epsilon_j = epsilonPSP_std*sigmaHat;
        Dhat{j+1} = caucau_psp_gaussian_vectorized( ...
            D{j+1},tau_j,sigmaHat,epsilon_j,tgrid);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,4) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 5. Variance-matched DE CauCau posterior mean
    % ---------------------------------------------------------------------
    Dhat = D;
    for j = J-1:-1:L
        tau_j = C_mean*sHatDE*2^(-alpha_mean*j);
        Dhat{j+1} = caucau_de_mean_vectorized(D{j+1},tau_j,aHatDE,tgrid);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,5) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 6. Variance-matched DE CauCau posterior median
    % ---------------------------------------------------------------------
    Dhat = D;
    for j = J-1:-1:L
        tau_j = C_med*sHatDE*2^(-alpha_med*j);
        Dhat{j+1} = caucau_de_median_vectorized(D{j+1},tau_j,aHatDE,tgrid);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,6) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 7. Variance-matched DE CauCau PSP
    %    thetaHat = d P(|theta| > sHatDE | d)
    % ---------------------------------------------------------------------
    Dhat = D;
    for j = J-1:-1:L
        tau_j = C_mean*sHatDE*2^(-alpha_mean*j);
        epsilon_j = epsilonPSP_std*sHatDE;
        Dhat{j+1} = caucau_psp_de_vectorized( ...
            D{j+1},tau_j,aHatDE,epsilon_j,tgrid);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,7) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 8. BAMS
    % ---------------------------------------------------------------------
    fhat = bams_embedded(y,h,J-L,BAMS_Ctukey,BAMS_gam);
    fhat = fhat(:).';
    MSE(m,8) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 9. Johnstone-Silverman EBayes posterior median
    % ---------------------------------------------------------------------
    Dhat = D;
    for j = J-1:-1:L
        Dhat{j+1} = ebayes_laplace_median_level( ...
            D{j+1},sigmaHat,EB_a,EB_estimate_a,EB_a_grid);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,9) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 10. Raised-Cosine Bayes
    % ---------------------------------------------------------------------
    Dhat = D;
    tauRC = 0;
    for j = J-1:-1:L
        tauRC = max(tauRC,max(abs(D{j+1})));
    end
    tauRC = max(tauRC,1e-8);
    for j = J-1:-1:L
        alphaRC = 1 - 1/(j-L+1)^gammaRC;
        alphaRC = min(max(alphaRC,0),1-1e-12);
        Dhat{j+1} = raised_cosine_bayes_level( ...
            D{j+1},sigmaHat,alphaRC,tauRC,xGL,wGL);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,10) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 11. Epanechnikov Bayes
    % ---------------------------------------------------------------------
    Dhat = D;
    sFine = max(std(D{J},0),1e-8);
    lambdaEP = 1/sFine^2 + (cEP/tauEP)*exp(-sFine/tauEP);
    for j = J-1:-1:L
        alphaEP = 1 - 1/(j-L+ellEP)^gammaEP;
        alphaEP = min(max(alphaEP,0),1-1e-12);
        betaEP = max(max(abs(D{j+1})),1e-8);
        Dhat{j+1} = epanechnikov_bayes_level( ...
            D{j+1},alphaEP,betaEP,lambdaEP,xGL,wGL);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,11) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 12. Smooth SCAD-SURE
    % ---------------------------------------------------------------------
    Dhat = D;
    for j = J-1:-1:L
        Dhat{j+1} = smooth_scad_sure_level( ...
            D{j+1},sigmaHat,aSCAD_grid,nLambdaSCAD,cSCAD_floor,N);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,12) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 13. NCPmean-2
    % ---------------------------------------------------------------------
    Dhat = ncpmean2_shrink(D,L,J,sigmaHat,NCP_useKnownSigma,xGL,wGL);
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,13) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 14. LNWS posterior mean
    % ---------------------------------------------------------------------
    [aLN,D_LN] = my_periodic_dwt(y,h,g,L-1);
    if LNWS_useKnownSigma
        sigmaLN = sigmaHat;
    else
        sigmaLN = robust_sigma_gaussian(D_LN{J});
    end
    DhatLN = lnws_posterior_mean( ...
        D_LN,L,J,sigmaLN,LNWS_b_min,LNWS_b_max,LNWS_b_ngrid);
    fhat = my_periodic_idwt(aLN,DhatLN,h,g,L-1,J);
    MSE(m,14) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 15. SureShrink
    % ---------------------------------------------------------------------
    Dhat = D;
    for j = J-1:-1:L
        Dhat{j+1} = sureshrink_level(D{j+1},sigmaHat);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,15) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 16. BayesShrink
    % ---------------------------------------------------------------------
    Dhat = D;
    for j = J-1:-1:L
        Dhat{j+1} = bayesshrink_level(D{j+1},sigmaHat);
    end
    fhat = my_periodic_idwt(aL,Dhat,h,g,L,J);
    MSE(m,16) = mean((fhat-f).^2);

    % ---------------------------------------------------------------------
    % 17. Sendur-Selesnick BiShrink
    % ---------------------------------------------------------------------
    [aSS,DSS] = my_periodic_dwt(y,h,g,L-1);
    DhatSS = sendur_selesnick_1d(DSS,L,J,sigmaHat,SS_window);
    fhat = my_periodic_idwt(aSS,DhatSS,h,g,L-1,J);
    MSE(m,17) = mean((fhat-f).^2);
end

%% ========================================================================
% NUMERICAL RESULTS
% =========================================================================
AMSE   = mean(MSE,1).';
SD_MSE = std(MSE,0,1).';

Result = table(methodNames(:),AMSE,SD_MSE, ...
    'VariableNames',{'Method','AMSE','SD_MSE'});

fprintf('\n==============================================================\n');
fprintf('BLOCKS: 17-METHOD CAUCAU BENCHMARK\n');
fprintf('N=%d, sigmaTrue=%g, SNR=%g, M=%d, L=%d\n', ...
    N,sigmaTrue,SNRtarget,M,L);
fprintf('All shrinkage rules estimate scale from data.\n');
fprintf('==============================================================\n');
disp(Result);

fprintf('MAD sigmaHat: mean = %.6f, SD = %.6f, median = %.6f\n', ...
    mean(sigmaHatStore),std(sigmaHatStore),median(sigmaHatStore));
fprintf('DE VM muHat: mean = %.6f, SD = %.6f, median = %.6f\n\n', ...
    mean(muVMStore),std(muVMStore),median(muVMStore));

%% ========================================================================
% PAIRED GAUSSIAN vs DE CAUCAU COMPARISONS: MEAN, MEDIAN, PSP
% =========================================================================
PairRule = {'Mean';'Median';'PSP'};
idxG  = [1 2 4];
idxDE = [5 6 7];
meanDiff = NaN(3,1);
mcseDiff = NaN(3,1);
for r = 1:3
    delta = MSE(:,idxDE(r))-MSE(:,idxG(r));
    meanDiff(r) = mean(delta);
    mcseDiff(r) = std(delta)/sqrt(M);
end
PairComparison = table(PairRule,AMSE(idxG),AMSE(idxDE),meanDiff,mcseDiff, ...
    'VariableNames',{'Rule','AMSE_G','AMSE_DE_VM','DEminusG','MCSE_DEminusG'});

disp(PairComparison);

%% ========================================================================
% SAVE NUMERICAL RESULTS
% =========================================================================
amseFile = fullfile(amseDir,[signalTag '_CauCau_17_AMSE_SD_MSE.txt']);
fid = fopen(amseFile,'w');
if fid == -1
    error('Could not open AMSE output file: %s',amseFile);
end

fprintf(fid,'Signal: %s\n',signalTag);
fprintf(fid,'N = %d\n',N);
fprintf(fid,'L = %d\n',L);
fprintf(fid,'SNRtarget = %.8g\n',SNRtarget);
fprintf(fid,'M = %d\n',M);
fprintf(fid,'sigmaTrue = %.8g (simulation generation only)\n',sigmaTrue);
fprintf(fid,'C_mean = %.12g, alpha_mean = %.12g\n',C_mean,alpha_mean);
fprintf(fid,'C_med = %.12g, alpha_med = %.12g\n',C_med,alpha_med);
fprintf(fid,'C_spom_G = %.12g, alpha_spom = %.12g, gamma_G = %.12g\n', ...
    C_spom_G,alpha_spom,gammaG);
fprintf(fid,'PSP practical-null standardized radius = %.12g\n\n',epsilonPSP_std);

fprintf(fid,'%-28s\t%16s\t%16s\n','Method','AMSE','SD_MSE');
for k = 1:K
    fprintf(fid,'%-28s\t%16.10g\t%16.10g\n', ...
        methodNames{k},AMSE(k),SD_MSE(k));
end

fprintf(fid,'\nPAIRED G vs DE (DEminusG < 0 favors DE)\n');
fprintf(fid,'%-12s\t%14s\t%14s\t%14s\t%14s\n', ...
    'Rule','AMSE_G','AMSE_DE_VM','DEminusG','MCSE');
for r = 1:3
    fprintf(fid,'%-12s\t%14.10g\t%14.10g\t%14.10g\t%14.10g\n', ...
        PairComparison.Rule{r},PairComparison.AMSE_G(r), ...
        PairComparison.AMSE_DE_VM(r),PairComparison.DEminusG(r), ...
        PairComparison.MCSE_DEminusG(r));
end
fclose(fid);

%% ========================================================================
% STANDARD TUKEY BOX-AND-WHISKER PLOT -- 17 BOXPLOTS
% =========================================================================
figH = figure('Color','w','Position',[60 80 1600 600]);
hold on;

boxHalfWidth  = 0.27;
capHalfWidth  = 0.13;
meanHalfWidth = 0.22;

for k = 1:K
    xs = sort(MSE(:,k));
    q1  = empirical_quantile_sorted(xs,0.25);
    med = empirical_quantile_sorted(xs,0.50);
    q3  = empirical_quantile_sorted(xs,0.75);
    iqrk = q3-q1;

    lowerFence = q1 - 1.5*iqrk;
    upperFence = q3 + 1.5*iqrk;
    inside = (xs >= lowerFence) & (xs <= upperFence);
    xInside = xs(inside);

    if isempty(xInside)
        whiskLow  = q1;
        whiskHigh = q3;
    else
        whiskLow  = xInside(1);
        whiskHigh = xInside(end);
    end

    outliers = xs(~inside);

    line([k k],[whiskLow q1],'Color','k','LineWidth',1.15);
    line([k k],[q3 whiskHigh],'Color','k','LineWidth',1.15);
    line([k-capHalfWidth k+capHalfWidth],[whiskLow whiskLow], ...
        'Color','k','LineWidth',1.15);
    line([k-capHalfWidth k+capHalfWidth],[whiskHigh whiskHigh], ...
        'Color','k','LineWidth',1.15);
    rectangle('Position',[k-boxHalfWidth,q1,2*boxHalfWidth,q3-q1], ...
        'EdgeColor','k','LineWidth',1.35);
    line([k-boxHalfWidth k+boxHalfWidth],[med med], ...
        'Color','k','LineWidth',1.15);
    line([k-meanHalfWidth k+meanHalfWidth],[AMSE(k) AMSE(k)], ...
        'Color','r','LineWidth',1.8);

    if ~isempty(outliers)
        plot(k*ones(size(outliers)),outliers,'ko', ...
            'MarkerSize',4,'LineWidth',0.8);
    end
end
hold off;

xlim([0.5 K+0.5]);
set(gca, ...
    'XTick',1:K, ...
    'XTickLabel',methodLabels, ...
    'XTickLabelRotation',34, ...
    'TickLabelInterpreter','latex', ...
    'FontSize',10.5, ...
    'Box','on');
grid off;

drawnow;
pdfFile = fullfile(figDir,[signalTag '_CauCau_17_MSE_Boxplot.pdf']);
pngFile = fullfile(figDir,[signalTag '_CauCau_17_MSE_Boxplot.png']);
exportgraphics(figH,pdfFile,'ContentType','vector');
exportgraphics(figH,pngFile,'Resolution',300);

fprintf('AMSE results saved to: %s\n',amseFile);
fprintf('Figure saved to: %s\n',pdfFile);
fprintf('Figure saved to: %s\n',pngFile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function sigmaHat = robust_sigma_gaussian(dFine)

x = dFine(:);
medx = median(x);
madx = median(abs(x-medx));

z075 = 0.6744897501960817;

sigmaHat = madx/z075;
sigmaHat = max(sigmaHat,1e-8);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STANDARDIZED Cau/Cau SPoM BIRTH GEOMETRY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function q = empirical_quantile_sorted(xs,p)
% Quantile of an already sorted vector using linear interpolation at
% r = 1 + (n-1)p.  No Statistics Toolbox required.

xs = xs(:);
n = numel(xs);

if n==0
    q = NaN;
    return;
end

if p<=0
    q = xs(1);
    return;
elseif p>=1
    q = xs(end);
    return;
end

r = 1 + (n-1)*p;
i1 = floor(r);
i2 = ceil(r);

if i1==i2
    q = xs(i1);
else
    w = r-i1;
    q = (1-w)*xs(i1) + w*xs(i2);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAMBDA-NEIGHBORHOOD WAVELET SHRINKAGE (LNWS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BAYESSHRINK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dhat,T] = bayesshrink_level(d,sigma)
% 1-D levelwise specialization of Chang-Yu-Vetterli BayesShrink.
%
% sigma_Y^2 = empirical second moment of noisy coefficients.
% sigma_X   = sqrt(max(sigma_Y^2-sigma^2,0)).
% T         = sigma^2/sigma_X.
% Apply soft thresholding.

sigmaY2 = mean(d(:).^2);

sigmaX = sqrt(max(sigmaY2-sigma^2,0));

if sigmaX <= sqrt(eps)*max(1,sigma)
    T = Inf;
else
    T = sigma^2/sigmaX;
end

dhat = soft_threshold(d,T);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SENDUR-SELESNICK BIVARIATE SHRINKAGE: 1-D ORTHOGONAL-DWT SPECIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SENDUR-SELESNICK BIVARIATE SHRINKAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Dhat,Tmean,Smean] = sendur_selesnick_1d( ...
                                      D,L,J,sigma,windowSize)
% Orthogonal-DWT 1-D specialization of Sendur-Selesnick BiShrink with
% local variance estimation.
%
% For child d and its parent p:
%
%       R = sqrt(d^2+p^2),
%       d_hat = d (R-T)_+ / R,
%       T = sqrt(3) sigma^2 / sigma_X.
%
% sigma_X is estimated locally from neighboring squared CHILD
% coefficients.  The original separable-DWT image implementation uses
% window size 7.  Here we use the corresponding 1-D periodic window.

Dhat = D;

Tmean = NaN(1,J);
Smean = NaN(1,J);

for j = J-1:-1:L

    child  = D{j+1}(:).';
    parent = D{j}(:).';

    if numel(child) ~= 2*numel(parent)
        error('Sendur-Selesnick parent/child size mismatch at level j=%d.',j);
    end

    % Each coarser parent is associated with its two dyadic children.
    parentExpanded = repelem(parent,2);

    % Local observed energy using periodic neighborhood.
    Wsig = periodic_local_mean(child.^2,windowSize);

    % Local signal standard deviation.
    Ssig = sqrt(max(Wsig-sigma^2,eps));

    T = sqrt(3)*sigma^2 ./ Ssig;

    R = sqrt(child.^2 + parentExpanded.^2);

    Rminus = max(R-T,0);

    fac = zeros(size(R));

    nz = (R > realmin);
    fac(nz) = Rminus(nz)./R(nz);

    out = child .* fac;

    Dhat{j+1} = reshape(out,size(D{j+1}));

    Tmean(j+1) = mean(T);
    Smean(j+1) = mean(Ssig);

end

% D{L} and the scaling coefficients at L-1 are untouched.  Together they
% reconstruct the same coarse V_L component used by the other methods.

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function z = periodic_local_mean(x,w)
% Centered periodic moving average with odd window length w.
% No toolbox required.

x = x(:).';

r = (w-1)/2;

z = zeros(size(x));

for k = -r:r
    z = z + circshift(x,[0 k]);
end

z = z/w;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PERIODIC ORTHONORMAL DWT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = soft_threshold(x,T)

out = sign(x).*max(abs(x)-T,0);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p = Phi0(x)
% Standard normal CDF without Statistics Toolbox.

p = 0.5*erfc(-x/sqrt(2));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = PhiInv0(p)
% Standard normal inverse CDF without Statistics Toolbox.

p = min(max(p,1e-15),1-1e-15);

x = -sqrt(2)*erfcinv(2*p);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = phi0(x)
% Standard normal density.

y = exp(-0.5*x.^2)/sqrt(2*pi);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EBayes-MED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dhat,wHat,aHat] = ebayes_laplace_median_level( ...
              d,sigma,aDefault,estimateA,aGrid)
% Johnstone-Silverman empirical Bayes posterior median.
%
% Standardize z=d/sigma and use the Laplace slab
%
%   gamma(theta;a) = (a/2) exp(-a |theta|).
%
% w is the prior probability of a NONZERO coefficient.
%
% If estimateA=false, a=aDefault and w is MML.
% If estimateA=true, a is chosen by profile MML over aGrid, with w MML
% at every candidate.  The latter is an intentionally transparent
% numerical profile-MML implementation.

z = d(:).'/sigma;

if ~estimateA

    aHat = aDefault;
    [wHat,~] = ebayes_fit_w(z,aHat);

else

    bestLL = -Inf;
    wHat   = 0.5;
    aHat   = aDefault;

    for ia = 1:length(aGrid)

        aa = aGrid(ia);

        [ww,ll] = ebayes_fit_w(z,aa);

        if ll > bestLL
            bestLL = ll;
            wHat   = ww;
            aHat   = aa;
        end

    end

end

muhat = ebayes_postmed_laplace(z,wHat,aHat);

dhat = sigma*muhat;
dhat = reshape(dhat,size(d));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [wHat,llHat] = ebayes_fit_w(z,a)
% Profile MML estimate of w for fixed Laplace rate a.
% Constraint: posterior-median threshold <= sqrt(2 log n).

z = z(:).';
n = length(z);

x = abs(z);

[ph,g,~,~] = laplace_marginal_parts(x,a);

% Lower admissible w determined by universal threshold
tU = sqrt(2*log(max(n,2)));

[phT,~,ApT,AmT] = laplace_marginal_parts(tU,a);

denT = phT + (ApT-AmT);

if denT <= 0 || ~isfinite(denT)
    wMin = 1e-8;
else
    wMin = phT/denT;
end

wMin = min(max(wMin,1e-8),1-1e-8);

% Score S(w) = sum (g-phi)/((1-w)phi+w g), decreasing in w.
score = @(w) sum((g-ph)./max((1-w)*ph+w*g,realmin));

sLo = score(wMin);
sHi = score(1-1e-10);

if sLo <= 0

    wHat = wMin;

elseif sHi >= 0

    wHat = 1-1e-10;

else

    lo = wMin;
    hi = 1-1e-10;

    for it = 1:60

        mid = 0.5*(lo+hi);
        sm  = score(mid);

        if sm > 0
            lo = mid;
        else
            hi = mid;
        end

    end

    wHat = 0.5*(lo+hi);

end

marg = (1-wHat)*ph + wHat*g;

llHat = sum(log(max(marg,realmin)));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function mu = ebayes_postmed_laplace(z,w,a)
% Explicit posterior median under spike + Laplace slab.

sz = size(z);

z = z(:).';

sgn = sign(z);
x   = abs(z);

[ph,g,Ap,~] = laplace_marginal_parts(x,a);

marg = (1-w)*ph + w*g;
wpost = w*g./max(marg,realmin);

% Conditional slab posterior upper-tail probability at zero
Ftail0 = Ap./max(g,realmin);

isZero = (wpost.*Ftail0 <= 0.5);

mabs = zeros(size(x));

idx = find(~isZero & x>0);

if ~isempty(idx)

    xx = x(idx);
    wp = wpost(idx);

    % D = exp(-a x) Phi(x-a) + exp(a x) Phi(-x-a)
    D = exp(-a*xx).*Phi0(xx-a) + ...
        exp( a*xx).*Phi0(-xx-a);

    % From Ftilde_1(m|x) = 1/(2 wpost):
    %
    % exp(-a x) Phi(x-a-m)/D = 1/(2 wpost).
    u = D.*exp(a*xx)./(2*wp);

    u = min(max(u,1e-14),1-1e-14);

    mm = xx - a - PhiInv0(u);

    mabs(idx) = max(mm,0);

end

mu = sgn.*mabs;
mu = reshape(mu,sz);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ph,g,Ap,Am] = laplace_marginal_parts(x,a)
% For x>=0:
%
% Ap = integral_{theta>0} Laplace(theta;a) phi(x-theta) dtheta
% Am = integral_{theta<0} Laplace(theta;a) phi(x-theta) dtheta
% g  = Ap+Am.

x = abs(x);

ph = phi0(x);

Ap = 0.5*a*exp(0.5*a^2-a*x).*Phi0(x-a);
Am = 0.5*a*exp(0.5*a^2+a*x).*Phi0(-x-a);

% Numerical guard against Inf*0 in extreme tails
bad = ~isfinite(Ap);
if any(bad)
    Ap(bad) = 0;
end

bad = ~isfinite(Am);
if any(bad)
    Am(bad) = 0;
end

g = Ap+Am;
g = max(g,realmin);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RAISED-COSINE BAYES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dhat = raised_cosine_bayes_level( ...
                   d,sigma,alpha,tau,xGL,wGL)
% Posterior mean under
%
% pi(theta)=alpha delta_0 +(1-alpha) g_RC(theta;tau),
%
% g_RC(theta;tau) =
%   [1 + cos(pi theta/tau)]/(2 tau), |theta|<tau.
%
% Bounded-support integrals are evaluated by vectorized Gauss-Legendre
% quadrature; this is mathematically the same posterior mean as the paper's
% analytic-plus-integral representation.

sz = size(d);
d  = d(:).';

if tau <= 1e-12
    dhat = zeros(sz);
    return;
end

theta = tau*xGL(:);
qw    = tau*wGL(:);

gprior = (1 + cos(pi*theta/tau))/(2*tau);

% nq x nd matrix
E = exp(-0.5*((theta-d)/sigma).^2);

slabMarg = sum((qw.*gprior).*E,1);
slabNum  = sum((qw.*gprior.*theta).*E,1);

zeroLik = exp(-0.5*(d/sigma).^2);

den = alpha*zeroLik + (1-alpha)*slabMarg;

out = (1-alpha)*slabNum./max(den,realmin);

dhat = reshape(out,sz);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EPANECHNIKOV BAYES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dhat = epanechnikov_bayes_level( ...
                   d,alpha,beta,lambda,xGL,wGL)
% Posterior mean under spike + Epanechnikov slab.
%
% The Epanechnikov paper places Exp(lambda) on sigma^2 and integrates it
% out. The resulting likelihood in d-theta is double exponential with rate
%
%        r = sqrt(2 lambda).
%
% Constants common to atom and slab cancel in the posterior mean.

sz = size(d);
d  = d(:).';

if beta <= 1e-12
    dhat = zeros(sz);
    return;
end

theta = beta*xGL(:);
qw    = beta*wGL(:);

gprior = (3/(4*beta))*(1-(theta/beta).^2);

rate = sqrt(2*lambda);

E = exp(-rate*abs(theta-d));

slabMarg = sum((qw.*gprior).*E,1);
slabNum  = sum((qw.*gprior.*theta).*E,1);

zeroLik = exp(-rate*abs(d));

den = alpha*zeroLik + (1-alpha)*slabMarg;

out = (1-alpha)*slabNum./max(den,realmin);

dhat = reshape(out,sz);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SMOOTH SCAD-SURE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dhat,lambdaHat,aHat,lambdaSURE] = ...
    smooth_scad_sure_level(d,sigma,aGrid,nLambda,cFloor,Nglobal)
% Adaptive/stabilized levelwise SURE for the smooth raised-cosine SCAD rule.
%
% Jointly select a and the raw SURE threshold over
%
%       a in aGrid,   0 <= lambda <= lambda_U,
%
% where
%
%       lambda_U = sigma*sqrt(2*log(Nglobal)).
%
% The selected threshold is stabilized as
%
%       lambdaHat = min(lambda_U,max(lambdaSURE,cFloor*sigma)).
%
% The benchmark settings use
%
%       aGrid  = [2 2.5 3 3.5],   cFloor = 2.

sz = size(d);
d  = d(:).';

n = length(d);

if n == 0
    dhat = reshape(d,sz);
    lambdaHat = 0;
    aHat = aGrid(1);
    lambdaSURE = 0;
    return;
end

if nargin < 6 || isempty(Nglobal)
    Nglobal = n;
end

if nargin < 5 || isempty(cFloor)
    cFloor = 2;
end

aGrid = aGrid(:).';

if any(aGrid <= 1)
    error('Smooth SCAD requires every a candidate to exceed 1.');
end

lambdaU = sigma*sqrt(2*log(max(Nglobal,2)));

lamGrid = [0,linspace(lambdaU/nLambda,lambdaU,nLambda)];

r = abs(d);

bestSURE   = Inf;
aHat       = aGrid(1);
lambdaSURE = 0;

for ia = 1:numel(aGrid)

    a = aGrid(ia);

    % lambda=0 => identity estimator; SURE = n sigma^2
    if n*sigma^2 < bestSURE
        bestSURE   = n*sigma^2;
        aHat       = a;
        lambdaSURE = 0;
    end

    for il = 2:length(lamGrid)

        lam = lamGrid(il);

        low = (r <= lam);
        mid = (r > lam & r < a*lam);
        hi  = (r >= a*lam);

        F = zeros(size(d));

        % rho=0
        F(low) = d(low).^2 - sigma^2;

        % rho=d-h
        if any(mid)

            s = (r(mid)-lam)/((a-1)*lam);

            hmag = lam*cos(0.5*pi*s).^2;

            % derivative dh/dd is even and equals:
            hp = -pi/(2*(a-1))*sin(pi*s);

            F(mid) = hmag.^2 + sigma^2 - 2*sigma^2*hp;

        end

        % rho=d
        F(hi) = sigma^2;

        sureVal = sum(F);

        if sureVal < bestSURE
            bestSURE   = sureVal;
            aHat       = a;
            lambdaSURE = lam;
        end

    end
end

lambdaHat = min(lambdaU,max(lambdaSURE,cFloor*sigma));

dhat = smooth_scad_apply(d,lambdaHat,aHat);
dhat = reshape(dhat,sz);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = smooth_scad_apply(d,lambda,a)

if lambda <= 0
    out = d;
    return;
end

r = abs(d);

out = zeros(size(d));

mid = (r > lambda & r < a*lambda);
hi  = (r >= a*lambda);

if any(mid)

    s = (r(mid)-lambda)/((a-1)*lambda);

    hmag = lambda*cos(0.5*pi*s).^2;

    out(mid) = d(mid) - sign(d(mid)).*hmag;

end

out(hi) = d(hi);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NCPmean-2: WANG-WOOD EMPIRICAL-BAYES BLOCK SHRINKAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Dhat,pHat,lambdaHat,sigmaHat] = ncpmean2_shrink( ...
                     D,L,J,sigmaKnown,useKnownSigma,xGL,wGL)
% Wang-Wood NCPmean with block size m=2.
%
% Reference construction:
%   z_B = ||d_B||^2
% is modeled through the noncentral chi-square representation.  The prior
% for the true block energy has a point mass at zero with probability p
% plus the Wang-Wood power prior.  The posterior mean Pmean(z) is used.
%
% Hyperparameters:
%   * sigma^2 is estimated from finest-level block energies unless
%     useKnownSigma=true;
%   * one global power-prior shape lambda is obtained by the
%     Wang-Wood quick moment/median construction;
%   * p_j is estimated separately at each level by marginal MLE.
%
% The estimated block vector is
%
%       d_B_hat = sqrt(Pmean(z_B)/z_B) d_B.
%
% No cap at one is imposed: the published posterior-mean rule is not a
% strict shrinkage rule, although it has bounded-shrinkage behavior.

m = 2;

% Gauss-Legendre rule on (0,1) for the C_eta integrals.
u  = 0.5*(xGL(:) + 1);
wu = 0.5*wGL(:);

% -------------------------------------------------------------------------
% Block energies at every processed level.
% -------------------------------------------------------------------------
Z = cell(1,J);
allz = [];

for j = L:J-1

    d = D{j+1}(:).';

    if mod(numel(d),m) ~= 0
        error('NCPmean-2 requires an even number of coefficients at each level.');
    end

    db = reshape(d,m,[]);
    z  = sum(db.^2,1);

    Z{j+1} = z;
    allz = [allz,z]; %#ok<AGROW>

end

% -------------------------------------------------------------------------
% Noise variance.  For m=2, median(chi-square_2) = 2 log 2 exactly.
% -------------------------------------------------------------------------
if useKnownSigma

    sigma2Hat = sigmaKnown^2;

else

    dFine = D{J}(:).';
    zFine = sum(reshape(dFine,m,[]).^2,1);

    chi2med_m2 = 2*log(2);
    sigma2Hat = median(zFine)/chi2med_m2;

    sigma2Hat = max(sigma2Hat,1e-12);

end

sigmaHat = sqrt(sigma2Hat);

% -------------------------------------------------------------------------
% Wang-Wood quick-and-dirty global power-prior shape.
%
% Preliminary zero-mass estimate:
%
%   p0 = #{z_i < 2 m sigma^2 log(Nblocks)} / Nblocks.
%
% Moment matching gives the central beta:
%
%   sum z_i = sigma^2 sum m_i + (1-p0) sum m_i / betaBar.
%
% Transform uBar = betaBar*sigma^2/(1+betaBar*sigma^2).  Under the
% power prior U~Beta(lambda+1,1), so choosing betaBar as the prior median
% gives
%
%   lambda = -log(2)/log(uBar) - 1.
% -------------------------------------------------------------------------
Nblocks = numel(allz);

if Nblocks == 0
    error('NCPmean-2 found no detail blocks.');
end

threshold0 = 2*m*sigma2Hat*log(max(Nblocks,2));
p0 = mean(allz < threshold0);

sumM = m*Nblocks;
excess = sum(allz) - sigma2Hat*sumM;

if excess <= 1e-12*sigma2Hat*sumM
    % Essentially no positive empirical excess energy.  The limiting
    % moment estimate betaBar is very large; use a finite numerical proxy.
    betaBar = 1e8/max(sigma2Hat,1e-12);
else
    betaBar = (1-p0)*sumM/excess;
    betaBar = max(betaBar,1e-12/max(sigma2Hat,1e-12));
end

uBar = betaBar*sigma2Hat/(1 + betaBar*sigma2Hat);
uBar = min(max(uBar,1e-10),1-1e-10);

lambdaHat = -log(2)/log(uBar) - 1;
lambdaHat = max(lambdaHat,-1+1e-8);

% -------------------------------------------------------------------------
% Levelwise marginal-MLE p_j and posterior-mean block shrinkage.
% -------------------------------------------------------------------------
Dhat = D;
pHat = NaN(1,J);

for j = L:J-1

    d = D{j+1}(:).';
    db = reshape(d,m,[]);
    z  = Z{j+1};

    [logR,A1,A2] = ncp_power_quantities( ...
                      z,sigma2Hat,lambdaHat,m,u,wu);

    pJ = ncp_fit_zero_mass_mle(logR);
    pHat(j+1) = pJ;

    % Posterior probability that the true block energy is zero.
    logitP = log(pJ) - log(1-pJ);
    pi0 = logistic_stable(logitP - logR);

    % Wang-Wood posterior mean of true block energy, Eq. (3.43):
    %
    % (1-pi0) [m*sigma^2 + z
    %          -(m*sigma^2+2z) A_{eta,1} + z A_{eta,2}].
    Pmean = (1-pi0) .* ...
            (m*sigma2Hat + z ...
             - (m*sigma2Hat + 2*z).*A1 ...
             + z.*A2);

    % Numerical guard only; theoretical posterior energy is nonnegative.
    Pmean = max(Pmean,0);

    fac = zeros(size(z));
    nz = (z > 1e-14);
    fac(nz) = sqrt(Pmean(nz)./z(nz));

    dbhat = db .* fac;

    Dhat{j+1} = reshape(dbhat,1,[]);
    Dhat{j+1} = reshape(Dhat{j+1},size(D{j+1}));

end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [logR,A1,A2] = ncp_power_quantities(z,sigma2,lambda,m,u,wu)
% Quantities for the Wang-Wood power prior.
%
% Let r=z/(2 sigma^2), eta=m/2+lambda, and
%
%   C_eta(r) = integral_0^1 u^eta exp(-r u) du.
%
% Then
%
%   R(z) = f_1(z)/f_0(z)
%        = (lambda+1) exp(r) C_eta(r),
%
% and
%
%   A_{eta,1}=C_{eta+1}/C_eta,   A_{eta,2}=C_{eta+2}/C_eta.
%
% We evaluate these by stable vectorized Gauss-Legendre quadrature.

z = z(:).';
r = z/(2*sigma2);
eta = m/2 + lambda;

if eta <= -1
    error('NCP power-prior integral requires eta > -1.');
end

logu = log(u);
logwu = log(wu);

% nq x nb matrix
logBase = logwu + eta*logu - u*r;

mx = max(logBase,[],1);
W = exp(logBase - mx);

den = sum(W,1);
den = max(den,realmin);

A1 = sum(W.*u,1)./den;
A2 = sum(W.*(u.^2),1)./den;

logC = mx + log(den);

logR = log(lambda+1) + r + logC;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pHat = ncp_fit_zero_mass_mle(logR)
% Marginal MLE of p, the prior probability of zero block energy.
%
% Apart from the central-chi-square factor, the level likelihood is
%
%   prod_i [ p + (1-p) R_i ].
%
% The likelihood is concave in p.  fminbnd is used on the log scale
% expression, and the two boundary limits are checked explicitly.

logR = logR(:).';

obj = @(p) ncp_neg_profile_loglik(p,logR);

opts = optimset('Display','off','TolX',1e-9);
pInt = fminbnd(obj,1e-9,1-1e-9,opts);

cand = [1e-10,pInt,1-1e-10];
val  = [obj(cand(1)),obj(cand(2)),obj(cand(3))];

[~,ii] = min(val);
pHat = cand(ii);

pHat = min(max(pHat,1e-10),1-1e-10);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nll = ncp_neg_profile_loglik(p,logR)

a = log(p);
b = log(1-p) + logR;

mx = max(a,b);
ll = sum(mx + log(exp(a-mx) + exp(b-mx)));

nll = -ll;

if ~isfinite(nll)
    nll = realmax/1e100;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = logistic_stable(x)

y = zeros(size(x));

ip = (x >= 0);
im = ~ip;

y(ip) = 1./(1+exp(-x(ip)));

ex = exp(x(im));
y(im) = ex./(1+ex);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAMBDA-NEIGHBORHOOD WAVELET SHRINKAGE (LNWS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Dhat,epsHat,bHat] = lnws_posterior_mean( ...
              D,L,J,sigma,bMin,bMax,nBgrid)
% Remenyi-Vidakovic Lambda-neighborhood empirical-Bayes shrinkage.
%
% The likelihood is exactly the marginal likelihood in Eq. (15) of the
% LNWS paper.  Instead of requiring Optimization Toolbox/fmincon, we profile
% out epsilon_j at each candidate b.  This maximizes the same likelihood:
%
%   for fixed b, each epsilon_j occurs only in level j and is optimized
%   independently; b is then optimized globally over the profile likelihood.
%
% Posterior-mean energy (Eq. 9) is used, followed by the coefficient
% shrinkage rule in Eq. (11).

s2 = sigma^2;

% -------------------------------------------------------------------------
% Form all Lambda-neighborhood energy statistics x_{j,l}.
% Child pairs are nonoverlapping: (1,2), (3,4), ...
% Their common parents are positions 1,2,...
% -------------------------------------------------------------------------

X = cell(1,J);

allx = [];

for j = L:J-1

    child  = D{j+1}(:).';
    parent = D{j}(:).';      % detail level j-1

    nchild = length(child);
    npair  = nchild/2;

    c1 = child(1:2:end);
    c2 = child(2:2:end);

    if length(parent) ~= npair
        error('LNWS parent/child indexing mismatch at level j=%d.',j);
    end

    x = (c1.^2 + c2.^2 + parent.^2)/s2;

    x = max(x,1e-14);

    X{j+1} = x;

    allx = [allx,x]; %#ok<AGROW>

end

% -------------------------------------------------------------------------
% Global b: paper suggests starting b0 = 1/mean(x).
% We use a broad log-grid and then refine around the best profile point.
% This avoids requiring Optimization Toolbox and is substantially more
% stable than a single unconstrained multidimensional optimization.
% -------------------------------------------------------------------------

b0 = 1/max(mean(allx),1e-12);

bLo = min(bMin,b0/100);
bHi = max(bMax,b0*100);

bLo = max(bLo,1e-10);
bHi = min(bHi,1e4);

lg = linspace(log(bLo),log(bHi),nBgrid);

prof = zeros(size(lg));

for ib = 1:length(lg)
    prof(ib) = lnws_profile_nll(exp(lg(ib)),X,L,J);
end

[~,imin] = min(prof);

ilo = max(1,imin-1);
ihi = min(length(lg),imin+1);

if ilo == ihi

    logbHat = lg(imin);

else

    obj = @(z) lnws_profile_nll(exp(z),X,L,J);

    opts = optimset('Display','off','TolX',1e-7);

    logbHat = fminbnd(obj,lg(ilo),lg(ihi),opts);

end

bHat = exp(logbHat);

% Final levelwise epsilon estimates at bHat
epsHat = NaN(1,J);

for j = L:J-1

    x = X{j+1};

    epsHat(j+1) = lnws_fit_epsilon(x,bHat);

end

% -------------------------------------------------------------------------
% Apply posterior-mean energy shrinkage.
%
% Only child coefficients are shrunk at level j.  Their parent is used to
% construct x but is left unchanged until its own level is processed.
% Detail level L-1 is never shrunk.
% -------------------------------------------------------------------------

Dhat = D;

for j = L:J-1

    child = D{j+1}(:).';

    x   = X{j+1};
    eps = epsHat(j+1);

    delta = lnws_delta_mean(x,eps,bHat);

    fac = sqrt(max(delta,0)./max(x,1e-14));

    % The published rule is exactly sqrt(delta/x); no artificial cap at 1.
    out = child;

    out(1:2:end) = fac .* child(1:2:end);
    out(2:2:end) = fac .* child(2:2:end);

    Dhat{j+1} = reshape(out,size(D{j+1}));

end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nll = lnws_profile_nll(b,X,L,J)
% Profile negative marginal log-likelihood for global rate b.

if ~(isfinite(b) && b>0)

    nll = realmax/1e100;
    return;

end

nll = 0;

for j = L:J-1

    x = X{j+1};

    epsj = lnws_fit_epsilon(x,b);

    nll = nll + lnws_level_nll(epsj,b,x);

end

if ~isfinite(nll)
    nll = realmax/1e100;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function epsHat = lnws_fit_epsilon(x,b)
% Marginal-MLE epsilon_j for fixed b.
%
% epsilon is the PRIOR MASS AT ZERO, matching the LNWS notation.

obj = @(e) lnws_level_nll(e,b,x);

opts = optimset('Display','off','TolX',1e-8);

% Permit solutions arbitrarily close to either boundary without evaluating
% exactly at 0 or 1.
epsHat = fminbnd(obj,1e-8,1-1e-8,opts);

% Explicitly check the two limiting boundary cases.
vals = [obj(1e-10),obj(epsHat),obj(1-1e-10)];
[~,ii] = min(vals);

if ii==1
    epsHat = 1e-10;
elseif ii==3
    epsHat = 1-1e-10;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nll = lnws_level_nll(epsj,b,x)
% Negative log of
%
% m(x) = eps_j m0(x) + (1-eps_j) m1(x)
%
% with
%
% m0 = central chi-square_3 density,
%
% m1 = marginal under lambda ~ Exp(rate=b).

[logm0,logm1] = lnws_log_marginals(x,b);

A = log(max(epsj,realmin))       + logm0;
B = log(max(1-epsj,realmin))     + logm1;

mx = max(A,B);

logm = mx + log(exp(A-mx)+exp(B-mx));

nll = -sum(logm);

if ~isfinite(nll)
    nll = realmax/1e100;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [logm0,logm1] = lnws_log_marginals(x,b)
% Stable log marginals from Eq. (8).
%
% m0(x) = sqrt[x/(2*pi)] exp(-x/2)
%
% m1(x) = b/sqrt(1+2b)
%         exp[-b*x/(1+2b)]
%         erf( sqrt[x/(2+4b)] ).

x = max(x,1e-300);

q = 1+2*b;

logm0 = 0.5*log(x) - 0.5*log(2*pi) - 0.5*x;

z = sqrt(x/(2*q));

E = erf(z);
E = max(E,realmin);

logm1 = log(b) - 0.5*log(q) - (b/q)*x + log(E);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function delta = lnws_delta_mean(x,epsj,b)
% Posterior mean of noncentrality lambda, Eq. (9).
%
% delta(x) = [1-p_j(x)] delta_E(x),
%
% where p_j is posterior probability of lambda=0 and
%
% delta_E(x) =
%
% [1+2b+x
%   + sqrt{2x(1+2b)/pi}
%     exp{-x/(2+4b)} / erf{sqrt[x/(2+4b)]}]
% / (1+2b)^2.

x = max(x,1e-14);

q = 1+2*b;

[logm0,logm1] = lnws_log_marginals(x,b);

A = log(max(epsj,realmin))   + logm0;
B = log(max(1-epsj,realmin)) + logm1;

mx = max(A,B);

p0 = exp(A-mx)./(exp(A-mx)+exp(B-mx));

z = sqrt(x/(2*q));

E = erf(z);
E = max(E,realmin);

ratioTerm = sqrt(2*x*q/pi).*exp(-x/(2*q))./E;

deltaE = (q + x + ratioTerm)/(q^2);

delta = (1-p0).*deltaE;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SURESHRINK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dhat,lambdaHat] = sureshrink_level(d,sigma)
% Levelwise SURE soft threshold with levelwise universal cap.

sz = size(d);

d = d(:).';

n = length(d);

if n == 0

    dhat = reshape(d,sz);
    lambdaHat = 0;
    return;

end

z  = abs(d)/sigma;
zs = sort(z,'ascend');

z2 = zs.^2;
cs = cumsum(z2);

k = (1:n)';

sure = n - 2*k + cs(:) + (n-k).*(z2(:));

[~,imin] = min(sure);

tSure = zs(imin);
tUniv = sqrt(2*log(max(n,2)));

t = min(tSure,tUniv);

lambdaHat = sigma*t;

dhat = soft_threshold(d,lambdaHat);
dhat = reshape(dhat,sz);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GAUSS-LEGENDRE RULE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x,w] = gauss_legendre_rule(n)
% n-point Gauss-Legendre rule on [-1,1], computed by Golub-Welsch.

k = 1:(n-1);

beta = k./sqrt(4*k.^2-1);

T = diag(beta,1) + diag(beta,-1);

[V,D] = eig(T);

x = diag(D);

[x,idx] = sort(x);

V = V(:,idx);

w = 2*(V(1,:).^2).';

x = x(:);
w = w(:);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PERIODIC ORTHONORMAL DWT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [aL,D] = my_periodic_dwt(x,h,g,L)

x = x(:).';

N = length(x);

J = round(log2(N));

D = cell(1,J);

a = x;

for j = J-1:-1:L

    [a,d] = analysis_step_periodic(a,h,g);

    D{j+1} = d;

end

aL = a;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = my_periodic_idwt(aL,D,h,g,L,J)

a = aL(:).';

for j = L:J-1

    d = D{j+1};

    a = synthesis_step_periodic(a,d,h,g);

end

x = a;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a,d] = analysis_step_periodic(x,h,g)

x = x(:).';

n = length(x);

if mod(n,2) ~= 0
    error('Signal length must be even.');
end

nh = length(h);

n2 = n/2;

a = zeros(1,n2);
d = zeros(1,n2);

offset = 0:(nh-1);

for k = 0:(n2-1)

    ind = mod(2*k+offset,n)+1;

    xx = x(ind);

    a(k+1) = sum(h.*xx);
    d(k+1) = sum(g.*xx);

end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = synthesis_step_periodic(a,d,h,g)

a = a(:).';
d = d(:).';

n2 = length(a);

n = 2*n2;

nh = length(h);

x = zeros(1,n);

offset = 0:(nh-1);

for k = 0:(n2-1)

    ind = mod(2*k+offset,n)+1;

    x(ind) = x(ind) ...
           + a(k+1)*h ...
           + d(k+1)*g;

end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cau/Cau SECOND POSTERIOR MODE (2mod / SPoM)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f = bams_embedded(sig,filt,L,Ctukey,gam)
% Embedded stand-alone BAMS implementation.
%
% Inputs:
%   sig     noisy signal
%   filt    orthogonal low-pass wavelet filter
%   L       number of decomposition levels
%   Ctukey  default 2
%   gam     default 2
%
% This reproduces the supplied BAMS rule.  The wavelet matrix is cached
% after its first construction because filt, N and L are fixed throughout
% a Monte Carlo run; caching changes runtime only, not the numerical rule.

n = length(sig);
J = log2(n);

if nargin == 4
    gam = 2;
end

if nargin == 3
    Ctukey = 2;
    gam = 2;
end

if nargin == 2
    Ctukey = 2;
    gam = 2;
    L = J-3;
end

if nargin == 1
    Ctukey = 2;
    gam = 2;
    L = J-3;
    filt = [-0.07576571478934  -0.02963552764595 ...
             0.49761866763246   0.80373875180522 ...
             0.29785779560554  -0.09921954357694 ...
            -0.01260396726226   0.03222310060407];
end

persistent Wcache ncache Lcache filtcache

needW = isempty(Wcache) || isempty(ncache) || isempty(Lcache) || ...
        isempty(filtcache) || ncache ~= n || Lcache ~= L || ...
        ~isequal(filtcache(:).',filt(:).');

if needW
    Wcache    = bams_Wavmat(filt,n,L,2);
    ncache    = n;
    Lcache    = L;
    filtcache = filt(:).';
end

W = Wcache;

sw = W*sig(:);

coarsest = J-L;
swt = sw;

% ---------- setting the parameter mu ----------
indx = (2^(J-1)+1):2^J;
niz  = sw(indx);
nizo = sort(niz);
mm   = length(nizo);

low  = floor(mm/4);
high = floor(3*mm/4);

pseudos = abs(nizo(high)-nizo(low))/Ctukey;
mu = 1/pseudos^2;

% ---------- setting tau and epsilon levelwise ----------
finest_lev = J-1;
tauu = sqrt(max(std(sw).^2 - 1/mu,0.0001));

for i = finest_lev:-1:coarsest

    eps_i = 1 - 1/(i-coarsest+1)^gam;
    tau_i = tauu*i^(1/2);

    ind = (2^i+1):2^(i+1);
    a = sw(ind);
    b = zeros(size(a));

    for jj = 1:length(a)
        [~,b(jj)] = bams_bayesrule(a(jj),mu,tau_i,eps_i);
    end

    swt(ind) = b;

end

f = W'*swt;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [me,bre] = bams_bayesrule(d,mu,tau,epsi)
% BAMS marginal distribution and posterior-mean Bayes rule.

de = 0.5*sqrt(2*mu)*exp(-abs(d)*sqrt(2*mu));

m = (tau*exp(-abs(d)/tau) - 1/sqrt(2*mu)* ...
     exp(-sqrt(2*mu)*abs(d))) / (2*tau^2 - 1/mu);

me = epsi*de + (1-epsi)*m;

if d >= 0

    br = (tau*(tau^2 - 1/(2*mu))*d*exp(-d/tau) + ...
          tau^2/mu*(exp(-sqrt(2*mu)*d)-exp(-d/tau))) / ...
         ((tau^2 - 1/(2*mu))* ...
          (tau*exp(-d/tau)-1/sqrt(2*mu)*exp(-sqrt(2*mu)*d)));

else

    br = (tau*(tau^2 - 1/(2*mu))*d*exp(d/tau) - ...
          tau^2/mu*(exp(sqrt(2*mu)*d)-exp(d/tau))) / ...
         ((tau^2 - 1/(2*mu))* ...
          (tau*exp(d/tau)-1/sqrt(2*mu)*exp(sqrt(2*mu)*d)));

end

bre = (1-epsi)*m*br / ((1-epsi)*m + epsi*de);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function W = bams_Wavmat(h,N,k0,shift)
% Matrix representation of the orthogonal discrete wavelet transform used
% in the supplied BAMS implementation.

if nargin == 3
    shift = 2;
end

J = log2(N);

if nargin == 2
    shift = 2;
    k0 = J;
end

h = h(:).';
g = fliplr(conj(h).* (-1).^(1:length(h)));

if J ~= floor(J)
    error('For standard DWT, N has to be a power of 2.')
end

h = [h,zeros(1,N)];
g = [g,zeros(1,N)];

oldmat = eye(2^(J-k0));

for k = k0:-1:1

    ubJk  = 2^(J-k);
    ubJk1 = 2^(J-k+1);

    gmat = zeros(ubJk1,ubJk);
    hmat = zeros(ubJk1,ubJk);

    for jj = 1:ubJk
        for ii = 1:ubJk1

            modulus = mod(N+ii-2*jj+shift,ubJk1);
            modulus = modulus + (modulus==0)*ubJk1;

            hmat(ii,jj) = h(modulus);
            gmat(ii,jj) = g(modulus);

        end
    end

    W = [oldmat*hmat'; gmat'];
    oldmat = W;

end

end

function [t0,dcritStd] = caucau_2mod_birth_standardized(tauStd)
% Gaussian second-mode birth geometry in standardized coordinates.
%
% For theta=sigma*u and tau=sigma*tauStd,
%
%   dcrit/sigma = min_t [tauStd exp(t) - H(t)/tauStd].
%
% No physical sigma is an argument of this standardized geometry.

obj = @(tt) tauStd.*exp(tt) - caucau_2mod_scaled_score(tt)./tauStd;
opts = optimset('Display','off','TolX',1e-11);

lo = -20;
hi = 20;

for iexpand = 1:5
    [t0,~,exitflag] = fminbnd(obj,lo,hi,opts);
    if exitflag <= 0
        error('Standardized Gaussian SPoM birth minimization failed.');
    end

    edgeTol = 1e-3*(hi-lo);
    if (t0-lo) < edgeTol
        lo = lo-10;
    elseif (hi-t0) < edgeTol
        hi = hi+10;
    else
        break;
    end
end

dcritStd = obj(t0);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function q = caucau_qprior(tgrid)

q = tgrid./(pi^2*sinh(tgrid));
iz = abs(tgrid) < 1e-10;
q(iz) = 1/pi^2;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DOUBLE-EXPONENTIAL POSTERIOR MEAN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function muhat = caucau_de_mean_vectorized(d,tau,aDE,tgrid)

original_size = size(d);
d = d(:).';
sgn = sign(d);
x   = abs(d);
tgrid = tgrid(:);

avec = tau*exp(tgrid);
q = caucau_qprior(tgrid);

Wplus  = q.*exp(-aDE*abs(avec-x));
Wminus = q.*exp(-aDE*(avec+x));

den = trapz(tgrid,Wplus+Wminus,1);
num = trapz(tgrid,avec.*(Wplus-Wminus),1);

mabs = zeros(size(x));
good = den > realmin & isfinite(den) & isfinite(num);
mabs(good) = num(good)./den(good);
mabs = max(mabs,0);

muhat = sgn.*mabs;
muhat = reshape(muhat,original_size);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DOUBLE-EXPONENTIAL POSTERIOR MEDIAN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function medhat = caucau_de_median_vectorized(d,tau,aDE,tgrid)

original_size = size(d);
d = d(:).';
sgn = sign(d);
x   = abs(d);
tgrid = tgrid(:);

avec = tau*exp(tgrid);
q = caucau_qprior(tgrid);

Wminus = q.*exp(-aDE*(avec+x));
Pminus = trapz(tgrid,Wminus,1);

Wplus = q.*exp(-aDE*abs(avec-x));
Pplus = trapz(tgrid,Wplus,1);

target = 0.5*(Pplus-Pminus);
Cum = cumtrapz(tgrid,Wplus,1);

nc = length(x);
mabs = zeros(1,nc);

for k = 1:nc

    if x(k) < 1e-14
        mabs(k) = 0;
        continue;
    end

    T = target(k);

    if T <= 0
        mabs(k) = 0;
        continue;
    end

    if T <= Cum(1,k)
        tm = tgrid(1);
    elseif T >= Cum(end,k)
        tm = tgrid(end);
    else
        ii = find(Cum(:,k)>=T,1,'first');
        i1 = ii-1;
        i2 = ii;
        c1 = Cum(i1,k);
        c2 = Cum(i2,k);
        t1 = tgrid(i1);
        t2 = tgrid(i2);

        if c2 > c1
            frac = (T-c1)/(c2-c1);
            tm = t1 + frac*(t2-t1);
        else
            tm = 0.5*(t1+t2);
        end
    end

    mabs(k) = tau*exp(tm);
end

medhat = sgn.*mabs;
medhat = reshape(medhat,original_size);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GAUSSIAN Cau/Cau SECOND POSTERIOR MODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dhat = caucau_psp_gaussian_vectorized(d,tau,sigma,epsilon,tgrid)
% Smooth Gaussian PSP rule:
%   dhat = d * P(|theta|>epsilon | d)
% with the continuous symmetric Cau/Cau prior of scale tau.

original_size = size(d);
d0 = d(:).';
x  = abs(d0);
tgrid = tgrid(:);

avec = tau*exp(tgrid);
q = caucau_qprior(tgrid);

Zplus  = (avec-x)/sigma;
Zminus = (avec+x)/sigma;
W = q.*(exp(-0.5*Zplus.^2) + exp(-0.5*Zminus.^2));

den = trapz(tgrid,W,1);
outside = (avec > epsilon);

if sum(outside) < 2
    num = zeros(size(den));
else
    num = trapz(tgrid(outside),W(outside,:),1);
end

pSignal = zeros(size(den));
good = den > realmin & isfinite(den) & isfinite(num);
pSignal(good) = num(good)./den(good);
pSignal = min(max(pSignal,0),1);

dhat = d0.*pSignal;
dhat = reshape(dhat,original_size);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dhat = caucau_psp_de_vectorized(d,tau,aDE,epsilon,tgrid)
% Smooth variance-matched DE PSP rule:
%   dhat = d * P(|theta|>epsilon | d)
% under the DE likelihood exp(-aDE*|d-theta|).

original_size = size(d);
d0 = d(:).';
x  = abs(d0);
tgrid = tgrid(:);

avec = tau*exp(tgrid);
q = caucau_qprior(tgrid);

W = q.*(exp(-aDE*abs(avec-x)) + exp(-aDE*(avec+x)));

den = trapz(tgrid,W,1);
outside = (avec > epsilon);

if sum(outside) < 2
    num = zeros(size(den));
else
    num = trapz(tgrid(outside),W(outside,:),1);
end

pSignal = zeros(size(den));
good = den > realmin & isfinite(den) & isfinite(num);
pSignal(good) = num(good)./den(good);
pSignal = min(max(pSignal,0),1);

dhat = d0.*pSignal;
dhat = reshape(dhat,original_size);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dhat = caucau_2mod_vectorized(d,tau,sigma,t0,dcrit,gammaInfl)
% Screen at gammaInfl*dcrit. Above it retain the genuine larger posterior
% stationary point solving B(theta)=|d|.

original_size = size(d);
d0 = d;
r  = abs(d0);
out = zeros(size(r));

lambda2mod = gammaInfl*dcrit;
idx = find(r > lambda2mod);

for kk = 1:numel(idx)

    ii = idx(kk);
    x  = r(ii);

    tlo = t0;
    thi = max(t0+0.5,log(max(x/tau,realmin))+0.5);

    fhi = caucau_2mod_birth_value(thi,tau,sigma)-x;
    nExpand = 0;

    while fhi <= 0 && nExpand < 80
        thi = thi+0.75;
        fhi = caucau_2mod_birth_value(thi,tau,sigma)-x;
        nExpand = nExpand+1;
    end

    if ~(isfinite(fhi) && fhi>0)
        error('Could not bracket Cau/Cau SPoM larger root.');
    end

    for it = 1:70
        tm = 0.5*(tlo+thi);
        fm = caucau_2mod_birth_value(tm,tau,sigma)-x;

        if fm > 0
            thi = tm;
        else
            tlo = tm;
        end
    end

    troot = 0.5*(tlo+thi);
    out(ii) = tau*exp(troot);
end

dhat = sign(d0).*out;
dhat = reshape(dhat,original_size);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function B = caucau_2mod_birth_value(t,tau,sigma)

H = caucau_2mod_scaled_score(t);
B = tau.*exp(t) - (sigma^2/tau).*H;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function H = caucau_2mod_scaled_score(t)

H = zeros(size(t));
small = abs(t) < 1e-4;
ts = t(small);

H(small) = -1 ...
           + (2/3)*ts ...
           - (1/6)*ts.^2 ...
           + (1/45)*ts.^3 ...
           - (1/120)*ts.^4 ...
           + (13/3780)*ts.^5 ...
           - (1/5040)*ts.^6;

tb = t(~small);

if ~isempty(tb)
    H(~small) = -1./sinh(tb) + exp(-tb)./tb;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GAUSSIAN Cau/Cau POSTERIOR MEAN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function muhat = caucau_mean_vectorized(d,tau,sigma,tgrid)

original_size = size(d);
d = d(:).';
sgn  = sign(d);
dabs = abs(d);
tgrid = tgrid(:);

avec = tau*exp(tgrid);
q = caucau_qprior(tgrid);

Zplus  = (avec-dabs)/sigma;
Zminus = (avec+dabs)/sigma;

Wplus  = q.*exp(-0.5*Zplus.^2);
Wminus = q.*exp(-0.5*Zminus.^2);

den = trapz(tgrid,Wplus+Wminus,1);
num = trapz(tgrid,avec.*(Wplus-Wminus),1);

muabs = zeros(size(dabs));
good = den > realmin & isfinite(den) & isfinite(num);
muabs(good) = num(good)./den(good);
muabs = max(muabs,0);

muhat = sgn.*muabs;
muhat = reshape(muhat,original_size);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GAUSSIAN Cau/Cau POSTERIOR MEDIAN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function medhat = caucau_median_vectorized(d,tau,sigma,tgrid)

original_size = size(d);
d = d(:).';
sgn  = sign(d);
dabs = abs(d);
tgrid = tgrid(:);

avec = tau*exp(tgrid);
q = caucau_qprior(tgrid);

Zminus = (avec+dabs)/sigma;
Wminus = q.*exp(-0.5*Zminus.^2);
Pminus = trapz(tgrid,Wminus,1);

Zplus = (avec-dabs)/sigma;
Wplus = q.*exp(-0.5*Zplus.^2);
Pplus = trapz(tgrid,Wplus,1);

target = 0.5*(Pplus-Pminus);
Cum = cumtrapz(tgrid,Wplus,1);

nc = length(dabs);
mabs = zeros(1,nc);

for k = 1:nc

    if dabs(k)<1e-14
        mabs(k)=0;
        continue;
    end

    T = target(k);

    if T <= 0
        mabs(k)=0;
        continue;
    end

    if T <= Cum(1,k)
        tm = tgrid(1);
    elseif T >= Cum(end,k)
        tm = tgrid(end);
    else
        ii = find(Cum(:,k)>=T,1,'first');
        i1 = ii-1;
        i2 = ii;
        c1 = Cum(i1,k);
        c2 = Cum(i2,k);
        t1 = tgrid(i1);
        t2 = tgrid(i2);

        if c2>c1
            frac = (T-c1)/(c2-c1);
            tm = t1+frac*(t2-t1);
        else
            tm = 0.5*(t1+t2);
        end
    end

    mabs(k) = tau*exp(tm);
end

medhat = sgn.*mabs;
medhat = reshape(medhat,original_size);

end

