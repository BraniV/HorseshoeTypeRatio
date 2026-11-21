function PLOT_HORSESHOE_KAPPA_GEOMETRY_V2
% PLOT_HORSESHOE_KAPPA_GEOMETRY_V2
%
% Revised four-figure set for denominator-induced horseshoe geometry
% and Cau/Cau effective shrinkage.
%
% No titles.
% Mathematical axis labels and legends use MATLAB LaTeX.
% Tick numbers use Times New Roman.
%
% Output:
%   Figs/KAPPA01_THREE_HORSESHOES.pdf/.png
%   Figs/KAPPA02_CONDITIONAL.pdf/.png
%   Figs/KAPPA03_TWO_SHRINKAGES.pdf/.png
%   Figs/KAPPA04_ASYMPTOTICS.pdf/.png

close all;

%% ------------------------------------------------------------
% Parameters
% ------------------------------------------------------------
tau   = 1/50;
sigma = 1;

outdir = 'Figs';
if ~exist(outdir,'dir')
    mkdir(outdir);
end

fontName = 'Times New Roman';
fontSize = 13;
lineWidth = 1.9;

%% ============================================================
% FIGURE 1
% Three U-shaped shrinkage-coordinate densities.
%
% Beta(a,b) has
%       h(kappa) ~ kappa^(a-1)          as kappa -> 0
%       h(kappa) ~ (1-kappa)^(b-1)      as kappa -> 1.
%
% Thus Beta(1/2,1/2) is the classical horseshoe, while the two
% asymmetric examples have a stronger singularity at one endpoint.
% ============================================================
k = linspace(0.002,0.998,3000);

a1 = 1/2; b1 = 1/2;
a2 = 1/4; b2 = 1/2;
a3 = 1/2; b3 = 1/4;

h1 = beta_density_no_toolbox(k,a1,b1);
h2 = beta_density_no_toolbox(k,a2,b2);
h3 = beta_density_no_toolbox(k,a3,b3);

figure('Color','w','Position',[120 120 680 520]);

plot(k,h1,'-','LineWidth',2.1);
hold on;
plot(k,h2,'--','LineWidth',lineWidth);
plot(k,h3,'-.','LineWidth',lineWidth);
hold off;

xlabel('$\kappa_Y$','Interpreter','latex','FontSize',15);
ylabel('$h_Y(\kappa)$','Interpreter','latex','FontSize',15);

lgd = legend({'$\mathrm{Beta}(1/2,1/2)$',...
              '$\mathrm{Beta}(1/4,1/2)$',...
              '$\mathrm{Beta}(1/2,1/4)$'},...
             'Interpreter','latex','Location','north');
set(lgd,'Box','off','FontSize',11);

xlim([0 1]);
ylim([0 5]);
box on;
grid on;
format_axes(gca,fontName,fontSize);

export_pair(gcf,outdir,'KAPPA01_THREE_HORSESHOES');


%% ============================================================
% FIGURE 2
% Exact conditional density p(kappa_Y | Theta=theta), Cau/Cau.
%
% Let r=|theta|/tau. Then
%
%   p(kappa | theta)
%      = (r^2-1)/log(r^2) * 1/[1+(r^2-1)kappa].
%
% At r=1, the density is Uniform(0,1).
%
% Moderate ratios are used so the curves remain visually informative;
% very large ratios create extreme endpoint peaks.
% ============================================================
k2 = linspace(0.001,0.999,3000);
ratios = [0.2 1 2 5];

figure('Color','w','Position',[140 140 680 520]);
hold on;

styles = {'-','--','-.',':'};

for j = 1:numel(ratios)
    r = ratios(j);
    pk = caucau_kappa_given_ratio(k2,r);
    plot(k2,pk,styles{j},'LineWidth',lineWidth);
end

hold off;

xlabel('$\kappa_Y$','Interpreter','latex','FontSize',15);
ylabel('$p(\kappa_Y\mid\Theta=\theta)$',...
       'Interpreter','latex','FontSize',15);

lgd = legend({'$|\theta|/\tau=0.2$',...
              '$|\theta|/\tau=1$',...
              '$|\theta|/\tau=2$',...
              '$|\theta|/\tau=5$'},...
             'Interpreter','latex','Location','north');
set(lgd,'Box','off','FontSize',11);

xlim([0 1]);
ylim([0 8]);
box on;
grid on;
format_axes(gca,fontName,fontSize);

export_pair(gcf,outdir,'KAPPA02_CONDITIONAL');


%% ============================================================
% Numerical quadrature setup for Figures 3 and 4
%
% Positive-half-line logarithmic coordinate:
%
%       theta = tau exp(t)
%
% and
%
%       pi_tau(tau exp(t)) tau exp(t)
%          = q(t) = t/[pi^2 sinh(t)].
%
% For Cau/Cau,
%
%       E(kappa_Y | Theta=tau exp(t))
%          = 1/(2t) - 1/[exp(2t)-1].
% ============================================================
tmin = -24;
tmax =  12;
Nt   = 24001;

t = linspace(tmin,tmax,Nt);
s = tau*exp(t);

% Stable evaluation of q(t)
q = t ./ (pi^2*sinh(t));
small = abs(t) < 1e-4;
q(small) = (1/pi^2).*(1 - t(small).^2/6);

% Stable evaluation of E(kappa_Y | Theta=tau exp(t))
EkTheta = 1./(2*t) - 1./expm1(2*t);
EkTheta(small) = 1/2 - t(small)/6 + t(small).^3/90;

q(~isfinite(q)) = 1/pi^2;
EkTheta(~isfinite(EkTheta)) = 1/2;


%% ============================================================
% FIGURE 3
% Cau/Cau:
%
%   E(kappa_Y | D=d)
%
% versus
%
%   kappa_eff(d) = 1 - E(Theta|d)/d.
%
% The log-log scale displays the very different rates:
%
%   E(kappa_Y|d)  ~ 1/log(d^2/tau^2),
%   kappa_eff(d) ~ 2 sigma^2/d^2.
% ============================================================
d3 = logspace(log10(0.10),log10(20),180);

[EkD3,kEff3] = caucau_shrinkage_curves(d3,t,s,q,EkTheta,sigma);

figure('Color','w','Position',[160 160 680 520]);

loglog(d3,EkD3,'-','LineWidth',2.0);
hold on;
loglog(d3,kEff3,'--','LineWidth',2.0);
hold off;

xlabel('$|d|$','Interpreter','latex','FontSize',15);
ylabel('$\kappa$','Interpreter','latex','FontSize',15);

lgd = legend({'$E(\kappa_Y\mid D=d)$',...
              '$\kappa_{\mathrm{eff}}(d)$'},...
             'Interpreter','latex','Location','southwest');
set(lgd,'Box','off','FontSize',11);

box on;
grid on;
format_axes(gca,fontName,fontSize);

export_pair(gcf,outdir,'KAPPA03_TWO_SHRINKAGES');


%% ============================================================
% FIGURE 4
% Scaled large-signal diagnostics:
%
%   log(d^2/tau^2) E(kappa_Y|d)          -> 1,
%
%   d^2 kappa_eff(d)/(2 sigma^2)         -> 1.
% ============================================================
d4 = logspace(log10(0.75),log10(50),180);

[EkD4,kEff4] = caucau_shrinkage_curves(d4,t,s,q,EkTheta,sigma);

diagLog  = log((d4/tau).^2).*EkD4;
diagQuad = (d4.^2/(2*sigma^2)).*kEff4;

figure('Color','w','Position',[180 180 680 520]);

semilogx(d4,diagLog,'-','LineWidth',2.0);
hold on;
semilogx(d4,diagQuad,'--','LineWidth',2.0);
yline(1,':','LineWidth',1.2);
hold off;

xlabel('$|d|$','Interpreter','latex','FontSize',15);
ylabel('scaled quantity','FontName',fontName,'FontSize',15);

lgd = legend({'$\log(d^2/\tau^2)\,E(\kappa_Y\mid D=d)$',...
              '$d^2\kappa_{\mathrm{eff}}(d)/(2\sigma^2)$',...
              '$1$'},...
             'Interpreter','latex','Location','best');
set(lgd,'Box','off','FontSize',10.5);

box on;
grid on;
format_axes(gca,fontName,fontSize);

export_pair(gcf,outdir,'KAPPA04_ASYMPTOTICS');

end


%% ============================================================
% Local functions
% ============================================================

function h = beta_density_no_toolbox(k,a,b)
% Beta density without requiring the Statistics Toolbox.

B = gamma(a)*gamma(b)/gamma(a+b);
h = k.^(a-1).*(1-k).^(b-1)/B;

end


function pk = caucau_kappa_given_ratio(k,r)
% Exact density p(kappa_Y | Theta=theta) for r=|theta|/tau.

r2 = r^2;

if abs(r2-1) < 1e-12
    pk = ones(size(k));
else
    c  = (r2-1)/log(r2);
    pk = c ./ (1 + (r2-1)*k);
end

end


function [EkD,kEff] = caucau_shrinkage_curves(d,t,s,q,EkTheta,sigma)
% Posterior E(kappa_Y | D=d) and effective posterior shrinkage
%
%   kappa_eff(d) = 1 - E(Theta|d)/d.
%
% Integrals use the positive-half-line logarithmic representation.

d = d(:).';
EkD  = zeros(size(d));
kEff = zeros(size(d));

normconst = 1/(sqrt(2*pi)*sigma);

for j = 1:numel(d)

    dj = d(j);

    phip = normconst*exp(-0.5*((dj-s)/sigma).^2);
    phim = normconst*exp(-0.5*((dj+s)/sigma).^2);

    wsum = q.*(phip+phim);
    wdif = s.*q.*(phip-phim);

    m = trapz(t,wsum);

    EkD(j) = trapz(t,EkTheta.*wsum)/m;

    postMean = trapz(t,wdif)/m;

    if abs(dj) > 1e-12
        kEff(j) = 1-postMean/dj;
    else
        kEff(j) = NaN;
    end
end

end


function format_axes(ax,fontName,fontSize)
% Tick numbers in Times New Roman. Mathematical labels are set separately.

set(ax,...
    'FontName',fontName,...
    'FontSize',fontSize,...
    'LineWidth',0.9,...
    'TickDir','out',...
    'TickLabelInterpreter','tex');

end


function export_pair(fig,outdir,baseName)
% Save vector PDF and high-resolution PNG.

pdfName = fullfile(outdir,[baseName '.pdf']);
pngName = fullfile(outdir,[baseName '.png']);

exportgraphics(fig,pdfName,'ContentType','vector');
exportgraphics(fig,pngName,'Resolution',300);

end
