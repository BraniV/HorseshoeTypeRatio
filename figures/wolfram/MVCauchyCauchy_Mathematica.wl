(* Multivariate Cauchy/Cauchy prior, p = 2, Sigma = IdentityMatrix[2].
   Prior: X ~ t_2(1,0,tau^2 I_2), Y ~ Cauchy(0,1), Theta=X/Y.
   Gaussian likelihood: D|Theta ~ Normal_2(Theta, sigma^2 I_2).
   Posterior median: spatial/geometric median, minimizing posterior E||Theta-a||.
   All plots are computed directly, without reading any external data.
   Numerical quadrature uses radial midpoints and angular midpoints.
*)

ClearAll["Global`*"];
tau = 1/50.;
sigma = 1.;

(* Exact elementary expression for the radial bivariate prior.
   pi(theta) = J(||theta||/tau)/(Pi^2 tau^2).
   Its density diverges at the origin, but is integrable there. *)

jFun[u_?NumericQ] := Module[{v = N[u], h},
  Which[
    v <= 0, Infinity,
    v < 0.995,
      (1/v - ArcCos[v]/Sqrt[1 - v^2])/(1 - v^2),
    v > 1.005,
      (ArcCosh[v]/Sqrt[v^2 - 1] - 1/v)/(v^2 - 1),
    True,
      h = v - 1;
      1/3 - 3 h/5 + 27 h^2/35 - 55 h^3/63 + 215 h^4/231
  ]
];

priorR[r_?NumericQ] := jFun[Abs[N[r]]/tau]/(Pi^2 tau^2);

(* Figure 1: logarithm of the bivariate prior density. *)
fig1 = DensityPlot[
  Log[10, priorR[Max[10^-6, Sqrt[th1^2 + th2^2]]]],
  {th1, -1.2, 1.2}, {th2, -1.2, 1.2},
  PlotPoints -> 70, MaxRecursion -> 2,
  PlotLegends -> Automatic,
  FrameLabel -> {"theta1", "theta2"},
  PlotLabel -> "Bivariate Cauchy/Cauchy prior: log10 density",
  AspectRatio -> 1, ImageSize -> 650
];

(* Quadrature for the posterior: integration in polar coordinates.
   With d=(s,0), theta=(r cos(phi), r sin(phi)), the radial
   posterior weight is proportional to
   r*pi(r)*Exp[-(r-s)^2/(2 sigma^2)]
             *Exp[s*r*(Cos[phi]-1)/sigma^2].
   The omitted multiplicative constant cancels in posterior ratios.
*)

r1 = Table[(k - 0.5)*0.001, {k, 1, 300}];
r2 = Table[0.3 + (k - 0.5)*0.01, {k, 1, 120}];
r3 = Table[1.5 + (k - 0.5)*0.025, {k, 1, 440}];
rNodes = Join[r1, r2, r3];
deltaR = Join[ConstantArray[0.001, Length[r1]],
              ConstantArray[0.01, Length[r2]],
              ConstantArray[0.025, Length[r3]]];
radialBase = N[rNodes*(priorR /@ rNodes)*deltaR];

nPhi = 160;
phis = Table[2. Pi (k - 0.5)/nPhi, {k, 1, nPhi}];
cosPhi = Cos[phis]; sinPhi = Sin[phis];
xGrid = Outer[Times, rNodes, cosPhi];
yGrid = Outer[Times, rNodes, sinPhi];
ones = ConstantArray[1., nPhi];

posteriorStats[0.] := {0., 0.};
posteriorStats[s_?NumericQ] := Module[{w0, wt, mean, grad, h, med},
  If[s <= 0, Return[{0., 0.}]];
  w0 = radialBase*Exp[-(rNodes - s)^2/(2 sigma^2)];
  wt = Outer[Times, w0, ones]*
       Exp[Outer[Times, s*rNodes/sigma^2, cosPhi - 1]];
  mean = Total[Flatten[wt*xGrid]]/Total[Flatten[wt]];
  grad[h0_?NumericQ] := Total[Flatten[
    wt*(h0 - xGrid)/Sqrt[(h0 - xGrid)^2 + yGrid^2]
  ]];
  med = h /. FindRoot[grad[h] == 0,
      {h, 0., s + 2.}, Method -> "Brent", MaxIterations -> 100];
  N[{mean, med}]
];

(* Radial symmetry gives E[Theta|d] = mean(s) d/s.
   The spatial median is similarly med(s) d/s, s=||d||. *)
sGrid = N[Range[0, 8, 1/10]];
curves = Table[Join[{s}, posteriorStats[s]], {s, sGrid}];
meanCurve = curves[[All, {1, 2}]];
medianCurve = curves[[All, {1, 3}]];
identityCurve = ({#, #} & /@ sGrid);

(* Figure 2: these radial curves fully specify the two bivariate rules. *)
fig2 = ListLinePlot[
  {meanCurve, medianCurve, identityCurve},
  PlotLegends -> {"Posterior mean", "Posterior spatial median", "No shrinkage"},
  Frame -> True,
  FrameLabel -> {"||d||", "Magnitude of estimate"},
  PlotLabel -> "Posterior mean and spatial median; tau=1/50, sigma=1",
  PlotRange -> All, ImageSize -> 750
];

(* Optional Figures 3 and 4: full vector fields in the data plane. *)
meanInterp = Interpolation[meanCurve, InterpolationOrder -> 3];
medInterp = Interpolation[medianCurve, InterpolationOrder -> 3];
meanVector[x_?NumericQ, y_?NumericQ] := Module[{s = Sqrt[x^2 + y^2]},
  If[s < 10^-9, {0., 0.}, (meanInterp[s]/s)*{x, y}]
];
medianVector[x_?NumericQ, y_?NumericQ] := Module[{s = Sqrt[x^2 + y^2]},
  If[s < 10^-9, {0., 0.}, (medInterp[s]/s)*{x, y}]
];
fig3 = VectorPlot[meanVector[d1, d2], {d1, -5.5, 5.5}, {d2, -5.5, 5.5},
   VectorPoints -> 19, FrameLabel -> {"d1", "d2"},
   PlotLabel -> "Posterior mean vector field", ImageSize -> 620];
fig4 = VectorPlot[medianVector[d1, d2], {d1, -5.5, 5.5}, {d2, -5.5, 5.5},
   VectorPoints -> 19, FrameLabel -> {"d1", "d2"},
   PlotLabel -> "Posterior spatial median vector field", ImageSize -> 620];

outDir = Directory[];
Export[FileNameJoin[{outDir, "Fig1_MVCauchyCauchy_prior_Mathematica.png"}], fig1,
       ImageResolution -> 220];
Export[FileNameJoin[{outDir, "Fig2_MVCauchyCauchy_rules_Mathematica.png"}], fig2,
       ImageResolution -> 220];
Export[FileNameJoin[{outDir, "Fig3_MVCauchyCauchy_mean_Mathematica.png"}], fig3,
       ImageResolution -> 220];
Export[FileNameJoin[{outDir, "Fig4_MVCauchyCauchy_median_Mathematica.png"}], fig4,
       ImageResolution -> 220];
Export[FileNameJoin[{outDir, "MVCauchyCauchy_curves_Mathematica.csv"}],
       Prepend[curves, {"||d||", "Posterior mean magnitude", "Spatial median magnitude"}]];
Print["Saved four figures and CSV in ", outDir];
