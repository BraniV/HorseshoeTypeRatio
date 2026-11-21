(* ============================================================ *)
(* BIVARIATE CAUCHY/CAUCHY (p = 2): ONLY TWO REVISED FIGURES     *)
(* Changes: true 3D prior surface with larger labels; posterior *)
(* rule curve unchanged except AspectRatio -> 1.                 *)
(* Same distribution, quadrature and posterior estimators.        *)
(* Requires Mathematica and an installed MaTeX package.         *)
(* ============================================================ *)
ClearAll["Global`*"];
Needs["MaTeX`"];
SetOptions[MaTeX,
  "Preamble" -> {"\\usepackage{amsmath,amssymb}",
                 "\\usepackage{mathptmx}"},
  FontSize -> 23
];

tau = 1/50.;
sigma = 1.;
outDir = Directory[];
fontFamily = "Times New Roman";
axesStyle = Directive[Black, AbsoluteThickness[1.15]];

(* The original exact bivariate radial prior density. *)
jFun[u_?NumericQ] := Module[{v = N[u], h},
  Which[
    v <= 0., Infinity,
    v < 0.995,
      (1/v - ArcCos[v]/Sqrt[1 - v^2])/(1 - v^2),
    v > 1.005,
      (ArcCosh[v]/Sqrt[v^2 - 1] - 1/v)/(v^2 - 1),
    True,
      h = v - 1.;
      1/3 - 3 h/5 + 27 h^2/35 - 55 h^3/63 +
      215 h^4/231 - 413 h^5/429
  ]
];
priorR[r_?NumericQ] := jFun[Abs[N[r]]/tau]/(Pi^2 tau^2);

(* FIGURE 1: TRUE 3D PRIOR SURFACE, NOT A 2D PROJECTION. *)
priorTextStyle = Directive[FontFamily -> fontFamily, FontSize -> 22, Black];
priorTickStyle = Directive[FontFamily -> fontFamily, FontSize -> 20, Black];
zCap = 900.;
plotExtent = 0.20;
figPrior = Plot3D[
  Min[zCap, priorR[Sqrt[th1^2 + th2^2]]],
  {th1, -plotExtent, plotExtent},
  {th2, -plotExtent, plotExtent},
  PlotPoints -> 85, MaxRecursion -> 2,
  PlotRange -> {0, zCap},
  Mesh -> None, ColorFunction -> "SolarColors",
  ColorFunctionScaling -> True, Lighting -> "Neutral",
  Boxed -> False, Axes -> True,
  AxesLabel -> {
    MaTeX["\\theta_1", FontSize -> 25],
    MaTeX["\\theta_2", FontSize -> 25],
    MaTeX["\\pi_{2,\\tau}(\\theta)", FontSize -> 25]
  },
  LabelStyle -> priorTextStyle,
  TicksStyle -> priorTickStyle, AxesStyle -> axesStyle,
  BoxRatios -> {1, 1, .9}, ViewPoint -> {2.2, -2.6, 1.7},
  ImageSize -> 850,
  ImagePadding -> {{100, 110}, {95, 65}}
];

(* Original posterior quadrature (unchanged). *)
r1 = Table[(k - .5)*.001, {k, 1, 300}];
r2 = Table[.3 + (k - .5)*.01, {k, 1, 120}];
r3 = Table[1.5 + (k - .5)*.025, {k, 1, 440}];
rNodes = Join[r1, r2, r3];
deltaR = Join[
  ConstantArray[.001, Length[r1]],
  ConstantArray[.01, Length[r2]],
  ConstantArray[.025, Length[r3]]
];
radialBase = N[rNodes*(priorR /@ rNodes)*deltaR];

nPhi = 320;
phis = Table[2. Pi (k - .5)/nPhi, {k, 1, nPhi}];
cosPhi = Cos[phis];
sinPhi = Sin[phis];
xGrid = Outer[Times, rNodes, cosPhi];
yGrid = Outer[Times, rNodes, sinPhi];
ones = ConstantArray[1., nPhi];

posteriorStats[0.] = {0., 0.};
posteriorStats[s_?NumericQ] :=
  Module[{w0, wt, denom, mean, grad, h, med},
    If[s <= 0., Return[{0., 0.}]];
    w0 = radialBase*Exp[-(rNodes - s)^2/(2 sigma^2)];
    wt = Outer[Times, w0, ones]*
      Exp[Outer[Times, s*rNodes/sigma^2, cosPhi - 1.]];
    denom = Total[Flatten[wt]];
    mean = Total[Flatten[wt*xGrid]]/denom;
    grad[h0_?NumericQ] := Total[Flatten[
      wt*(h0 - xGrid)/Sqrt[(h0 - xGrid)^2 + yGrid^2]
    ]];
    med = h /. FindRoot[
      grad[h] == 0., {h, 0., s + 2.},
      Method -> "Brent", MaxIterations -> 100
    ];
    N[{mean, med}]
  ];

(* 401 observations, spacing 0.02; values are not changed by font edits. *)
sGrid = N[Subdivide[0., 8., 400]];
curves = Table[Join[{s}, posteriorStats[s]], {s, sGrid}];
meanInterp = Interpolation[curves[[All, {1, 2}]], InterpolationOrder -> 3];
medInterp  = Interpolation[curves[[All, {1, 3}]], InterpolationOrder -> 3];

(* FIGURE 2: larger ticks, legend, axis math and sufficient padding. *)
rulesTextStyle = Directive[FontFamily -> fontFamily, FontSize -> 24, Black];
rulesTickStyle = Directive[FontFamily -> fontFamily, FontSize -> 21, Black];
styles = {
  Directive[Blue, AbsoluteThickness[2.5]],
  Directive[Red, AbsoluteThickness[2.5]],
  Directive[Black, AbsoluteThickness[1.5]]
};
fig2 = Plot[
  Evaluate[{meanInterp[s], medInterp[s], s}], {s, 0., 8.},
  PlotPoints -> 150, MaxRecursion -> 3,
  PlotStyle -> styles,
  PlotLegends -> Placed[
    LineLegend[
      styles,
      {MaTeX["\\text{Posterior mean}", FontSize -> 24],
       MaTeX["\\text{Posterior spatial median}", FontSize -> 24],
       MaTeX["\\text{No shrinkage}", FontSize -> 24]},
      LegendMarkerSize -> 45, LegendLayout -> "Column"
    ],
    {0.29, 0.78}
  ],
  Frame -> False, Axes -> True, AxesOrigin -> {0., 0.},
  AxesLabel -> {
    MaTeX["\\lVert d\\rVert", FontSize -> 29],
    MaTeX["\\lVert\\widehat{\\theta}(d)\\rVert", FontSize -> 29]
  },
  LabelStyle -> rulesTextStyle, TicksStyle -> rulesTickStyle,
  AxesStyle -> axesStyle,
  PlotRange -> All,
  ImageSize -> 900,
  ImagePadding -> {{125, 110}, {125, 80}},
  AspectRatio -> 1
];

(* Only the two requested figures are displayed/exported. *)
Print[figPrior];
Print[fig2];
Do[
  Module[{name = item[[1]], fig = item[[2]]},
    Export[FileNameJoin[{outDir, name <> ".pdf"}], fig];
    Export[FileNameJoin[{outDir, name <> ".png"}], fig,
           ImageResolution -> 300];
    Print["Saved: ", name, " (.pdf and .png)"]
  ],
  {item, {
    {"MVCauchyCauchy_prior_3D_fonts_fixed", figPrior},
    {"MVCauchyCauchy_rules_square", fig2}
  }}
];
