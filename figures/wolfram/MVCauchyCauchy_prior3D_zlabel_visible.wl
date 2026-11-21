(* Corrected TRUE 3D Cauchy/Cauchy prior density (p=2).
   Standalone: recomputes only the prior surface, NOT the posterior rules.
   The z-axis label is in the 2D right-hand label area, so Plot3D's
   perspective label placement cannot clip it. The Figure 2 file stays as is.
   Requires MaTeX installed in Mathematica.
*)

ClearAll["Global`*"];
Needs["MaTeX`"];
SetOptions[MaTeX,
  "Preamble" -> {"\\usepackage{amsmath,amssymb}", "\\usepackage{mathptmx}"},
  FontSize -> 23
];

tau = 1/50.;
plotExtent = 0.20;
zCap = 900.;
fontFamily = "Times New Roman";
axesStyle = Directive[Black, AbsoluteThickness[1.15]];
priorTextStyle = Directive[FontFamily -> fontFamily, FontSize -> 22, Black];
priorTickStyle = Directive[FontFamily -> fontFamily, FontSize -> 20, Black];

jFun[u_?NumericQ] := Module[{v = N[u], h},
  Which[
    v <= 0., Infinity,
    v < 0.995, (1/v - ArcCos[v]/Sqrt[1 - v^2])/(1 - v^2),
    v > 1.005, (ArcCosh[v]/Sqrt[v^2 - 1] - 1/v)/(v^2 - 1),
    True,
      h = v - 1.;
      1/3 - 3 h/5 + 27 h^2/35 - 55 h^3/63 +
      215 h^4/231 - 413 h^5/429
  ]
];
priorR[r_?NumericQ] := jFun[Abs[N[r]]/tau]/(Pi^2 tau^2);

(* The camera, capped height and density formula are unchanged.
   Use NONE for the third AxesLabel: Mathematica can clip a 3D label
   at perspective boundaries; the Labeled wrapper places it outside.
   The overall output is approximately square, like the rules figure.
*)
priorSurface = Plot3D[
  Min[zCap, priorR[Sqrt[th1^2 + th2^2]]],
  {th1, -plotExtent, plotExtent},
  {th2, -plotExtent, plotExtent},
  PlotPoints -> 85, MaxRecursion -> 2,
  PlotRange -> {0, zCap}, PlotRangePadding -> Scaled[0.02],
  Mesh -> None, ColorFunction -> "SolarColors",
  ColorFunctionScaling -> True, Lighting -> "Neutral",
  Boxed -> False, Axes -> True,
  AxesLabel -> {
    MaTeX["\\theta_1", FontSize -> 30],
    MaTeX["\\theta_2", FontSize -> 30],
    None
  },
  LabelStyle -> priorTextStyle,
  TicksStyle -> priorTickStyle,
  AxesStyle -> axesStyle,
  BoxRatios -> {1, 1, 0.9},
  ViewPoint -> {2.2, -2.6, 1.7},
  SphericalRegion -> True,
  ImageSize -> 820,
  ImagePadding -> {{80, 35}, {85, 65}}
];

(* Outside label is not subject to three-dimensional perspective clipping. *)
figPrior = Labeled[
  priorSurface,
  Rotate[MaTeX["\\pi_{2,\\tau}(\\theta)", FontSize -> 28], Pi/2],
  Right,
  Spacings -> 0
];

Print[figPrior];
outDir = Directory[];
Export[FileNameJoin[{outDir, "MVCauchyCauchy_prior_3D_zlabel_visible.pdf"}], figPrior];
Export[FileNameJoin[{outDir, "MVCauchyCauchy_prior_3D_zlabel_visible.png"}], figPrior,
  ImageResolution -> 300];
Print["Saved the prior only. Figure 2 was not modified."];
