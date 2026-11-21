(* Bivariate Cauchy/Cauchy prior: true 3D density, p=2.
   Recompute the PRIOR ONLY. Figure 2 is not modified.
   Requires MaTeX installed. *)

ClearAll["Global`*"];
Needs["MaTeX`"];
SetOptions[MaTeX,
  "Preamble" -> {"\\usepackage{amsmath,amssymb}", "\\usepackage{mathptmx}"},
  FontSize -> 23];

tau = 1/50.;
plotExtent = 0.20;
zCap = 900.;

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

(* Explicit two-dimensional annotations stay inside export bounds.
   AxesOrigin makes the x and y coordinate axes intersect at zero. *)
figPrior = Plot3D[
  Min[zCap, priorR[Sqrt[th1^2 + th2^2]]],
  {th1, -plotExtent, plotExtent},
  {th2, -plotExtent, plotExtent},
  PlotPoints -> 85, MaxRecursion -> 2,
  PlotRange -> {0, zCap},
  PlotRangePadding -> Scaled[0.06],
  PlotRangeClipping -> False,
  Mesh -> None,
  ColorFunction -> "SolarColors",
  ColorFunctionScaling -> True,
  Lighting -> "Neutral",
  Boxed -> False, Axes -> True,
  AxesOrigin -> {0, 0, 0},
  AxesLabel -> {None, None, None},
  LabelStyle -> Directive[FontFamily -> "Times New Roman", FontSize -> 22, Black],
  TicksStyle -> Directive[FontFamily -> "Times New Roman", FontSize -> 20, Black],
  AxesStyle -> Directive[Black, AbsoluteThickness[1.15]],
  BoxRatios -> {1, 1, 0.9},
  ViewPoint -> {2.2, -2.6, 1.7},
  SphericalRegion -> True,
  ImageSize -> 960,
  ImagePadding -> {{140, 90}, {120, 95}},
  Epilog -> {
    Inset[MaTeX["\\theta_1", FontSize -> 30], Scaled[{0.29, 0.13}]],
    Inset[MaTeX["\\theta_2", FontSize -> 30], Scaled[{0.81, 0.17}]],
    Inset[Rotate[MaTeX["\\pi_{2,\\tau}(\\boldsymbol{\\theta})", FontSize -> 28], Pi/2],
          Scaled[{0.075, 0.57}]]
  }
];
Print[figPrior];
outDir = Directory[];
Export[FileNameJoin[{outDir, "MVCauchyCauchy_prior_3D_origin_axes.pdf"}], figPrior];
Export[FileNameJoin[{outDir, "MVCauchyCauchy_prior_3D_origin_axes.png"}], figPrior,
  ImageResolution -> 300];
Print["Saved prior only. Figure 2 not modified."];
