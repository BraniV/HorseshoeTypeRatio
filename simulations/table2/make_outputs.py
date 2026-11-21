import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os; HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'res')
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
SIGS = ['Blocks', 'Bumps', 'Doppler', 'HeaviSine']
LABEL = {
    'CaCaMean': r'Ca/Ca posterior mean, fixed $\tau_j$',
    'CaCaMMLE': r'Ca/Ca posterior mean, MMLE $\tau_j$',
    'Horseshoe': r'Horseshoe posterior mean (MMLE $\tau_j$)',
    'BAMS': r'BAMS',
    'EBayesMed': r'EBayesThresh, posterior median',
    'EBayesMean': r'EBayesThresh, posterior mean',
    'BayesThresh': r'BayesThresh, posterior median',
    'RaisedCosine': r'Raised-cosine Bayes',
    'Epanechnikov': r'Epanechnikov Bayes',
    'NCPmean2': r'NCPmean ($m=2$)',
    'LNWS': r'$\Lambda$-neighborhood (LNWS)',
    'BayesShrink': r'BayesShrink',
    'BiShrink': r'Bivariate shrinkage (BiShrink)',
    'SureShrink': r'SureShrink',
    'SmoothSCAD': r'Smooth SCAD--SURE',
    'BlockJS': r'BlockJS',
    'VisuShrink': r'VisuShrink',
}
CITE = {
    'CaCaMean': '', 'Horseshoe': r'\citep{carvalho2010horseshoe,vanderPasSzaboVaart2017}',
    'BAMS': r'\citep{VidakovicRuggeri2001}', 'EBayesMed': r'\citep{JohnstoneSilverman2005}',
    'EBayesMean': r'\citep{JohnstoneSilverman2005}', 'BayesThresh': r'\citep{AbramovichSapatinasSilverman1998}',
    'RaisedCosine': r'\citep{ReinaSousa2025}', 'Epanechnikov': r'\citep{CausilSousa2025}',
    'NCPmean2': r'\citep{WangWood2006}', 'LNWS': r'\citep{RemenyiVidakovic2013}',
    'BayesShrink': r'\citep{ChangYuVetterli2000}', 'BiShrink': r'\citep{SendurSelesnick2002}',
    'SureShrink': r'\citep{DonohoJohnstone1995}', 'SmoothSCAD': r'\citep{Kulkarni2026}',
    'BlockJS': r'\citep{Cai1999}', 'VisuShrink': r'\citep{DonohoJohnstone1994}',
}
BAYES = {'CaCaMean', 'CaCaMMLE', 'Horseshoe', 'BAMS', 'EBayesMed', 'EBayesMean', 'BayesThresh',
         'RaisedCosine', 'Epanechnikov', 'NCPmean2', 'LNWS', 'BayesShrink', 'BiShrink'}

prefix = sys.argv[1] if len(sys.argv) > 1 else 'main'
R = {s: np.load(f'{RES}/{prefix}_{s}.npz') for s in SIGS}
methods = list(R['Blocks']['methods'])
M = R['Blocks']['MSE'].shape[0]
import os
if prefix == 'main' and all(os.path.exists(f'{RES}/mmle_{s}.npz') for s in SIGS):
    R = {s: dict(R[s]) for s in SIGS}
    for s in SIGS:
        R[s]['MSE'] = np.column_stack([R[s]['MSE'], np.load(f'{RES}/mmle_{s}.npz')['MSE']])
    methods = methods + ['CaCaMMLE']
A = np.array([[R[s]['MSE'][:, k].mean() for s in SIGS] for k in range(len(methods))])
SE = np.array([[R[s]['MSE'][:, k].std(ddof=1) / np.sqrt(M) for s in SIGS] for k in range(len(methods))])
ranks = np.argsort(np.argsort(A, axis=0), axis=0) + 1

# paired comparison of every method with Ca/Ca mean: mean diff and SE
iC = methods.index('CaCaMean')
pairSE = np.array([[np.std(R[s]['MSE'][:, k] - R[s]['MSE'][:, iC], ddof=1) / np.sqrt(M) for s in SIGS]
                   for k in range(len(methods))])

with open(f'{OUT}/{prefix}_summary.txt', 'w') as fh:
    fh.write(f'M={M}\n')
    for s in SIGS:
        fh.write(f"{s}: sigmaHat mean {R[s]['SIG'].mean():.4f} sd {R[s]['SIG'].std(ddof=1):.4f}; bams_bad {int(R[s]['bams_bad'])}\n")
    for k, m in enumerate(methods):
        fh.write(f'{m:13s} ' + ' '.join(f'{A[k,i]:.4f}({SE[k,i]:.4f})[{ranks[k,i]:2d}] d={A[k,i]-A[iC,i]:+.4f}+-{pairSE[k,i]:.4f}' for i in range(4))
                 + f'  meanrank={ranks[k].mean():.2f}\n')
print(open(f'{OUT}/{prefix}_summary.txt').read())

if prefix != 'main':
    sys.exit()

# ---------------- LaTeX table ----------------
top = ['CaCaMean', 'CaCaMMLE']
order = sorted(range(len(methods)), key=lambda k: (methods[k] not in top, top.index(methods[k]) if methods[k] in top else 0, ranks[k].mean()))
lines = []
for k in order:
    cells = []
    for i in range(4):
        v = f'{A[k,i]:.4f}'
        if ranks[k, i] == 1:
            v = r'\textbf{' + v + '}'
        cells.append(f'{v} & ({ranks[k,i]})')
    name = LABEL[methods[k]] + (r'$^{\ast}$' if methods[k] not in BAYES else '')
    row = f'{name} & ' + ' & '.join(cells) + f' & {ranks[k].mean():.2f} \\\\'
    lines.append(row)
    if methods[k] == 'CaCaMMLE':
        lines.append(r'\midrule')
open(f'{OUT}/table_rows.tex', 'w').write('\n'.join(lines) + '\n')
maxse = SE.max()
open(f'{OUT}/table_meta.txt', 'w').write(f'max MC SE {maxse:.5f}\n')

# Figure: see fig_amse.py
