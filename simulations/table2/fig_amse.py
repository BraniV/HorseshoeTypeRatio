import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import djbench as db
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SIGS = ['Blocks','Bumps','Doppler','HeaviSine']; SEEDS = [20261104,20261105,20261106,20261107]
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'font.family': 'serif', 'mathtext.fontset': 'cm'})
fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.0), constrained_layout=True)
t = np.arange(1, 1025) / 1025
for ax, s, seed in zip(axs.ravel(), SIGS, SEEDS):
    r = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res', f'main_{s}.npz')); k = list(r['methods']).index('CaCaMean'); mse = r['MSE'][:, k]
    amse = mse.mean(); im = int(np.argsort(mse)[len(mse) // 2])        # median-MSE replication
    f0 = db.dj_signal(s, 1024); f = f0 * 5 / np.std(f0)
    h, g = db.FILTERS[db.SIGNAL_FILTER[s]]; W = db.dwt_matrix(h, g, 1024, 3)
    rng = np.random.default_rng(seed)
    for _ in range(im): rng.standard_normal(1024)
    y = f + rng.standard_normal(1024); w = W @ y; sig = db.mad_sigma(w[db.lev(9)]); wh = w.copy()
    for j in range(3, 10):
        wh[db.lev(j)] = db.InterpRule(lambda z, tt=45 * 2.0 ** (-1.45 * j): db.caca_mean_std(z, tt))(w[db.lev(j)], sig)
    fh = W.T @ wh
    assert abs(np.mean((fh - f) ** 2) - mse[im]) < 1e-10, (s, np.mean((fh - f) ** 2), mse[im])
    ax.plot(t, y, '.', ms=1.4, color='#b9b8b3', zorder=1, label='noisy data')
    ax.plot(t, f, '-', lw=0.8, color='#0b0b0b', zorder=2, label='true signal')
    ax.plot(t, fh, '-', lw=1.0, color='#2a78d6', zorder=3, label='Ca/Ca posterior mean')
    ax.set_title(f'{s}:  AMSE {amse:.4f},  MSE {mse[im]:.4f}', fontsize=8.5, loc='left')
    ax.set_xlim(0, 1); ax.tick_params(length=2, width=0.5, labelsize=7, colors='#52514e')
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    print(s, f'AMSE {amse:.4f}', 'plotted rep', im, f'MSE {mse[im]:.4f}')
hl, ll = axs[0, 0].get_legend_handles_labels()
fig.legend(hl, ll, loc='outside upper center', ncol=3, fontsize=7.5, frameon=False, markerscale=5, handlelength=2.0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'DJ_CaCa_mean_examples.pdf')); fig.savefig(os.path.join(OUT, 'DJ_CaCa_mean_examples.png'), dpi=200)
