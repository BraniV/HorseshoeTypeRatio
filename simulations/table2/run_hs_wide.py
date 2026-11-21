import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import djbench as db
db.HS_TAUS = np.exp(np.linspace(np.log(1 / 512), np.log(16), 57))
signal, M, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
B = db.Bench.__new__(db.Bench)
B.N, B.L, B.J = 1024, 3, 10
f0 = db.dj_signal(signal, 1024); B.f = f0 * 5 / np.std(f0)
h, g = db.FILTERS[db.SIGNAL_FILTER[signal]]; B.W = db.dwt_matrix(h, g, 1024, 3)
hs = db.HorseshoeTables()
rng = np.random.default_rng(seed); MSE = np.zeros(M); TH = np.zeros((M, 10))
for m in range(M):
    y = B.f + rng.standard_normal(1024); w = B.W @ y
    sig = db.mad_sigma(w[db.lev(9)]); wh = w.copy()
    for j in range(3, 10):
        wh[db.lev(j)], TH[m, j] = hs.level(w[db.lev(j)], sig)
    MSE[m] = np.mean((B.W.T @ wh - B.f) ** 2)
np.savez(out, MSE=MSE, TAUHAT=TH); print('done', signal, MSE.mean(), np.median(TH, 0)[3:])
