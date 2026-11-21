"""Ca/Ca posterior mean with levelwise MMLE of tau_j (no tuning constants).
Re-generates exactly the same noisy realizations as run_mc.py (same seed)."""
import sys, time
import numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from djbench import Bench, lev, mad_sigma, TGRID, QCACA, PI

TAUS = np.exp(np.linspace(np.log(1 / 512), np.log(16), 57))
ZG = np.arange(0, 40 + 1e-9, 0.01)


def caca_logm_mean(z, tau):
    a = tau * np.exp(TGRID)[:, None]
    lm = np.empty(z.size); mu = np.empty(z.size)
    for s in range(0, z.size, 300):
        zz = z[s:s + 300][None, :]
        Wp = QCACA[:, None] * np.exp(-0.5 * (a - zz) ** 2)
        Wm = QCACA[:, None] * np.exp(-0.5 * (a + zz) ** 2)
        den = np.trapezoid(Wp + Wm, TGRID, axis=0)
        num = np.trapezoid(a * (Wp - Wm), TGRID, axis=0)
        lm[s:s + 300] = np.log(np.maximum(den, 1e-300)) - 0.5 * np.log(2 * PI)
        mu[s:s + 300] = np.maximum(num / np.maximum(den, 1e-300), 0)
    return lm, mu


TAB = [caca_logm_mean(ZG, t) for t in TAUS]


def ev(z, k):
    lm = np.interp(np.minimum(z, 40), ZG, TAB[k][0])
    mu = np.interp(np.minimum(z, 40), ZG, TAB[k][1])
    big = z > 40
    if big.any():
        lm[big], mu[big] = caca_logm_mean(z[big], TAUS[k])
    return lm, mu


signal, M, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
B = Bench(signal, snr=5.0, snr_type='sd')
rng = np.random.default_rng(seed)
MSE = np.zeros(M); TAUHAT = np.zeros((M, B.J))
for m in range(M):
    y = B.f + rng.standard_normal(B.N)
    w = B.W @ y
    sig = mad_sigma(w[lev(B.J - 1)])
    wh = w.copy()
    for j in range(B.L, B.J):
        d = w[lev(j)]; z = np.abs(d) / sig
        ks = [k for k, t in enumerate(TAUS) if t >= 1 / z.size - 1e-12]
        ll = [ev(z, k)[0].sum() for k in ks]
        kb = ks[int(np.argmax(ll))]
        TAUHAT[m, j] = TAUS[kb]
        wh[lev(j)] = sig * np.sign(d) * ev(z, kb)[1]
    MSE[m] = np.mean((B.W.T @ wh - B.f) ** 2)
np.savez(out, MSE=MSE, TAUHAT=TAUHAT)
print('done', signal, MSE.mean(), np.median(TAUHAT, 0))
