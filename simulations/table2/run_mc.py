import sys, time, json
import numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from djbench import Bench, METHODS

signal, M, snr, snr_type, seed, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6]
B = Bench(signal, snr=snr, snr_type=snr_type)
rng = np.random.default_rng(seed)
MSE = np.zeros((M, len(METHODS)))
SIG = np.zeros(M)
t0 = time.time()
keep = None
for m in range(M):
    y = B.f + rng.standard_normal(B.N)
    est, sig = B.run(y)
    SIG[m] = sig
    for k, name in enumerate(METHODS):
        MSE[m, k] = np.mean((est[name] - B.f) ** 2)
    if m == 0:
        keep = dict(y=y, f=B.f, caca=est['CaCaMean'], hs=est['Horseshoe'])
    if (m + 1) % 50 == 0:
        print(signal, m + 1, f'{time.time()-t0:.0f}s', flush=True)
np.savez(out, MSE=MSE, SIG=SIG, methods=np.array(METHODS), bams_bad=B.bams_bad, **keep)
print('done', signal, time.time() - t0, 'bams_bad', B.bams_bad)
