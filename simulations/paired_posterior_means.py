"""Independent reconstruction of the paper's paired posterior-mean comparison.

The simulation design follows Section 8. Seeds and numerical grids are explicit
choices for this reconstruction. This script does not implement all 17 rules.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import ndtri

N = 1024
COARSE_LEVEL = 3
SIGNALS = ("Blocks", "Bumps", "Doppler", "HeaviSine")
METHODS = ("caucau_fixed", "caucau_mmle", "horseshoe_mmle", "horseshoe_wide_mmle")
# Analysis filters follow the periodic transform used in the archived pilot.
HAAR = [1, 1]
DB3 = [
    .3326705529500826, .8068915093110926, .4598775021184916,
    -.1350110200102546, -.08544127388202666, .03522629188570954,
]
SYM4 = [
    -.07576571478950221, -.029635527646002493, .497618667632775,
    .8037387518051321, .29785779560530606, -.09921954357663353,
    -.012603967262031304, .032223100604051466,
]
FILTERS = dict(zip(SIGNALS, (HAAR, DB3, SYM4, SYM4)))


def wavelet_matrix(signal):
    h = np.asarray(FILTERS[signal], dtype=float)
    h /= np.linalg.norm(h)
    g = (h * (-1.0) ** np.arange(1, h.size + 1))[::-1]
    matrix = sparse.eye(2**COARSE_LEVEL, format="csr")
    for n in (16, 32, 64, 128, 256, 512, 1024):
        rows = np.repeat(np.arange(n // 2), h.size)
        taps = np.tile(np.arange(h.size), n // 2)
        cols = (2 * rows + taps) % n
        low = sparse.csr_matrix((h[taps], (rows, cols)), shape=(n // 2, n))
        high = sparse.csr_matrix((g[taps], (rows, cols)), shape=(n // 2, n))
        matrix = sparse.vstack((matrix @ low, high), format="csr")
    return matrix


def test_signal(name):
    t = np.arange(1, N + 1) / (N + 1)
    locations = np.array([.10, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
    if name == "Blocks":
        heights = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
        f = heights @ (t[None, :] >= locations[:, None])
    elif name == "Bumps":
        heights = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
        widths = np.array([.005, .005, .006, .010, .010, .030, .010, .005, .008, .005, .005])
        f = heights @ (1 + np.abs((t - locations[:, None]) / widths[:, None])) ** -4
    elif name == "Doppler":
        f = np.sqrt(t * (1 - t)) * np.sin(2 * np.pi * 1.05 / (t + .05))
    else:
        f = 4 * np.sin(4 * np.pi * t) - np.sign(t - .3) - np.sign(.72 - t)
    return 5 * f / f.std()


def lookup(taus, zgrid, family):
    # Integrate on the log absolute coefficient (Ca/Ca) or log local scale
    # (horseshoe) in noise SD units. Shift by log(tau) in the prior weights.
    u = np.linspace(-27, 9, 4501)
    t = u - np.log(taus)[:, None]
    if family == "caucau":
        weights = np.full_like(t, 2 / np.pi**2)
        np.divide(2 * t, np.pi**2 * np.sinh(t), out=weights, where=np.abs(t) > 1e-12)
    else:
        weights = 1 / (np.pi * np.cosh(t))
    weights *= u[1] - u[0]
    weights[:, [0, -1]] *= .5
    marginal = np.empty((len(taus), len(zgrid)))
    mean = np.empty_like(marginal)
    a = np.exp(u)[:, None]
    for start in range(0, len(zgrid), 128):
        z = zgrid[None, start:start + 128]
        if family == "caucau":
            positive = np.exp(-.5 * (z - a)**2)
            negative = np.exp(-.5 * (z + a)**2)
            den = weights @ (positive + negative)
            num = weights @ (a * (positive - negative))
        else:
            variance = 1 + a**2
            kernel = np.exp(-.5 * z**2 / variance) / np.sqrt(variance)
            den = weights @ kernel
            num = weights @ (kernel * z * (1 - 1 / variance))
        marginal[:, start:start + z.size] = np.log(den)
        mean[:, start:start + z.size] = num / den
    return marginal, mean


def fitted_mean(z, zgrid, log_marginal, means):
    scores = np.stack([np.interp(z, zgrid, row).sum(axis=1) for row in log_marginal])
    best = scores.argmax(axis=0)
    estimate = np.empty_like(z)
    for index in np.unique(best):
        rows = best == index
        estimate[rows] = np.interp(z[rows], zgrid, means[index])
    return estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--tau-grid-size", type=int, default=61)
    parser.add_argument("--output", type=Path, default=Path("runs/paired"))
    args = parser.parse_args()
    if args.replications < 2 or args.tau_grid_size < 2 or args.seed < 0:
        parser.error("use at least 2 replications and grid points, and a nonnegative seed")

    args.output.mkdir(parents=True, exist_ok=True)
    instances = {}
    zmax = 0
    for name, seed in zip(SIGNALS, np.random.SeedSequence(args.seed).spawn(len(SIGNALS))):
        wavelet = wavelet_matrix(name)
        f = test_signal(name)
        y = f + np.random.default_rng(seed).standard_normal((args.replications, N))
        d = (wavelet @ y.T).T
        fine = d[:, N // 2:]
        sigma = np.median(np.abs(fine - np.median(fine, axis=1, keepdims=True)), axis=1) / ndtri(.75)
        zmax = max(zmax, np.max(np.abs(d) / sigma[:, None]))
        instances[name] = wavelet, f, y, d, sigma[:, None]
    zgrid = np.arange(0, int(np.ceil(zmax)) + .04, .04)
    est = {name: np.repeat(vals[3][None, :, :], len(METHODS), axis=0) for name, vals in instances.items()}
    for j in range(COARSE_LEVEL, 10):
        n = 2**j
        band = slice(n, 2*n)
        narrow = np.geomspace(1/n, 1, args.tau_grid_size)
        wide = np.geomspace(1/n, 16, args.tau_grid_size)
        fixed = [45 * 2**(-1.45*j)]
        cc_log, cc_means = lookup(np.r_[fixed, wide], zgrid, "caucau")
        hs_log, hs_means = lookup(np.r_[narrow, wide], zgrid, "horseshoe")
        cut = len(narrow)
        for name, (_, _, _, d, sigma) in instances.items():
            z = np.abs(d[:, band]) / sigma
            sign_scale = np.sign(d[:, band]) * sigma
            est[name][0, :, band] = sign_scale * np.interp(z, zgrid, cc_means[0])
            est[name][1, :, band] = sign_scale * fitted_mean(z, zgrid, cc_log[1:], cc_means[1:])
            est[name][2, :, band] = sign_scale * fitted_mean(z, zgrid, hs_log[:cut], hs_means[:cut])
            est[name][3, :, band] = sign_scale * fitted_mean(z, zgrid, hs_log[cut:], hs_means[cut:])
        print(f"Completed level {j}", flush=True)

    losses = {}
    examples = {}
    with (args.output / "summary.csv").open("w", newline="") as f:
        csv_out = csv.writer(f)
        csv_out.writerow(["signal", "method", "mean_mse", "mcse", "paired_difference_vs_horseshoe", "paired_mcse"])
        for name, (wavelet, truth, observed, _, _) in instances.items():
            loss = np.mean((est[name] - (wavelet @ truth)[None, None, :])**2, axis=-1)
            assert np.isfinite(loss).all()
            losses[name] = loss
            se_scale = np.sqrt(args.replications)
            for method, values in zip(METHODS, loss):
                differ = values - loss[2]
                csv_out.writerow([name, method, values.mean(), values.std(ddof=1)/se_scale,
                                  differ.mean(), differ.std(ddof=1)/se_scale])
            pick = np.argsort(loss[0])[len(loss[0]) // 2]
            examples[name] = np.stack((truth, observed[pick], wavelet.T @ est[name][0, pick]))
    np.savez_compressed(args.output / "replication_mse.npz", method=np.asarray(METHODS), **losses)
    np.savez_compressed(args.output / "median_rank_examples.npz", **examples)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
