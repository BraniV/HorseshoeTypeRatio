"""Regenerate numerical counterparts of Figures 1 and 2 from the formulas."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import beta, roots_legendre


def uniform_cauchy_curves(d):
    # Integrate over positive and negative theta together. Split at the
    # origin and around the likelihood peak to resolve both contributions.
    def integrands(x):
        inv = 1 / x**2
        log_term = np.log1p(inv)
        prior = log_term / (2*np.pi)
        latent = 1 - inv / ((1 + inv) * log_term)
        positive = np.exp(-.5*(x - d)**2)
        negative = np.exp(-.5*(x + d)**2)
        return (prior*(positive + negative),
                prior*latent*(positive + negative),
                prior*((d - x)*positive + (d + x)*negative))

    points = sorted({0., max(0., d-8), d, d+8, d+12})
    norm, latent, correction = [
        quad(lambda x: integrands(x)[i], 0, points[-1], points=points[1:-1],
             epsabs=1e-12)[0]
        for i in range(3)
    ]
    return d**2 * latent / norm, d * correction / norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/scalar_figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "mathtext.fontset": "stix"})

    # In the standardized horseshoe posterior, u=sqrt(1-kappa)
    # eliminates the integrable singularity at kappa=1.
    d = np.linspace(0, 12, 241)
    nodes, weights = roots_legendre(160)
    kappa = 1 - ((nodes + 1) / 2)**2
    weighted_posterior = (weights / 2) * np.exp(-.5 * d[:, None]**2 * kappa)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(d, (weighted_posterior @ kappa) / weighted_posterior.sum(axis=1))
    ax.set(xlabel=r"$|d|$", ylabel=r"$E(\kappa\mid d)$")
    fig.tight_layout()
    fig.savefig(args.output / "ekappad.pdf")
    plt.close(fig)

    grid = np.linspace(.002, .998, 600)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for a, b, style in ((.5, .5, "-"), (.25, .5, "--"), (.5, .25, ":")):
        ax.plot(grid, grid**(a-1)*(1-grid)**(b-1)/beta(a,b), style,
                label=rf"$\mathrm{{Beta}}({a:g},{b:g})$")
    ax.set(xlabel=r"$\kappa_Y$", ylabel=r"$h_Y(\kappa)$", ylim=(0, 5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "KAPPA01_THREE_HORSESHOES.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(grid, 1 / (np.pi * np.sqrt(grid * (1 - grid))))
    ax.set(xlabel=r"$\kappa$", ylabel="Prior density")
    fig.tight_layout()
    fig.savefig(args.output / "hsshoe.pdf")
    plt.close(fig)

    d = np.geomspace(.3, 150, 140)
    latent, effective = np.asarray([uniform_cauchy_curves(x) for x in d]).T
    np.testing.assert_allclose([latent[-1], effective[-1]], [.5, 2], rtol=.005)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.semilogx(d, latent, label=r"$d^2 E(\kappa_Y^\star\mid d)$")
    ax.semilogx(d, effective, "--", label=r"$d^2\kappa_{\rm eff}(d)$")
    for limit in (.5, 2):
        ax.axhline(limit, color=".6", linewidth=.7)
    ax.set(xlabel=r"$|d|$", ylabel="Normalized shrinkage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "KAPPA02_UNIFORM_CAUCHY.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
