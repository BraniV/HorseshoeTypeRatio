"""Verify Table 2 against the original replication losses and write paired summaries."""
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIGNALS = ("Blocks", "Bumps", "Doppler", "HeaviSine")
# The manuscript prints the two Cauchy/Cauchy rules first and sorts the rest
# by mean rank. This maps the corresponding rows to the original method IDs.
MANUSCRIPT_METHODS = (
    "CaCaMMLE", "CaCaMean", "BAMS", "Horseshoe", "NCPmean2", "LNWS",
    "EBayesMed", "BayesThresh", "EBayesMean", "BiShrink", "RaisedCosine",
    "SmoothSCAD", "Epanechnikov", "SureShrink", "BayesShrink", "BlockJS",
    "VisuShrink",
)


def main():
    with (HERE.parents[1] / "tables/dj_amse.csv").open(newline="", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    if len(table) != len(MANUSCRIPT_METHODS):
        raise ValueError("Manuscript row count does not match the method list.")

    comparisons = []
    for signal in SIGNALS:
        with (
            np.load(HERE / f"res/main_{signal}.npz") as main_run,
            np.load(HERE / f"res/mmle_{signal}.npz") as caca_run,
            np.load(HERE / f"res/hswide_{signal}.npz") as hs_wide_run,
        ):
            methods = [*main_run["methods"], "CaCaMMLE"]
            losses = np.column_stack([main_run["MSE"], caca_run["MSE"]])
            assert losses.shape == (1000, 17)
            assert set(MANUSCRIPT_METHODS) == set(methods)
            amse = losses.mean(axis=0)
            ranks = np.argsort(np.argsort(amse)) + 1
            for row, name in zip(table, MANUSCRIPT_METHODS):
                i = methods.index(name)
                assert f"{amse[i]:.4f}" == row[f"{signal}_AMSE"], (signal, name, "AMSE")
                assert ranks[i] == int(row[f"{signal}_rank"]), (signal, name, "rank")
            for label, horseshoe in (
                ("standard", losses[:, methods.index("Horseshoe")]),
                ("wide", hs_wide_run["MSE"]),
            ):
                diff = caca_run["MSE"] - horseshoe
                comparisons.append((signal, label, f"{diff.mean():.10f}",
                                    f"{diff.std(ddof=1) / np.sqrt(diff.size):.10f}",
                                    f"{np.mean(diff < 0):.3f}"))
    out = HERE / "out/paired_vs_horseshoe.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["signal", "horseshoe_grid", "mean_mse_difference", "paired_mcse", "fraction_lower_mse"])
        writer.writerows(comparisons)
    print("All 68 Table 2 AMSE entries and ranks match. Wrote paired_vs_horseshoe.csv.")


if __name__ == "__main__":
    main()
