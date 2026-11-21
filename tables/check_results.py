"""Check manuscript ranks and reproduce the archived pilot summary from its losses."""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ("Blocks", "Bumps", "Doppler", "HeaviSine")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


table = read_csv(ROOT / "tables/dj_amse.csv")
assert len(table) == 17
for signal in SIGNALS:
    ordered = sorted(table, key=lambda row: float(row[f"{signal}_AMSE"]))
    assert [int(row[f"{signal}_rank"]) for row in ordered] == list(range(1, 18))
for row in table:
    assert float(row["mean_rank"]) == sum(int(row[f"{signal}_rank"]) for signal in SIGNALS) / 4

pilot = ROOT / "archive/pilot"
expected = read_csv(pilot / "CauCau_DJ_M1000_AMSE.csv")
with np.load(pilot / "CauCau_DJ_M1000_replication_MSE.npz") as data:
    assert data["method"].tolist() == [row["Method"] for row in expected]
    for signal in SIGNALS:
        loss = data[signal]
        assert loss.shape == (1000, len(expected))
        for statistic, values in [("AMSE", loss.mean(axis=0)), ("SD_MSE", loss.std(axis=0, ddof=1))]:
            reported = [float(row[f"{signal}_{statistic}"]) for row in expected]
            np.testing.assert_allclose(values, reported, atol=5.1e-9, rtol=0)
print("Verified all Table 2 ranks and all archived pilot AMSEs and replication SDs.")
