from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyampute.exploration.mcar_statistical_tests import MCARTest


def _make_mcar(n_obs: int, n_vars: int, missing_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_obs, n_vars))
    miss = rng.random((n_obs, n_vars)) < missing_rate
    data[miss] = np.nan
    cols = [f"v{i}" for i in range(n_vars)]
    return pd.DataFrame(data, columns=cols)


def _make_mar(n_obs: int, n_vars: int, missing_pct: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_obs, n_vars))
    threshold = np.percentile(data[:, 0], missing_pct * 100)
    miss = data[:, 0] < threshold
    data[miss, -1] = np.nan
    cols = [f"v{i}" for i in range(n_vars)]
    return pd.DataFrame(data, columns=cols)


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "data" / "preprocessing" / "mcar_refs"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "mcar_small": _make_mcar(100, 10, 0.10, seed=42),
        "mar_small": _make_mar(100, 10, 0.10, seed=42),
        "mcar_medium_high_missing": _make_mcar(900, 50, 0.50, seed=7),
        "ttest_mar": _make_mar(200, 8, 0.20, seed=99),
    }

    little = MCARTest(method="little")
    ttest = MCARTest(method="ttest")

    little_expected: dict[str, float] = {}

    for name, df in scenarios.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        little_expected[name] = float(little(df))

    ttest_matrix = ttest(scenarios["ttest_mar"])
    ttest_matrix.to_csv(out_dir / "ttest_mar_expected.csv")

    with (out_dir / "little_expected.json").open("w", encoding="utf-8") as f:
        json.dump(little_expected, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
