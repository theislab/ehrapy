import numpy as np
import pytest

import ehrapy as ep
from tests.ml.conftest import TASKS, longitudinal


@pytest.mark.parametrize("model", ["gradient_boosting", "gru"])
def test_permutation_importance_finds_signal(model):
    if model == "gru":
        pytest.importorskip("torch")
        model = ep.ml.GRU(trainer=ep.ml.Trainer(max_epochs=30, batch_size=32, device="cpu"))
    edata = longitudinal()
    ep.ml.split(edata)
    predictor = ep.ml.fit(edata, TASKS["binary"], model=model, var_names=["signal", "noise"])

    ep.ml.permutation_importance(edata, predictor, n_repeats=3)

    importance = edata.var["permutation_importance"]
    assert importance["signal"] > 0.2
    assert abs(importance["noise"]) < 0.05
    assert np.isnan(importance["leak"])
    assert edata.varm["permutation_importance"].shape == (3, 3)


def test_permutation_importance_is_reproducible():
    edata = longitudinal()
    ep.ml.split(edata)
    predictor = ep.ml.fit(edata, TASKS["regression"])

    first, second = (
        ep.ml.permutation_importance(edata, predictor, copy=True).varm["permutation_importance"] for _ in range(2)
    )
    np.testing.assert_array_equal(first, second)
