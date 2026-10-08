import sys

import dask.array as da
import ehrdata as ed
import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from tests.conftest import forbid_dask_compute
from tests.ml.conftest import TASKS, longitudinal, static

torch = pytest.importorskip("torch")

MODELS = {
    "mlp": ep.ml.MLP,
    "gru": ep.ml.GRU,
    "lstm": ep.ml.LSTM,
    "grud": ep.ml.GRUD,
    "tcn": ep.ml.TCN,
    "transformer": ep.ml.Transformer,
    "retain": ep.ml.RETAIN,
}
QUICK = ep.ml.Trainer(max_epochs=2, batch_size=16, device="cpu")


def _split(n_obs: int) -> ed.EHRData:
    edata = longitudinal(n_obs)
    ep.ml.split(edata)
    return edata


@pytest.mark.parametrize("kind", TASKS)
@pytest.mark.parametrize("name", MODELS)
def test_deep_models_fit_every_kind(name, kind):
    edata = _split(60)
    predictor = ep.ml.fit(edata, TASKS[kind], model=MODELS[name](trainer=QUICK), obs_keys=["sex"])
    ep.ml.predict(edata, predictor)

    predictions = edata.obsm["prediction"] if kind == "multilabel" else edata.obs["prediction"]
    assert not pd.isna(np.asarray(predictions)).any()
    assert edata.obsm["X_prediction"].shape == (60, predictor.model.n_embedding)
    assert ("prediction_attention" in edata.obsm) == (name in {"transformer", "retain"})


@pytest.mark.parametrize("name", MODELS)
def test_deep_models_find_signal(name):
    edata = _split(300)
    trainer = ep.ml.Trainer(max_epochs=30, batch_size=32, device="cpu")
    ep.ml.predict(edata, ep.ml.fit(edata, TASKS["binary"], model=MODELS[name](trainer=trainer)))

    assert ep.ml.evaluate(edata, TASKS["binary"], n_bootstrap=0).loc["auroc", "value"] > 0.8


def test_attention_covers_the_observation_window():
    edata = _split(60)
    ep.ml.predict(edata, ep.ml.fit(edata, TASKS["binary"], model=ep.ml.RETAIN(trainer=QUICK)))

    attention = edata.obsm["prediction_attention"]
    assert list(attention.columns) == ["2", "3", "4", "5"]
    np.testing.assert_allclose(attention.sum(axis=1), 1, rtol=1e-5)


def test_training_stops_early():
    edata = _split(100)
    trainer = ep.ml.Trainer(max_epochs=500, patience=2, device="cpu")

    assert ep.ml.fit(edata, TASKS["binary"], model=ep.ml.MLP(trainer=trainer)).model.n_epochs < 500


def test_deep_models_are_reproducible():
    edata = _split(60)
    predictions = [
        ep.ml.predict(
            edata, ep.ml.fit(edata, TASKS["binary"], model=ep.ml.GRU(trainer=QUICK), random_state=seed), copy=True
        ).obs["prediction"]
        for seed in (0, 0, 1)
    ]

    pd.testing.assert_series_equal(predictions[0], predictions[1])
    assert not predictions[0].equals(predictions[2])


def test_class_weight_raises_rare_class_probabilities():
    edata = _split(100)
    task = ep.ml.Task("rare", prediction_time=6, observation_window=4)
    edata.obs["rare"] = (edata.obs["value"] > 1.5).astype(int)
    models = (ep.ml.MLP(trainer=ep.ml.Trainer(max_epochs=5, device="cpu", class_weight=w)) for w in (None, "balanced"))
    mean_predictions = [
        ep.ml.predict(edata, ep.ml.fit(edata, task, model=model), copy=True).obs["prediction"].mean()
        for model in models
    ]

    assert mean_predictions[1] > mean_predictions[0]


def test_torch_module():
    class Mean(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4)

        def forward(self, values, mask, time_since_observed, static):
            return self.linear(values.mean(dim=1))

    edata = _split(60)
    module = Mean()
    predictor = ep.ml.fit(edata, TASKS["regression"], model=module)
    ep.ml.predict(edata, predictor)

    assert edata.obsm["X_prediction"].shape == (60, 4)
    assert predictor.model.module.network is not module


def test_model_names():
    edata = _split(60)

    assert isinstance(ep.ml.fit(edata, TASKS["binary"], model="gru").model.module.network.rnn, torch.nn.GRU)


def test_models_of_time_series_need_longitudinal_data():
    with pytest.raises(ValueError, match="need longitudinal data"):
        ep.ml.fit(static(), ep.ml.Task("y"), model=ep.ml.GRU(trainer=QUICK))


def test_dask():
    reference = _split(60)
    edata = ed.EHRData(X=da.from_array(reference.X, chunks=(20, 3, 10)), obs=reference.obs.copy(), var=reference.var)
    expected = ep.ml.predict(
        reference, ep.ml.fit(reference, TASKS["binary"], model=ep.ml.GRU(trainer=QUICK)), copy=True
    )

    with forbid_dask_compute(allowed=1):
        predictor = ep.ml.fit(edata, TASKS["binary"], model=ep.ml.GRU(trainer=QUICK))
    with forbid_dask_compute(allowed=1):
        ep.ml.predict(edata, predictor)

    np.testing.assert_allclose(edata.obs["prediction"], expected.obs["prediction"], rtol=1e-5)
    np.testing.assert_allclose(edata.obsm["X_prediction"], expected.obsm["X_prediction"], rtol=1e-5)


def test_missing_torch_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ImportError, match=r"ehrapy\[ml\]"):
        ep.ml.fit(_split(60), TASKS["binary"], model="gru")
