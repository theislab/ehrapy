import dask.array as da
import ehrdata as ed
import holoviews as hv
import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from tests.conftest import forbid_dask_compute

torch = pytest.importorskip("torch")

QUICK = ep.ml.Trainer(max_epochs=2, batch_size=16, device="cpu")
TRAINER = ep.ml.Trainer(max_epochs=30, batch_size=32, device="cpu")
SIGNAL = ["signal", "noise"]


def _rolling(n_obs: int = 200, *, n_timepoints: int = 12, seed: int = 0) -> ed.EHRData:
    """A `label` and `value` at every timepoint given by the mean of `signal` at the two timepoints before it."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n_obs, n_timepoints))
    score = np.full((n_obs, n_timepoints), np.nan)
    score[:, 2:] = (signal[:, :-2] + signal[:, 1:-1]) / 2 + 0.2 * rng.normal(size=(n_obs, n_timepoints - 2))
    X = np.stack([signal, rng.normal(size=signal.shape), np.where(np.isnan(score), np.nan, score > 0), score], axis=1)
    X[:, :2][rng.random((n_obs, 2, n_timepoints)) < 0.1] = np.nan
    obs = pd.DataFrame(
        {"sex": rng.choice(["f", "m"], n_obs), "patient": [str(i) for i in range(n_obs)]},
        index=[str(i) for i in range(n_obs)],
    )
    edata = ed.EHRData(X=X, obs=obs, var=pd.DataFrame(index=["signal", "noise", "label", "value"]))
    ep.ml.split(edata)
    return edata


def _model(name: str, trainer: ep.ml.Trainer = QUICK) -> str | ep.ml.DeepModel:
    models = {
        "gru": ep.ml.GRU,
        "lstm": ep.ml.LSTM,
        "grud": ep.ml.GRUD,
        "tcn": ep.ml.TCN,
        "transformer": ep.ml.Transformer,
    }
    return models[name](trainer=trainer) if name in models else name


@pytest.mark.parametrize(
    ("name", "window"), [("gradient_boosting", 2), ("mlp", 2), ("gru", None), ("transformer", None)]
)
def test_rolling_finds_signal(name, window):
    edata = _rolling()
    task = ep.ml.Task("label", rolling=True, observation_window=window)
    model = ep.ml.MLP(trainer=TRAINER) if name == "mlp" else _model(name, TRAINER)
    ep.ml.predict(edata, ep.ml.fit(edata, task, model=model, var_names=SIGNAL, obs_keys=["sex"]))

    predictions = edata.obsm["prediction"]
    assert predictions.shape == (200, 12)
    assert np.isnan(predictions[:, 0]).all()
    assert not np.isnan(predictions[:, 1:]).any()
    assert ep.ml.evaluate(edata, task, n_bootstrap=0).loc["auroc", "value"] > 0.8


def test_rolling_regression():
    edata = _rolling()
    task = ep.ml.Task("value", kind="regression", rolling=True, observation_window=2)
    ep.ml.predict(edata, ep.ml.fit(edata, task, var_names=SIGNAL))

    assert ep.ml.evaluate(edata, task, n_bootstrap=0).loc["r2", "value"] > 0.5


@pytest.mark.parametrize("name", ["gradient_boosting", "gru", "lstm", "grud", "tcn", "transformer"])
def test_rolling_predictions_only_read_earlier_timepoints(name):
    edata = _rolling(60)
    task = ep.ml.Task("label", rolling=True, gap=2)
    predictor = ep.ml.fit(edata, task, model=_model(name), var_names=SIGNAL)
    ep.ml.predict(edata, predictor)
    changed = edata.copy()
    changed.X[:, :2, 6:] = 0
    ep.ml.predict(changed, predictor)

    assert np.isnan(edata.obsm["prediction"][:, :3]).all()
    np.testing.assert_array_equal(changed.obsm["prediction"][:, :9], edata.obsm["prediction"][:, :9])
    assert not np.array_equal(changed.obsm["prediction"][:, 9:], edata.obsm["prediction"][:, 9:])


@pytest.mark.parametrize(
    ("task", "kwargs", "match"),
    [
        pytest.param(
            ep.ml.Task("label", rolling=True), {"var_names": ["signal", "label"]}, "cannot be features", id="label"
        ),
        pytest.param(ep.ml.Task("label", rolling=True), {"model": ep.ml.RETAIN(trainer=QUICK)}, "RETAIN", id="retain"),
        pytest.param(
            ep.ml.Task("label", rolling=True, observation_window=3),
            {"model": ep.ml.GRU(trainer=QUICK)},
            "observation_window",
            id="window",
        ),
        pytest.param(ep.ml.Task("value", rolling=True), {}, "marks positive timepoints", id="not 0 and 1"),
    ],
)
def test_rolling_fit_raises(task, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ep.ml.fit(_rolling(60), task, **kwargs)


def test_rolling_task_raises():
    with pytest.raises(ValueError, match="rolling task"):
        ep.ml.Task("label", kind="multiclass", rolling=True)
    with pytest.raises(ValueError, match="rolling task"):
        ep.ml.Task("label", rolling=True, prediction_time=3)


def test_rolling_excludes_label_from_features():
    edata = _rolling(60)

    assert ep.ml.fit(edata, ep.ml.Task("label", rolling=True, observation_window=2)).var_names == [
        "signal",
        "noise",
        "value",
    ]


def test_rolling_evaluate_resamples_patients():
    edata = _rolling()
    task = ep.ml.Task("label", rolling=True, observation_window=2)
    ep.ml.predict(edata, ep.ml.fit(edata, task, var_names=SIGNAL))

    by_observation, by_patient = (
        ep.ml.evaluate(edata, task, patient_key=key, n_bootstrap=50) for key in (None, "patient")
    )
    pd.testing.assert_frame_equal(by_observation, by_patient)
    by_sex = ep.ml.evaluate(edata, task, groupby="sex", n_bootstrap=10)
    assert set(by_sex.index.get_level_values("sex")) == {"f", "m", "difference"}
    assert isinstance(ep.pl.prediction_performance(edata, task), hv.Layout)


def test_rolling_calibration_and_conformal_prediction():
    edata = _rolling(400)
    ep.ml.split(edata, fractions=(0.4, 0.3, 0.3))
    task = ep.ml.Task("label", rolling=True, observation_window=2)
    predictor = ep.ml.calibrate(edata, ep.ml.fit(edata, task, var_names=SIGNAL), method="platt")
    ep.ml.predict(edata, ep.ml.conformalize(edata, predictor, alpha=0.1))

    sets = edata.obsm["prediction_set"]
    assert sets.shape == (400, 12, 2)
    held_out = (edata.obs["split"] == "held_out").to_numpy()[:, None] & ~np.isnan(edata.X[:, 2])
    labels = edata.X[:, 2][held_out].astype(int)
    assert sets[held_out][np.arange(len(labels)), labels].mean() > 0.85

    regression = ep.ml.Task("value", kind="regression", rolling=True, observation_window=2)
    ep.ml.predict(edata, ep.ml.conformalize(edata, ep.ml.fit(edata, regression, var_names=SIGNAL)))
    assert (edata.obsm["prediction_lower"] <= edata.obsm["prediction_upper"]).sum() == (
        ~np.isnan(edata.obsm["prediction"])
    ).sum()


@pytest.mark.parametrize("name", ["gradient_boosting", "gru"])
def test_rolling_permutation_importance(name):
    edata = _rolling()
    task = ep.ml.Task("label", rolling=True, observation_window=None if name == "gru" else 2)
    predictor = ep.ml.fit(edata, task, model=_model(name, TRAINER), var_names=SIGNAL)

    ep.ml.permutation_importance(edata, predictor, n_repeats=2)

    importance = edata.var["permutation_importance"]
    assert importance["signal"] > 0.2 > abs(importance["noise"])


@pytest.mark.parametrize("name", ["gradient_boosting", "gru"])
def test_rolling_dask(name):
    reference = _rolling(60)
    edata = ed.EHRData(X=da.from_array(reference.X, chunks=(20, 4, 12)), obs=reference.obs.copy(), var=reference.var)
    task = ep.ml.Task("label", rolling=True, observation_window=None if name == "gru" else 2)
    expected = ep.ml.predict(reference, ep.ml.fit(reference, task, model=_model(name), var_names=SIGNAL), copy=True)

    with forbid_dask_compute(allowed=1):
        predictor = ep.ml.fit(edata, task, model=_model(name), var_names=SIGNAL)
    with forbid_dask_compute(allowed=1):
        ep.ml.predict(edata, predictor)

    np.testing.assert_allclose(edata.obsm["prediction"], expected.obsm["prediction"], rtol=1e-5)
