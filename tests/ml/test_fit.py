import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute
from tests.ml.conftest import TASKS, longitudinal, static

MAIN_METRICS = {
    "binary": "auroc",
    "multiclass": "auroc_macro",
    "multilabel": "auroc_macro",
    "regression": "r2",
    "survival": "c_index",
}


@pytest.mark.parametrize(
    ("kind", "model", "minimum"),
    [
        *(
            (kind, model, 0.8)
            for kind in ("binary", "multiclass", "multilabel")
            for model in ("logistic", "gradient_boosting", "random_forest")
        ),
        *(("regression", model, 0.5) for model in ("linear", "gradient_boosting", "random_forest")),
        ("survival", "cox", 0.6),
    ],
)
def test_fit_predict_finds_signal(kind, model, minimum):
    edata = longitudinal()
    ep.ml.split(edata)
    task = TASKS[kind]
    predictor = ep.ml.fit(edata, task, model=model, obs_keys=["sex"])
    ep.ml.predict(edata, predictor)

    assert predictor.feature_names[-2:] == ["sex_f", "sex_m"]
    if kind in {"multiclass", "multilabel"}:
        assert list(edata.obsm["prediction"].columns) == [str(c) for c in predictor.classes]
    if kind != "multilabel":
        assert edata.obs["prediction"].notna().all()
    assert ep.ml.evaluate(edata, task, n_bootstrap=0).loc[MAIN_METRICS[kind], "value"] > minimum


def test_multiclass_predicts_most_probable_class():
    edata = longitudinal()
    ep.ml.split(edata)
    ep.ml.predict(edata, ep.ml.fit(edata, TASKS["multiclass"], model="logistic"))

    probabilities = edata.obsm["prediction"]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert (edata.obs["prediction"].astype(str) == probabilities.idxmax(axis=1)).all()


def test_features_end_before_prediction_time_and_gap():
    edata = longitudinal()
    ep.ml.split(edata)
    leaking = ep.ml.Task("label")
    ep.ml.predict(edata, ep.ml.fit(edata, leaking))
    assert ep.ml.evaluate(edata, leaking, n_bootstrap=0).loc["auroc", "value"] > 0.95

    predictor = ep.ml.fit(edata, ep.ml.Task("label", prediction_time=8, gap=2))
    ep.ml.predict(edata, predictor)
    changed = edata.copy()
    changed.X[:, :, 6:] = 0
    ep.ml.predict(changed, predictor)
    np.testing.assert_array_equal(changed.obs["prediction"], edata.obs["prediction"])


def test_fit_with_longitudinal_statistics():
    edata = longitudinal()
    ep.ml.split(edata, stratify="label")

    predictor = ep.ml.fit(edata, TASKS["binary"], model="logistic", statistics=["count", "std", "slope"])

    assert predictor.feature_names == [f"{var}_{stat}" for var in edata.var_names for stat in ("count", "std", "slope")]


def test_preprocessing_is_fit_on_train():
    edata = static()
    is_train = (edata.obs["split"] == "train").to_numpy()
    edata.X[~is_train, 1] += 100
    predictor = ep.ml.fit(edata, ep.ml.Task("y"), var_names=["noise"])

    train = edata.X[is_train, 1]
    imputer, scaler = predictor.preprocessing
    np.testing.assert_allclose(imputer.statistics_, [np.nanmedian(train)])
    np.testing.assert_allclose(scaler.mean_, [np.where(np.isnan(train), np.nanmedian(train), train).mean()])


def test_fit_excludes_targets_from_features():
    edata = static()

    assert ep.ml.fit(edata, ep.ml.Task("y")).var_names == ["x", "noise"]
    with pytest.raises(ValueError, match="cannot be features"):
        ep.ml.fit(edata, ep.ml.Task("y"), var_names=["x", "y"])
    with pytest.raises(ValueError, match="cannot be features"):
        ep.ml.fit(edata, ep.ml.Task("outcome", kind="regression"), obs_keys=["outcome"])


def test_fit_is_reproducible():
    edata = longitudinal()
    ep.ml.split(edata)
    predictions = [
        ep.ml.predict(edata, ep.ml.fit(edata, TASKS["binary"], max_train_obs=100, random_state=seed), copy=True).obs[
            "prediction"
        ]
        for seed in (0, 0, 1)
    ]

    pd.testing.assert_series_equal(predictions[0], predictions[1])
    assert not predictions[0].equals(predictions[2])


@pytest.mark.parametrize(
    ("edata", "task", "kwargs", "error", "match"),
    [
        pytest.param(longitudinal(), TASKS["binary"], {}, KeyError, "Split the observations first", id="no split"),
        pytest.param(static(), ep.ml.Task("y", prediction_time=1), {}, ValueError, "only apply", id="2D timepoints"),
        pytest.param(static(), ep.ml.Task("y"), {"model": "linear"}, ValueError, "Unknown `model`", id="model"),
        pytest.param(static(), ep.ml.Task("outcome"), {}, ValueError, "exactly two values", id="not binary"),
    ],
)
def test_fit_raises(edata, task, kwargs, error, match):
    with pytest.raises(error, match=match):
        ep.ml.fit(edata, task, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"label": "label", "kind": "ordinal"}, "`kind`"),
        ({"label": "label", "kind": "multilabel"}, "several label columns"),
        ({"label": ["a", "b"]}, "several label columns"),
        ({"label": "time", "kind": "survival"}, "`event`"),
    ],
)
def test_task_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ep.ml.Task(**kwargs)


def test_task_window_raises():
    edata = longitudinal()
    ep.ml.split(edata)

    with pytest.raises(ValueError, match="not within"):
        ep.ml.fit(edata, ep.ml.Task("label", prediction_time=4, observation_window=6))


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
def test_array_types(array_type, ndim):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    reference = longitudinal() if ndim == 3 else static()
    ep.ml.split(reference)
    task = TASKS["multiclass"] if ndim == 3 else ep.ml.Task("y")
    edata = ed.EHRData(X=array_type(reference.X), obs=reference.obs.copy(), var=reference.var)
    expected = ep.ml.predict(reference, ep.ml.fit(reference, task), copy=True)

    with forbid_dask_compute(allowed=1):
        predictor = ep.ml.fit(edata, task)
    with forbid_dask_compute(allowed=1):
        ep.ml.predict(edata, predictor)

    actual, wanted = ((data.obsm if ndim == 3 else data.obs)["prediction"] for data in (edata, expected))
    np.testing.assert_allclose(actual, wanted)


def test_physionet2012_mortality():
    edata = ed.dt.physionet2012()
    ep.ml.split(edata, stratify="In-hospital_death")
    task = ep.ml.Task("In-hospital_death")
    ep.ml.predict(edata, ep.ml.fit(edata, task, obs_keys=["Age", "Gender", "ICUType"]))

    result = ep.ml.evaluate(edata, task, n_bootstrap=20)
    assert result.loc["auroc", "value"] > 0.8
    assert result.loc["auprc", "value"] > 0.4
