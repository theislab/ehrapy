import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute


def _longitudinal(n_obs: int = 300, *, seed: int = 0) -> ed.EHRData:
    """A label given by the mean of `signal` at timepoints 2 to 5, which `leak` reveals from timepoint 6 on."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs, 3, 10))
    label = X[:, 0, 2:6].mean(axis=1) + 0.2 * rng.normal(size=n_obs) > 0
    X[:, 1, 6:] = label[:, None]
    X[rng.random(X.shape) < 0.1] = np.nan
    obs = pd.DataFrame(
        {
            "label": label.astype(int),
            "patient": rng.integers(0, n_obs // 3, n_obs).astype(str),
            "sex": rng.choice(["f", "m"], n_obs),
            "admission": rng.permutation(n_obs),
        },
        index=[str(i) for i in range(n_obs)],
    )
    return ed.EHRData(X=X, obs=obs, var=pd.DataFrame(index=["signal", "leak", "noise"]))


def _static(n_obs: int = 300, *, seed: int = 0) -> ed.EHRData:
    """A label `y` that is a variable as well, and the numeric `outcome` that `x` determines."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_obs)
    y = (x + 0.5 * rng.normal(size=n_obs) > 0).astype(int)
    X = np.column_stack([x, rng.normal(size=n_obs), y]).astype(float)
    X[rng.random(X.shape) < 0.1] = np.nan
    obs = pd.DataFrame({"y": y, "outcome": 2 * x + 0.1 * rng.normal(size=n_obs)}, index=[str(i) for i in range(n_obs)])
    edata = ed.EHRData(X=X, obs=obs, var=pd.DataFrame(index=["x", "noise", "y"]))
    ep.ml.split(edata)
    return edata


TASK = ep.ml.Task("label", prediction_time=6, observation_window=4)


def test_split_fractions_and_reproducibility():
    edata = _longitudinal()
    ep.ml.split(edata)
    assert edata.obs["split"].value_counts().to_dict() == {"train": 210, "tuning": 45, "held_out": 45}

    other = ep.ml.split(edata, random_state=0, copy=True)
    pd.testing.assert_series_equal(other.obs["split"], edata.obs["split"])
    assert not ep.ml.split(edata, random_state=1, copy=True).obs["split"].equals(edata.obs["split"])


def test_split_groups_are_disjoint():
    edata = _longitudinal()
    ep.ml.split(edata, groupby="patient")

    assert edata.obs.groupby("patient")["split"].nunique().max() == 1


def test_split_stratify():
    edata = _longitudinal()
    ep.ml.split(edata, stratify="label")

    prevalence = edata.obs.groupby("split", observed=True)["label"].mean()
    np.testing.assert_allclose(prevalence, edata.obs["label"].mean(), atol=0.02)
    with pytest.raises(ValueError, match="differs within groups"):
        ep.ml.split(edata, groupby="patient", stratify="label")


def test_split_by_time():
    edata = _longitudinal()
    ep.ml.split(edata, time_key="admission")

    admission = edata.obs.groupby("split", observed=True)["admission"]
    assert admission.max()["train"] < admission.min()["tuning"]
    assert admission.max()["tuning"] < admission.min()["held_out"]


@pytest.mark.parametrize("model", ["logistic", "gradient_boosting"])
def test_fit_predict_finds_signal(model):
    edata = _longitudinal()
    ep.ml.split(edata, stratify="label")
    predictor = ep.ml.fit(edata, TASK, model=model, obs_keys=["sex"])
    ep.ml.predict(edata, predictor)

    assert predictor.feature_names[-2:] == ["sex_f", "sex_m"]
    assert edata.obs["prediction"].between(0, 1).all()
    assert ep.ml.evaluate(edata, TASK, n_bootstrap=0).loc["auroc", "value"] > 0.8


def test_features_end_before_prediction_time_and_gap():
    edata = _longitudinal()
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


def test_preprocessing_is_fit_on_train():
    edata = _static()
    is_train = (edata.obs["split"] == "train").to_numpy()
    edata.X[~is_train, 1] += 100
    predictor = ep.ml.fit(edata, ep.ml.Task("y"), var_names=["noise"])

    train = edata.X[is_train, 1]
    imputer, scaler, _ = predictor.pipeline
    np.testing.assert_allclose(imputer.statistics_, [np.nanmedian(train)])
    np.testing.assert_allclose(scaler.mean_, [np.where(np.isnan(train), np.nanmedian(train), train).mean()])


def test_fit_excludes_label_from_features():
    edata = _static()

    assert ep.ml.fit(edata, ep.ml.Task("y")).var_names == ["x", "noise"]
    with pytest.raises(ValueError, match="cannot be a feature"):
        ep.ml.fit(edata, ep.ml.Task("y"), var_names=["x", "y"])
    with pytest.raises(ValueError, match="cannot be a feature"):
        ep.ml.fit(edata, ep.ml.Task("outcome", kind="regression"), obs_keys=["outcome"])


def test_fit_is_reproducible():
    edata = _longitudinal()
    ep.ml.split(edata)
    predictions = [
        ep.ml.predict(edata, ep.ml.fit(edata, TASK, max_train_obs=100, random_state=seed), copy=True).obs["prediction"]
        for seed in (0, 0, 1)
    ]

    pd.testing.assert_series_equal(predictions[0], predictions[1])
    assert not predictions[0].equals(predictions[2])


@pytest.mark.parametrize("model", ["linear", "gradient_boosting"])
def test_regression(model):
    edata = _static()
    task = ep.ml.Task("outcome", kind="regression")
    ep.ml.predict(edata, ep.ml.fit(edata, task, model=model, var_names=["x", "noise"]))

    result = ep.ml.evaluate(edata, task, n_bootstrap=0)
    assert list(result.index) == ["mae", "rmse", "r2"]
    assert result.loc["r2", "value"] > 0.6


@pytest.mark.parametrize(
    ("edata", "task", "kwargs", "error", "match"),
    [
        pytest.param(_longitudinal(), TASK, {}, KeyError, "Split the observations first", id="no split"),
        pytest.param(_static(), ep.ml.Task("y", prediction_time=1), {}, ValueError, "only apply", id="2D timepoints"),
        pytest.param(_static(), ep.ml.Task("y"), {"model": "linear"}, ValueError, "Unknown `model`", id="model"),
        pytest.param(_static(), ep.ml.Task("outcome"), {}, ValueError, "exactly two values", id="not binary"),
    ],
)
def test_fit_raises(edata, task, kwargs, error, match):
    with pytest.raises(error, match=match):
        ep.ml.fit(edata, task, **kwargs)


def test_task_window_raises():
    edata = _longitudinal()
    ep.ml.split(edata)

    with pytest.raises(ValueError, match="not within"):
        ep.ml.fit(edata, ep.ml.Task("label", prediction_time=4, observation_window=6))
    with pytest.raises(ValueError, match="`kind`"):
        ep.ml.Task("label", kind="multiclass")


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
def test_array_types(array_type, ndim):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    reference = _longitudinal() if ndim == 3 else _static()
    ep.ml.split(reference)
    task = TASK if ndim == 3 else ep.ml.Task("y")
    edata = ed.EHRData(X=array_type(reference.X), obs=reference.obs.copy(), var=reference.var)
    expected = ep.ml.predict(reference, ep.ml.fit(reference, task), copy=True)

    with forbid_dask_compute(allowed=1):
        predictor = ep.ml.fit(edata, task)
    with forbid_dask_compute(allowed=1):
        ep.ml.predict(edata, predictor)

    np.testing.assert_allclose(edata.obs["prediction"], expected.obs["prediction"])


def test_evaluate():
    rng = np.random.default_rng(0)
    probability = rng.uniform(0.05, 0.95, 2000)
    obs = pd.DataFrame({"y": rng.binomial(1, probability), "group": rng.choice(["a", "b"], 2000)})
    obs["prediction"] = probability
    edata = ed.EHRData(obs=obs.set_index(obs.index.astype(str)))
    task = ep.ml.Task("y")

    result = ep.ml.evaluate(edata, task, split=None, n_bootstrap=200)
    assert (result["ci_lower"] <= result["value"]).all()
    assert (result["value"] <= result["ci_upper"]).all()
    assert result.loc["calibration_intercept", "ci_lower"] < 0 < result.loc["calibration_intercept", "ci_upper"]
    assert result.loc["calibration_slope", "ci_lower"] < 1 < result.loc["calibration_slope", "ci_upper"]
    pd.testing.assert_frame_equal(result, ep.ml.evaluate(edata, task, split=None, n_bootstrap=200))

    by_group = ep.ml.evaluate(edata, task, split=None, groupby="group", n_bootstrap=0)
    assert by_group.index.names == ["group", "metric"]
    assert by_group["ci_lower"].isna().all()

    edata.obs["prediction"] = edata.obs["y"]
    assert ep.ml.evaluate(edata, task, split=None, n_bootstrap=0).loc[["auroc", "auprc"], "value"].tolist() == [1, 1]
    with pytest.raises(KeyError, match="Predict first"):
        ep.ml.evaluate(edata, task, key="missing")


def test_physionet2012_mortality():
    edata = ed.dt.physionet2012()
    ep.ml.split(edata, stratify="In-hospital_death")
    task = ep.ml.Task("In-hospital_death")
    ep.ml.predict(edata, ep.ml.fit(edata, task, obs_keys=["Age", "Gender", "ICUType"]))

    result = ep.ml.evaluate(edata, task, n_bootstrap=20)
    assert result.loc["auroc", "value"] > 0.8
    assert result.loc["auprc", "value"] > 0.4
