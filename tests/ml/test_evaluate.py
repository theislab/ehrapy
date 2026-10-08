import ehrdata as ed
import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from ehrapy.ml._evaluate import _resample


def _calibrated(n_obs: int = 2000, *, seed: int = 0) -> ed.EHRData:
    """Binary labels drawn with the predicted probabilities, in two groups and with three observations per patient."""
    rng = np.random.default_rng(seed)
    probability = rng.uniform(0.05, 0.95, n_obs)
    obs = pd.DataFrame(
        {
            "y": rng.binomial(1, probability),
            "prediction": probability,
            "group": rng.choice(["a", "b"], n_obs),
            "patient": np.arange(n_obs) // 3,
        },
        index=[str(i) for i in range(n_obs)],
    )
    return ed.EHRData(obs=obs)


def test_evaluate_binary():
    edata = _calibrated()
    task = ep.ml.Task("y")

    result = ep.ml.evaluate(edata, task, split=None, n_bootstrap=200)
    assert list(result.index) == [
        "auroc",
        "auprc",
        "brier",
        "ece",
        "calibration_intercept",
        "calibration_slope",
        "sensitivity",
        "specificity",
        "f1",
        "positive_rate",
    ]
    unbiased = result.drop("ece")
    assert (unbiased["ci_lower"] <= unbiased["value"]).all()
    assert (unbiased["value"] <= unbiased["ci_upper"]).all()
    assert result.loc["calibration_intercept", "ci_lower"] < 0 < result.loc["calibration_intercept", "ci_upper"]
    assert result.loc["calibration_slope", "ci_lower"] < 1 < result.loc["calibration_slope", "ci_upper"]
    assert result.loc["ece", "value"] < 0.05
    pd.testing.assert_frame_equal(result, ep.ml.evaluate(edata, task, split=None, n_bootstrap=200))

    edata.obs["prediction"] = edata.obs["y"]
    perfect = ep.ml.evaluate(edata, task, split=None, n_bootstrap=0)
    assert perfect.loc[["auroc", "auprc", "sensitivity", "specificity", "f1"], "value"].tolist() == [1] * 5
    with pytest.raises(KeyError, match="Predict first"):
        ep.ml.evaluate(edata, task, key="missing")


def test_evaluate_subgroups():
    edata = _calibrated()
    edata.obs.loc[edata.obs["group"] == "b", "prediction"] = 0.9

    result = ep.ml.evaluate(edata, ep.ml.Task("y"), split=None, groupby="group", n_bootstrap=20)

    assert result.index.names == ["group", "metric"]
    assert set(result.index.get_level_values("group")) == {"a", "b", "difference"}
    difference = result.loc["difference", "value"]
    assert difference["positive_rate"] == pytest.approx(
        1 - (edata.obs.query("group == 'a'")["prediction"] >= 0.5).mean()
    )
    assert difference["equalized_odds"] == max(difference["sensitivity"], difference["specificity"])


def test_evaluate_resamples_patients():
    patients = np.repeat(np.arange(5), [1, 2, 3, 4, 5])
    rows = _resample(patients, np.random.default_rng(0))

    drawn = patients[rows]
    assert all((drawn == patient).sum() % (patients == patient).sum() == 0 for patient in np.unique(drawn))
    edata = _calibrated()
    by_row, by_patient = (
        ep.ml.evaluate(edata, ep.ml.Task("y"), split=None, patient_key=key, n_bootstrap=50) for key in (None, "patient")
    )
    pd.testing.assert_series_equal(by_row["value"], by_patient["value"])
    assert not by_row["ci_lower"].equals(by_patient["ci_lower"])


def test_evaluate_multiclass_and_multilabel():
    y = np.array([0, 1, 2, 2, 1, 0])
    probabilities = np.eye(3)[y] * 0.7 + 0.1
    obs = pd.DataFrame({"y": y, "a": y == 0, "b": y == 1}, index=[str(i) for i in range(6)])
    edata = ed.EHRData(obs=obs, obsm={"prediction": probabilities, "labels": probabilities[:, :2]})

    multiclass = ep.ml.evaluate(edata, ep.ml.Task("y", kind="multiclass"), split=None, n_bootstrap=0)
    multilabel = ep.ml.evaluate(
        edata, ep.ml.Task(["a", "b"], kind="multilabel"), key="labels", split=None, n_bootstrap=0
    )

    assert (multiclass["value"] == 1).all()
    assert (multilabel["value"] == 1).all()


def test_evaluate_survival():
    rng = np.random.default_rng(0)
    risk = rng.normal(size=100)
    obs = pd.DataFrame(
        {"time": np.argsort(np.argsort(-risk)) + 1.0, "event": True, "prediction": risk},
        index=[str(i) for i in range(100)],
    )
    edata = ed.EHRData(obs=obs)

    result = ep.ml.evaluate(edata, ep.ml.Task("time", kind="survival", event="event"), split=None, n_bootstrap=0)
    assert result.loc["c_index", "value"] == 1
