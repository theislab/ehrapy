import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import ehrapy as ep
from tests.ml.conftest import TASKS, longitudinal


@pytest.fixture(scope="module")
def edata():
    edata = longitudinal(1000)
    ep.ml.split(edata, fractions=(0.4, 0.3, 0.3))
    return edata


@pytest.mark.parametrize("method", ["platt", "isotonic", "temperature"])
def test_calibrate_improves_calibration(edata, method):
    edata.obs["rare"] = (edata.obs["value"] > 1.5).astype(int)
    task = ep.ml.Task("rare", prediction_time=6, observation_window=4)
    predictor = ep.ml.fit(edata, task, model=LogisticRegression(class_weight="balanced"))
    uncalibrated = ep.ml.predict(edata, predictor, copy=True)
    calibrated = ep.ml.predict(edata, ep.ml.calibrate(edata, predictor, method=method), copy=True)

    brier, auroc = (
        [ep.ml.evaluate(data, task, n_bootstrap=0).loc[metric, "value"] for data in (uncalibrated, calibrated)]
        for metric in ("brier", "auroc")
    )
    assert brier[1] < brier[0]
    if method != "isotonic":
        assert auroc[1] == pytest.approx(auroc[0])


def test_temperature_scaling_of_multiclass_predictions(edata):
    predictor = ep.ml.fit(edata, TASKS["multiclass"], model="random_forest")
    calibrated = ep.ml.calibrate(edata, predictor, method="temperature")
    ep.ml.predict(edata, calibrated)

    np.testing.assert_allclose(edata.obsm["prediction"].sum(axis=1), 1)
    assert calibrated.calibrator.temperature != pytest.approx(1)
    with pytest.raises(ValueError, match="does not apply to multiclass"):
        ep.ml.calibrate(edata, predictor, method="platt")


@pytest.mark.parametrize("kind", ["binary", "multiclass", "regression"])
def test_conformal_prediction_covers_held_out_targets(edata, kind):
    task = TASKS[kind]
    predictor = ep.ml.conformalize(edata, ep.ml.fit(edata, task), alpha=0.1)
    ep.ml.predict(edata, predictor)

    held_out = (edata.obs["split"] == "held_out").to_numpy()
    if kind == "regression":
        lower, upper, value = (
            edata.obs[column].to_numpy()[held_out] for column in ("prediction_lower", "prediction_upper", task.label)
        )
        coverage = ((lower <= value) & (value <= upper)).mean()
    else:
        sets = edata.obsm["prediction_set"][held_out]
        coverage = sets.to_numpy()[
            np.arange(len(sets)), sets.columns.get_indexer(edata.obs[task.label][held_out].astype(str))
        ].mean()
    assert coverage > 0.85


def test_calibrate_drops_conformal_prediction(edata):
    predictor = ep.ml.conformalize(edata, ep.ml.fit(edata, TASKS["binary"]))

    assert ep.ml.calibrate(edata, predictor).conformal is None


@pytest.mark.parametrize("function", [ep.ml.calibrate, ep.ml.conformalize])
def test_uncertainty_raises_for_survival(edata, function):
    with pytest.raises(ValueError, match="survival"):
        function(edata, ep.ml.fit(edata, TASKS["survival"]))
