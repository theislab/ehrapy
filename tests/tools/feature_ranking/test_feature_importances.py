import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import CATEGORICAL_TAG, DEFAULT_TEM_LAYER_NAME, FEATURE_TYPE_KEY, NUMERIC_TAG

from ehrapy.tools import rank_features_supervised


def test_continuous_prediction():
    target = np.random.default_rng().random(1000)
    X = np.stack((target, target * 2, [1] * 1000)).T

    edata = ed.EHRData(X)
    edata.var_names = ["target", "feature1", "feature2"]
    edata.var[FEATURE_TYPE_KEY] = [NUMERIC_TAG] * 3

    for model in ["regression", "svm", "rf"]:
        rank_features_supervised(edata, predicted_feature="target", model=model, var_names="all")
        assert "feature_importances" in edata.var
        assert edata.var["feature_importances"]["feature1"] > 0
        assert edata.var["feature_importances"]["feature2"] == 0
        assert pd.isna(edata.var["feature_importances"]["target"])


def test_categorical_prediction():
    target = np.random.default_rng().integers(2, size=1000)
    X = np.stack((target, target, [1] * 1000)).T.astype(np.float32)

    edata = ed.EHRData(X)
    edata.var_names = ["target", "feature1", "feature2"]
    edata.var[FEATURE_TYPE_KEY] = [CATEGORICAL_TAG] * 3

    for model in ["regression", "svm", "rf"]:
        rank_features_supervised(edata, predicted_feature="target", model=model, var_names="all")
        assert "feature_importances" in edata.var
        assert edata.var["feature_importances"]["feature1"] > 0
        assert edata.var["feature_importances"]["feature2"] == 0
        assert pd.isna(edata.var["feature_importances"]["target"])


def test_multiclass_prediction():
    target = np.random.default_rng().integers(4, size=1000)
    X = np.stack((target, target, [1] * 1000)).T.astype(np.float32)

    edata = ed.EHRData(X)
    edata.var_names = ["target", "feature1", "feature2"]
    edata.var[FEATURE_TYPE_KEY] = [CATEGORICAL_TAG] * 3

    rank_features_supervised(edata, predicted_feature="target", model="rf", var_names="all")
    assert "feature_importances" in edata.var
    assert edata.var["feature_importances"]["feature1"] > 0
    assert edata.var["feature_importances"]["feature2"] == 0
    assert pd.isna(edata.var["feature_importances"]["target"])

    for invalid_model in ["regression", "svm"]:
        with pytest.raises(ValueError) as excinfo:
            rank_features_supervised(edata, predicted_feature="target", model=invalid_model, var_names="all")
        assert str(excinfo.value).startswith("Feature target has more than two categories.")


def test_continuous_prediction_3D_edata(edata_blob_small):
    rank_features_supervised(edata_blob_small, predicted_feature="feature_9", model="regression", layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        rank_features_supervised(
            edata_blob_small, predicted_feature="feature_9", model="regression", layer=DEFAULT_TEM_LAYER_NAME
        )


def test_var_names_subset():
    target = np.random.default_rng(0).random(100)
    edata = ed.EHRData(np.stack((target, target * 2, target * 3)).T)
    edata.var_names = ["target", "feature1", "feature2"]
    edata.var[FEATURE_TYPE_KEY] = [NUMERIC_TAG] * 3

    rank_features_supervised(edata, predicted_feature="target", model="regression", var_names=("feature1",))
    assert edata.var["feature_importances"]["feature1"] > 0
    assert pd.isna(edata.var["feature_importances"]["feature2"])


def test_copy():
    target = np.random.default_rng(0).random(100)
    edata = ed.EHRData(np.stack((target, target * 2)).T)
    edata.var_names = ["target", "feature1"]
    edata.var[FEATURE_TYPE_KEY] = [NUMERIC_TAG] * 2

    edata_copy = rank_features_supervised(edata, predicted_feature="target", model="regression", copy=True)
    assert "feature_importances" in edata_copy.var
    assert "feature_importances" not in edata.var

    assert rank_features_supervised(edata, predicted_feature="target", model="regression") is None
    assert "feature_importances" in edata.var


@pytest.mark.parametrize(("target_type", "metric"), [(NUMERIC_TAG, "r2"), (CATEGORICAL_TAG, "accuracy")])
def test_score_stored_in_uns(target_type, metric):
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, 100).astype(float)
    edata = ed.EHRData(np.stack((target, target + rng.normal(scale=0.1, size=100))).T)
    edata.var_names = ["target", "feature1"]
    edata.var[FEATURE_TYPE_KEY] = [target_type, NUMERIC_TAG]

    rank_features_supervised(edata, predicted_feature="target", model="regression", key_added="importances")

    result = edata.uns["importances"]
    assert result["predicted_feature"] == "target"
    assert result["metric"] == metric
    assert 0.5 < result["score"] <= 1
