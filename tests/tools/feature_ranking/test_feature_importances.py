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
        rank_features_supervised(edata, "target", model=model, input_features="all")
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
        rank_features_supervised(edata, "target", model=model, input_features="all")
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

    rank_features_supervised(edata, "target", model="rf", input_features="all")
    assert "feature_importances" in edata.var
    assert edata.var["feature_importances"]["feature1"] > 0
    assert edata.var["feature_importances"]["feature2"] == 0
    assert pd.isna(edata.var["feature_importances"]["target"])

    for invalid_model in ["regression", "svm"]:
        with pytest.raises(ValueError) as excinfo:
            rank_features_supervised(edata, "target", model=invalid_model, input_features="all")
        assert str(excinfo.value).startswith("Feature target has more than two categories.")


def test_continuous_prediction_3D_edata(edata_blob_small):
    rank_features_supervised(edata_blob_small, "feature_9", model="regression", layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        rank_features_supervised(edata_blob_small, "feature_9", model="regression", layer=DEFAULT_TEM_LAYER_NAME)
