import unittest

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from ehrapy.anndata._constants import CATEGORICAL_TAG, CONTINUOUS_TAG, DATE_TAG, FEATURE_TYPE_KEY
from ehrapy.tools import rank_features_supervised


def test_continuous_prediction():
    target = np.random.rand(1000)
    X = np.stack((target, target * 2, [1] * 1000)).T

    adata = AnnData(X)
    adata.var_names = ["target", "feature1", "feature2"]
    adata.var[FEATURE_TYPE_KEY] = [CONTINUOUS_TAG] * 3

    for model in ["regression", "svm", "rf"]:
        rank_features_supervised(adata, "target", model, "all")
        assert "feature_importances" in adata.var
        assert adata.var["feature_importances"]["feature1"] > 0
        assert adata.var["feature_importances"]["feature2"] == 0
        assert pd.isna(adata.var["feature_importances"]["target"])


def test_categorical_prediction():
    target = np.random.randint(2, size=1000)
    X = np.stack((target, target, [1] * 1000)).T

    adata = AnnData(X)
    adata.var_names = ["target", "feature1", "feature2"]
    adata.var[FEATURE_TYPE_KEY] = [CATEGORICAL_TAG] * 3

    for model in ["regression", "svm", "rf"]:
        rank_features_supervised(adata, "target", model, "all")
        assert "feature_importances" in adata.var
        assert adata.var["feature_importances"]["feature1"] > 0
        assert adata.var["feature_importances"]["feature2"] == 0
        assert pd.isna(adata.var["feature_importances"]["target"])


def test_multiclass_prediction():
    target = np.random.randint(4, size=1000)
    X = np.stack((target, target, [1] * 1000)).T

    adata = AnnData(X)
    adata.var_names = ["target", "feature1", "feature2"]
    adata.var[FEATURE_TYPE_KEY] = [CATEGORICAL_TAG] * 3

    rank_features_supervised(adata, "target", "rf", "all")
    assert "feature_importances" in adata.var
    assert adata.var["feature_importances"]["feature1"] > 0
    assert adata.var["feature_importances"]["feature2"] == 0
    assert pd.isna(adata.var["feature_importances"]["target"])

    for invalid_model in ["regression", "svm"]:
        with pytest.raises(ValueError) as excinfo:
            rank_features_supervised(adata, "target", invalid_model, "all")
        assert str(excinfo.value).startswith("Feature target has more than two categories.")
