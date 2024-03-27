import unittest

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from ehrapy.tools import feature_importances


def test_continuous_prediction():
    target = np.random.rand(1000)
    X = np.stack((target, target * 2, [1] * 1000)).T
    adata = AnnData(X)
    adata.var_names = ["target", "feature1", "feature2"]

    for model in ["regression", "svm", "rf"]:
        feature_importances(adata, "target", "continuous", model, "all")
        assert "feature_importances" in adata.var
        assert adata.var["feature_importances"]["feature1"] > 0
        assert adata.var["feature_importances"]["feature2"] == 0
        assert pd.isna(adata.var["feature_importances"]["target"])


def test_categorical_prediction():
    target = np.random.randint(2, size=1000)
    X = np.stack((target, target, [1] * 1000)).T

    adata = AnnData(X)
    adata.var_names = ["target", "feature1", "feature2"]

    for model in ["regression", "svm", "rf"]:
        feature_importances(adata, "target", "categorical", model, "all")
        assert "feature_importances" in adata.var
        assert adata.var["feature_importances"]["feature1"] > 0
        assert adata.var["feature_importances"]["feature2"] == 0
        assert pd.isna(adata.var["feature_importances"]["target"])


def test_multiclass_prediction():
    target = np.random.randint(4, size=1000)
    X = np.stack((target, target, [1] * 1000)).T

    adata = AnnData(X)
    adata.var_names = ["target", "feature1", "feature2"]

    feature_importances(adata, "target", "categorical", "rf", "all")
    assert "feature_importances" in adata.var
    assert adata.var["feature_importances"]["feature1"] > 0
    assert adata.var["feature_importances"]["feature2"] == 0
    assert pd.isna(adata.var["feature_importances"]["target"])

    for invalid_model in ["regression", "svm"]:
        with pytest.raises(ValueError) as excinfo:
            feature_importances(adata, "target", "categorical", invalid_model, "all")
        assert str(excinfo.value).startswith("Feature target has more than two categories.")
