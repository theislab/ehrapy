import ehrdata as ed
import numpy as np
import pytest
import requests
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep
from ehrapy.preprocessing._highly_variable_features import highly_variable_features


def test_highly_variable_features_3D_edata(edata_blob_small):
    edata_blob_small.X = np.abs(edata_blob_small.X)
    edata_blob_small.layers["layer_2"] = np.abs(edata_blob_small.layers["layer_2"])
    edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME] = np.abs(edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME])
    highly_variable_features(edata_blob_small, span=1, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        highly_variable_features(edata_blob_small, span=1, layer=DEFAULT_TEM_LAYER_NAME)


def test_highly_variable_features(clean_up_plots):
    try:
        edata = ed.dt.diabetes_130_fairlearn()
        edata = ep.pp.encode(edata, autodetect=True)
    except requests.exceptions.HTTPError as e:
        if "403" in str(e):
            pytest.skip("Dataset download failed with 403 Forbidden")
        raise

    ep.pp.knn_impute(edata)
    highly_variable_features(edata)

    assert "highly_variable" in edata.var.columns
    assert "highly_variable_rank" in edata.var.columns
    assert "means" in edata.var.columns
    assert "variances" in edata.var.columns
    assert "variances_norm" in edata.var.columns

    edata = ed.dt.diabetes_130_fairlearn()
    edata = ep.pp.encode(edata, autodetect=True)
    ep.pp.knn_impute(edata)
    highly_variable_features(edata, top_features_percentage=0.5)
    assert edata.var["highly_variable"].sum() == 31
