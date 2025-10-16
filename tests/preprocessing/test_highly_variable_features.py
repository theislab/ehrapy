import ehrdata as ed
import numpy as np
import pytest
import requests

import ehrapy as ep
from ehrapy.preprocessing._highly_variable_features import highly_variable_features


def test_highly_variable_features_3D_edata(edata_blob_small):
    edata_blob_small.X = np.abs(edata_blob_small.X)
    edata_blob_small.layers["layer_2"] = np.abs(edata_blob_small.layers["layer_2"])
    edata_blob_small.layers["R_layer"] = np.abs(edata_blob_small.layers["R_layer"])
    highly_variable_features(edata_blob_small, span=1, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        highly_variable_features(edata_blob_small, span=1, layer="R_layer")


def test_highly_variable_features(clean_up_plots):
    try:
        adata = ed.dt.diabetes_130_fairlearn()
        adata = ep.pp.encode(adata, autodetect=True)
    except requests.exceptions.HTTPError as e:
        if "403" in str(e):
            pytest.skip("Dataset download failed with 403 Forbidden")
        raise

    ep.pp.knn_impute(adata)
    highly_variable_features(adata)

    assert "highly_variable" in adata.var.columns
    assert "highly_variable_rank" in adata.var.columns
    assert "means" in adata.var.columns
    assert "variances" in adata.var.columns
    assert "variances_norm" in adata.var.columns

    adata = ep.dt.dermatology(encoded=True)
    ep.pp.knn_impute(adata)
    highly_variable_features(adata, top_features_percentage=0.5)
    assert adata.var["highly_variable"].sum() == 17
