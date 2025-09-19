import pytest
from anndata import AnnData
from ehrdata import EHRData

import ehrapy as ep
from ehrapy._compat import _cast_adata_to_match_data_type


@pytest.mark.parametrize("data_type", [EHRData(), AnnData()])
def test_obs_df(data_type):
    adata = ep.dt.mimic_2(encoded=True)
    adata = _cast_adata_to_match_data_type(adata, data_type)
    df = ep.get.obs_df(adata, keys=["age"])
    # since pass through of scanpy, merely testing shape
    assert df.shape == (len(adata), 1)


@pytest.mark.parametrize("data_type", [EHRData(), AnnData()])
def test_rank_features_groups_df(data_type):
    adata = ep.dt.mimic_2(encoded=True)
    adata = _cast_adata_to_match_data_type(adata, data_type)
    ep.tl.rank_features_groups(adata, "service_unit")
    df = ep.get.rank_features_groups_df(adata, group="FICU")
    # since pass through of scanpy, merely testing shape
    assert df.shape == (54, 5)


@pytest.mark.parametrize("data_type", [EHRData(), AnnData()])
def test_var_df(data_type):
    adata = ep.dt.mimic_2(encoded=True)
    adata = _cast_adata_to_match_data_type(adata, data_type)
    df = ep.get.var_df(adata, keys=["0", "1", "2", "3"])
    # since pass through of scanpy, merely testing shape
    assert df.shape == (len(adata.var), 4)
