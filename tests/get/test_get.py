import ehrdata as ed
import pytest
from anndata import AnnData
from ehrdata import EHRData

import ehrapy as ep
from ehrapy._compat import _cast_edata_to_match_data_type


@pytest.mark.parametrize("data_type", [EHRData(), AnnData()])
def test_obs_df(data_type):
    edata = ed.dt.mimic_2()
    edata = ep.pp.encode(edata, autodetect=True)
    edata = _cast_edata_to_match_data_type(edata, data_type)
    df = ep.get.obs_df(edata, keys=["age"])
    # since pass through of scanpy, merely testing shape
    assert df.shape == (len(edata), 1)


@pytest.mark.parametrize("data_type", [EHRData(), AnnData()])
def test_rank_features_groups_df(data_type):
    edata = ed.dt.mimic_2()
    edata = ep.pp.encode(edata, autodetect=True)
    edata = _cast_edata_to_match_data_type(edata, data_type)
    ep.tl.rank_features_groups(edata, "service_unit")
    df = ep.get.rank_features_groups_df(edata, group="FICU")
    # since pass through of scanpy, merely testing shape
    assert df.shape == (54, 5)


@pytest.mark.parametrize("data_type", [EHRData(), AnnData()])
def test_var_df(data_type):
    edata = ed.dt.mimic_2()
    edata = ep.pp.encode(edata, autodetect=True)
    edata = _cast_edata_to_match_data_type(edata, data_type)
    df = ep.get.var_df(edata, keys=["0", "1", "2", "3"])
    # since pass through of scanpy, merely testing shape
    assert df.shape == (len(edata.var), 4)
