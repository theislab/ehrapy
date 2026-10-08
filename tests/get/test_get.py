import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute


def test_obs_df():
    edata = ed.dt.mimic_2()
    edata = ep.pp.encode(edata, autodetect=True)
    df = ep.get.obs_df(edata, keys=["age"])
    # since pass through of scanpy, merely testing shape
    assert df.shape == (len(edata), 1)


def test_obs_df_feature_symbols():
    edata = ed.dt.mimic_2()
    edata.var["symbol"] = [f"symbol_{name}" for name in edata.var_names]
    df = ep.get.obs_df(edata, keys=["symbol_age"], feature_symbols="symbol")
    assert df.columns.tolist() == ["symbol_age"]
    np.testing.assert_array_equal(df["symbol_age"].to_numpy(), edata[:, "age"].X.ravel())


def test_rank_features_groups_df():
    edata = ed.dt.mimic_2()
    edata = ep.pp.encode(edata, autodetect=True)
    ep.tl.rank_features_groups(edata, groupby="service_unit")
    df = ep.get.rank_features_groups_df(edata, group="FICU")
    # since pass through of scanpy, merely testing shape
    assert df.shape == (54, 5)


def test_var_df():
    edata = ed.dt.mimic_2()
    edata = ep.pp.encode(edata, autodetect=True)
    df = ep.get.var_df(edata, keys=["0", "1", "2", "3"])
    # since pass through of scanpy, merely testing shape
    assert df.shape == (len(edata.var), 4)


def test_obs_df_3d_obs_keys(edata_blobs_timeseries_small):
    df = ep.get.obs_df(edata_blobs_timeseries_small, keys=["cluster"], layer=DEFAULT_TEM_LAYER_NAME)

    pd.testing.assert_series_equal(df["cluster"], edata_blobs_timeseries_small.obs["cluster"])


def test_obs_df_var_df_3d_var_keys_raise(edata_blobs_timeseries_small):
    with pytest.raises(ValueError, match="only supports 2D data"):
        ep.get.obs_df(edata_blobs_timeseries_small, keys=["cluster", "feature_0"], layer=DEFAULT_TEM_LAYER_NAME)
    with pytest.raises(ValueError, match="only supports 2D data"):
        ep.get.var_df(edata_blobs_timeseries_small, keys=["0"], layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize(("getter", "keys"), [("obs_df", ["c", "group"]), ("var_df", ["p4", "p1"])])
def test_get_array_types(array_type, getter, keys, rng):
    X = np.where(rng.random((6, 3)) < 0.4, 0, rng.standard_normal((6, 3)))
    obs = pd.DataFrame({"group": list("aabbcc")}, index=[f"p{i}" for i in range(6)])
    var = pd.DataFrame(index=["a", "b", "c"])
    get = getattr(ep.get, getter)
    expected = get(ed.EHRData(X=X, obs=obs, var=var), keys=keys)
    edata = ed.EHRData(X=array_type(X), obs=obs, var=var)

    with forbid_dask_compute(allowed=1):
        result = get(edata, keys=keys)

    pd.testing.assert_frame_equal(result, expected)
