import dask.array as da
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


@pytest.mark.parametrize("statistic", ["first", "last", "mean", "max", "slope"])
def test_obs_df_3d_var_keys(edata_blobs_timeseries_small, statistic):
    edata = edata_blobs_timeseries_small
    edata.layers[DEFAULT_TEM_LAYER_NAME][0, 1, :3] = np.nan
    summary = ep.pp.summarize_measurements(
        edata, layer=DEFAULT_TEM_LAYER_NAME, var_names=["feature_1", "feature_0"], statistics=[statistic]
    )

    df = ep.get.obs_df(
        edata, keys=["cluster", "feature_1", "feature_0"], layer=DEFAULT_TEM_LAYER_NAME, statistic=statistic
    )

    assert df.columns.tolist() == ["cluster", "feature_1", "feature_0"]
    pd.testing.assert_series_equal(df["cluster"], edata.obs["cluster"])
    np.testing.assert_array_equal(df[["feature_1", "feature_0"]].to_numpy(), summary.X)


def test_obs_df_3d_timepoint(edata_blobs_timeseries_small):
    edata = edata_blobs_timeseries_small

    df = ep.get.obs_df(edata[:, :, [4]], keys=["feature_2"], layer=DEFAULT_TEM_LAYER_NAME)

    np.testing.assert_array_equal(df["feature_2"], edata.layers[DEFAULT_TEM_LAYER_NAME][:, 2, 4])


def test_obs_df_3d_dask(edata_blobs_timeseries_small):
    edata = edata_blobs_timeseries_small
    expected = ep.get.obs_df(edata, keys=["feature_0", "cluster"], layer=DEFAULT_TEM_LAYER_NAME, statistic="mean")
    edata.layers[DEFAULT_TEM_LAYER_NAME] = da.from_array(edata.layers[DEFAULT_TEM_LAYER_NAME], chunks=(5, -1, -1))

    with forbid_dask_compute(allowed=1):
        result = ep.get.obs_df(edata, keys=["feature_0", "cluster"], layer=DEFAULT_TEM_LAYER_NAME, statistic="mean")

    pd.testing.assert_frame_equal(result, expected)


def test_var_df_3d_raises(edata_blobs_timeseries_small):
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
