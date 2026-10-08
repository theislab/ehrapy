import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
import scipy.stats.mstats
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from fast_array_utils.conv import to_dense
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute


def test_winsorize_var(mimic_2_10):
    winsorized_edata = ep.pp.winsorize(mimic_2_10, var_names=["age"], limits=[0.2, 0.2], copy=True)
    expected = np.array(
        [71.43198, 64.92076, 36.5, 44.49191, 25.41667, 36.54657, 25.41667, 71.43198, 71.43198, 25.41667]
    ).reshape((10, 1))

    np.testing.assert_allclose(np.array(winsorized_edata[:, "age"].X, dtype=np.float32), expected)


def test_winsorized_obs(mimic_2_10):
    to_winsorize_obs = ed.move_to_obs(mimic_2_10, "age", copy=True)
    winsorized_edata = ep.pp.winsorize(to_winsorize_obs, obs_cols=["age"], limits=[0.2, 0.2], copy=True)
    expected = np.array(
        [71.43198, 64.92076, 36.5, 44.49191, 25.41667, 36.54657, 25.41667, 71.43198, 71.43198, 25.41667]
    )

    np.testing.assert_allclose(np.array(winsorized_edata.obs["age"]), expected)


@pytest.mark.parametrize("inclusive", [(True, True), (False, False)])
def test_winsorize_matches_scipy_ignoring_missing_values(rng, inclusive):
    x = rng.normal(size=40)
    x[rng.random(40) < 0.25] = np.nan
    edata = ed.EHRData(X=x[:, None].copy())

    ep.pp.winsorize(edata, var_names=edata.var_names, limits=(0.1, 0.2), inclusive=inclusive)

    valid = ~np.isnan(x)
    expected = scipy.stats.mstats.winsorize(x[valid], limits=(0.1, 0.2), inclusive=inclusive)
    np.testing.assert_array_equal(edata.X[valid, 0], expected)
    assert np.isnan(edata.X[~valid, 0]).all()


def test_winsorize_3D_pools_timepoints(rng):
    X = rng.normal(size=(20, 3, 4))
    edata = ed.EHRData(shape=(20, 3), layers={DEFAULT_TEM_LAYER_NAME: X.copy()})
    flat = ed.EHRData(X=np.moveaxis(X, 1, 2).reshape(-1, 3))

    ep.pp.winsorize(edata, var_names=edata.var_names, limits=(0.1, 0.1), layer=DEFAULT_TEM_LAYER_NAME)
    ep.pp.winsorize(flat, var_names=flat.var_names, limits=(0.1, 0.1))

    np.testing.assert_array_equal(edata.layers[DEFAULT_TEM_LAYER_NAME], np.moveaxis(flat.X.reshape(20, 4, 3), 1, 2))


def test_winsorize_invalid_limits(mimic_2_10):
    with pytest.raises(ValueError, match="between 0 and 1"):
        ep.pp.winsorize(mimic_2_10, var_names=["age"], limits=(0.1, 1.5))


def test_clip_var(mimic_2_10):
    age_before = mimic_2_10[:, "age"].X.copy()
    clipped_edata = ep.pp.clip_quantile(mimic_2_10, var_names=["age"], limits=(25, 50), copy=True)
    expected = np.array([50, 50, 36.5, 44.49191, 25, 36.54657, 25, 50, 50, 25.41667]).reshape((10, 1))

    np.testing.assert_allclose(np.array(clipped_edata[:, "age"].X, dtype=np.float32), expected)
    np.testing.assert_array_equal(mimic_2_10[:, "age"].X, age_before)


def test_clip_obs(mimic_2_10):
    to_clip_obs = ed.move_to_obs(mimic_2_10, "age", copy=True)
    clipped_edata = ep.pp.clip_quantile(to_clip_obs, obs_cols=["age"], limits=(25, 50), copy=True)
    expected = np.array([50, 50, 36.5, 44.49191, 25, 36.54657, 25, 50, 50, 25.41667], dtype=np.float32)
    np.testing.assert_allclose(np.array(clipped_edata.obs["age"].values.astype(np.float32)), expected)


OUTLIER_FUNCTIONS = [
    pytest.param(ep.pp.winsorize, {"limits": (0.1, 0.2)}, True, id="winsorize"),
    pytest.param(ep.pp.winsorize, {"limits": (0.9, 0.0)}, False, id="winsorize-bounds-exclude-zero"),
    pytest.param(ep.pp.clip_quantile, {"limits": (-1.0, 2.0)}, True, id="clip"),
    pytest.param(ep.pp.clip_quantile, {"limits": (0.5, 2.0)}, False, id="clip-limits-exclude-zero"),
]


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(("func", "kwargs", "sparse_support"), OUTLIER_FUNCTIONS)
def test_outliers_array_types(array_type, ndim, func, kwargs, sparse_support, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    shape = (20, 5) if ndim == 2 else (20, 5, 3)
    X = np.where(rng.random(shape) < 0.6, 0, rng.gamma(2, size=shape))
    X[rng.random(shape) < 0.1] = np.nan
    X[:, 1] = 3.0
    X[:, 2] = np.nan
    X[:, 3] *= np.where(rng.random(X[:, 3].shape) < 0.5, -1, 1)
    X[:, 4] = 10.0
    # the last variable is not selected and must stay unchanged
    kwargs = {**kwargs, "var_names": ["0", "1", "2", "3"], "layer": DEFAULT_TEM_LAYER_NAME}

    def make_edata(X):
        return ed.EHRData(shape=X.shape[:2], layers={DEFAULT_TEM_LAYER_NAME: X})

    expected = func(make_edata(X.copy()), copy=True, **kwargs).layers[DEFAULT_TEM_LAYER_NAME]
    edata = make_edata(array_type(X))

    if array_type.flags & Flags.Sparse and not sparse_support:
        with pytest.raises(NotImplementedError):
            to_dense(func(edata, copy=True, **kwargs).layers[DEFAULT_TEM_LAYER_NAME], to_cpu_memory=True)
        return

    with forbid_dask_compute():
        result = func(edata, copy=True, **kwargs).layers[DEFAULT_TEM_LAYER_NAME]

    assert isinstance(result, array_type.cls)
    if array_type.flags & Flags.Dask:
        assert type(result._meta) is type(edata.layers[DEFAULT_TEM_LAYER_NAME]._meta)
    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, equal_nan=True)
    np.testing.assert_array_equal(to_dense(edata.layers[DEFAULT_TEM_LAYER_NAME], to_cpu_memory=True), X)

    with forbid_dask_compute():
        func(edata, **kwargs)

    assert isinstance(edata.layers[DEFAULT_TEM_LAYER_NAME], array_type.cls)
    np.testing.assert_allclose(
        to_dense(edata.layers[DEFAULT_TEM_LAYER_NAME], to_cpu_memory=True), expected, equal_nan=True
    )


def test_winsorize_default_limits_cut_one_percent_per_side():
    values = np.arange(100, dtype=float)
    edata = ed.EHRData(X=values[:, None].copy(), var=pd.DataFrame(index=["value"]))

    ep.pp.winsorize(edata, var_names=["value"])

    np.testing.assert_array_equal(edata.X[:, 0], np.clip(values, 1, 98))
