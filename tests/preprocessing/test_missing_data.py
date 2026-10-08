from __future__ import annotations

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_missing_data_mask_nan(array_type, missing_values_edata):
    edata = missing_values_edata
    edata.X = array_type(edata.X)

    with forbid_dask_compute():
        ep.pp.missing_data_mask(edata)

    expected = np.array([[False, True, False], [True, True, False]])
    assert isinstance(edata.layers["missing_data_mask"], array_type.cls)
    assert isinstance(edata.X, array_type.cls)
    assert np.array_equal(to_dense(edata.layers["missing_data_mask"], to_cpu_memory=True), expected)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("mask_values", [None, [], [-1.0, 999.0], [0.0]])
def test_missing_data_mask_array_types(array_type, ndim, mask_values, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    shape = (20, 4) if ndim == 2 else (20, 4, 3)
    X = np.where(rng.random(shape) < 0.5, 0.0, rng.gamma(2, size=shape))
    X[rng.random(shape) < 0.1] = -1.0
    X[rng.random(shape) < 0.2] = np.nan
    X[:, 0] = np.nan
    X[:, 1] = 999.0
    expected = ep.pp.missing_data_mask(ed.EHRData(X=X), mask_values=mask_values, copy=True).layers["missing_data_mask"]
    edata = ed.EHRData(X=array_type(X))

    if array_type.flags & Flags.Sparse and mask_values == [0.0]:
        with pytest.raises(NotImplementedError, match="sparse arrays"):
            ep.pp.missing_data_mask(edata, mask_values=mask_values)
        return

    with forbid_dask_compute():
        result = ep.pp.missing_data_mask(edata, mask_values=mask_values, copy=True).layers["missing_data_mask"]

    assert isinstance(result, array_type.cls)
    assert result.dtype == bool
    if isinstance(result, CSBase):
        assert result.nnz == expected.sum()
    np.testing.assert_array_equal(to_dense(result, to_cpu_memory=True), expected)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_missing_data_mask_object_dtype(array_type):
    # EHR data often arrives as an object array because columns mix
    # numeric and categorical values; np.isnan would error on this dtype,
    # so the function must fall back to a dtype-agnostic check.
    X = np.array([[1.0, np.nan, "A"], ["B", 5.0, np.nan]], dtype=object)
    edata = ed.EHRData(
        X=array_type(X),
        obs=pd.DataFrame({"id": ["a", "b"]}),
        var=pd.DataFrame(index=["v1", "v2", "v3"]),
    )

    with forbid_dask_compute():
        ep.pp.missing_data_mask(edata)

    expected = np.array([[False, True, False], [False, False, True]])
    assert isinstance(edata.layers["missing_data_mask"], array_type.cls)
    assert np.array_equal(to_dense(edata.layers["missing_data_mask"], to_cpu_memory=True), expected)


def test_missing_data_mask_no_missing():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    edata = ed.EHRData(
        X=X,
        obs=pd.DataFrame({"id": ["a", "b"]}),
        var=pd.DataFrame(index=["v1", "v2"]),
    )

    ep.pp.missing_data_mask(edata)

    assert np.all(~edata.layers["missing_data_mask"])


def test_missing_data_mask_single_sentinel(missing_values_edata):
    ep.pp.missing_data_mask(missing_values_edata, mask_values=[-1])

    expected = np.array([[False, True, False], [True, True, False]])
    assert np.array_equal(missing_values_edata.layers["missing_data_mask"], expected)


def test_missing_data_mask_sentinel_present():
    X = np.array([[1.0, -1.0, 3.0], [0.0, 5.0, np.nan]], dtype=np.float64)
    edata = ed.EHRData(
        X=X,
        obs=pd.DataFrame({"id": ["a", "b"]}),
        var=pd.DataFrame(index=["v1", "v2", "v3"]),
    )

    ep.pp.missing_data_mask(edata, mask_values=[-1, 0])

    expected = np.array([[False, True, False], [True, False, True]])
    assert np.array_equal(edata.layers["missing_data_mask"], expected)


def test_missing_data_mask_multiple_sentinels():
    X = np.array([[999.0, 2.0], [-1.0, 0.0]], dtype=np.float64)
    edata = ed.EHRData(
        X=X,
        obs=pd.DataFrame({"id": ["a", "b"]}),
        var=pd.DataFrame(index=["v1", "v2"]),
    )

    ep.pp.missing_data_mask(edata, mask_values=[999, -1])

    expected = np.array([[True, False], [True, False]])
    assert np.array_equal(edata.layers["missing_data_mask"], expected)


def test_missing_data_mask_layer_parameter():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    layer_data = np.array([[np.nan, 2.0], [3.0, np.nan]], dtype=np.float64)
    edata = ed.EHRData(
        X=X,
        obs=pd.DataFrame({"id": ["a", "b"]}),
        var=pd.DataFrame(index=["v1", "v2"]),
        layers={"raw": layer_data},
    )

    ep.pp.missing_data_mask(edata, layer="raw")

    expected = np.array([[True, False], [False, True]])
    assert np.array_equal(edata.layers["missing_data_mask"], expected)


def test_missing_data_mask_copy_false_modifies_inplace(missing_values_edata):
    result = ep.pp.missing_data_mask(missing_values_edata)

    assert result is None
    assert "missing_data_mask" in missing_values_edata.layers


def test_missing_data_mask_copy_true_returns_new_object(missing_values_edata):
    result = ep.pp.missing_data_mask(missing_values_edata, copy=True)

    assert result is not None
    assert result is not missing_values_edata
    assert "missing_data_mask" in result.layers
    assert "missing_data_mask" not in missing_values_edata.layers


def test_missing_data_mask_custom_key(missing_values_edata):
    ep.pp.missing_data_mask(missing_values_edata, key_added="my_mask")

    assert "my_mask" in missing_values_edata.layers
    assert "missing_data_mask" not in missing_values_edata.layers
