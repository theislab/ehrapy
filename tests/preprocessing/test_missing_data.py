from __future__ import annotations

import dask.array as da
import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import ehrapy as ep
from ehrapy._types import ARRAY_TYPES_NONNUMERIC, asarray
from tests.conftest import as_dense_dask_array


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_missing_data_mask_nan(array_type, missing_values_edata):
    edata = missing_values_edata
    edata.X = array_type(edata.X)

    ep.pp.missing_data_mask(edata)

    expected = np.array([[False, True, False], [True, True, False]])
    assert "missing_data_mask" in edata.layers
    assert np.array_equal(asarray(edata.layers["missing_data_mask"]), expected)


def test_missing_data_mask_preserves_dask(missing_values_edata):
    edata = missing_values_edata
    edata.X = as_dense_dask_array(edata.X)

    ep.pp.missing_data_mask(edata)

    assert isinstance(edata.layers["missing_data_mask"], da.Array)


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


@pytest.mark.parametrize("sparse_type", [sp.csr_array, sp.csc_array])
def test_missing_data_mask_sparse_nan_preserves_sparse(sparse_type):
    # NaN must live among the explicitly stored entries; implicit zeros
    # cannot encode NaN in CSR/CSC.
    dense = np.array([[np.nan, 0.0, 1.0], [0.0, np.nan, 0.0]], dtype=np.float64)
    edata = ed.EHRData(
        X=sparse_type(dense),
        obs=pd.DataFrame({"id": ["a", "b"]}),
        var=pd.DataFrame(index=["v1", "v2", "v3"]),
    )

    ep.pp.missing_data_mask(edata)

    mask = edata.layers["missing_data_mask"]
    assert sp.issparse(mask)
    assert isinstance(mask, sparse_type)
    assert mask.dtype == np.bool_
    expected = np.array([[True, False, False], [False, True, False]])
    assert np.array_equal(mask.toarray(), expected)


def test_missing_data_mask_empty_mask_values_is_noop_on_sparse():
    # Regression: an empty sentinel iterable must not trigger the sparse
    # NotImplementedError path — there are no sentinels to apply.
    X = sp.csr_array(np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float64))
    edata = ed.EHRData(
        X=X,
        obs=pd.DataFrame({"id": ["a", "b"]}),
        var=pd.DataFrame(index=["v1", "v2"]),
    )

    ep.pp.missing_data_mask(edata, mask_values=[])

    mask = edata.layers["missing_data_mask"]
    assert sp.issparse(mask)
    assert np.array_equal(mask.toarray(), np.array([[True, False], [False, False]]))


def test_missing_data_mask_sparse_sentinels_raises():
    X = sp.csr_array(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
    edata = ed.EHRData(
        X=X,
        obs=pd.DataFrame({"id": ["a", "b"]}),
        var=pd.DataFrame(index=["v1", "v2"]),
    )

    with pytest.raises(NotImplementedError, match="sparse arrays"):
        ep.pp.missing_data_mask(edata, mask_values=[0])
