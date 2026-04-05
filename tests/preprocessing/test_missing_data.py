from __future__ import annotations

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from ehrapy._types import ARRAY_TYPES_NONNUMERIC
from tests.conftest import as_dense_dask_array


@pytest.fixture
def edata_with_nan():
    X = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]], dtype=np.float64)
    return ed.EHRData(
        X=X,
        obs=pd.DataFrame({"id": ["a", "b"]}),
        var=pd.DataFrame(index=["v1", "v2", "v3"]),
    )


class TestMissingDataMaskDefault:
    """Test default NaN masking."""

    @pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
    def test_nan_masking(self, array_type, edata_with_nan):
        edata = edata_with_nan
        edata.X = array_type(edata.X)

        ep.pp.missing_data_mask(edata)

        expected = np.array([[False, True, False], [True, False, True]])
        assert "missing_data_mask" in edata.layers
        assert np.array_equal(edata.layers["missing_data_mask"], expected)

    def test_no_missing(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        edata = ed.EHRData(
            X=X,
            obs=pd.DataFrame({"id": ["a", "b"]}),
            var=pd.DataFrame(index=["v1", "v2"]),
        )

        ep.pp.missing_data_mask(edata)

        assert np.all(~edata.layers["missing_data_mask"])


class TestMissingDataMaskCustomValues:
    """Test custom mask_values parameter."""

    def test_single_sentinel(self, edata_with_nan):
        edata = edata_with_nan

        ep.pp.missing_data_mask(edata, mask_values=[-1])

        # Only NaN values should be masked (no -1 in the data)
        expected = np.array([[False, True, False], [True, False, True]])
        assert np.array_equal(edata.layers["missing_data_mask"], expected)

    def test_sentinel_present(self):
        X = np.array([[1.0, -1.0, 3.0], [0.0, 5.0, np.nan]], dtype=np.float64)
        edata = ed.EHRData(
            X=X,
            obs=pd.DataFrame({"id": ["a", "b"]}),
            var=pd.DataFrame(index=["v1", "v2", "v3"]),
        )

        ep.pp.missing_data_mask(edata, mask_values=[-1, 0])

        expected = np.array([[False, True, False], [True, False, True]])
        assert np.array_equal(edata.layers["missing_data_mask"], expected)

    def test_multiple_sentinels(self):
        X = np.array([[999.0, 2.0], [-1.0, 0.0]], dtype=np.float64)
        edata = ed.EHRData(
            X=X,
            obs=pd.DataFrame({"id": ["a", "b"]}),
            var=pd.DataFrame(index=["v1", "v2"]),
        )

        ep.pp.missing_data_mask(edata, mask_values=[999, -1])

        expected = np.array([[True, False], [True, False]])
        assert np.array_equal(edata.layers["missing_data_mask"], expected)


class TestMissingDataMaskLayer:
    """Test layer parameter."""

    def test_layer_parameter(self):
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


class TestMissingDataMaskCopy:
    """Test copy parameter."""

    def test_copy_false_modifies_inplace(self, edata_with_nan):
        result = ep.pp.missing_data_mask(edata_with_nan)

        assert result is None
        assert "missing_data_mask" in edata_with_nan.layers

    def test_copy_true_returns_new_object(self, edata_with_nan):
        result = ep.pp.missing_data_mask(edata_with_nan, copy=True)

        assert result is not None
        assert result is not edata_with_nan
        assert "missing_data_mask" in result.layers
        assert "missing_data_mask" not in edata_with_nan.layers


class TestMissingDataMaskKeyAdded:
    """Test key_added parameter."""

    def test_custom_key(self, edata_with_nan):
        ep.pp.missing_data_mask(edata_with_nan, key_added="my_mask")

        assert "my_mask" in edata_with_nan.layers
        assert "missing_data_mask" not in edata_with_nan.layers
