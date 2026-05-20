from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import CATEGORICAL_TAG, DEFAULT_TEM_LAYER_NAME, FEATURE_TYPE_KEY, NUMERIC_TAG
from pandas import CategoricalDtype, DataFrame
from pandas.testing import assert_frame_equal

from ehrapy._compat import DaskArray
from ehrapy.preprocessing._encoding import _reorder_encodings, encode
from tests.conftest import ARRAY_TYPES_NONNUMERIC, TEST_DATA_PATH, as_dense_dask_array


def _convert_edata_arrays(edata, array_type):
    """Wrap ``edata.X`` and ``edata.layers['layer_2']`` with ``array_type``."""
    edata.X = array_type(edata.X)
    if "layer_2" in edata.layers:
        edata.layers["layer_2"] = array_type(edata.layers["layer_2"])
    return edata


CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{TEST_DATA_PATH}/encode"


def test_encode_3D_edata(edata_blob_small):
    encode(edata_blob_small, autodetect=True, layer="layer_2")
    # 3D longitudinal layers are now supported; the encoded layer keeps its time axis.
    n_time = edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME].shape[2]
    encoded = encode(edata_blob_small, autodetect=True, layer=DEFAULT_TEM_LAYER_NAME)
    encoded_layer = encoded.layers[DEFAULT_TEM_LAYER_NAME]
    assert encoded_layer.ndim == 3
    assert encoded_layer.shape[0] == edata_blob_small.n_obs
    assert encoded_layer.shape[2] == n_time


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_encode_3D_longitudinal_one_hot(edata_mini_3D_missing_values, array_type):
    """One-hot encode a 3D layer with categorical columns.

    The encoder must fit on values stacked across time so the category space is shared,
    the time axis is preserved, and ``obs`` stores the first-timepoint value.
    """
    edata = edata_mini_3D_missing_values
    layer = DEFAULT_TEM_LAYER_NAME
    n_obs, n_vars, n_time = edata.layers[layer].shape
    edata.var_names = ["n1", "n2", "n3", "n4", "letter", "yn"]

    edata.layers[layer] = array_type(edata.layers[layer])

    encoded = encode(edata, autodetect=False, encodings={"one-hot": ["letter", "yn"]}, layer=layer)

    encoded_layer = encoded.layers[layer]
    if array_type is as_dense_dask_array:
        assert isinstance(encoded_layer, DaskArray)
        assert isinstance(encoded.layers["original"], DaskArray)

    assert encoded_layer.ndim == 3
    assert encoded_layer.shape[0] == n_obs
    assert encoded_layer.shape[2] == n_time
    # one-hot expansion: letter (A, B, nan) + yn (Yes, No, nan) replaces 2 cols, adds 6.
    assert encoded_layer.shape[1] == n_vars - 2 + 6

    # obs holds the first-timepoint value for each encoded categorical.
    assert list(encoded.obs["letter"]) == ["A", "A", None, "B"] or list(encoded.obs["letter"]) == [
        "A",
        "A",
        np.nan,
        "B",
    ]
    assert list(encoded.obs["yn"]) == ["Yes", "Yes", "Yes", "Yes"]


def test_encode_3D_reencode_not_supported(edata_mini_3D_missing_values):
    edata = edata_mini_3D_missing_values
    edata.var_names = ["n1", "n2", "n3", "n4", "letter", "yn"]
    encoded = encode(edata, autodetect=False, encodings={"one-hot": ["letter", "yn"]}, layer=DEFAULT_TEM_LAYER_NAME)
    with pytest.raises(NotImplementedError, match="Re-encoding 3D"):
        encode(encoded, autodetect=False, encodings={"label": ["letter"]}, layer=DEFAULT_TEM_LAYER_NAME)


def test_unknown_encode_mode(encode_ds_1_edata):
    with pytest.raises(ValueError):
        encoded_edata = encode(encode_ds_1_edata, autodetect=False, encodings={"unknown_mode": ["survival"]})  # noqa: F841


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_duplicate_column_encoding(encode_ds_1_edata, layer):
    with pytest.raises(ValueError):
        encoded_edata = encode(  # noqa: F841
            encode_ds_1_edata,
            autodetect=False,
            encodings={"label": ["survival"], "one-hot": ["survival"]},
            layer=layer,
        )


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_autodetect_encode(encode_ds_1_edata, layer, array_type):
    _convert_edata_arrays(encode_ds_1_edata, array_type)
    # break .X to ensure its not used
    if layer is not None:
        encode_ds_1_edata.X = None
    encoded_edata = encode(encode_ds_1_edata, autodetect=True, layer=layer)
    encoded_X = encoded_edata.X if layer is None else encoded_edata.layers[layer]
    # dask inputs must keep the encoded result lazy
    if array_type is as_dense_dask_array:
        assert isinstance(encoded_X, DaskArray)
        assert isinstance(encoded_edata.layers["original"], DaskArray)
    assert list(encoded_edata.obs.columns) == ["survival", "clinic_day"]
    assert set(encoded_edata.var_names) == {
        "ehrapycat_survival_False",
        "ehrapycat_survival_True",
        "ehrapycat_clinic_day_Monday",
        "ehrapycat_clinic_day_Friday",
        "ehrapycat_clinic_day_Saturday",
        "ehrapycat_clinic_day_Sunday",
        "patient_id",
        "los_days",
        "b12_values",
    }

    assert np.all(
        encoded_edata.var["unencoded_var_names"]
        == [
            "survival",
            "survival",
            "clinic_day",
            "clinic_day",
            "clinic_day",
            "clinic_day",
            "patient_id",
            "los_days",
            "b12_values",
        ]
    )

    assert np.all(encoded_edata.var["encoding_mode"][:6] == ["one-hot"] * 6)
    assert np.all(enc is None for enc in encoded_edata.var["encoding_mode"][6:])

    X = encoded_edata.X if layer is None else encoded_edata.layers[layer]
    assert id(X) != id(encoded_edata.layers["original"])
    assert (
        encode_ds_1_edata is not None
        and X is not None
        and encode_ds_1_edata.obs is not None
        and encode_ds_1_edata.uns is not None
    )
    assert id(encoded_edata) != id(encode_ds_1_edata)
    assert id(encoded_edata.obs) != id(encode_ds_1_edata.obs)
    assert id(encoded_edata.uns) != id(encode_ds_1_edata.uns)
    assert id(encoded_edata.var) != id(encode_ds_1_edata.var)
    assert all(column in set(encoded_edata.obs.columns) for column in ["survival", "clinic_day"])
    assert not any(column in set(encode_ds_1_edata.obs.columns) for column in ["survival", "clinic_day"])

    assert_frame_equal(
        encode_ds_1_edata.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )

    assert np.all(
        encoded_edata.var[FEATURE_TYPE_KEY]
        == [
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
        ]
    )

    assert pd.api.types.is_bool_dtype(encoded_edata.obs["survival"].dtype)
    assert isinstance(encoded_edata.obs["clinic_day"].dtype, CategoricalDtype)


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_autodetect_num_only(capfd, encode_ds_2_edata, layer):
    if layer is not None:
        encode_ds_2_edata.X = None
    encoded_edata = encode(encode_ds_2_edata, autodetect=True, layer=layer)
    out, err = capfd.readouterr()
    assert id(encoded_edata) == id(encode_ds_2_edata)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_autodetect_custom_mode(encode_ds_1_edata, layer, array_type):
    _convert_edata_arrays(encode_ds_1_edata, array_type)
    if layer is not None:
        encode_ds_1_edata.X = None
    encoded_edata = encode(encode_ds_1_edata, autodetect=True, encodings="label", layer=layer)
    encoded_X = encoded_edata.X if layer is None else encoded_edata.layers[layer]
    if array_type is as_dense_dask_array:
        assert isinstance(encoded_X, DaskArray)
    assert list(encoded_edata.obs.columns) == ["survival", "clinic_day"]
    assert set(encoded_edata.var_names) == {
        "ehrapycat_survival",
        "ehrapycat_clinic_day",
        "patient_id",
        "los_days",
        "b12_values",
    }

    assert np.all(
        encoded_edata.var["unencoded_var_names"] == ["survival", "clinic_day", "patient_id", "los_days", "b12_values"]
    )
    assert np.all(encoded_edata.var["encoding_mode"][:2] == ["label"] * 2)
    assert np.all(enc is None for enc in encoded_edata.var["encoding_mode"][2:])

    X = encoded_edata.X if layer is None else encoded_edata.layers[layer]
    assert id(X) != id(encoded_edata.layers["original"])
    assert (
        encode_ds_1_edata is not None
        and X is not None
        and encode_ds_1_edata.obs is not None
        and encode_ds_1_edata.uns is not None
    )
    assert id(encoded_edata) != id(encode_ds_1_edata)
    assert id(encoded_edata.obs) != id(encode_ds_1_edata.obs)
    assert id(encoded_edata.uns) != id(encode_ds_1_edata.uns)
    assert id(encoded_edata.var) != id(encode_ds_1_edata.var)
    assert all(column in set(encoded_edata.obs.columns) for column in ["survival", "clinic_day"])
    assert not any(column in set(encode_ds_1_edata.obs.columns) for column in ["survival", "clinic_day"])

    assert_frame_equal(
        encode_ds_1_edata.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )

    assert np.all(
        encoded_edata.var[FEATURE_TYPE_KEY]
        == [
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
        ]
    )

    assert pd.api.types.is_bool_dtype(encoded_edata.obs["survival"].dtype)
    assert isinstance(encoded_edata.obs["clinic_day"].dtype, CategoricalDtype)


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_autodetect_encode_again(encode_ds_1_edata, layer):
    if layer is not None:
        encode_ds_1_edata.X = None
    encoded_edata = encode(encode_ds_1_edata, autodetect=True, layer=layer)
    encoded_edata_again = encode(encoded_edata, autodetect=True, layer=layer)
    assert id(encoded_edata_again) == id(encoded_edata)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_custom_encode(encode_ds_1_edata, layer, array_type):
    _convert_edata_arrays(encode_ds_1_edata, array_type)
    if layer is not None:
        encode_ds_1_edata.X = None
    encoded_edata = encode(
        encode_ds_1_edata,
        autodetect=False,
        encodings={"label": ["survival"], "one-hot": ["clinic_day"]},
        layer=layer,
    )
    X = encoded_edata.X if layer is None else encoded_edata.layers[layer]
    if array_type is as_dense_dask_array:
        assert isinstance(X, DaskArray)
    assert X.shape == (5, 8)
    assert list(encoded_edata.obs.columns) == ["survival", "clinic_day"]
    assert "ehrapycat_survival" in list(encoded_edata.var_names)
    assert all(
        clinic_day in list(encoded_edata.var_names)
        for clinic_day in [
            "ehrapycat_clinic_day_Friday",
            "ehrapycat_clinic_day_Monday",
            "ehrapycat_clinic_day_Saturday",
            "ehrapycat_clinic_day_Sunday",
        ]
    )

    assert np.all(
        encoded_edata.var["unencoded_var_names"]
        == ["clinic_day", "clinic_day", "clinic_day", "clinic_day", "survival", "patient_id", "los_days", "b12_values"]
    )
    assert np.all(encoded_edata.var["encoding_mode"][:5] == ["one-hot"] * 4 + ["label"])
    assert np.all(enc is None for enc in encoded_edata.var["encoding_mode"][5:])

    assert id(X) != id(encoded_edata.layers["original"])
    assert (
        encode_ds_1_edata is not None
        and X is not None
        and encode_ds_1_edata.obs is not None
        and encode_ds_1_edata.uns is not None
    )
    assert id(encoded_edata) != id(encode_ds_1_edata)
    assert id(encoded_edata.obs) != id(encode_ds_1_edata.obs)
    assert id(encoded_edata.uns) != id(encode_ds_1_edata.uns)
    assert id(encoded_edata.var) != id(encode_ds_1_edata.var)
    assert all(column in set(encoded_edata.obs.columns) for column in ["survival", "clinic_day"])
    assert not any(column in set(encode_ds_1_edata.obs.columns) for column in ["survival", "clinic_day"])

    assert_frame_equal(
        encode_ds_1_edata.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )

    assert np.all(
        encoded_edata.var[FEATURE_TYPE_KEY]
        == [
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
        ]
    )

    assert pd.api.types.is_bool_dtype(encoded_edata.obs["survival"].dtype)
    assert isinstance(encoded_edata.obs["clinic_day"].dtype, CategoricalDtype)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_custom_encode_again_single_columns_encoding(encode_ds_1_edata, layer, array_type):
    _convert_edata_arrays(encode_ds_1_edata, array_type)
    if layer is not None:
        encode_ds_1_edata.X = None
    encoded_edata = encode(
        encode_ds_1_edata,
        autodetect=False,
        encodings={"label": ["survival"], "one-hot": ["clinic_day"]},
        layer=layer,
    )
    encoded_edata = encode(encoded_edata, autodetect=False, encodings={"label": ["clinic_day"]}, layer=layer)

    X = encoded_edata.X if layer is None else encoded_edata.layers[layer]
    if array_type is as_dense_dask_array:
        assert isinstance(X, DaskArray)
    assert X.shape == (5, 5)
    assert len(encoded_edata.obs.columns) == 2
    assert set(encoded_edata.obs.columns) == {"survival", "clinic_day"}
    assert "ehrapycat_survival" in list(encoded_edata.var_names)
    assert "ehrapycat_clinic_day" in list(encoded_edata.var_names)
    assert all(
        clinic_day not in list(encoded_edata.var_names)
        for clinic_day in [
            "ehrapycat_clinic_day_Friday",
            "ehrapycat_clinic_day_Monday",
            "ehrapycat_clinic_day_Saturday",
            "ehrapycat_clinic_day_Sunday",
        ]
    )

    assert np.all(
        encoded_edata.var["encoding_mode"].loc[["ehrapycat_survival", "ehrapycat_clinic_day"]] == ["label", "label"]
    )

    assert id(X) != id(encoded_edata.layers["original"])
    assert pd.api.types.is_bool_dtype(encoded_edata.obs["survival"].dtype)
    assert isinstance(encoded_edata.obs["clinic_day"].dtype, CategoricalDtype)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_custom_encode_again_multiple_columns_encoding(encode_ds_1_edata, layer, array_type):
    _convert_edata_arrays(encode_ds_1_edata, array_type)
    if layer is not None:
        encode_ds_1_edata.X = None
    encoded_edata = encode(
        encode_ds_1_edata, autodetect=False, encodings={"one-hot": ["clinic_day", "survival"]}, layer=layer
    )
    encoded_edata_again = encode(
        encoded_edata,
        autodetect=False,
        encodings={"label": ["survival"], "one-hot": ["clinic_day"]},
        layer=layer,
    )

    X = encoded_edata_again.X if layer is None else encoded_edata_again.layers[layer]
    if array_type is as_dense_dask_array:
        assert isinstance(X, DaskArray)
    assert X.shape == (5, 8)
    assert len(encoded_edata_again.obs.columns) == 2
    assert set(encoded_edata_again.obs.columns) == {"survival", "clinic_day"}
    assert "ehrapycat_survival" in list(encoded_edata_again.var_names)
    assert "ehrapycat_clinic_day_Friday" in list(encoded_edata_again.var_names)
    assert all(
        survival_outcome not in list(encoded_edata_again.var_names)
        for survival_outcome in ["ehrapycat_survival_False", "ehrapycat_survival_True"]
    )

    assert np.all(
        encoded_edata_again.var.loc[encoded_edata_again.var["unencoded_var_names"] == "survival", "encoding_mode"]
        == "label"
    )
    assert np.all(
        encoded_edata_again.var.loc[encoded_edata_again.var["unencoded_var_names"] == "clinic_day", "encoding_mode"]
        == "one-hot"
    )

    assert id(X) != id(encoded_edata_again.layers["original"])
    assert pd.api.types.is_bool_dtype(encoded_edata.obs["survival"].dtype)
    assert isinstance(encoded_edata.obs["clinic_day"].dtype, CategoricalDtype)


def test_update_encoding_scheme_1(encode_ds_1_edata):
    encode_ds_1_edata.var["unencoded_var_names"] = ["col1", "col2", "col3", "col4", "col5"]
    encode_ds_1_edata.var["encoding_mode"] = ["label", "label", "label", "one-hot", "one-hot"]

    new_encodings = {"one-hot": ["col1"], "label": ["col2", "col3", "col4"]}

    expected_encodings = {
        "label": ["col2", "col3", "col4"],
        "one-hot": ["col1", "col5"],
    }
    updated_encodings = _reorder_encodings(encode_ds_1_edata, new_encodings)

    assert expected_encodings == updated_encodings
