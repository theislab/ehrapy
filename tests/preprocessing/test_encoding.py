from pathlib import Path

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import CATEGORICAL_TAG, DEFAULT_TEM_LAYER_NAME, FEATURE_TYPE_KEY, NUMERIC_TAG
from fast_array_utils.conv import to_dense
from pandas import CategoricalDtype, DataFrame
from pandas.testing import assert_frame_equal
from testing.fast_array_utils import Flags

from ehrapy.preprocessing._encoding import _reorder_encodings, encode
from tests.conftest import TEST_DATA_PATH, forbid_dask_compute


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


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_encode_3D_longitudinal_one_hot(edata_mini_3D_missing_values, array_type):
    """One-hot encode a 3D layer with categorical columns.

    The encoder must fit on values stacked across time so the category space is shared, the time axis is preserved, and ``obs`` stores the first-timepoint value.
    """
    edata = edata_mini_3D_missing_values
    layer = DEFAULT_TEM_LAYER_NAME
    n_obs, n_vars, n_time = edata.layers[layer].shape
    edata.var_names = ["n1", "n2", "n3", "n4", "letter", "yn"]

    edata.layers[layer] = array_type(edata.layers[layer])

    with forbid_dask_compute(allowed=1):
        encoded = encode(edata, autodetect=False, encodings={"one-hot": ["letter", "yn"]}, layer=layer)

    encoded_layer = encoded.layers[layer]
    assert isinstance(encoded_layer, array_type.cls)
    assert isinstance(encoded.layers["original"], array_type.cls)
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


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_autodetect_encode(encode_ds_1_edata, layer, array_type):
    _convert_edata_arrays(encode_ds_1_edata, array_type)
    # break .X to ensure its not used
    if layer is not None:
        encode_ds_1_edata.X = None
    with forbid_dask_compute(allowed=1):
        encoded_edata = encode(encode_ds_1_edata, autodetect=True, layer=layer)
    encoded_X = encoded_edata.X if layer is None else encoded_edata.layers[layer]
    assert isinstance(encoded_X, array_type.cls)
    assert isinstance(encoded_edata.layers["original"], array_type.cls)
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

    assert FEATURE_TYPE_KEY not in encode_ds_1_edata.var

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


@pytest.mark.parametrize(
    ("autodetect", "encodings"),
    [(True, "one-hot"), (True, "label"), (False, {"label": ["survival"], "one-hot": ["clinic_day"]})],
)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_encode_does_not_modify_input(encode_ds_1_edata, autodetect, encodings, layer):
    edata_before = encode_ds_1_edata.copy()

    encoded_edata = encode(encode_ds_1_edata, autodetect=autodetect, encodings=encodings, layer=layer)

    assert "original" in encoded_edata.layers
    assert np.array_equal(encode_ds_1_edata.X, edata_before.X)
    assert encode_ds_1_edata.layers.keys() == edata_before.layers.keys()
    for key in edata_before.layers:
        assert np.array_equal(encode_ds_1_edata.layers[key], edata_before.layers[key])
    assert_frame_equal(encode_ds_1_edata.var, edata_before.var)
    assert_frame_equal(encode_ds_1_edata.obs, edata_before.obs)
    assert encode_ds_1_edata.uns.keys() == edata_before.uns.keys()


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_autodetect_num_only(capfd, encode_ds_2_edata, layer):
    if layer is not None:
        encode_ds_2_edata.X = None
    encoded_edata = encode(encode_ds_2_edata, autodetect=True, layer=layer)
    out, err = capfd.readouterr()
    assert id(encoded_edata) == id(encode_ds_2_edata)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_autodetect_custom_mode(encode_ds_1_edata, layer, array_type):
    _convert_edata_arrays(encode_ds_1_edata, array_type)
    if layer is not None:
        encode_ds_1_edata.X = None
    with forbid_dask_compute(allowed=1):
        encoded_edata = encode(encode_ds_1_edata, autodetect=True, encodings="label", layer=layer)
    encoded_X = encoded_edata.X if layer is None else encoded_edata.layers[layer]
    assert isinstance(encoded_X, array_type.cls)
    assert isinstance(encoded_edata.layers["original"], array_type.cls)
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

    assert FEATURE_TYPE_KEY not in encode_ds_1_edata.var

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


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_custom_encode(encode_ds_1_edata, layer, array_type):
    _convert_edata_arrays(encode_ds_1_edata, array_type)
    if layer is not None:
        encode_ds_1_edata.X = None
    with forbid_dask_compute(allowed=1):
        encoded_edata = encode(
            encode_ds_1_edata,
            autodetect=False,
            encodings={"label": ["survival"], "one-hot": ["clinic_day"]},
            layer=layer,
        )
    X = encoded_edata.X if layer is None else encoded_edata.layers[layer]
    assert isinstance(X, array_type.cls)
    assert isinstance(encoded_edata.layers["original"], array_type.cls)
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

    assert FEATURE_TYPE_KEY not in encode_ds_1_edata.var

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


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_custom_encode_again_single_columns_encoding(encode_ds_1_edata, layer, array_type):
    _convert_edata_arrays(encode_ds_1_edata, array_type)
    if layer is not None:
        encode_ds_1_edata.X = None
    with forbid_dask_compute(allowed=2):
        encoded_edata = encode(
            encode_ds_1_edata,
            autodetect=False,
            encodings={"label": ["survival"], "one-hot": ["clinic_day"]},
            layer=layer,
        )
        encoded_edata = encode(encoded_edata, autodetect=False, encodings={"label": ["clinic_day"]}, layer=layer)

    X = encoded_edata.X if layer is None else encoded_edata.layers[layer]
    assert isinstance(X, array_type.cls)
    assert isinstance(encoded_edata.layers["original"], array_type.cls)
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


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_custom_encode_again_multiple_columns_encoding(encode_ds_1_edata, layer, array_type):
    _convert_edata_arrays(encode_ds_1_edata, array_type)
    if layer is not None:
        encode_ds_1_edata.X = None
    with forbid_dask_compute(allowed=2):
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
    assert isinstance(X, array_type.cls)
    assert isinstance(encoded_edata_again.layers["original"], array_type.cls)
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


def test_encode_3D_single_timepoint(encode_ds_1_edata):
    expected = encode(encode_ds_1_edata, autodetect=True)
    edata = ed.EHRData(
        shape=encode_ds_1_edata.shape[:2],
        var=encode_ds_1_edata.var.copy(),
        layers={DEFAULT_TEM_LAYER_NAME: encode_ds_1_edata.X[:, :, None]},
    )

    encoded = encode(edata, autodetect=True, layer=DEFAULT_TEM_LAYER_NAME)

    assert encoded.var_names.tolist() == expected.var_names.tolist()
    np.testing.assert_array_equal(encoded.layers[DEFAULT_TEM_LAYER_NAME][:, :, 0], expected.X)


ENCODINGS = [
    pytest.param(True, "one-hot", id="autodetect-one-hot"),
    pytest.param(True, "label", id="autodetect-label"),
    pytest.param(False, {"label": ["survival"], "one-hot": ["clinic_day"]}, id="custom"),
]


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("feature_types", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(("autodetect", "encodings"), ENCODINGS)
def test_encode_array_types(encode_ds_1_edata, array_type, ndim, feature_types, autodetect, encodings):
    if array_type.flags & Flags.Sparse:
        if ndim == 3:
            pytest.skip("sparse arrays are 2D")
        X = np.zeros((20, 3))
        X[::4, 0] = 1
        X[1::5, 1] = 2.5
        X[2::3, 2] = np.nan
        edata = ed.EHRData(X=array_type(X))
        if not autodetect:
            with pytest.raises(NotImplementedError, match="densify"):
                encode(edata, autodetect=False, encodings={"one-hot": ["0"]})
            return
        with forbid_dask_compute():
            assert encode(edata, autodetect=True, encodings=encodings) is edata
        return

    X = encode_ds_1_edata.X
    if ndim == 3:
        X = np.stack([X, X[::-1]], axis=2)

    def make_edata(X):
        edata = ed.EHRData(shape=X.shape[:2], var=encode_ds_1_edata.var.copy(), layers={DEFAULT_TEM_LAYER_NAME: X})
        if feature_types:
            ed.infer_feature_types(edata, layer=DEFAULT_TEM_LAYER_NAME, output=None)
        return edata

    kwargs = {"autodetect": autodetect, "encodings": encodings, "layer": DEFAULT_TEM_LAYER_NAME}
    expected = encode(make_edata(X), **kwargs)
    edata = make_edata(array_type(X))

    with forbid_dask_compute(allowed=1):
        result = encode(edata, **kwargs)

    assert_frame_equal(result.obs, expected.obs)
    assert_frame_equal(result.var, expected.var)
    for key in [DEFAULT_TEM_LAYER_NAME, "original"]:
        assert isinstance(result.layers[key], array_type.cls)
        np.testing.assert_allclose(
            to_dense(result.layers[key], to_cpu_memory=True), expected.layers[key], equal_nan=True
        )


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_encode_again_array_types(encode_ds_1_edata, array_type):
    def encode_twice(edata):
        edata = encode(edata, autodetect=False, encodings={"one-hot": ["clinic_day", "survival"]})
        return encode(edata, autodetect=False, encodings={"label": ["survival"], "one-hot": ["clinic_day"]})

    expected = encode_twice(encode_ds_1_edata.copy())
    encode_ds_1_edata.X = array_type(encode_ds_1_edata.X)

    with forbid_dask_compute(allowed=2):
        result = encode_twice(encode_ds_1_edata)

    assert_frame_equal(result.obs, expected.obs)
    assert_frame_equal(result.var, expected.var)
    for X, expected_X in [(result.X, expected.X), (result.layers["original"], expected.layers["original"])]:
        assert isinstance(X, array_type.cls)
        np.testing.assert_allclose(to_dense(X, to_cpu_memory=True), expected_X, equal_nan=True)
