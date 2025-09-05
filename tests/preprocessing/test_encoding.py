from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import CategoricalDtype, DataFrame
from pandas.testing import assert_frame_equal

from ehrapy.anndata._constants import CATEGORICAL_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from ehrapy.io._read import read_csv
from ehrapy.preprocessing._encoding import _reorder_encodings, encode
from tests.conftest import TEST_DATA_PATH

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{TEST_DATA_PATH}/encode"


def test_encode_3D_edata(edata_blob_small):
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        encode(edata_blob_small, layer="R_layer", copy=True)


def test_unknown_encode_mode(encode_ds_1_adata):
    with pytest.raises(ValueError):
        encoded_adata = encode(encode_ds_1_adata, autodetect=False, encodings={"unknown_mode": ["survival"]})  # noqa: F841


def test_duplicate_column_encoding(encode_ds_1_adata):
    with pytest.raises(ValueError):
        encoded_adata = encode(  # noqa: F841
            encode_ds_1_adata,
            autodetect=False,
            encodings={"label": ["survival"], "one-hot": ["survival"]},
        )


def test_autodetect_encode(encode_ds_1_adata):
    encoded_adata = encode(encode_ds_1_adata, autodetect=True)
    assert list(encoded_adata.obs.columns) == ["survival", "clinic_day"]
    assert set(encoded_adata.var_names) == {
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
        encoded_adata.var["unencoded_var_names"]
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

    assert np.all(encoded_adata.var["encoding_mode"][:6] == ["one-hot"] * 6)
    assert np.all(enc is None for enc in encoded_adata.var["encoding_mode"][6:])

    assert id(encoded_adata.X) != id(encoded_adata.layers["original"])
    assert (
        encode_ds_1_adata is not None
        and encode_ds_1_adata.X is not None
        and encode_ds_1_adata.obs is not None
        and encode_ds_1_adata.uns is not None
    )
    assert id(encoded_adata) != id(encode_ds_1_adata)
    assert id(encoded_adata.obs) != id(encode_ds_1_adata.obs)
    assert id(encoded_adata.uns) != id(encode_ds_1_adata.uns)
    assert id(encoded_adata.var) != id(encode_ds_1_adata.var)
    assert all(column in set(encoded_adata.obs.columns) for column in ["survival", "clinic_day"])
    assert not any(column in set(encode_ds_1_adata.obs.columns) for column in ["survival", "clinic_day"])

    assert_frame_equal(
        encode_ds_1_adata.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )

    assert np.all(
        encoded_adata.var[FEATURE_TYPE_KEY]
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

    assert pd.api.types.is_bool_dtype(encoded_adata.obs["survival"].dtype)
    assert isinstance(encoded_adata.obs["clinic_day"].dtype, CategoricalDtype)


def test_autodetect_num_only(capfd, encode_ds_2_adata):
    encoded_adata = encode(encode_ds_2_adata, autodetect=True)
    out, err = capfd.readouterr()
    assert id(encoded_adata) == id(encode_ds_2_adata)


def test_autodetect_custom_mode(encode_ds_1_adata):
    encoded_adata = encode(encode_ds_1_adata, autodetect=True, encodings="label")
    assert list(encoded_adata.obs.columns) == ["survival", "clinic_day"]
    assert set(encoded_adata.var_names) == {
        "ehrapycat_survival",
        "ehrapycat_clinic_day",
        "patient_id",
        "los_days",
        "b12_values",
    }

    assert np.all(
        encoded_adata.var["unencoded_var_names"] == ["survival", "clinic_day", "patient_id", "los_days", "b12_values"]
    )
    assert np.all(encoded_adata.var["encoding_mode"][:2] == ["label"] * 2)
    assert np.all(enc is None for enc in encoded_adata.var["encoding_mode"][2:])

    assert id(encoded_adata.X) != id(encoded_adata.layers["original"])
    assert (
        encode_ds_1_adata is not None
        and encode_ds_1_adata.X is not None
        and encode_ds_1_adata.obs is not None
        and encode_ds_1_adata.uns is not None
    )
    assert id(encoded_adata) != id(encode_ds_1_adata)
    assert id(encoded_adata.obs) != id(encode_ds_1_adata.obs)
    assert id(encoded_adata.uns) != id(encode_ds_1_adata.uns)
    assert id(encoded_adata.var) != id(encode_ds_1_adata.var)
    assert all(column in set(encoded_adata.obs.columns) for column in ["survival", "clinic_day"])
    assert not any(column in set(encode_ds_1_adata.obs.columns) for column in ["survival", "clinic_day"])

    assert_frame_equal(
        encode_ds_1_adata.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )

    assert np.all(
        encoded_adata.var[FEATURE_TYPE_KEY]
        == [
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
        ]
    )

    assert pd.api.types.is_bool_dtype(encoded_adata.obs["survival"].dtype)
    assert isinstance(encoded_adata.obs["clinic_day"].dtype, CategoricalDtype)


def test_autodetect_encode_again(encode_ds_1_adata):
    encoded_adata = encode(encode_ds_1_adata, autodetect=True)
    encoded_adata_again = encode(encoded_adata, autodetect=True)
    assert id(encoded_adata_again) == id(encoded_adata)


def test_custom_encode(encode_ds_1_adata):
    encoded_adata = encode(
        encode_ds_1_adata,
        autodetect=False,
        encodings={"label": ["survival"], "one-hot": ["clinic_day"]},
    )
    assert encoded_adata.X.shape == (5, 8)
    assert list(encoded_adata.obs.columns) == ["survival", "clinic_day"]
    assert "ehrapycat_survival" in list(encoded_adata.var_names)
    assert all(
        clinic_day in list(encoded_adata.var_names)
        for clinic_day in [
            "ehrapycat_clinic_day_Friday",
            "ehrapycat_clinic_day_Monday",
            "ehrapycat_clinic_day_Saturday",
            "ehrapycat_clinic_day_Sunday",
        ]
    )

    assert np.all(
        encoded_adata.var["unencoded_var_names"]
        == ["clinic_day", "clinic_day", "clinic_day", "clinic_day", "survival", "patient_id", "los_days", "b12_values"]
    )
    assert np.all(encoded_adata.var["encoding_mode"][:5] == ["one-hot"] * 4 + ["label"])
    assert np.all(enc is None for enc in encoded_adata.var["encoding_mode"][5:])

    assert id(encoded_adata.X) != id(encoded_adata.layers["original"])
    assert (
        encode_ds_1_adata is not None
        and encode_ds_1_adata.X is not None
        and encode_ds_1_adata.obs is not None
        and encode_ds_1_adata.uns is not None
    )
    assert id(encoded_adata) != id(encode_ds_1_adata)
    assert id(encoded_adata.obs) != id(encode_ds_1_adata.obs)
    assert id(encoded_adata.uns) != id(encode_ds_1_adata.uns)
    assert id(encoded_adata.var) != id(encode_ds_1_adata.var)
    assert all(column in set(encoded_adata.obs.columns) for column in ["survival", "clinic_day"])
    assert not any(column in set(encode_ds_1_adata.obs.columns) for column in ["survival", "clinic_day"])

    assert_frame_equal(
        encode_ds_1_adata.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )

    assert np.all(
        encoded_adata.var[FEATURE_TYPE_KEY]
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

    assert pd.api.types.is_bool_dtype(encoded_adata.obs["survival"].dtype)
    assert isinstance(encoded_adata.obs["clinic_day"].dtype, CategoricalDtype)


def test_custom_encode_again_single_columns_encoding(encode_ds_1_adata):
    encoded_adata = encode(
        encode_ds_1_adata,
        autodetect=False,
        encodings={"label": ["survival"], "one-hot": ["clinic_day"]},
    )
    encoded_adata = encode(encoded_adata, autodetect=False, encodings={"label": ["clinic_day"]})
    assert encoded_adata.X.shape == (5, 5)
    assert len(encoded_adata.obs.columns) == 2
    assert set(encoded_adata.obs.columns) == {"survival", "clinic_day"}
    assert "ehrapycat_survival" in list(encoded_adata.var_names)
    assert "ehrapycat_clinic_day" in list(encoded_adata.var_names)
    assert all(
        clinic_day not in list(encoded_adata.var_names)
        for clinic_day in [
            "ehrapycat_clinic_day_Friday",
            "ehrapycat_clinic_day_Monday",
            "ehrapycat_clinic_day_Saturday",
            "ehrapycat_clinic_day_Sunday",
        ]
    )

    assert np.all(
        encoded_adata.var["encoding_mode"].loc[["ehrapycat_survival", "ehrapycat_clinic_day"]] == ["label", "label"]
    )

    assert id(encoded_adata.X) != id(encoded_adata.layers["original"])
    assert pd.api.types.is_bool_dtype(encoded_adata.obs["survival"].dtype)
    assert isinstance(encoded_adata.obs["clinic_day"].dtype, CategoricalDtype)


def test_custom_encode_again_multiple_columns_encoding(encode_ds_1_adata):
    encoded_adata = encode(encode_ds_1_adata, autodetect=False, encodings={"one-hot": ["clinic_day", "survival"]})
    encoded_adata_again = encode(
        encoded_adata,
        autodetect=False,
        encodings={"label": ["survival"], "one-hot": ["clinic_day"]},
    )
    assert encoded_adata_again.X.shape == (5, 8)
    assert len(encoded_adata_again.obs.columns) == 2
    assert set(encoded_adata_again.obs.columns) == {"survival", "clinic_day"}
    assert "ehrapycat_survival" in list(encoded_adata_again.var_names)
    assert "ehrapycat_clinic_day_Friday" in list(encoded_adata_again.var_names)
    assert all(
        survival_outcome not in list(encoded_adata_again.var_names)
        for survival_outcome in ["ehrapycat_survival_False", "ehrapycat_survival_True"]
    )

    assert np.all(
        encoded_adata_again.var.loc[encoded_adata_again.var["unencoded_var_names"] == "survival", "encoding_mode"]
        == "label"
    )
    assert np.all(
        encoded_adata_again.var.loc[encoded_adata_again.var["unencoded_var_names"] == "clinic_day", "encoding_mode"]
        == "one-hot"
    )

    assert id(encoded_adata_again.X) != id(encoded_adata_again.layers["original"])
    assert pd.api.types.is_bool_dtype(encoded_adata.obs["survival"].dtype)
    assert isinstance(encoded_adata.obs["clinic_day"].dtype, CategoricalDtype)


def test_update_encoding_scheme_1(encode_ds_1_adata):
    encode_ds_1_adata.var["unencoded_var_names"] = ["col1", "col2", "col3", "col4", "col5"]
    encode_ds_1_adata.var["encoding_mode"] = ["label", "label", "label", "one-hot", "one-hot"]

    new_encodings = {"one-hot": ["col1"], "label": ["col2", "col3", "col4"]}

    expected_encodings = {
        "label": ["col2", "col3", "col4"],
        "one-hot": ["col1", "col5"],
    }
    updated_encodings = _reorder_encodings(encode_ds_1_adata, new_encodings)

    assert expected_encodings == updated_encodings
