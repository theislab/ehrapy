from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import CategoricalDtype, DataFrame
from pandas.testing import assert_frame_equal

from ehrapy.anndata._constants import CATEGORICAL_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from ehrapy.io._read import read_csv
from ehrapy.preprocessing._encoding import DuplicateColumnEncodingError, _reorder_encodings, encode

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_encode"


def test_unknown_encode_mode():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    with pytest.raises(ValueError):
        encoded_ann_data = encode(adata, autodetect=False, encodings={"unknown_mode": ["survival"]})  # noqa: F841


def test_duplicate_column_encoding():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    with pytest.raises(ValueError):
        encoded_ann_data = encode(  # noqa: F841
            adata,
            autodetect=False,
            encodings={"label": ["survival"], "one-hot": ["survival"]},
        )


def test_autodetect_encode():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    encoded_ann_data = encode(adata, autodetect=True)
    assert list(encoded_ann_data.obs.columns) == ["survival", "clinic_day"]
    assert set(encoded_ann_data.var_names) == {
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
        encoded_ann_data.var["unencoded_var_names"]
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

    assert np.all(encoded_ann_data.var["encoding_mode"][:6] == ["one-hot"] * 6)
    assert np.all(enc is None for enc in encoded_ann_data.var["encoding_mode"][6:])

    assert id(encoded_ann_data.X) != id(encoded_ann_data.layers["original"])
    assert adata is not None and adata.X is not None and adata.obs is not None and adata.uns is not None
    assert id(encoded_ann_data) != id(adata)
    assert id(encoded_ann_data.obs) != id(adata.obs)
    assert id(encoded_ann_data.uns) != id(adata.uns)
    assert id(encoded_ann_data.var) != id(adata.var)
    assert all(column in set(encoded_ann_data.obs.columns) for column in ["survival", "clinic_day"])
    assert not any(column in set(adata.obs.columns) for column in ["survival", "clinic_day"])

    assert_frame_equal(
        adata.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )

    assert np.all(
        encoded_ann_data.var[FEATURE_TYPE_KEY]
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

    assert pd.api.types.is_bool_dtype(encoded_ann_data.obs["survival"].dtype)
    assert isinstance(encoded_ann_data.obs["clinic_day"].dtype, CategoricalDtype)


def test_autodetect_num_only(capfd):
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset2.csv")
    encoded_ann_data = encode(adata, autodetect=True)
    out, err = capfd.readouterr()
    assert id(encoded_ann_data) == id(adata)


def test_autodetect_custom_mode():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    encoded_ann_data = encode(adata, autodetect=True, encodings="label")
    assert list(encoded_ann_data.obs.columns) == ["survival", "clinic_day"]
    assert set(encoded_ann_data.var_names) == {
        "ehrapycat_survival",
        "ehrapycat_clinic_day",
        "patient_id",
        "los_days",
        "b12_values",
    }

    assert np.all(
        encoded_ann_data.var["unencoded_var_names"]
        == ["survival", "clinic_day", "patient_id", "los_days", "b12_values"]
    )
    assert np.all(encoded_ann_data.var["encoding_mode"][:2] == ["label"] * 2)
    assert np.all(enc is None for enc in encoded_ann_data.var["encoding_mode"][2:])

    assert id(encoded_ann_data.X) != id(encoded_ann_data.layers["original"])
    assert adata is not None and adata.X is not None and adata.obs is not None and adata.uns is not None
    assert id(encoded_ann_data) != id(adata)
    assert id(encoded_ann_data.obs) != id(adata.obs)
    assert id(encoded_ann_data.uns) != id(adata.uns)
    assert id(encoded_ann_data.var) != id(adata.var)
    assert all(column in set(encoded_ann_data.obs.columns) for column in ["survival", "clinic_day"])
    assert not any(column in set(adata.obs.columns) for column in ["survival", "clinic_day"])

    assert_frame_equal(
        adata.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )

    assert np.all(
        encoded_ann_data.var[FEATURE_TYPE_KEY]
        == [
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
        ]
    )

    assert pd.api.types.is_bool_dtype(encoded_ann_data.obs["survival"].dtype)
    assert isinstance(encoded_ann_data.obs["clinic_day"].dtype, CategoricalDtype)


def test_autodetect_encode_again():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    encoded_ann_data = encode(adata, autodetect=True)
    encoded_ann_data_again = encode(encoded_ann_data, autodetect=True)
    assert id(encoded_ann_data_again) == id(encoded_ann_data)


def test_custom_encode():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    encoded_ann_data = encode(
        adata,
        autodetect=False,
        encodings={"label": ["survival"], "one-hot": ["clinic_day"]},
    )
    assert encoded_ann_data.X.shape == (5, 8)
    assert list(encoded_ann_data.obs.columns) == ["survival", "clinic_day"]
    assert "ehrapycat_survival" in list(encoded_ann_data.var_names)
    assert all(
        clinic_day in list(encoded_ann_data.var_names)
        for clinic_day in [
            "ehrapycat_clinic_day_Friday",
            "ehrapycat_clinic_day_Monday",
            "ehrapycat_clinic_day_Saturday",
            "ehrapycat_clinic_day_Sunday",
        ]
    )

    assert np.all(
        encoded_ann_data.var["unencoded_var_names"]
        == ["clinic_day", "clinic_day", "clinic_day", "clinic_day", "survival", "patient_id", "los_days", "b12_values"]
    )
    assert np.all(encoded_ann_data.var["encoding_mode"][:5] == ["one-hot"] * 4 + ["label"])
    assert np.all(enc is None for enc in encoded_ann_data.var["encoding_mode"][5:])

    assert id(encoded_ann_data.X) != id(encoded_ann_data.layers["original"])
    assert adata is not None and adata.X is not None and adata.obs is not None and adata.uns is not None
    assert id(encoded_ann_data) != id(adata)
    assert id(encoded_ann_data.obs) != id(adata.obs)
    assert id(encoded_ann_data.uns) != id(adata.uns)
    assert id(encoded_ann_data.var) != id(adata.var)
    assert all(column in set(encoded_ann_data.obs.columns) for column in ["survival", "clinic_day"])
    assert not any(column in set(adata.obs.columns) for column in ["survival", "clinic_day"])

    assert_frame_equal(
        adata.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )

    assert np.all(
        encoded_ann_data.var[FEATURE_TYPE_KEY]
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

    assert pd.api.types.is_bool_dtype(encoded_ann_data.obs["survival"].dtype)
    assert isinstance(encoded_ann_data.obs["clinic_day"].dtype, CategoricalDtype)


def test_custom_encode_again_single_columns_encoding():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    encoded_ann_data = encode(
        adata,
        autodetect=False,
        encodings={"label": ["survival"], "one-hot": ["clinic_day"]},
    )
    encoded_ann_data_again = encode(encoded_ann_data, autodetect=False, encodings={"label": ["clinic_day"]})
    assert encoded_ann_data_again.X.shape == (5, 5)
    assert len(encoded_ann_data_again.obs.columns) == 2
    assert set(encoded_ann_data_again.obs.columns) == {"survival", "clinic_day"}
    assert "ehrapycat_survival" in list(encoded_ann_data_again.var_names)
    assert "ehrapycat_clinic_day" in list(encoded_ann_data_again.var_names)
    assert all(
        clinic_day not in list(encoded_ann_data_again.var_names)
        for clinic_day in [
            "ehrapycat_clinic_day_Friday",
            "ehrapycat_clinic_day_Monday",
            "ehrapycat_clinic_day_Saturday",
            "ehrapycat_clinic_day_Sunday",
        ]
    )

    assert np.all(
        encoded_ann_data_again.var["encoding_mode"].loc[["ehrapycat_survival", "ehrapycat_clinic_day"]]
        == ["label", "label"]
    )

    assert id(encoded_ann_data_again.X) != id(encoded_ann_data_again.layers["original"])
    assert pd.api.types.is_bool_dtype(encoded_ann_data.obs["survival"].dtype)
    assert isinstance(encoded_ann_data.obs["clinic_day"].dtype, CategoricalDtype)


def test_custom_encode_again_multiple_columns_encoding():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    encoded_ann_data = encode(adata, autodetect=False, encodings={"one-hot": ["clinic_day", "survival"]})
    encoded_ann_data_again = encode(
        encoded_ann_data,
        autodetect=False,
        encodings={"label": ["survival"], "one-hot": ["clinic_day"]},
    )
    assert encoded_ann_data_again.X.shape == (5, 8)
    assert len(encoded_ann_data_again.obs.columns) == 2
    assert set(encoded_ann_data_again.obs.columns) == {"survival", "clinic_day"}
    assert "ehrapycat_survival" in list(encoded_ann_data_again.var_names)
    assert "ehrapycat_clinic_day_Friday" in list(encoded_ann_data_again.var_names)
    assert all(
        survival_outcome not in list(encoded_ann_data_again.var_names)
        for survival_outcome in ["ehrapycat_survival_False", "ehrapycat_survival_True"]
    )

    assert np.all(
        encoded_ann_data_again.var.loc[encoded_ann_data_again.var["unencoded_var_names"] == "survival", "encoding_mode"]
        == "label"
    )
    assert np.all(
        encoded_ann_data_again.var.loc[
            encoded_ann_data_again.var["unencoded_var_names"] == "clinic_day", "encoding_mode"
        ]
        == "one-hot"
    )

    assert id(encoded_ann_data_again.X) != id(encoded_ann_data_again.layers["original"])
    assert pd.api.types.is_bool_dtype(encoded_ann_data.obs["survival"].dtype)
    assert isinstance(encoded_ann_data.obs["clinic_day"].dtype, CategoricalDtype)


def test_update_encoding_scheme_1():
    # just a dummy adata object that won't be used actually
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")

    adata.var["unencoded_var_names"] = ["col1", "col2", "col3", "col4", "col5"]
    adata.var["encoding_mode"] = ["label", "label", "label", "one-hot", "one-hot"]

    new_encodings = {"one-hot": ["col1"], "label": ["col2", "col3", "col4"]}

    expected_encodings = {
        "label": ["col2", "col3", "col4"],
        "one-hot": ["col1", "col5"],
    }
    updated_encodings = _reorder_encodings(adata, new_encodings)

    assert expected_encodings == updated_encodings
