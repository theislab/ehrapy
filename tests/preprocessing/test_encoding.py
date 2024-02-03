from pathlib import Path

import pandas as pd
import pytest
from pandas import CategoricalDtype, DataFrame
from pandas.testing import assert_frame_equal

from ehrapy.anndata._constants import EHRAPY_TYPE_KEY, NON_NUMERIC_ENCODED_TAG, NON_NUMERIC_TAG, NUMERIC_TAG
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
            encodings={"label": ["survival"], "count": ["survival"]},
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

    assert encoded_ann_data.uns["var_to_encoding"] == {
        "survival": "one-hot",
        "clinic_day": "one-hot",
    }
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
            {EHRAPY_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, NON_NUMERIC_TAG, NON_NUMERIC_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )
    assert_frame_equal(
        encoded_ann_data.var,
        DataFrame(
            {
                EHRAPY_TYPE_KEY: [
                    NON_NUMERIC_ENCODED_TAG,
                    NON_NUMERIC_ENCODED_TAG,
                    NON_NUMERIC_ENCODED_TAG,
                    NON_NUMERIC_ENCODED_TAG,
                    NON_NUMERIC_ENCODED_TAG,
                    NON_NUMERIC_ENCODED_TAG,
                    NUMERIC_TAG,
                    NUMERIC_TAG,
                    NUMERIC_TAG,
                ]
            },
            index=[
                "ehrapycat_survival_False",
                "ehrapycat_survival_True",
                "ehrapycat_clinic_day_Friday",
                "ehrapycat_clinic_day_Monday",
                "ehrapycat_clinic_day_Saturday",
                "ehrapycat_clinic_day_Sunday",
                "patient_id",
                "los_days",
                "b12_values",
            ],
        ),
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
    encoded_ann_data = encode(adata, autodetect=True, encodings="count")
    assert list(encoded_ann_data.obs.columns) == ["survival", "clinic_day"]
    assert set(encoded_ann_data.var_names) == {
        "ehrapycat_survival",
        "ehrapycat_clinic_day",
        "patient_id",
        "los_days",
        "b12_values",
    }

    assert encoded_ann_data.uns["var_to_encoding"] == {
        "survival": "count",
        "clinic_day": "count",
    }
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
            {EHRAPY_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, NON_NUMERIC_TAG, NON_NUMERIC_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )
    assert_frame_equal(
        encoded_ann_data.var,
        DataFrame(
            {
                EHRAPY_TYPE_KEY: [
                    NON_NUMERIC_ENCODED_TAG,
                    NON_NUMERIC_ENCODED_TAG,
                    NUMERIC_TAG,
                    NUMERIC_TAG,
                    NUMERIC_TAG,
                ]
            },
            index=[
                "ehrapycat_survival",
                "ehrapycat_clinic_day",
                "patient_id",
                "los_days",
                "b12_values",
            ],
        ),
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
    assert encoded_ann_data.uns["var_to_encoding"] == {
        "survival": "label",
        "clinic_day": "one-hot",
    }
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
            {EHRAPY_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, NON_NUMERIC_TAG, NON_NUMERIC_TAG]},
            index=["patient_id", "los_days", "b12_values", "survival", "clinic_day"],
        ),
    )
    assert_frame_equal(
        encoded_ann_data.var,
        DataFrame(
            {
                EHRAPY_TYPE_KEY: [
                    NON_NUMERIC_ENCODED_TAG,
                    NON_NUMERIC_ENCODED_TAG,
                    NON_NUMERIC_ENCODED_TAG,
                    NON_NUMERIC_ENCODED_TAG,
                    NON_NUMERIC_ENCODED_TAG,
                    NUMERIC_TAG,
                    NUMERIC_TAG,
                    NUMERIC_TAG,
                ]
            },
            index=[
                "ehrapycat_clinic_day_Friday",
                "ehrapycat_clinic_day_Monday",
                "ehrapycat_clinic_day_Saturday",
                "ehrapycat_clinic_day_Sunday",
                "ehrapycat_survival",
                "patient_id",
                "los_days",
                "b12_values",
            ],
        ),
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
    assert list(encoded_ann_data_again.obs.columns) == ["survival", "clinic_day"]
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
    assert encoded_ann_data_again.uns["var_to_encoding"] == {
        "survival": "label",
        "clinic_day": "label",
    }
    assert id(encoded_ann_data_again.X) != id(encoded_ann_data_again.layers["original"])
    assert pd.api.types.is_bool_dtype(encoded_ann_data.obs["survival"].dtype)
    assert isinstance(encoded_ann_data.obs["clinic_day"].dtype, CategoricalDtype)


def test_custom_encode_again_multiple_columns_encoding():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    encoded_ann_data = encode(adata, autodetect=False, encodings={"one-hot": ["clinic_day", "survival"]})
    encoded_ann_data_again = encode(
        encoded_ann_data,
        autodetect=False,
        encodings={"label": ["survival"], "count": ["clinic_day"]},
    )
    assert encoded_ann_data_again.X.shape == (5, 5)
    assert list(encoded_ann_data_again.obs.columns) == ["survival", "clinic_day"]
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
    assert all(
        survival_outcome not in list(encoded_ann_data_again.var_names)
        for survival_outcome in ["ehrapycat_survival_False", "ehrapycat_survival_True"]
    )
    assert encoded_ann_data_again.uns["var_to_encoding"] == {
        "survival": "label",
        "clinic_day": "count",
    }
    assert id(encoded_ann_data_again.X) != id(encoded_ann_data_again.layers["original"])
    assert pd.api.types.is_bool_dtype(encoded_ann_data.obs["survival"].dtype)
    assert isinstance(encoded_ann_data.obs["clinic_day"].dtype, CategoricalDtype)


def test_update_encoding_scheme_1():
    # just a dummy adata object that won't be used actually
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    adata.uns["encoding_to_var"] = {
        "label": ["col1", "col2", "col3"],
        "count": ["col4"],
        "hash": [["col5", "col6", "col7"], ["col8", "col9"]],
    }
    adata.uns["var_to_encoding"] = {
        "col1": "label",
        "col2": "label",
        "col3": "label",
        "col4": "count",
        "col5": "hash",
        "col6": "hash",
        "col7": "hash",
        "col8": "hash",
        "col9": "hash",
    }
    new_encodings = {"label": ["col4", "col5", "col6"], "hash": [["col1", "col2", "col3"]]}

    expected_encodings = {
        "label": ["col4", "col5", "col6"],
        "hash": [["col1", "col2", "col3"], ["col7"], ["col8", "col9"]],
    }
    updated_encodings = _reorder_encodings(adata, new_encodings)

    assert expected_encodings == updated_encodings


def test_update_encoding_scheme_2():
    # just a dummy adata object that won't be used actually
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    adata.uns["encoding_to_var"] = {
        "count": ["col4"],
        "hash": [["col5", "col6", "col7"], ["col8", "col9"]],
    }
    adata.uns["var_to_encoding"] = {
        "col4": "count",
        "col5": "hash",
        "col6": "hash",
        "col7": "hash",
        "col8": "hash",
        "col9": "hash",
    }
    new_encodings = {
        "label": ["col4", "col5", "col6", "col7", "col8", "col9"],
        "hash": [["col1", "col2", "col3"]],
    }

    expected_encodings = {
        "label": ["col4", "col5", "col6", "col7", "col8", "col9"],
        "hash": [["col1", "col2", "col3"]],
    }
    updated_encodings = _reorder_encodings(adata, new_encodings)

    assert expected_encodings == updated_encodings


def test_update_encoding_scheme_3():
    # just a dummy adata object that won't be used actually
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    adata.uns["encoding_to_var"] = {
        "label": ["col1", "col2", "col3"],
        "count": ["col4"],
        "hash": [["col5", "col6", "col7"], ["col8", "col9"]],
    }
    adata.uns["var_to_encoding"] = {
        "col1": "label",
        "col2": "label",
        "col3": "label",
        "col4": "count",
        "col5": "hash",
        "col6": "hash",
        "col7": "hash",
        "col8": "hash",
        "col9": "hash",
    }
    new_encodings = {
        "label": ["col10", "col11"],
        "hash": [["col12", "col13", "col14"]],
        "count": ["col15"],
    }

    expected_encodings = {
        "label": ["col10", "col11", "col1", "col2", "col3"],
        "hash": [["col12", "col13", "col14"], ["col5", "col6", "col7"], ["col8", "col9"]],
        "count": ["col15", "col4"],
    }
    updated_encodings = _reorder_encodings(adata, new_encodings)

    assert expected_encodings == updated_encodings


def test_update_encoding_scheme_4():
    # just a dummy adata objec that won't be used actually
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    adata.uns["encoding_to_var"] = {
        "label": ["col1", "col2", "col3"],
        "count": ["col4"],
        "hash": [["col5", "col6", "col7"], ["col8", "col9"]],
    }
    adata.uns["var_to_encoding"] = {
        "col1": "label",
        "col2": "label",
        "col3": "label",
        "col4": "count",
        "col5": "hash",
        "col6": "hash",
        "col7": "hash",
        "col8": "hash",
        "col9": "hash",
    }
    new_encodings = {
        "label": ["col1", "col2", "col3"],
        "count": ["col5", "col6", "col7", "col8", "col9"],
    }

    expected_encodings = {
        "label": ["col1", "col2", "col3"],
        "count": ["col5", "col6", "col7", "col8", "col9", "col4"],
    }
    updated_encodings = _reorder_encodings(adata, new_encodings)

    assert expected_encodings == updated_encodings


def test_update_encoding_scheme_5():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    adata.uns["encoding_to_var"] = {
        "label": ["col1", "col2", "col3"],
        "count": ["col4"],
        "hash": [["col5", "col6", "col7"], ["col8", "col9"]],
    }
    adata.uns["var_to_encoding"] = {
        "col1": "label",
        "col2": "label",
        "col3": "label",
        "col4": "count",
        "col5": "hash",
        "col6": "hash",
        "col7": "hash",
        "col8": "hash",
        "col9": "hash",
    }
    new_encodings = {"hash": [["col1", "col2", "col9"], ["col3", "col5"], ["col4", "col6", "col7"], ["col8"]]}

    expected_encodings = {"hash": [["col1", "col2", "col9"], ["col3", "col5"], ["col4", "col6", "col7"], ["col8"]]}
    updated_encodings = _reorder_encodings(adata, new_encodings)

    assert expected_encodings == updated_encodings


def test_update_encoding_scheme_duplicates_raise_error():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset1.csv")
    adata.uns["encoding_to_var"] = {
        "label": ["col1", "col2", "col3"],
        "count": ["col4"],
        "hash": [["col5", "col6", "col7"], ["col8", "col9"]],
    }
    adata.uns["var_to_encoding"] = {
        "col1": "label",
        "col2": "label",
        "col3": "label",
        "col4": "count",
        "col5": "hash",
        "col6": "hash",
        "col7": "hash",
        "col8": "hash",
        "col9": "hash",
    }
    new_encodings = {"hash": [["col1", "col2", "col9"], ["col3", "col9"], ["col4", "col6", "col7"], ["col8"]]}

    _ = {
        "label": ["col10"],
        "hash": [["col1", "col2", "col9"], ["col3", "col5"], ["col4", "col6", "col7"], ["col8"]],
    }
    with pytest.raises(DuplicateColumnEncodingError):
        _ = _reorder_encodings(adata, new_encodings)
