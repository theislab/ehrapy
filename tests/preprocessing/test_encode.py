from pathlib import Path

import pytest

from ehrapy.io._read import read
from ehrapy.preprocessing.encoding._encode import DuplicateColumnEncodingError, _reorder_encodings, encode

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_encode"


class TestEncode:
    def test_unknown_encode_mode(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        with pytest.raises(ValueError):
            encoded_ann_data = encode(adata, autodetect=False, encodings={"unknown_mode": ["survival"]})  # noqa: F841

    def test_duplicate_column_encoding(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        with pytest.raises(ValueError):
            encoded_ann_data = encode(  # noqa: F841
                adata,
                autodetect=False,
                encodings={"label_encoding": ["survival"], "count_encoding": ["survival"]},
            )

    def test_autodetect_encode(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        encoded_ann_data = encode(adata, autodetect=True)
        assert list(encoded_ann_data.obs.columns) == ["survival", "clinic_day"]
        assert set(encoded_ann_data.var_names) == {
            "ehrapycat_survival",
            "ehrapycat_clinic_day",
            "patient_id",
            "los_days",
            "b12_values",
        }

        assert encoded_ann_data.uns["var_to_encoding"] == {
            "survival": "label_encoding",
            "clinic_day": "label_encoding",
        }
        assert id(encoded_ann_data.X) != id(encoded_ann_data.layers["original"])

    def test_autodetect_encode_again(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        encoded_ann_data = encode(adata, autodetect=True, encodings={})
        encoded_ann_data_again = encode(encoded_ann_data, autodetect=True)  # noqa: F841
        assert encoded_ann_data_again is None

    def test_custom_encode(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        encoded_ann_data = encode(
            adata,
            autodetect=False,
            encodings={"label_encoding": ["survival"], "one_hot_encoding": ["clinic_day"]},
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
            "survival": "label_encoding",
            "clinic_day": "one_hot_encoding",
        }
        assert id(encoded_ann_data.X) != id(encoded_ann_data.layers["original"])

    def test_custom_encode_again_single_columns_encoding(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        encoded_ann_data = encode(
            adata,
            autodetect=False,
            encodings={"label_encoding": ["survival"], "one_hot_encoding": ["clinic_day"]},
        )
        encoded_ann_data_again = encode(
            encoded_ann_data, autodetect=False, encodings={"label_encoding": ["clinic_day"]}
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
        assert encoded_ann_data_again.uns["var_to_encoding"] == {
            "survival": "label_encoding",
            "clinic_day": "label_encoding",
        }
        assert id(encoded_ann_data_again.X) != id(encoded_ann_data_again.layers["original"])

    def test_custom_encode_again_multiple_columns_encoding(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        encoded_ann_data = encode(adata, autodetect=False, encodings={"one_hot_encoding": ["clinic_day", "survival"]})
        encoded_ann_data_again = encode(
            encoded_ann_data,
            autodetect=False,
            encodings={"label_encoding": ["survival"], "count_encoding": ["clinic_day"]},
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
            "survival": "label_encoding",
            "clinic_day": "count_encoding",
        }
        assert id(encoded_ann_data_again.X) != id(encoded_ann_data_again.layers["original"])

    def test_update_encoding_scheme_1(self):
        # just a dummy adata object that won't be used actually
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        adata.uns["encoding_to_var"] = {
            "label_encoding": ["col1", "col2", "col3"],
            "count_encoding": ["col4"],
            "hash_encoding": [["col5", "col6", "col7"], ["col8", "col9"]],
        }
        adata.uns["var_to_encoding"] = {
            "col1": "label_encoding",
            "col2": "label_encoding",
            "col3": "label_encoding",
            "col4": "count_encoding",
            "col5": "hash_encoding",
            "col6": "hash_encoding",
            "col7": "hash_encoding",
            "col8": "hash_encoding",
            "col9": "hash_encoding",
        }
        new_encodings = {"label_encoding": ["col4", "col5", "col6"], "hash_encoding": [["col1", "col2", "col3"]]}

        expected_encodings = {
            "label_encoding": ["col4", "col5", "col6"],
            "hash_encoding": [["col1", "col2", "col3"], ["col7"], ["col8", "col9"]],
        }
        updated_encodings = _reorder_encodings(adata, new_encodings)

        assert expected_encodings == updated_encodings

    def test_update_encoding_scheme_2(self):
        # just a dummy adata object that won't be used actually
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        adata.uns["encoding_to_var"] = {
            "count_encoding": ["col4"],
            "hash_encoding": [["col5", "col6", "col7"], ["col8", "col9"]],
        }
        adata.uns["var_to_encoding"] = {
            "col4": "count_encoding",
            "col5": "hash_encoding",
            "col6": "hash_encoding",
            "col7": "hash_encoding",
            "col8": "hash_encoding",
            "col9": "hash_encoding",
        }
        new_encodings = {
            "label_encoding": ["col4", "col5", "col6", "col7", "col8", "col9"],
            "hash_encoding": [["col1", "col2", "col3"]],
        }

        expected_encodings = {
            "label_encoding": ["col4", "col5", "col6", "col7", "col8", "col9"],
            "hash_encoding": [["col1", "col2", "col3"]],
        }
        updated_encodings = _reorder_encodings(adata, new_encodings)

        assert expected_encodings == updated_encodings

    def test_update_encoding_scheme_3(self):
        # just a dummy adata object that won't be used actually
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        adata.uns["encoding_to_var"] = {
            "label_encoding": ["col1", "col2", "col3"],
            "count_encoding": ["col4"],
            "hash_encoding": [["col5", "col6", "col7"], ["col8", "col9"]],
        }
        adata.uns["var_to_encoding"] = {
            "col1": "label_encoding",
            "col2": "label_encoding",
            "col3": "label_encoding",
            "col4": "count_encoding",
            "col5": "hash_encoding",
            "col6": "hash_encoding",
            "col7": "hash_encoding",
            "col8": "hash_encoding",
            "col9": "hash_encoding",
        }
        new_encodings = {
            "label_encoding": ["col10", "col11"],
            "hash_encoding": [["col12", "col13", "col14"]],
            "count_encoding": ["col15"],
        }

        expected_encodings = {
            "label_encoding": ["col10", "col11", "col1", "col2", "col3"],
            "hash_encoding": [["col12", "col13", "col14"], ["col5", "col6", "col7"], ["col8", "col9"]],
            "count_encoding": ["col15", "col4"],
        }
        updated_encodings = _reorder_encodings(adata, new_encodings)

        assert expected_encodings == updated_encodings

    def test_update_encoding_scheme_4(self):
        # just a dummy adata objec that won't be used actually
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        adata.uns["encoding_to_var"] = {
            "label_encoding": ["col1", "col2", "col3"],
            "count_encoding": ["col4"],
            "hash_encoding": [["col5", "col6", "col7"], ["col8", "col9"]],
        }
        adata.uns["var_to_encoding"] = {
            "col1": "label_encoding",
            "col2": "label_encoding",
            "col3": "label_encoding",
            "col4": "count_encoding",
            "col5": "hash_encoding",
            "col6": "hash_encoding",
            "col7": "hash_encoding",
            "col8": "hash_encoding",
            "col9": "hash_encoding",
        }
        new_encodings = {
            "label_encoding": ["col1", "col2", "col3"],
            "count_encoding": ["col5", "col6", "col7", "col8", "col9"],
        }

        expected_encodings = {
            "label_encoding": ["col1", "col2", "col3"],
            "count_encoding": ["col5", "col6", "col7", "col8", "col9", "col4"],
        }
        updated_encodings = _reorder_encodings(adata, new_encodings)

        assert expected_encodings == updated_encodings

    def test_update_encoding_scheme_5(self):
        # just a dummy adata objec that won't be used actually
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        adata.uns["encoding_to_var"] = {
            "label_encoding": ["col1", "col2", "col3"],
            "count_encoding": ["col4"],
            "hash_encoding": [["col5", "col6", "col7"], ["col8", "col9"]],
        }
        adata.uns["var_to_encoding"] = {
            "col1": "label_encoding",
            "col2": "label_encoding",
            "col3": "label_encoding",
            "col4": "count_encoding",
            "col5": "hash_encoding",
            "col6": "hash_encoding",
            "col7": "hash_encoding",
            "col8": "hash_encoding",
            "col9": "hash_encoding",
        }
        new_encodings = {
            "hash_encoding": [["col1", "col2", "col9"], ["col3", "col5"], ["col4", "col6", "col7"], ["col8"]]
        }

        expected_encodings = {
            "hash_encoding": [["col1", "col2", "col9"], ["col3", "col5"], ["col4", "col6", "col7"], ["col8"]]
        }
        updated_encodings = _reorder_encodings(adata, new_encodings)

        assert expected_encodings == updated_encodings

    def test_update_encoding_scheme_duplicates_raise_error(self):
        # just a dummy adata objec that won't be used actually
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        adata.uns["encoding_to_var"] = {
            "label_encoding": ["col1", "col2", "col3"],
            "count_encoding": ["col4"],
            "hash_encoding": [["col5", "col6", "col7"], ["col8", "col9"]],
        }
        adata.uns["var_to_encoding"] = {
            "col1": "label_encoding",
            "col2": "label_encoding",
            "col3": "label_encoding",
            "col4": "count_encoding",
            "col5": "hash_encoding",
            "col6": "hash_encoding",
            "col7": "hash_encoding",
            "col8": "hash_encoding",
            "col9": "hash_encoding",
        }
        new_encodings = {
            "hash_encoding": [["col1", "col2", "col9"], ["col3", "col9"], ["col4", "col6", "col7"], ["col8"]]
        }

        _ = {
            "label_encoding": ["col10"],
            "hash_encoding": [["col1", "col2", "col9"], ["col3", "col5"], ["col4", "col6", "col7"], ["col8"]],
        }
        with pytest.raises(DuplicateColumnEncodingError):
            _ = _reorder_encodings(adata, new_encodings)
