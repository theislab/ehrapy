from pathlib import Path

import pytest

from ehrapy.api.encode.encode import Encoder
from ehrapy.api.io.read import DataReader

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_encode"


class TestRead:
    def test_unknown_encode_mode(self):
        adata = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv")
        with pytest.raises(ValueError):
            encoded_ann_data = Encoder.encode(  # noqa: F841
                adata, autodetect=False, encodings={"unknown_mode": ["survival"]}
            )

    def test_duplicate_column_encoding(self):
        adata = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv")
        with pytest.raises(ValueError):
            encoded_ann_data = Encoder.encode(  # noqa: F841
                adata,
                autodetect=False,
                encodings={"label_encoding": ["survival"], "count_encoding": ["survival"]},
            )

    def test_autodetect_encode(self):
        adata = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv")
        encoded_ann_data = Encoder.encode(adata, autodetect=True, encodings={})
        assert list(encoded_ann_data.obs.columns) == ["survival", "clinic_day"]
        assert list(encoded_ann_data.var_names) == [
            "ehrapycat_survival",
            "ehrapycat_clinic_day",
            "patient_id",
            "los_days",
            "b12_values",
        ]

        assert encoded_ann_data.uns["current_encodings"] == {
            "survival": "one_hot_encoding",
            "clinic_day": "one_hot_encoding",
        }
        assert id(encoded_ann_data.X) != id(encoded_ann_data.layers["original"])

    def test_autodetect_encode_again(self):
        adata = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv")
        encoded_ann_data = Encoder.encode(adata, autodetect=True, encodings={})
        encoded_ann_data_again = Encoder.encode(encoded_ann_data, autodetect=True, encodings={})  # noqa: F841
        assert encoded_ann_data_again is None

    def test_custom_encode(self):
        adata = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv")
        encoded_ann_data = Encoder.encode(
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
        assert encoded_ann_data.uns["current_encodings"] == {
            "survival": "label_encoding",
            "clinic_day": "one_hot_encoding",
        }
        assert id(encoded_ann_data.X) != id(encoded_ann_data.layers["original"])

    def test_custom_encode_again_single_columns_encoding(self):
        adata = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv")
        encoded_ann_data = Encoder.encode(
            adata,
            autodetect=False,
            encodings={"label_encoding": ["survival"], "one_hot_encoding": ["clinic_day"]},
        )
        encoded_ann_data_again = Encoder.encode(
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
        assert encoded_ann_data_again.uns["current_encodings"] == {
            "survival": "label_encoding",
            "clinic_day": "label_encoding",
        }
        assert id(encoded_ann_data_again.X) != id(encoded_ann_data_again.layers["original"])

    def test_custom_encode_again_multiple_columns_encoding(self):
        adata = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv")
        encoded_ann_data = Encoder.encode(
            adata, autodetect=False, encodings={"one_hot_encoding": ["clinic_day", "survival"]}
        )
        encoded_ann_data_again = Encoder.encode(
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
        assert encoded_ann_data_again.uns["current_encodings"] == {
            "survival": "label_encoding",
            "clinic_day": "count_encoding",
        }
        assert id(encoded_ann_data_again.X) != id(encoded_ann_data_again.layers["original"])
