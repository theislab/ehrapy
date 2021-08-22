import warnings
from pathlib import Path

import numpy as np
import pytest

from ehrapy.api.io.read import ColumnNotFoundError, DataReader, IndexColumnWarning

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_io"


class TestRead:
    def test_read_csv(self):
        ann_data = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv")
        matrix = np.array([[14, 500, False], [7, 330, False], [10, 800, True], [11, 765, True], [3, 800, True]])
        assert ann_data.X.shape == (5, 3)
        assert (ann_data.X == matrix).all()
        assert ann_data.var_names.to_list() == ["los_days", "b12_values", "survival"]
        assert (ann_data.layers["original"] == matrix).all()
        assert id(ann_data.layers["original"]) != id(ann_data.X)
        assert list(ann_data.obs.index) == ["12", "13", "14", "15", "16"]

    def test_read_tsv(self):
        ann_data = DataReader.read(filename=f"{_TEST_PATH}/dataset2.tsv", delimiter="\t")
        matrix = np.array(
            [
                [54, 185.34, False],
                [25, 175.39, True],
                [36, 183.29, False],
                [44, 173.93, True],
                [27, 190.32, True],
            ]
        )
        assert ann_data.X.shape == (5, 3)
        assert (ann_data.X == matrix).all()
        assert ann_data.var_names.to_list() == ["age", "height", "gamer"]
        assert (ann_data.layers["original"] == matrix).all()
        assert id(ann_data.layers["original"]) != id(ann_data.X)
        assert list(ann_data.obs.index) == ["12", "13", "14", "15", "16"]

    def test_read_csv_without_index_column(self):
        with warnings.catch_warnings(record=True) as w:
            ann_data = DataReader.read(filename=f"{_TEST_PATH}/dataset3.csv")
            matrix = np.array(
                [[1, 14, 500, False], [2, 7, 330, False], [3, 10, 800, True], [4, 11, 765, True], [5, 3, 800, True]]
            )
            assert ann_data.X.shape == (5, 4)
            assert (ann_data.X == matrix).all()
            assert ann_data.var_names.to_list() == ["clinic_id", "los_days", "b12_values", "survival"]
            assert (ann_data.layers["original"] == matrix).all()
            assert id(ann_data.layers["original"]) != id(ann_data.X)
            assert list(ann_data.obs.index) == ["0", "1", "2", "3", "4"]
            assert len(w) == 1
            assert issubclass(w[-1].category, IndexColumnWarning)

    def test_set_default_index(self):
        with warnings.catch_warnings(record=True) as w:
            ann_data = DataReader.read(filename=f"{_TEST_PATH}/dataset3.csv")
            assert ann_data.X.shape == (5, 4)
            assert not ann_data.obs_names.name
            assert list(ann_data.obs.index.values) == [f"{i}" for i in range(5)]
            assert len(w) == 1
            assert issubclass(w[-1].category, IndexColumnWarning)

    def test_set_patient_id_index(self):
        ann_data = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv")
        assert ann_data.X.shape == (5, 3)
        assert ann_data.obs_names.name == "patient_id"
        assert list(ann_data.obs.index.values) == [f"{i}" for i in range(12, 17)]

    def test_set_given_str_index(self):
        ann_data = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv", index_column="los_days")
        assert ann_data.X.shape == (5, 3)
        assert ann_data.obs_names.name == "los_days"
        assert list(ann_data.obs.index.values) == ["14", "7", "10", "11", "3"]

    def test_set_given_int_index(self):
        ann_data = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv", index_column=1)
        assert ann_data.X.shape == (5, 3)
        assert ann_data.obs_names.name == "los_days"
        assert list(ann_data.obs.index.values) == ["14", "7", "10", "11", "3"]

    def test_move_single_column_misspelled(self):
        with pytest.raises(ColumnNotFoundError):
            ann_data = DataReader.read(  # noqa: F841
                filename=f"{_TEST_PATH}/dataset1.csv", columns_obs_only=["b11_values"]
            )

    def test_move_single_column_to_obs(self):
        ann_data = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv", columns_obs_only=["b12_values"])
        assert ann_data.X.shape == (5, 2)
        assert list(ann_data.obs.columns) == ["b12_values"]
        assert "b12_values" not in list(ann_data.var_names.values)

    def test_move_multiple_columns_to_obs(self):
        ann_data = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv", columns_obs_only=["b12_values", "survival"])
        assert ann_data.X.shape == (5, 1)
        assert list(ann_data.obs.columns) == ["b12_values", "survival"]
        assert "b12_values" not in list(ann_data.var_names.values) and "survival" not in list(ann_data.var_names.values)
