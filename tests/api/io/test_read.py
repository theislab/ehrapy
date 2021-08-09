import warnings
from pathlib import Path

import numpy as np

from ehrapy.api.io.read import DataReader, IndexColumnWarning

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
