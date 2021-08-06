from pathlib import Path

import numpy as np

from ehrapy.api.io.read import DataReader

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_io"


class TestRead:
    def test_read_csv(self):
        ann_data = DataReader.read(filename=f"{_TEST_PATH}/dataset1.csv")
        matrix = np.array(
            [[14, 500, "false"], [7, 330, "false"], [10, 800, "true"], [11, 765, "true"], [3, 800, "true"]]
        )
        assert ann_data.X.shape == (5, 3)
        assert (ann_data.X == matrix).all()
        assert ann_data.var_names.to_list() == ["los_days", "b12_values", "survival"]
        assert (ann_data.layers["original"] == matrix).all()
        assert id(ann_data.layers["original"]) != id(ann_data.X)

    def test_read_tsv(self):
        ann_data = DataReader.read(filename=f"{_TEST_PATH}/dataset2.tsv")
        matrix = np.array(
            [
                [54, 185.34, "FALSE"],
                [25, 175.39, "TRUE"],
                [36, 183.29, "FALSE"],
                [44, 173.93, "TRUE"],
                [27, 190.32, "TRUE"],
            ]
        )
        assert ann_data.X.shape == (5, 3)
        assert (ann_data.X == matrix).all()
        assert ann_data.var_names.to_list() == ["age", "height", "gamer"]
        assert (ann_data.layers["original"] == matrix).all()
        assert id(ann_data.layers["original"]) != id(ann_data.X)
