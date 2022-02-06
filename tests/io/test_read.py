import os
from pathlib import Path

import numpy as np
import pytest

from ehrapy._util import shell_command_accessible
from ehrapy.anndata_ext import ColumnNotFoundError
from ehrapy.io._read import read

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_io"


class TestRead:
    def test_read_csv(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv")
        matrix = np.array(
            [[12, 14, 500, False], [13, 7, 330, False], [14, 10, 800, True], [15, 11, 765, True], [16, 3, 800, True]]
        )
        assert adata.X.shape == (5, 4)
        assert (adata.X == matrix).all()
        assert adata.var_names.to_list() == ["patient_id", "los_days", "b12_values", "survival"]
        assert (adata.layers["original"] == matrix).all()
        assert id(adata.layers["original"]) != id(adata.X)

    def test_read_tsv(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset2.tsv", delimiter="\t")
        matrix = np.array(
            [
                [12, 54, 185.34, False],
                [13, 25, 175.39, True],
                [14, 36, 183.29, False],
                [15, 44, 173.93, True],
                [16, 27, 190.32, True],
            ]
        )
        assert adata.X.shape == (5, 4)
        assert (adata.X == matrix).all()
        assert adata.var_names.to_list() == ["patient_id", "age", "height", "gamer"]
        assert (adata.layers["original"] == matrix).all()
        assert id(adata.layers["original"]) != id(adata.X)

    def test_read_csv_without_index_column(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset3.csv")
        matrix = np.array(
            [[1, 14, 500, False], [2, 7, 330, False], [3, 10, 800, True], [4, 11, 765, True], [5, 3, 800, True]]
        )
        assert adata.X.shape == (5, 4)
        assert (adata.X == matrix).all()
        assert adata.var_names.to_list() == ["clinic_id", "los_days", "b12_values", "survival"]
        assert (adata.layers["original"] == matrix).all()
        assert id(adata.layers["original"]) != id(adata.X)
        assert list(adata.obs.index) == ["0", "1", "2", "3", "4"]

    @pytest.mark.skipif(
        (os.name != "nt" and not shell_command_accessible(["gs", "-h"]))
        or (os.name == "nt" and not shell_command_accessible(["gswin64c", " -v"])),
        reason="Requires ghostscript to be installed.",
    )
    def test_read_pdf(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_pdf.pdf")["test_pdf_0"]
        assert adata.X.shape == (32, 11)
        assert adata.var_names.to_list() == [
            "mpg",
            "cyl",
            "disp",
            "hp",
            "drat",
            "wt",
            "qsec",
            "vs",
            "am",
            "gear",
            "carb",
        ]
        assert id(adata.layers["original"]) != id(adata.X)

    @pytest.mark.skipif(
        (os.name != "nt" and not shell_command_accessible(["gs", "-h"]))
        or (os.name == "nt" and not shell_command_accessible(["gswin64c", " -v"])),
        reason="Requires ghostscript to be installed.",
    )
    def test_read_pdf_no_index(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_pdf.pdf")["test_pdf_1"]
        assert adata.X.shape == (6, 5)
        assert adata.var_names.to_list() == [
            "Sepal.Length",
            "Sepal.Width",
            "Petal.Length",
            "Petal.Width",
            "Species",
        ]
        assert id(adata.layers["original"]) != id(adata.X)

    def test_set_default_index(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset3.csv")
        assert adata.X.shape == (5, 4)
        assert not adata.obs_names.name
        assert list(adata.obs.index.values) == [f"{i}" for i in range(5)]

    def test_set_given_str_index(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv", index_column="los_days")
        assert adata.X.shape == (5, 3)
        assert adata.obs_names.name == "los_days"
        assert list(adata.obs.index.values) == ["14", "7", "10", "11", "3"]

    def test_set_given_int_index(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv", index_column=1)
        assert adata.X.shape == (5, 3)
        assert adata.obs_names.name == "los_days"
        assert list(adata.obs.index.values) == ["14", "7", "10", "11", "3"]

    def test_move_single_column_misspelled(self):
        with pytest.raises(ColumnNotFoundError):
            adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv", columns_obs_only=["b11_values"])  # noqa: F841

    def test_move_single_column_to_obs(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv", columns_obs_only=["b12_values"])
        assert adata.X.shape == (5, 3)
        assert list(adata.obs.columns) == ["b12_values"]
        assert "b12_values" not in list(adata.var_names.values)

    def test_move_multiple_columns_to_obs(self):
        adata = read(dataset_path=f"{_TEST_PATH}/dataset1.csv", columns_obs_only=["b12_values", "survival"])
        assert adata.X.shape == (5, 2)
        assert list(adata.obs.columns) == ["b12_values", "survival"]
        assert "b12_values" not in list(adata.var_names.values) and "survival" not in list(adata.var_names.values)
