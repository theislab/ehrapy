import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from ehrapy.api.io import read
from ehrapy.api.preprocessing import explicit_impute, knn_impute, miss_forest_impute, simple_impute
from ehrapy.api.preprocessing._data_imputation import ImputeStrategyNotAvailableError

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_imputation"


class TestImputation:
    def test_mean_impute_no_copy(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = simple_impute(adata)

        assert id(adata) == id(adata_imputed)
        assert not np.isnan(adata_imputed.X).any()

    def test_mean_impute_copy(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = simple_impute(adata, copy=True)

        assert id(adata) != id(adata_imputed)
        assert not np.isnan(adata_imputed.X).any()

    def test_mean_impute_throws_error_non_numerical(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute.csv")

        with pytest.raises(ImputeStrategyNotAvailableError):
            _ = simple_impute(adata)

    def test_mean_impute_subset(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = simple_impute(adata, var_names=["intcol", "indexcol"])

        assert not np.all([item != item for item in adata_imputed.X[::, 1:2]])
        assert np.any([item != item for item in adata_imputed.X[::, 3:4]])

    def test_median_impute_no_copy(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = simple_impute(adata, strategy="median")

        assert id(adata) == id(adata_imputed)
        assert not np.isnan(adata_imputed.X).any()

    def test_median_impute_copy(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = simple_impute(adata, strategy="median", copy=True)

        assert id(adata) != id(adata_imputed)
        assert not np.isnan(adata_imputed.X).any()

    def test_median_impute_throws_error_non_numerical(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute.csv")

        with pytest.raises(ImputeStrategyNotAvailableError):
            _ = simple_impute(adata, strategy="median")

    def test_median_impute_subset(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = simple_impute(adata, var_names=["intcol", "indexcol"], strategy="median")

        assert not np.all([item != item for item in adata_imputed.X[::, 1:2]])
        assert np.any([item != item for item in adata_imputed.X[::, 3:4]])

    def test_most_frequent_impute_no_copy(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = simple_impute(adata, strategy="most_frequent")

        assert id(adata) == id(adata_imputed)
        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_most_frequent_impute_copy(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = simple_impute(adata, strategy="most_frequent", copy=True)

        assert id(adata) != id(adata_imputed)
        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_most_frequent_impute_subset(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = simple_impute(adata, var_names=["intcol", "strcol"], strategy="most_frequent")

        assert not (np.all([item != item for item in adata_imputed.X[::, 1:3]]))

    def test_knn_impute_no_copy(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = knn_impute(adata)

        assert id(adata) == id(adata_imputed)

    def test_knn_impute_copy(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = knn_impute(adata, copy=True)

        assert id(adata) != id(adata_imputed)

    def test_knn_impute_non_numerical_data(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = knn_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_knn_impute_numerical_data(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = knn_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_missforest_impute_non_numerical_data(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = miss_forest_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_missforest_impute_numerical_data(self):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = miss_forest_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_missforest_impute_subset(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = miss_forest_impute(adata, var_names={"non_numerical": ["intcol"], "numerical": ["strcol"]})

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_explicit_impute_all(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = explicit_impute(adata, replacement=1011)

        assert (adata_imputed.X == 1011).sum() == 3

    def test_explicit_impute_subset(self):
        adata = read(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = explicit_impute(adata, replacement={"strcol": "REPLACED", "intcol": 1011})

        assert (adata_imputed.X == 1011).sum() == 1
        assert (adata_imputed.X == "REPLACED").sum() == 1
