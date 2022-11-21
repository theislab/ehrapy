import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from ehrapy.io._read import read_csv
from ehrapy.preprocessing._data_imputation import (
    ImputeStrategyNotAvailableError,
    _warn_imputation_threshold,
    explicit_impute,
    iterative_svd_impute,
    knn_impute,
    matrix_factorization_impute,
    mice_forest_impute,
    miss_forest_impute,
    nuclear_norm_minimization_impute,
    simple_impute,
    soft_impute,
)

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_imputation"


class TestImputation:
    def test_mean_impute_no_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        simple_impute(adata)

        assert not np.isnan(adata.X).any()

    def test_mean_impute_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = simple_impute(adata, copy=True)

        assert id(adata) != id(adata_imputed)
        assert not np.isnan(adata_imputed.X).any()

    def test_mean_impute_throws_error_non_numerical(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")

        with pytest.raises(ImputeStrategyNotAvailableError):
            simple_impute(adata)

    def test_mean_impute_subset(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = simple_impute(adata, var_names=["intcol", "indexcol"], copy=True)

        assert not np.all([item != item for item in adata_imputed.X[::, 1:2]])
        assert np.any([item != item for item in adata_imputed.X[::, 3:4]])

    def test_median_impute_no_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        simple_impute(adata, strategy="median")

        assert not np.isnan(adata.X).any()

    def test_median_impute_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = simple_impute(adata, strategy="median", copy=True)

        assert id(adata) != id(adata_imputed)
        assert not np.isnan(adata_imputed.X).any()

    def test_median_impute_throws_error_non_numerical(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")

        with pytest.raises(ImputeStrategyNotAvailableError):
            simple_impute(adata, strategy="median")

    def test_median_impute_subset(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = simple_impute(adata, var_names=["intcol", "indexcol"], strategy="median", copy=True)

        assert not np.all([item != item for item in adata_imputed.X[::, 1:2]])
        assert np.any([item != item for item in adata_imputed.X[::, 3:4]])

    def test_most_frequent_impute_no_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        simple_impute(adata, strategy="most_frequent")

        assert not (np.all([item != item for item in adata.X]))

    def test_most_frequent_impute_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = simple_impute(adata, strategy="most_frequent", copy=True)

        assert id(adata) != id(adata_imputed)
        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_most_frequent_impute_subset(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = simple_impute(adata, var_names=["intcol", "strcol"], strategy="most_frequent", copy=True)

        assert not (np.all([item != item for item in adata_imputed.X[::, 1:3]]))

    def test_knn_impute_no_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        knn_impute(adata)

        assert not (np.all([item != item for item in adata.X]))

    def test_knn_impute_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = knn_impute(adata, n_neighbours=3, copy=True)

        assert id(adata) != id(adata_imputed)
        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_knn_impute_non_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = knn_impute(adata, n_neighbours=3, copy=True)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_knn_impute_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = knn_impute(adata, copy=True)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_knn_impute_list_str(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = knn_impute(adata, var_names=["intcol", "strcol", "boolcol"], copy=True)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_missforest_impute_non_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = miss_forest_impute(adata, copy=True)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_missforest_impute_numerical_data(self):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = miss_forest_impute(adata, copy=True)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_missforest_impute_subset(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = miss_forest_impute(
            adata, var_names={"non_numerical": ["intcol"], "numerical": ["strcol"]}, copy=True
        )

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_missforest_impute_list_str(self):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = miss_forest_impute(adata, var_names=["col1", "col2", "col3"], copy=True)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_missforest_impute_dict(self):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = miss_forest_impute(
            adata, var_names={"numerical": ["intcol", "datetime"], "non_numerical": ["strcol", "boolcol"]}, copy=True
        )

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_soft_impute_no_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = soft_impute(adata)

        assert id(adata) == id(adata_imputed)

    def test_soft_impute_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = soft_impute(adata, copy=True)

        assert id(adata) != id(adata_imputed)

    def test_soft_impute_non_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = soft_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_soft_impute_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = soft_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_soft_impute_list_str(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = soft_impute(adata, var_names=["intcol", "strcol", "boolcol"])

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_IterativeSVD_impute_no_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = iterative_svd_impute(adata, rank=2)

        assert id(adata) == id(adata_imputed)

    def test_IterativeSVD_impute_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = iterative_svd_impute(adata, rank=2, copy=True)

        assert id(adata) != id(adata_imputed)

    def test_IterativeSVD_impute_non_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = iterative_svd_impute(adata, rank=3)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_IterativeSVD_impute_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = iterative_svd_impute(adata, rank=2)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_IterativeSVD_impute_list_str(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = iterative_svd_impute(adata, var_names=["intcol", "strcol", "boolcol"], rank=2)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_matrix_factorization_impute_no_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = matrix_factorization_impute(adata)

        assert id(adata) == id(adata_imputed)

    def test_matrix_factorization_impute_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = matrix_factorization_impute(adata, copy=True)

        assert id(adata) != id(adata_imputed)

    def test_matrix_factorization_impute_non_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = matrix_factorization_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_matrix_factorization_impute_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = matrix_factorization_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_matrix_factorization_impute_list_str(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = matrix_factorization_impute(adata, var_names=["intcol", "strcol", "boolcol"])

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_nuclear_norm_minimization_impute_no_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = nuclear_norm_minimization_impute(adata)

        assert id(adata) == id(adata_imputed)

    def test_nuclear_norm_minimization_impute_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = nuclear_norm_minimization_impute(adata, copy=True)

        assert id(adata) != id(adata_imputed)

    def test_nuclear_norm_minimization_impute_non_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = nuclear_norm_minimization_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_nuclear_norm_minimization_impute_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = nuclear_norm_minimization_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_nuclear_norm_minimization_impute_list_str(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = nuclear_norm_minimization_impute(adata, var_names=["intcol", "strcol", "boolcol"])

        assert not (np.all([item != item for item in adata_imputed.X]))

    @pytest.mark.skipif(os.name == "posix", reason="miceforest Imputation not supported by MacOS.")
    def test_miceforest_impute_no_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_iris.csv")
        adata_imputed = mice_forest_impute(adata)

        assert id(adata) == id(adata_imputed)

    @pytest.mark.skipif(os.name == "posix", reason="miceforest Imputation not supported by MacOS.")
    def test_miceforest_impute_copy(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_iris.csv")
        adata_imputed = mice_forest_impute(adata, copy=True)

        assert id(adata) != id(adata_imputed)

    @pytest.mark.skipif(os.name == "posix", reason="miceforest Imputation not supported by MacOS.")
    def test_miceforest_impute_non_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_titanic.csv")
        adata_imputed = mice_forest_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    @pytest.mark.skipif(os.name == "posix", reason="miceforest Imputation not supported by MacOS.")
    def test_miceforest_impute_numerical_data(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_iris.csv")
        adata_imputed = mice_forest_impute(adata)

        assert not (np.all([item != item for item in adata_imputed.X]))

    @pytest.mark.skipif(os.name == "posix", reason="miceforest Imputation not supported by MacOS.")
    def test_miceforest_impute_list_str(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_titanic.csv")
        adata_imputed = mice_forest_impute(adata, var_names=["Cabin", "Age"])

        assert not (np.all([item != item for item in adata_imputed.X]))

    def test_explicit_impute_all(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        adata_imputed = explicit_impute(adata, replacement=1011, copy=True)

        assert (adata_imputed.X == 1011).sum() == 3

    def test_explicit_impute_subset(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
        adata_imputed = explicit_impute(adata, replacement={"strcol": "REPLACED", "intcol": 1011}, copy=True)

        assert (adata_imputed.X == 1011).sum() == 1
        assert (adata_imputed.X == "REPLACED").sum() == 1

    def test_warning(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
        warning_results = _warn_imputation_threshold(adata, threshold=20, var_names=None)
        assert warning_results == {"col1": 25, "col3": 50}
