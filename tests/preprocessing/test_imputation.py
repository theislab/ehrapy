import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from ehrapy.io._read import read_csv
from ehrapy.preprocessing._imputation import (
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


@pytest.fixture
def impute_num_adata():
    return read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")


@pytest.fixture
def impute_adata():
    return read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")


@pytest.fixture
def impute_iris():
    return read_csv(dataset_path=f"{_TEST_PATH}/test_impute_iris.csv")


@pytest.fixture
def impute_titanic():
    return read_csv(dataset_path=f"{_TEST_PATH}/test_impute_titanic.csv")


def test_mean_impute_no_copy(impute_num_adata):
    simple_impute(impute_num_adata)

    assert not np.isnan(impute_num_adata.X).any()


def test_mean_impute_copy(impute_num_adata):
    adata_imputed = simple_impute(impute_num_adata, copy=True)

    assert id(impute_num_adata) != id(adata_imputed)
    assert not np.isnan(adata_imputed.X).any()


def test_mean_impute_throws_error_non_numerical(impute_adata):
    with pytest.raises(ValueError):
        simple_impute(impute_adata)


def test_mean_impute_subset(impute_adata):
    adata_imputed = simple_impute(impute_adata, var_names=["intcol", "indexcol"], copy=True)

    assert not np.all([item != item for item in adata_imputed.X[::, 1:2]])
    assert np.any([item != item for item in adata_imputed.X[::, 3:4]])


def test_median_impute_no_copy(impute_num_adata):
    simple_impute(impute_num_adata, strategy="median")

    assert not np.isnan(impute_num_adata.X).any()


def test_median_impute_copy(impute_num_adata):
    adata_imputed = simple_impute(impute_num_adata, strategy="median", copy=True)

    assert id(impute_adata) != id(adata_imputed)
    assert not np.isnan(adata_imputed.X).any()


def test_median_impute_throws_error_non_numerical(impute_adata):
    with pytest.raises(ValueError):
        simple_impute(impute_adata, strategy="median")


def test_median_impute_subset(impute_adata):
    adata_imputed = simple_impute(impute_adata, var_names=["intcol", "indexcol"], strategy="median", copy=True)

    assert not np.all([item != item for item in adata_imputed.X[::, 1:2]])
    assert np.any([item != item for item in adata_imputed.X[::, 3:4]])


def test_most_frequent_impute_no_copy(impute_adata):
    simple_impute(impute_adata, strategy="most_frequent")

    assert not (np.all([item != item for item in impute_adata.X]))


def test_most_frequent_impute_copy(impute_adata):
    adata_imputed = simple_impute(impute_adata, strategy="most_frequent", copy=True)

    assert id(impute_adata) != id(adata_imputed)
    assert not (np.all([item != item for item in adata_imputed.X]))


def test_most_frequent_impute_subset(impute_adata):
    adata_imputed = simple_impute(impute_adata, var_names=["intcol", "strcol"], strategy="most_frequent", copy=True)

    assert not (np.all([item != item for item in adata_imputed.X[::, 1:3]]))


def test_knn_impute_no_copy(impute_num_adata):
    knn_impute(impute_num_adata)

    assert not (np.all([item != item for item in impute_num_adata.X]))


def test_knn_impute_copy(impute_num_adata):
    adata_imputed = knn_impute(impute_num_adata, n_neighbours=3, copy=True)

    assert id(impute_num_adata) != id(adata_imputed)
    assert not (np.all([item != item for item in adata_imputed.X]))


def test_knn_impute_non_numerical_data(impute_adata):
    adata_imputed = knn_impute(impute_adata, n_neighbours=3, copy=True)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_knn_impute_numerical_data(impute_num_adata):
    adata_imputed = knn_impute(impute_num_adata, copy=True)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_knn_impute_list_str(impute_adata):
    adata_imputed = knn_impute(impute_adata, var_names=["intcol", "strcol", "boolcol"], copy=True)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_missforest_impute_non_numerical_data(impute_adata):
    adata_imputed = miss_forest_impute(impute_adata, copy=True)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_missforest_impute_numerical_data(impute_num_adata):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    adata_imputed = miss_forest_impute(impute_num_adata, copy=True)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_missforest_impute_subset(impute_num_adata):
    adata_imputed = miss_forest_impute(
        impute_num_adata, var_names={"non_numerical": ["intcol"], "numerical": ["strcol"]}, copy=True
    )

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_missforest_impute_list_str(impute_num_adata):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    adata_imputed = miss_forest_impute(impute_num_adata, var_names=["col1", "col2", "col3"], copy=True)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_missforest_impute_dict(impute_adata):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    adata_imputed = miss_forest_impute(
        impute_adata, var_names={"numerical": ["intcol", "datetime"], "non_numerical": ["strcol", "boolcol"]}, copy=True
    )

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_soft_impute_no_copy(impute_num_adata):
    adata_imputed = soft_impute(impute_num_adata)

    assert id(impute_num_adata) == id(adata_imputed)


def test_soft_impute_copy(impute_num_adata):
    adata_imputed = soft_impute(impute_num_adata, copy=True)

    assert id(impute_num_adata) != id(adata_imputed)


def test_soft_impute_non_numerical_data(impute_adata):
    adata_imputed = soft_impute(impute_adata)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_soft_impute_numerical_data(impute_num_adata):
    adata_imputed = soft_impute(impute_num_adata)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_soft_impute_list_str(impute_adata):
    adata_imputed = soft_impute(impute_adata, var_names=["intcol", "strcol", "boolcol"])

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_IterativeSVD_impute_no_copy(impute_num_adata):
    adata_imputed = iterative_svd_impute(impute_num_adata, rank=2)

    assert id(impute_num_adata) == id(adata_imputed)


def test_IterativeSVD_impute_copy(impute_num_adata):
    adata_imputed = iterative_svd_impute(impute_num_adata, rank=2, copy=True)

    assert id(impute_adata) != id(adata_imputed)


def test_IterativeSVD_impute_non_numerical_data(impute_adata):
    adata_imputed = iterative_svd_impute(impute_adata, rank=3)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_IterativeSVD_impute_numerical_data(impute_num_adata):
    adata_imputed = iterative_svd_impute(impute_num_adata, rank=2)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_IterativeSVD_impute_list_str(impute_adata):
    adata_imputed = iterative_svd_impute(impute_adata, var_names=["intcol", "strcol", "boolcol"], rank=2)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_matrix_factorization_impute_no_copy(impute_num_adata):
    adata_imputed = matrix_factorization_impute(impute_num_adata)

    assert id(impute_num_adata) == id(adata_imputed)


def test_matrix_factorization_impute_copy(impute_num_adata):
    adata_imputed = matrix_factorization_impute(impute_num_adata, copy=True)

    assert id(impute_num_adata) != id(adata_imputed)


def test_matrix_factorization_impute_non_numerical_data(impute_adata):
    adata_imputed = matrix_factorization_impute(impute_adata)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_matrix_factorization_impute_numerical_data(impute_adata):
    adata_imputed = matrix_factorization_impute(impute_adata)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_matrix_factorization_impute_list_str(impute_adata):
    adata_imputed = matrix_factorization_impute(impute_adata, var_names=["intcol", "strcol", "boolcol"])

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_nuclear_norm_minimization_impute_no_copy(impute_num_adata):
    adata_imputed = nuclear_norm_minimization_impute(impute_num_adata)

    assert id(impute_num_adata) == id(adata_imputed)


def test_nuclear_norm_minimization_impute_copy(impute_num_adata):
    adata_imputed = nuclear_norm_minimization_impute(impute_num_adata, copy=True)

    assert id(impute_num_adata) != id(adata_imputed)


def test_nuclear_norm_minimization_impute_non_numerical_data(impute_adata):
    adata_imputed = nuclear_norm_minimization_impute(impute_adata)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_nuclear_norm_minimization_impute_numerical_data(impute_num_adata):
    adata_imputed = nuclear_norm_minimization_impute(impute_num_adata)

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_nuclear_norm_minimization_impute_list_str(impute_adata):
    adata_imputed = nuclear_norm_minimization_impute(impute_adata, var_names=["intcol", "strcol", "boolcol"])

    assert not (np.all([item != item for item in adata_imputed.X]))


@pytest.mark.skipif(os.name == "posix", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_no_copy(impute_iris):
    adata_imputed = mice_forest_impute(impute_iris)

    assert id(impute_iris) == id(adata_imputed)


@pytest.mark.skipif(os.name == "posix", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_copy(impute_iris):
    adata_imputed = mice_forest_impute(impute_iris, copy=True)

    assert id(impute_iris) != id(adata_imputed)


@pytest.mark.skipif(os.name == "posix", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_non_numerical_data(impute_titanic):
    adata_imputed = mice_forest_impute(impute_titanic)

    assert not (np.all([item != item for item in adata_imputed.X]))


@pytest.mark.skipif(os.name == "posix", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_numerical_data(impute_iris):
    adata_imputed = mice_forest_impute(impute_iris)

    assert not (np.all([item != item for item in adata_imputed.X]))


@pytest.mark.skipif(os.name == "posix", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_list_str(impute_titanic):
    adata_imputed = mice_forest_impute(impute_titanic, var_names=["Cabin", "Age"])

    assert not (np.all([item != item for item in adata_imputed.X]))


def test_explicit_impute_all(impute_num_adata):
    warnings.filterwarnings("ignore", category=FutureWarning)
    adata_imputed = explicit_impute(impute_num_adata, replacement=1011, copy=True)

    assert (adata_imputed.X == 1011).sum() == 3


def test_explicit_impute_subset(impute_adata):
    adata_imputed = explicit_impute(impute_adata, replacement={"strcol": "REPLACED", "intcol": 1011}, copy=True)

    assert (adata_imputed.X == 1011).sum() == 1
    assert (adata_imputed.X == "REPLACED").sum() == 1


def test_warning(impute_num_adata):
    warning_results = _warn_imputation_threshold(impute_num_adata, threshold=20, var_names=None)
    assert warning_results == {"col1": 25, "col3": 50}
