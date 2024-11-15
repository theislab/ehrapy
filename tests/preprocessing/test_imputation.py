import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from ehrapy.preprocessing._imputation import (
    _warn_imputation_threshold,
    explicit_impute,
    knn_impute,
    mice_forest_impute,
    miss_forest_impute,
    simple_impute,
)
from tests.conftest import TEST_DATA_PATH

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{TEST_DATA_PATH}/imputation"


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


def test_median_impute_copy(impute_num_adata, impute_adata):
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


def test_knn_impute_check_backend(impute_num_adata):
    knn_impute(impute_num_adata, backend="faiss")
    knn_impute(impute_num_adata, backend="scikit-learn")
    with pytest.raises(
        ValueError,
        match="Unknown backend 'invalid_backend' for KNN imputation. Choose between 'scikit-learn' and 'faiss'.",
    ):
        knn_impute(impute_num_adata, backend="invalid_backend")


def test_knn_impute_no_copy(impute_num_adata):
    knn_impute(impute_num_adata)

    assert not (np.all([item != item for item in impute_num_adata.X]))


def test_knn_impute_copy(impute_num_adata):
    adata_imputed = knn_impute(impute_num_adata, n_neighbors=3, copy=True)

    assert id(impute_num_adata) != id(adata_imputed)
    assert not (np.all([item != item for item in adata_imputed.X]))


def test_knn_impute_non_numerical_data(impute_adata):
    adata_imputed = knn_impute(impute_adata, n_neighbors=3, copy=True)

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


@pytest.mark.skipif(os.name == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_no_copy(impute_iris_adata):
    adata_imputed = mice_forest_impute(impute_iris_adata)

    assert id(impute_iris_adata) == id(adata_imputed)


@pytest.mark.skipif(os.name == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_copy(impute_iris_adata):
    adata_imputed = mice_forest_impute(impute_iris_adata, copy=True)

    assert id(impute_iris_adata) != id(adata_imputed)


@pytest.mark.skipif(os.name == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_non_numerical_data(impute_titanic_adata):
    adata_imputed = mice_forest_impute(impute_titanic_adata)

    assert not (np.all([item != item for item in adata_imputed.X]))


@pytest.mark.skipif(os.name == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_numerical_data(impute_iris_adata):
    adata_imputed = mice_forest_impute(impute_iris_adata)

    assert not (np.all([item != item for item in adata_imputed.X]))


@pytest.mark.skipif(os.name == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_list_str(impute_titanic_adata):
    adata_imputed = mice_forest_impute(impute_titanic_adata, var_names=["Cabin", "Age"])

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
