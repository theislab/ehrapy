import os
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from sklearn.exceptions import ConvergenceWarning

from ehrapy._utils_data import _are_ndarrays_equal, _is_val_missing, _to_dense_matrix
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


def _base_check_imputation(
    adata_before_imputation: AnnData,
    adata_after_imputation: AnnData,
    before_imputation_layer: str | None = None,
    after_imputation_layer: str | None = None,
    imputed_var_names: Iterable[str] | None = None,
):
    """Provides a base check for all imputations:

    - Imputation doesn't leave any NaN behind
    - Imputation doesn't modify anything in non-imputated columns (if the imputation on a subset was requested)
    - Imputation doesn't modify any data that wasn't NaN

    Args:
        adata_before_imputation: AnnData before imputation
        adata_after_imputation: AnnData after imputation
        before_imputation_layer: Layer to consider in the original ``AnnData``, ``X`` if not specified
        after_imputation_layer: Layer to consider in the imputated ``AnnData``, ``X`` if not specified
        imputed_var_names: Names of the features that were imputated, will consider all of them if not specified

    Raises:
        AssertionError: If any of the checks fail.
    """

    layer_before = _to_dense_matrix(adata_before_imputation, before_imputation_layer)
    layer_after = _to_dense_matrix(adata_after_imputation, after_imputation_layer)

    if layer_before.shape != layer_after.shape:
        raise AssertionError("The shapes of the two layers do not match")

    var_indices = (
        np.arange(layer_before.shape[1])
        if imputed_var_names is None
        else [
            adata_before_imputation.var_names.get_loc(var_name)
            for var_name in imputed_var_names
            if var_name in imputed_var_names
        ]
    )

    before_nan_mask = _is_val_missing(layer_before)
    imputed_mask = np.zeros(layer_before.shape[1], dtype=bool)
    imputed_mask[var_indices] = True

    # Ensure no NaN remains in the imputed columns of layer_after
    if np.any(before_nan_mask[:, imputed_mask] & _is_val_missing(layer_after[:, imputed_mask])):
        raise AssertionError("NaN found in imputed columns of layer_after.")

    # Ensure unchanged values outside imputed columns
    unchanged_mask = ~imputed_mask
    if not _are_ndarrays_equal(layer_before[:, unchanged_mask], layer_after[:, unchanged_mask]):
        raise AssertionError("Values outside imputed columns were modified.")

    # Ensure imputation does not alter non-NaN values in the imputed columns
    imputed_non_nan_mask = (~before_nan_mask) & imputed_mask
    if not _are_ndarrays_equal(layer_before[imputed_non_nan_mask], layer_after[imputed_non_nan_mask]):
        raise AssertionError("Non-NaN values in imputed columns were modified.")

    # All checks passed


def test_base_check_imputation_incompatible_shapes(impute_num_adata):
    adata_imputed = knn_impute(impute_num_adata, copy=True)
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_adata, adata_imputed[1:, :])
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_adata, adata_imputed[:, 1:])


def test_base_check_imputation_nan_detected_after_complete_imputation(impute_num_adata):
    adata_imputed = knn_impute(impute_num_adata, copy=True)
    adata_imputed.X[0, 2] = np.nan
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_adata, adata_imputed)


def test_base_check_imputation_nan_detected_after_partial_imputation(impute_num_adata):
    var_names = ("col2", "col3")
    adata_imputed = knn_impute(impute_num_adata, var_names=var_names, copy=True)
    adata_imputed.X[0, 2] = np.nan
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_adata, adata_imputed, imputed_var_names=var_names)


def test_base_check_imputation_nan_ignored_if_not_in_imputed_column(impute_num_adata):
    var_names = ("col2", "col3")
    adata_imputed = knn_impute(impute_num_adata, var_names=var_names, copy=True)
    # col1 has a NaN at row 2, should get ignored
    _base_check_imputation(impute_num_adata, adata_imputed, imputed_var_names=var_names)


def test_base_check_imputation_change_detected_in_non_imputed_column(impute_num_adata):
    var_names = ("col2", "col3")
    adata_imputed = knn_impute(impute_num_adata, var_names=var_names, copy=True)
    # col1 has a NaN at row 2, let's simulate it has been imputed by mistake
    adata_imputed.X[2, 0] = 42.0
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_adata, adata_imputed, imputed_var_names=var_names)


def test_base_check_imputation_change_detected_in_imputed_column(impute_num_adata):
    adata_imputed = knn_impute(impute_num_adata, copy=True)
    # col3 didn't have a NaN at row 1, let's simulate it has been modified by mistake
    adata_imputed.X[1, 2] = 42.0
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_adata, adata_imputed)


def test_mean_impute_no_copy(impute_num_adata):
    adata_not_imputed = impute_num_adata.copy()
    adata_imputed_no_copy = simple_impute(impute_num_adata)

    _base_check_imputation(adata_not_imputed, adata_imputed_no_copy)
    assert id(adata_imputed_no_copy) == id(impute_num_adata)


def test_mean_impute_copy(impute_num_adata):
    adata_imputed = simple_impute(impute_num_adata, copy=True)

    assert id(impute_num_adata) != id(adata_imputed)
    _base_check_imputation(impute_num_adata, adata_imputed)


def test_mean_impute_throws_error_non_numerical(impute_adata):
    with pytest.raises(ValueError):
        simple_impute(impute_adata)


def test_mean_impute_subset(impute_adata):
    var_names = ("intcol", "indexcol")
    adata_imputed = simple_impute(impute_adata, var_names=var_names, copy=True)

    _base_check_imputation(impute_adata, adata_imputed, imputed_var_names=var_names)
    assert np.any([item != item for item in adata_imputed.X[::, 3:4]])


def test_median_impute_no_copy(impute_num_adata):
    adata_not_imputed = impute_num_adata.copy()
    adata_imputed_no_copy = simple_impute(impute_num_adata, strategy="median")

    _base_check_imputation(adata_not_imputed, adata_imputed_no_copy)
    assert id(adata_imputed_no_copy) == id(impute_num_adata)


def test_median_impute_copy(impute_num_adata):
    adata_imputed = simple_impute(impute_num_adata, strategy="median", copy=True)

    _base_check_imputation(impute_num_adata, adata_imputed)
    assert id(impute_num_adata) != id(adata_imputed)


def test_median_impute_throws_error_non_numerical(impute_adata):
    with pytest.raises(ValueError):
        simple_impute(impute_adata, strategy="median")


def test_median_impute_subset(impute_adata):
    var_names = ("intcol", "indexcol")
    adata_imputed = simple_impute(impute_adata, var_names=var_names, strategy="median", copy=True)

    _base_check_imputation(impute_adata, adata_imputed, imputed_var_names=var_names)


def test_most_frequent_impute_no_copy(impute_adata):
    adata_not_imputed = impute_adata.copy()
    adata_imputed_no_copy = simple_impute(impute_adata, strategy="most_frequent")

    _base_check_imputation(adata_not_imputed, adata_imputed_no_copy)
    assert id(adata_imputed_no_copy) == id(impute_adata)


def test_most_frequent_impute_copy(impute_adata):
    adata_imputed = simple_impute(impute_adata, strategy="most_frequent", copy=True)

    _base_check_imputation(impute_adata, adata_imputed)
    assert id(impute_adata) != id(adata_imputed)


def test_unknown_simple_imputation_strategy(impute_adata):
    with pytest.raises(ValueError):
        simple_impute(impute_adata, strategy="invalid_strategy", copy=True)  # type: ignore


def test_most_frequent_impute_subset(impute_adata):
    var_names = ("intcol", "strcol")
    adata_imputed = simple_impute(impute_adata, var_names=var_names, strategy="most_frequent", copy=True)

    _base_check_imputation(impute_adata, adata_imputed, imputed_var_names=var_names)


def test_knn_impute_check_backend(impute_num_adata):
    knn_impute(impute_num_adata, backend="faiss", copy=True)
    knn_impute(impute_num_adata, backend="scikit-learn", copy=True)
    with pytest.raises(
        ValueError,
        match="Unknown backend 'invalid_backend' for KNN imputation. Choose between 'scikit-learn' and 'faiss'.",
    ):
        knn_impute(impute_num_adata, backend="invalid_backend")  # type: ignore


def test_knn_impute_no_copy(impute_num_adata):
    adata_not_imputed = impute_num_adata.copy()
    adata_imputed_no_copy = knn_impute(impute_num_adata)

    _base_check_imputation(adata_not_imputed, adata_imputed_no_copy)
    assert id(adata_imputed_no_copy) == id(impute_num_adata)


def test_knn_impute_copy(impute_num_adata):
    adata_imputed = knn_impute(impute_num_adata, n_neighbors=3, copy=True)

    _base_check_imputation(impute_num_adata, adata_imputed)
    assert id(impute_num_adata) != id(adata_imputed)


def test_knn_impute_non_numerical_data(impute_adata):
    with pytest.raises(ValueError):
        knn_impute(impute_adata, n_neighbors=3, copy=True)


def test_knn_impute_numerical_data(impute_num_adata):
    adata_imputed = knn_impute(impute_num_adata, copy=True)

    _base_check_imputation(impute_num_adata, adata_imputed)


def test_missforest_impute_non_numerical_data(impute_adata):
    with pytest.raises(ValueError):
        miss_forest_impute(impute_adata, copy=True)


def test_missforest_impute_numerical_data(impute_num_adata):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    adata_imputed = miss_forest_impute(impute_num_adata, copy=True)

    _base_check_imputation(impute_num_adata, adata_imputed)


def test_missforest_impute_subset(impute_num_adata):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    var_names = ("col2", "col3")
    adata_imputed = miss_forest_impute(impute_num_adata, var_names=var_names, copy=True)

    _base_check_imputation(impute_num_adata, adata_imputed, imputed_var_names=var_names)


@pytest.mark.skipif(os.name == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_no_copy(impute_iris_adata):
    adata_not_imputed = impute_iris_adata.copy()
    adata_imputed_no_copy = mice_forest_impute(impute_iris_adata)

    _base_check_imputation(adata_not_imputed, adata_imputed_no_copy)
    assert id(impute_iris_adata) == id(adata_imputed_no_copy)


@pytest.mark.skipif(os.name == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_copy(impute_iris_adata):
    adata_imputed = mice_forest_impute(impute_iris_adata, copy=True)

    _base_check_imputation(impute_iris_adata, adata_imputed)
    assert id(impute_iris_adata) != id(adata_imputed)


@pytest.mark.skipif(os.name == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_non_numerical_data(impute_titanic_adata):
    with pytest.raises(ValueError):
        mice_forest_impute(impute_titanic_adata)


@pytest.mark.skipif(os.name == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_numerical_data(impute_iris_adata):
    adata_imputed = mice_forest_impute(impute_iris_adata)

    _base_check_imputation(impute_iris_adata, adata_imputed)


def test_explicit_impute_all(impute_num_adata):
    warnings.filterwarnings("ignore", category=FutureWarning)
    adata_imputed = explicit_impute(impute_num_adata, replacement=1011, copy=True)

    _base_check_imputation(impute_num_adata, adata_imputed)
    assert np.sum([adata_imputed.X == 1011]) == 3


def test_explicit_impute_subset(impute_adata):
    adata_imputed = explicit_impute(impute_adata, replacement={"strcol": "REPLACED", "intcol": 1011}, copy=True)

    _base_check_imputation(impute_adata, adata_imputed, imputed_var_names=("strcol", "intcol"))
    assert np.sum([adata_imputed.X == 1011]) == 1
    assert np.sum([adata_imputed.X == "REPLACED"]) == 1


def test_warning(impute_num_adata):
    warning_results = _warn_imputation_threshold(impute_num_adata, threshold=20, var_names=None)
    assert warning_results == {"col1": 25, "col3": 50}
