import platform
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest
from anndata import AnnData
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from fast_array_utils.conv import to_dense
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning

from ehrapy.preprocessing._imputation import (
    _warn_imputation_threshold,
    explicit_impute,
    knn_impute,
    mice_forest_impute,
    miss_forest_impute,
    simple_impute,
)
from tests.conftest import ARRAY_TYPES, TEST_DATA_PATH

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

    def _are_ndarrays_equal(arr1: np.ndarray, arr2: np.ndarray) -> np.bool_:
        return np.all(np.equal(arr1, arr2, dtype=object) | ((arr1 != arr1) & (arr2 != arr2)))

    def _is_val_missing(data: np.ndarray) -> np.ndarray[Any, np.dtype[np.bool_]]:
        return np.isin(data, [None, ""]) | (data != data)

    # Convert dask arrays to numpy arrays
    if isinstance(adata_before_imputation.X, da.Array):
        adata_before_imputation.X = adata_before_imputation.X.compute()
    if isinstance(adata_after_imputation.X, da.Array):
        adata_after_imputation.X = adata_after_imputation.X.compute()

    layer_before = to_dense(adata_before_imputation.layers.get(before_imputation_layer, adata_before_imputation.X))
    layer_after = to_dense(adata_after_imputation.layers.get(after_imputation_layer, adata_after_imputation.X))

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

    # If reaching here: all checks passed
    return


def test_base_check_imputation_incompatible_shapes(impute_num_edata):
    adata_imputed = knn_impute(impute_num_edata, copy=True)
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, adata_imputed[1:, :])
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, adata_imputed[:, 1:])


def test_base_check_imputation_nan_detected_after_complete_imputation(impute_num_edata):
    adata_imputed = knn_impute(impute_num_edata, copy=True)
    adata_imputed.X[0, 2] = np.nan
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, adata_imputed)


def test_base_check_imputation_nan_detected_after_partial_imputation(impute_num_edata):
    var_names = ("col2", "col3")
    adata_imputed = knn_impute(impute_num_edata, var_names=var_names, copy=True)
    adata_imputed.X[0, 2] = np.nan
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, adata_imputed, imputed_var_names=var_names)


def test_base_check_imputation_nan_ignored_if_not_in_imputed_column(impute_num_edata):
    var_names = ("col2", "col3")
    adata_imputed = knn_impute(impute_num_edata, var_names=var_names, copy=True)
    # col1 has a NaN at row 2, should get ignored
    _base_check_imputation(impute_num_edata, adata_imputed, imputed_var_names=var_names)


def test_base_check_imputation_change_detected_in_non_imputed_column(impute_num_edata):
    var_names = ("col2", "col3")
    adata_imputed = knn_impute(impute_num_edata, var_names=var_names, copy=True)
    # col1 has a NaN at row 2, let's simulate it has been imputed by mistake
    adata_imputed.X[2, 0] = 42.0
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, adata_imputed, imputed_var_names=var_names)


def test_base_check_imputation_change_detected_in_imputed_column(impute_num_edata):
    adata_imputed = knn_impute(impute_num_edata, copy=True)
    # col3 didn't have a NaN at row 1, let's simulate it has been modified by mistake
    adata_imputed.X[1, 2] = 42.0
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, adata_imputed)


def test_mean_impute_no_copy(impute_num_edata):
    adata_not_imputed = impute_num_edata.copy()
    simple_impute(impute_num_edata)

    _base_check_imputation(adata_not_imputed, impute_num_edata)


def test_simple_impute_3D_edata(edata_blob_small):
    simple_impute(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        simple_impute(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


def test_mean_impute_copy(impute_num_edata):
    adata_imputed = simple_impute(impute_num_edata, copy=True)

    assert id(impute_num_edata) != id(adata_imputed)
    _base_check_imputation(impute_num_edata, adata_imputed)


def test_mean_impute_throws_error_non_numerical(impute_edata):
    with pytest.raises(ValueError):
        simple_impute(impute_edata)


def test_mean_impute_subset(impute_edata):
    var_names = ("intcol", "indexcol")
    adata_imputed = simple_impute(impute_edata, var_names=var_names, copy=True)

    _base_check_imputation(impute_edata, adata_imputed, imputed_var_names=var_names)
    assert np.any([item != item for item in adata_imputed.X[::, 3:4]])


def test_median_impute_no_copy(impute_num_edata):
    adata_not_imputed = impute_num_edata.copy()
    simple_impute(impute_num_edata, strategy="median")

    _base_check_imputation(adata_not_imputed, impute_num_edata)


def test_median_impute_copy(impute_num_edata):
    adata_imputed = simple_impute(impute_num_edata, strategy="median", copy=True)

    _base_check_imputation(impute_num_edata, adata_imputed)
    assert id(impute_num_edata) != id(adata_imputed)


def test_median_impute_throws_error_non_numerical(impute_edata):
    with pytest.raises(ValueError):
        simple_impute(impute_edata, strategy="median")


def test_median_impute_subset(impute_edata):
    var_names = ("intcol", "indexcol")
    adata_imputed = simple_impute(impute_edata, var_names=var_names, strategy="median", copy=True)

    _base_check_imputation(impute_edata, adata_imputed, imputed_var_names=var_names)


def test_most_frequent_impute_no_copy(impute_edata):
    adata_not_imputed = impute_edata.copy()
    simple_impute(impute_edata, strategy="most_frequent")

    _base_check_imputation(adata_not_imputed, impute_edata)


def test_most_frequent_impute_copy(impute_edata):
    adata_imputed = simple_impute(impute_edata, strategy="most_frequent", copy=True)

    _base_check_imputation(impute_edata, adata_imputed)
    assert id(impute_edata) != id(adata_imputed)


def test_unknown_simple_imputation_strategy(impute_edata):
    with pytest.raises(ValueError):
        simple_impute(impute_edata, strategy="invalid_strategy", copy=True)  # type: ignore


def test_most_frequent_impute_subset(impute_edata):
    var_names = ("intcol", "strcol")
    adata_imputed = simple_impute(impute_edata, var_names=var_names, strategy="most_frequent", copy=True)

    _base_check_imputation(impute_edata, adata_imputed, imputed_var_names=var_names)


def test_knn_impute_3D_edata(edata_blob_small):
    knn_impute(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        knn_impute(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


def test_knn_impute_check_backend(impute_num_edata):
    knn_impute(impute_num_edata, backend="faiss", copy=True)
    knn_impute(impute_num_edata, backend="scikit-learn", copy=True)
    with pytest.raises(
        ValueError,
        match="Unknown backend 'invalid_backend' for KNN imputation. Choose between 'scikit-learn' and 'faiss'.",
    ):
        knn_impute(impute_num_edata, backend="invalid_backend")  # type: ignore


def test_knn_impute_no_copy(impute_num_edata):
    adata_not_imputed = impute_num_edata.copy()
    knn_impute(impute_num_edata)

    _base_check_imputation(adata_not_imputed, impute_num_edata)


def test_knn_impute_copy(impute_num_edata):
    adata_imputed = knn_impute(impute_num_edata, n_neighbors=3, copy=True)

    _base_check_imputation(impute_num_edata, adata_imputed)
    assert id(impute_num_edata) != id(adata_imputed)


def test_knn_impute_non_numerical_data(impute_edata):
    with pytest.raises(ValueError):
        knn_impute(impute_edata, n_neighbors=3, copy=True)


def test_knn_impute_numerical_data(impute_num_edata):
    adata_imputed = knn_impute(impute_num_edata, copy=True)

    _base_check_imputation(impute_num_edata, adata_imputed)


def test_missforest_impute_3D_edata(edata_blob_small):
    miss_forest_impute(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        miss_forest_impute(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


def test_missforest_impute_non_numerical_data(impute_edata):
    with pytest.raises(ValueError):
        miss_forest_impute(impute_edata, copy=True)


def test_missforest_impute_numerical_data(impute_num_edata):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    adata_imputed = miss_forest_impute(impute_num_edata, copy=True)

    _base_check_imputation(impute_num_edata, adata_imputed)


def test_missforest_impute_subset(impute_num_edata):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    var_names = ("col2", "col3")
    adata_imputed = miss_forest_impute(impute_num_edata, var_names=var_names, copy=True)

    _base_check_imputation(impute_num_edata, adata_imputed, imputed_var_names=var_names)


@pytest.mark.parametrize(
    "array_type,expected_error",
    [
        (np.array, None),
        (da.from_array, NotImplementedError),
        (sparse.csr_matrix, NotImplementedError),
    ],
)
@pytest.mark.skipif(platform.system() == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_array_types(impute_num_edata, array_type, expected_error):
    impute_num_edata.X = array_type(impute_num_edata.X)
    if expected_error:
        with pytest.raises(expected_error):
            mice_forest_impute(impute_num_edata, copy=True)


@pytest.mark.skipif(platform.system() == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_3D_edata(edata_blob_small):
    edata_blob_small.X[3:5, 4:6] = np.nan
    edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME][3:5, 4:6] = np.nan
    mice_forest_impute(edata_blob_small)
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        mice_forest_impute(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.skipif(platform.system() == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_no_copy(impute_iris_edata):
    adata_not_imputed = impute_iris_edata.copy()
    mice_forest_impute(impute_iris_edata)

    _base_check_imputation(adata_not_imputed, impute_iris_edata)


@pytest.mark.skipif(platform.system() == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_copy(impute_iris_edata):
    adata_imputed = mice_forest_impute(impute_iris_edata, copy=True)

    _base_check_imputation(impute_iris_edata, adata_imputed)
    assert id(impute_iris_edata) != id(adata_imputed)


@pytest.mark.skipif(platform.system() == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_non_numerical_data(impute_titanic_edata):
    with pytest.raises(ValueError):
        mice_forest_impute(impute_titanic_edata)


@pytest.mark.skipif(platform.system() == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_numerical_data(impute_iris_edata):
    adata_not_imputed = impute_iris_edata.copy()
    mice_forest_impute(impute_iris_edata)

    _base_check_imputation(adata_not_imputed, impute_iris_edata)


@pytest.mark.parametrize(
    "array_type,expected_error",
    [
        (np.array, None),
        (da.from_array, None),
        (sparse.csr_matrix, NotImplementedError),
    ],
)
def test_explicit_impute_array_types(impute_num_edata, array_type, expected_error):
    impute_num_edata.X = array_type(impute_num_edata.X)
    if expected_error:
        with pytest.raises(expected_error):
            explicit_impute(impute_num_edata, replacement=1011, copy=True)


def test_explicit_impute_3D_edata(edata_blob_small):
    explicit_impute(edata_blob_small, replacement=1011, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        explicit_impute(edata_blob_small, replacement=1011, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_explicit_impute_all(array_type, impute_num_edata):
    impute_num_edata.X = array_type(impute_num_edata.X)
    warnings.filterwarnings("ignore", category=FutureWarning)
    adata_imputed = explicit_impute(impute_num_edata, replacement=1011, copy=True)

    _base_check_imputation(impute_num_edata, adata_imputed)
    assert np.sum([adata_imputed.X == 1011]) == 3


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_explicit_impute_subset(impute_edata, array_type):
    impute_edata.X = array_type(impute_edata.X)
    adata_imputed = explicit_impute(impute_edata, replacement={"strcol": "REPLACED", "intcol": 1011}, copy=True)

    _base_check_imputation(impute_edata, adata_imputed, imputed_var_names=("strcol", "intcol"))
    assert np.sum([adata_imputed.X == 1011]) == 1
    assert np.sum([adata_imputed.X == "REPLACED"]) == 1


def test_warning(impute_num_edata):
    warning_results = _warn_imputation_threshold(impute_num_edata, threshold=20, var_names=None)
    assert warning_results == {"col1": 25, "col3": 50}
