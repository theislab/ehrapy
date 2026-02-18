import platform
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest
from ehrdata import EHRData
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from fast_array_utils.conv import to_dense
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning

from ehrapy.preprocessing._imputation import (
    _warn_imputation_threshold,
    explicit_impute,
    knn_impute,
    locf_impute,
    mice_forest_impute,
    miss_forest_impute,
    simple_impute,
)
from tests.conftest import ARRAY_TYPES_NONNUMERIC, ARRAY_TYPES_NUMERIC, ARRAY_TYPES_NUMERIC_3D_ABLE, TEST_DATA_PATH

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{TEST_DATA_PATH}/imputation"


def _base_check_imputation(
    edata_before_imputation: EHRData,
    edata_after_imputation: EHRData,
    before_imputation_layer: str | None = None,
    after_imputation_layer: str | None = None,
    imputed_var_names: Iterable[str] | None = None,
):
    """Provides a base check for all imputations:

    - Imputation doesn't leave any NaN behind
    - Imputation doesn't modify anything in non-imputated columns (if the imputation on a subset was requested)
    - Imputation doesn't modify any data that wasn't NaN

    Args:
        edata_before_imputation: EHRData before imputation
        edata_after_imputation: EHRData after imputation
        before_imputation_layer: Layer to consider in the original ``EHRData``, ``X`` if not specified
        after_imputation_layer: Layer to consider in the imputated ``EHRData``, ``X`` if not specified
        imputed_var_names: Names of the features that were imputated, will consider all of them if not specified

    Raises:
        AssertionError: If any of the checks fail.
    """

    def _are_ndarrays_equal(arr1: np.ndarray, arr2: np.ndarray) -> np.bool_:
        return np.all(np.equal(arr1, arr2, dtype=object) | ((arr1 != arr1) & (arr2 != arr2)))

    def _is_val_missing(data: np.ndarray) -> np.ndarray[Any, np.dtype[np.bool_]]:
        return np.isin(data, [None, ""]) | (data != data)

    layer_before = to_dense(edata_before_imputation.layers.get(before_imputation_layer, edata_before_imputation.X))
    layer_after = to_dense(edata_after_imputation.layers.get(after_imputation_layer, edata_after_imputation.X))

    if isinstance(layer_before, da.Array):
        layer_before = layer_before.compute()
    if isinstance(layer_after, da.Array):
        layer_after = layer_after.compute()

    if layer_before.shape != layer_after.shape:
        raise AssertionError("The shapes of the two layers do not match")

    var_indices = (
        np.arange(layer_before.shape[1])
        if imputed_var_names is None
        else [
            edata_before_imputation.var_names.get_loc(var_name)
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
    imputed_non_nan_mask = (~before_nan_mask) & (
        imputed_mask[None, :] if layer_before.ndim == 2 else imputed_mask[None, :, None]
    )
    if not _are_ndarrays_equal(layer_before[imputed_non_nan_mask], layer_after[imputed_non_nan_mask]):
        raise AssertionError("Non-NaN values in imputed columns were modified.")

    # If reaching here: all checks passed
    return


def test_base_check_imputation_incompatible_shapes(impute_num_edata):
    edata_imputed = knn_impute(impute_num_edata, copy=True)
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, edata_imputed[1:, :])
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, edata_imputed[:, 1:])


def test_base_check_imputation_nan_detected_after_complete_imputation(impute_num_edata):
    edata_imputed = knn_impute(impute_num_edata, copy=True)
    edata_imputed.X[0, 2] = np.nan
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, edata_imputed)


def test_base_check_imputation_nan_detected_after_partial_imputation(impute_num_edata):
    var_names = ("col2", "col3")
    edata_imputed = knn_impute(impute_num_edata, var_names=var_names, copy=True)
    edata_imputed.X[0, 2] = np.nan
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, edata_imputed, imputed_var_names=var_names)


def test_base_check_imputation_nan_ignored_if_not_in_imputed_column(impute_num_edata):
    var_names = ("col2", "col3")
    edata_imputed = knn_impute(impute_num_edata, var_names=var_names, copy=True)
    # col1 has a NaN at row 2, should get ignored
    _base_check_imputation(impute_num_edata, edata_imputed, imputed_var_names=var_names)


def test_base_check_imputation_change_detected_in_non_imputed_column(impute_num_edata):
    var_names = ("col2", "col3")
    edata_imputed = knn_impute(impute_num_edata, var_names=var_names, copy=True)
    # col1 has a NaN at row 2, let's simulate it has been imputed by mistake
    edata_imputed.X[2, 0] = 42.0
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, edata_imputed, imputed_var_names=var_names)


def test_base_check_imputation_change_detected_in_imputed_column(impute_num_edata):
    edata_imputed = knn_impute(impute_num_edata, copy=True)
    # col3 didn't have a NaN at row 1, let's simulate it has been modified by mistake
    edata_imputed.X[1, 2] = 42.0
    with pytest.raises(AssertionError):
        _base_check_imputation(impute_num_edata, edata_imputed)


@pytest.mark.parametrize(
    "array_type,expected_error",
    [
        (np.array, None),
        (da.array, None),
        (sparse.csr_array, None),
        (sparse.csc_array, None),
        # (sparse.coo_array, None) # not yet supported by AnnData
    ],
)
def test_simple_impute_array_types(impute_num_edata, array_type, expected_error):
    impute_num_edata.X = array_type(impute_num_edata.X)

    if expected_error:
        with pytest.raises(expected_error):
            simple_impute(impute_num_edata, strategy="mean")


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NUMERIC)
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_impute_basic(impute_num_edata, array_type, strategy):
    impute_num_edata.X = array_type(impute_num_edata.X)

    if isinstance(impute_num_edata.X, da.Array) and strategy != "mean":
        with pytest.raises(ValueError):
            edata_imputed = simple_impute(impute_num_edata, strategy=strategy, copy=True)

    else:
        edata_imputed = simple_impute(impute_num_edata, strategy=strategy, copy=True)
        _base_check_imputation(impute_num_edata, edata_imputed)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_impute_copy(impute_num_edata, strategy):
    edata_imputed = simple_impute(impute_num_edata, strategy=strategy, copy=True)

    assert id(impute_num_edata) != id(edata_imputed)
    _base_check_imputation(impute_num_edata, edata_imputed)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_impute_subset(impute_edata, array_type, strategy):
    impute_edata.X = array_type(impute_edata.X)
    var_names = ("intcol", "indexcol")
    if isinstance(impute_edata.X, da.Array) and strategy != "mean":
        with pytest.raises(ValueError):
            edata_imputed = simple_impute(impute_edata, var_names=var_names, strategy=strategy, copy=True)
    else:
        edata_imputed = simple_impute(impute_edata, var_names=var_names, strategy=strategy, copy=True)

        _base_check_imputation(impute_edata, edata_imputed, imputed_var_names=var_names)
        assert np.any([item != item for item in edata_imputed.X[::, 3:4]])

        # manually verified computation result
        if strategy == "mean":
            assert edata_imputed.X[0, 1] == 3.0
        elif strategy == "most_frequent":
            assert edata_imputed.X[0, 1] == 2.0  # if multiple equally frequent values, return minimum


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NUMERIC_3D_ABLE)
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_impute_3D_edata(mcar_edata, array_type, strategy):
    mcar_edata.layers[DEFAULT_TEM_LAYER_NAME] = array_type(mcar_edata.layers[DEFAULT_TEM_LAYER_NAME])

    if isinstance(mcar_edata.layers[DEFAULT_TEM_LAYER_NAME], da.Array) and strategy != "mean":
        with pytest.raises(ValueError):
            edata_imputed = simple_impute(mcar_edata, layer=DEFAULT_TEM_LAYER_NAME, strategy=strategy, copy=True)

    else:
        edata_imputed = simple_impute(mcar_edata, layer=DEFAULT_TEM_LAYER_NAME, strategy=strategy, copy=True)
        _base_check_imputation(
            mcar_edata,
            edata_imputed,
            before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
            after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        )

        # manually verify computation result for 1 value
        if strategy in {"mean", "median"}:
            element = edata_imputed[9, 0, 0].layers[DEFAULT_TEM_LAYER_NAME]

            if strategy == "mean":
                reference_value = np.nanmean(mcar_edata[:, 0, :].layers[DEFAULT_TEM_LAYER_NAME])
            elif strategy == "median":
                reference_value = np.nanmedian(mcar_edata[:, 0, :].layers[DEFAULT_TEM_LAYER_NAME])

            assert np.isclose(element, reference_value)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_impute_3D_edata_nonnumeric(edata_mini_3D_missing_values, array_type, strategy):
    edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME] = array_type(
        edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME]
    )

    if strategy == "most_frequent" and not isinstance(
        edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME], da.Array
    ):
        edata_imputed = simple_impute(
            edata_mini_3D_missing_values, layer=DEFAULT_TEM_LAYER_NAME, strategy=strategy, copy=True
        )
        _base_check_imputation(
            edata_mini_3D_missing_values,
            edata_imputed,
            before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
            after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        )
    else:
        with pytest.raises(ValueError):
            edata_imputed = simple_impute(
                edata_mini_3D_missing_values, layer=DEFAULT_TEM_LAYER_NAME, strategy=strategy, copy=True
            )


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_simple_impute_throws_error_non_numerical(impute_edata, strategy):
    with pytest.raises(ValueError):
        simple_impute(impute_edata, strategy=strategy)


def test_simple_impute_invalid_strategy(impute_edata):
    with pytest.raises(ValueError):
        simple_impute(impute_edata, strategy="invalid_strategy", copy=True)  # type: ignore


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
    edata_not_imputed = impute_num_edata.copy()
    knn_impute(impute_num_edata)

    _base_check_imputation(edata_not_imputed, impute_num_edata)


def test_knn_impute_copy(impute_num_edata):
    edata_imputed = knn_impute(impute_num_edata, n_neighbors=3, copy=True)

    _base_check_imputation(impute_num_edata, edata_imputed)
    assert id(impute_num_edata) != id(edata_imputed)


def test_knn_impute_non_numerical_data(impute_edata):
    with pytest.raises(ValueError):
        knn_impute(impute_edata, n_neighbors=3, copy=True)


def test_knn_impute_numerical_data(impute_num_edata):
    edata_imputed = knn_impute(impute_num_edata, copy=True)

    _base_check_imputation(impute_num_edata, edata_imputed)


def test_missforest_impute_3D_edata(edata_blob_small):
    miss_forest_impute(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        miss_forest_impute(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


def test_missforest_impute_non_numerical_data(impute_edata):
    with pytest.raises(ValueError):
        miss_forest_impute(impute_edata, copy=True)


def test_missforest_impute_numerical_data(impute_num_edata):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    edata_imputed = miss_forest_impute(impute_num_edata, copy=True)

    _base_check_imputation(impute_num_edata, edata_imputed)


def test_missforest_impute_subset(impute_num_edata):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    var_names = ("col2", "col3")
    edata_imputed = miss_forest_impute(impute_num_edata, var_names=var_names, copy=True)

    _base_check_imputation(impute_num_edata, edata_imputed, imputed_var_names=var_names)


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
    edata_not_imputed = impute_iris_edata.copy()
    mice_forest_impute(impute_iris_edata)

    _base_check_imputation(edata_not_imputed, impute_iris_edata)


@pytest.mark.skipif(platform.system() == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_copy(impute_iris_edata):
    edata_imputed = mice_forest_impute(impute_iris_edata, copy=True)

    _base_check_imputation(impute_iris_edata, edata_imputed)
    assert id(impute_iris_edata) != id(edata_imputed)


@pytest.mark.skipif(platform.system() == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_non_numerical_data(impute_titanic_edata):
    with pytest.raises(ValueError):
        mice_forest_impute(impute_titanic_edata)


@pytest.mark.skipif(platform.system() == "Darwin", reason="miceforest Imputation not supported by MacOS.")
def test_miceforest_impute_numerical_data(impute_iris_edata):
    edata_not_imputed = impute_iris_edata.copy()
    mice_forest_impute(impute_iris_edata)

    _base_check_imputation(edata_not_imputed, impute_iris_edata)


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


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_explicit_impute_all(array_type, impute_num_edata):
    impute_num_edata.X = array_type(impute_num_edata.X)
    warnings.filterwarnings("ignore", category=FutureWarning)
    edata_imputed = explicit_impute(impute_num_edata, replacement=1011, copy=True)

    _base_check_imputation(impute_num_edata, edata_imputed)
    assert np.sum([edata_imputed.X == 1011]) == 3


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_explicit_impute_subset(impute_edata, array_type):
    impute_edata.X = array_type(impute_edata.X)
    edata_imputed = explicit_impute(impute_edata, replacement={"strcol": "REPLACED", "intcol": 1011}, copy=True)

    _base_check_imputation(impute_edata, edata_imputed, imputed_var_names=("strcol", "intcol"))
    assert np.sum([edata_imputed.X == 1011]) == 1
    assert np.sum([edata_imputed.X == "REPLACED"]) == 1


def test_warning(impute_num_edata):
    warning_results = _warn_imputation_threshold(impute_num_edata, threshold=20, var_names=None)
    assert warning_results == {"col1": 25, "col3": 50}


# ── LOCF impute tests ──────────────────────────────────────────────────


@pytest.fixture
def locf_edata_3d():
    """3D data with known NaN positions for deterministic LOCF testing.

    Shape: (2 patients, 3 vars, 4 time steps)

    Patient 0:
      var0: [1.0, NaN, 3.0, NaN]    -> ffill -> [1, 1, 3, 3]
      var1: [NaN, 2.0, NaN, 4.0]    -> ffill -> [NaN, 2, 2, 4]  (leading NaN -> mean)
      var2: [5.0, 6.0, 7.0, 8.0]    -> ffill -> [5, 6, 7, 8]    (no NaN)

    Patient 1:
      var0: [NaN, NaN, 3.0, NaN]    -> ffill -> [NaN, NaN, 3, 3] (leading NaN -> mean)
      var1: [1.0, NaN, NaN, NaN]    -> ffill -> [1, 1, 1, 1]
      var2: [NaN, 2.0, NaN, 4.0]    -> ffill -> [NaN, 2, 2, 4]  (leading NaN -> mean)
    """
    data_3d = np.array(
        [
            [
                [1.0, np.nan, 3.0, np.nan],
                [np.nan, 2.0, np.nan, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            [
                [np.nan, np.nan, 3.0, np.nan],
                [1.0, np.nan, np.nan, np.nan],
                [np.nan, 2.0, np.nan, 4.0],
            ],
        ]
    )
    return EHRData(shape=(2, 3), layers={DEFAULT_TEM_LAYER_NAME: data_3d})


def test_locf_impute_forward_fill(locf_edata_3d):
    original = locf_edata_3d.copy()
    result = locf_impute(locf_edata_3d, layer=DEFAULT_TEM_LAYER_NAME, copy=True)
    imputed = result.layers[DEFAULT_TEM_LAYER_NAME]

    assert not np.any(np.isnan(imputed))

    assert imputed[0, 0, 1] == 1.0
    assert imputed[0, 0, 3] == 3.0
    assert imputed[1, 1, 1] == 1.0
    assert imputed[1, 1, 2] == 1.0
    assert imputed[1, 1, 3] == 1.0

    _base_check_imputation(
        original,
        result,
        before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
    )


def test_locf_impute_mean_fallback(locf_edata_3d):
    result = locf_impute(locf_edata_3d, layer=DEFAULT_TEM_LAYER_NAME, copy=True)
    imputed = result.layers[DEFAULT_TEM_LAYER_NAME]

    original = locf_edata_3d.layers[DEFAULT_TEM_LAYER_NAME].astype(float)
    feature_means = np.nanmean(original, axis=(0, 2))

    assert np.isclose(imputed[0, 1, 0], feature_means[1])
    assert np.isclose(imputed[1, 0, 0], feature_means[0])
    assert np.isclose(imputed[1, 0, 1], feature_means[0])
    assert np.isclose(imputed[1, 2, 0], feature_means[2])


def test_locf_impute_inplace(locf_edata_3d):
    original_data = locf_edata_3d.layers[DEFAULT_TEM_LAYER_NAME].copy()
    result = locf_impute(locf_edata_3d, layer=DEFAULT_TEM_LAYER_NAME, copy=False)

    assert result is None
    assert not np.any(np.isnan(locf_edata_3d.layers[DEFAULT_TEM_LAYER_NAME]))
    assert not np.array_equal(locf_edata_3d.layers[DEFAULT_TEM_LAYER_NAME], original_data)


def test_locf_impute_var_names(locf_edata_3d):
    original = locf_edata_3d.copy()
    var_to_impute = [locf_edata_3d.var_names[0]]
    result = locf_impute(locf_edata_3d, var_names=var_to_impute, layer=DEFAULT_TEM_LAYER_NAME, copy=True)
    imputed = result.layers[DEFAULT_TEM_LAYER_NAME]

    assert not np.any(np.isnan(imputed[:, 0, :]))

    orig_layer = original.layers[DEFAULT_TEM_LAYER_NAME]
    np.testing.assert_array_equal(imputed[:, 1, :], orig_layer[:, 1, :])
    np.testing.assert_array_equal(imputed[:, 2, :], orig_layer[:, 2, :])


def test_locf_impute_requires_3d(mcar_edata):
    with pytest.raises(ValueError, match="requires 3D data"):
        locf_impute(mcar_edata)


def test_locf_impute_invalid_fallback(locf_edata_3d):
    with pytest.raises(ValueError, match="Unsupported fallback method"):
        locf_impute(locf_edata_3d, layer=DEFAULT_TEM_LAYER_NAME, fallback_method="invalid")
