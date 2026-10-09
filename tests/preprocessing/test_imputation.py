import warnings
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest
from ehrdata import EHRData
from ehrdata._logger import logger
from ehrdata.core.constants import CATEGORICAL_TAG, DEFAULT_TEM_LAYER_NAME, FEATURE_TYPE_KEY, NUMERIC_TAG
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase, DaskArray
from sklearn import config_context
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from testing.fast_array_utils import Flags

from ehrapy.preprocessing._imputation import (
    _warn_imputation_threshold,
    explicit_impute,
    gradient_boosting_impute,
    knn_impute,
    locf_impute,
    miss_forest_impute,
    simple_impute,
)
from tests.conftest import TEST_DATA_PATH, forbid_dask_compute

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{TEST_DATA_PATH}/imputation"
ALL_NAN_VAR = 1


def _array_types_data(rng: np.random.Generator, ndim: int) -> np.ndarray:
    """Values with many zeros and ties, missing values, an all-missing variable and a constant variable."""
    shape = (20, 4) if ndim == 2 else (20, 4, 3)
    X = np.where(rng.random(shape) < 0.5, 0.0, rng.integers(1, 4, size=shape))
    X[rng.random(shape) < 0.2] = np.nan
    X[:, ALL_NAN_VAR] = np.nan
    X[:, 2] = np.where(np.isnan(X[:, 2]), np.nan, 5.0)
    return X


def _continuous_data(rng: np.random.Generator, ndim: int) -> np.ndarray:
    """Distinct values with zeros, missing values, an all-missing variable and a complete variable, so that nearest neighbors are unique."""
    shape = (30, 6) if ndim == 2 else (20, 6, 3)
    X = np.where(rng.random(shape) < 0.3, 0.0, rng.gamma(2, size=shape))
    X[rng.random(shape) < 0.15] = np.nan
    X[:, ALL_NAN_VAR] = np.nan
    X[:, -1] = rng.gamma(2, size=X[:, -1].shape)
    return X


def _numeric_edata(X) -> EHRData:
    edata = EHRData(X=X)
    edata.var[FEATURE_TYPE_KEY] = NUMERIC_TAG
    return edata


def _assert_same_array_type(result, X) -> None:
    assert type(result) is type(X)
    if isinstance(X, DaskArray):
        assert type(result._meta) is type(X._meta)


def _stored_positions(X: CSBase | DaskArray) -> set[tuple[int, int]]:
    coo = (X.compute() if isinstance(X, DaskArray) else X).tocoo()
    return set(zip(coo.row.tolist(), coo.col.tolist(), strict=True))


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

    layer_before = to_dense(
        edata_before_imputation.layers.get(before_imputation_layer, edata_before_imputation.X), to_cpu_memory=True
    )
    layer_after = to_dense(
        edata_after_imputation.layers.get(after_imputation_layer, edata_after_imputation.X), to_cpu_memory=True
    )

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


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
@pytest.mark.parametrize("var_names", [None, ["0", "1"]], ids=["all", "subset"])
def test_simple_impute_array_types(array_type, ndim, strategy, var_names, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    X = _array_types_data(rng, ndim)
    expected = simple_impute(_numeric_edata(X), var_names=var_names, strategy=strategy, copy=True).X
    edata = _numeric_edata(array_type(X))

    with forbid_dask_compute():
        result = simple_impute(edata, var_names=var_names, strategy=strategy, copy=True).X

    assert isinstance(result, array_type.cls)
    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, equal_nan=True)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("mean", [7 / 3, np.nan, 3.0]), ("median", [3.0, np.nan, 2.0]), ("most_frequent", [3.0, np.nan, 2.0])],
)
def test_simple_impute_all_nan_variable(strategy, expected):
    X = np.array([[1.0, np.nan, 2.0], [np.nan, np.nan, 2.0], [3.0, np.nan, np.nan], [3.0, np.nan, 5.0]])
    imputed = simple_impute(EHRData(X=X), strategy=strategy, copy=True).X

    np.testing.assert_allclose(imputed[1, 0], expected[0])
    np.testing.assert_allclose(imputed[2, 2], expected[2])
    assert np.isnan(imputed[:, 1]).all()


def test_simple_impute_unknown_var_name(impute_num_edata):
    with pytest.raises(KeyError, match="nope"):
        simple_impute(impute_num_edata, var_names=["nope"])

    assert np.isnan(impute_num_edata.X[:, -1]).any()


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_impute_basic(impute_num_edata, array_type, strategy):
    impute_num_edata.X = array_type(impute_num_edata.X)

    with forbid_dask_compute():
        edata_imputed = simple_impute(impute_num_edata, strategy=strategy, copy=True)

    assert isinstance(edata_imputed.X, array_type.cls)
    _base_check_imputation(impute_num_edata, edata_imputed)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_impute_copy(impute_num_edata, strategy):
    edata_imputed = simple_impute(impute_num_edata, strategy=strategy, copy=True)

    assert id(impute_num_edata) != id(edata_imputed)
    _base_check_imputation(impute_num_edata, edata_imputed)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_impute_subset(impute_edata, array_type, strategy):
    impute_edata.X = array_type(impute_edata.X)
    var_names = ("intcol", "indexcol")
    with forbid_dask_compute():
        edata_imputed = simple_impute(impute_edata, var_names=var_names, strategy=strategy, copy=True)

    assert isinstance(edata_imputed.X, array_type.cls)
    _base_check_imputation(impute_edata, edata_imputed, imputed_var_names=var_names)
    X = to_dense(edata_imputed.X, to_cpu_memory=True)
    assert np.any([item != item for item in X[::, 3:4]])

    # manually verified computation result
    if strategy == "mean":
        assert X[0, 1] == 3.0
    elif strategy == "most_frequent":
        assert X[0, 1] == 2.0  # if multiple equally frequent values, return minimum


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_impute_3D_edata(mcar_edata, array_type, strategy):
    mcar_edata.layers[DEFAULT_TEM_LAYER_NAME] = array_type(mcar_edata.layers[DEFAULT_TEM_LAYER_NAME])
    with forbid_dask_compute():
        edata_imputed = simple_impute(mcar_edata, layer=DEFAULT_TEM_LAYER_NAME, strategy=strategy, copy=True)

    assert isinstance(edata_imputed.layers[DEFAULT_TEM_LAYER_NAME], array_type.cls)
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
            reference_value = np.nanmean(
                to_dense(mcar_edata[:, 0, :].layers[DEFAULT_TEM_LAYER_NAME], to_cpu_memory=True)
            )
        elif strategy == "median":
            reference_value = np.nanmedian(
                to_dense(mcar_edata[:, 0, :].layers[DEFAULT_TEM_LAYER_NAME], to_cpu_memory=True)
            )

        assert np.isclose(to_dense(element, to_cpu_memory=True), reference_value)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_impute_3D_edata_nonnumeric(edata_mini_3D_missing_values, array_type, strategy):
    edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME] = array_type(
        edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME]
    )

    if strategy == "most_frequent":
        with forbid_dask_compute():
            edata_imputed = simple_impute(
                edata_mini_3D_missing_values, layer=DEFAULT_TEM_LAYER_NAME, strategy=strategy, copy=True
            )
        assert isinstance(edata_imputed.layers[DEFAULT_TEM_LAYER_NAME], array_type.cls)
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
            to_dense(edata_imputed.layers[DEFAULT_TEM_LAYER_NAME], to_cpu_memory=True)


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_simple_impute_throws_error_non_numerical(impute_edata, strategy):
    with pytest.raises(ValueError):
        simple_impute(impute_edata, strategy=strategy)


def test_simple_impute_invalid_strategy(impute_edata):
    with pytest.raises(ValueError):
        simple_impute(impute_edata, strategy="invalid_strategy", copy=True)  # type: ignore


@pytest.mark.parametrize("edata_mini_3D_missing_values", [True], indirect=True)
def test_knn_impute_3d_numerical(edata_mini_3D_missing_values):
    edata = edata_mini_3D_missing_values.copy()
    edata_imputed = knn_impute(edata, layer=DEFAULT_TEM_LAYER_NAME, copy=True)
    _base_check_imputation(
        edata_mini_3D_missing_values,
        edata_imputed,
        before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
    )


@pytest.mark.parametrize("edata_mini_3D_missing_values", [True], indirect=True)
def test_knn_impute_3d_scikit_backend(edata_mini_3D_missing_values):
    edata = edata_mini_3D_missing_values.copy()
    edata_imputed = knn_impute(edata, layer=DEFAULT_TEM_LAYER_NAME, copy=True, backend="scikit-learn")
    _base_check_imputation(
        edata_mini_3D_missing_values,
        edata_imputed,
        before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
    )


def test_knn_impute_3d_var_names_subset(edata_mini_3D_missing_values):
    edata = edata_mini_3D_missing_values.copy()
    imputed = knn_impute(edata, layer=DEFAULT_TEM_LAYER_NAME, var_names=["1", "2"], copy=True)
    edata_imputed = imputed[:, :2].copy()
    _base_check_imputation(
        edata_mini_3D_missing_values[:, :2],
        edata_imputed,
        before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
    )


def test_knn_impute_3d_layer_none(edata_mini_3D_missing_values):
    with pytest.raises(ValueError, match="requires a layer"):
        knn_impute(edata_mini_3D_missing_values, copy=True)


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
        knn_impute(impute_edata, var_names=["strcol"], n_neighbors=3, copy=True)


def test_knn_impute_numerical_data(impute_num_edata):
    edata_imputed = knn_impute(impute_num_edata, copy=True)

    _base_check_imputation(impute_num_edata, edata_imputed)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("var_names", [None, ["0", "2"]], ids=["all", "subset"])
def test_knn_impute_array_types(array_type, ndim, var_names, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    X = _continuous_data(rng, ndim)
    expected = knn_impute(_numeric_edata(X), var_names=var_names, backend="scikit-learn", copy=True).X
    X = array_type(X)

    # a small working memory imputes sparse arrays in several batches
    with forbid_dask_compute(), config_context(working_memory=0.01):
        result = knn_impute(_numeric_edata(X), var_names=var_names, backend="scikit-learn", copy=True).X

    _assert_same_array_type(result, X)
    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, equal_nan=True)
    if array_type.flags & Flags.Sparse:
        assert _stored_positions(result) == _stored_positions(X)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_knn_impute_without_shared_variables(array_type):
    X = np.array([[1.0, np.nan, np.nan], [np.nan, 2.0, 0.0], [np.nan, 4.0, 0.0], [2.0, np.nan, np.nan]])
    expected = knn_impute(_numeric_edata(X), backend="scikit-learn", copy=True).X
    X = array_type(X)

    with forbid_dask_compute():
        result = knn_impute(_numeric_edata(X), backend="scikit-learn", copy=True).X

    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected)
    np.testing.assert_allclose(expected[0], [1.0, 3.0, 0.0])
    if array_type.flags & Flags.Sparse:
        assert _stored_positions(result) == _stored_positions(X)


@pytest.mark.array_type(Flags.Dask, skip=Flags.Disk | Flags.Gpu)
def test_knn_impute_uneven_chunks(array_type, rng):
    X = _continuous_data(rng, 2)
    expected = knn_impute(_numeric_edata(X), backend="scikit-learn", copy=True).X

    with forbid_dask_compute():
        result = knn_impute(
            _numeric_edata(array_type(X).rechunk(((2, 3, 11, 14), (1, 5)))), backend="scikit-learn", copy=True
        ).X

    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, equal_nan=True)


@pytest.mark.array_type(Flags.Sparse | Flags.Dask, skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize(
    "kwargs",
    [{"backend": "faiss"}, {"backend": "scikit-learn", "backend_kwargs": {"weights": "distance"}}],
    ids=["faiss", "backend_kwargs"],
)
def test_knn_impute_numpy_only_options(array_type, kwargs, rng):
    edata = _numeric_edata(array_type(_continuous_data(rng, 2)))

    with pytest.raises(NotImplementedError, match="only with backend='scikit-learn' and without backend_kwargs"):
        knn_impute(edata, **kwargs)


class _DenseExtraTreesRegressor(ExtraTreesRegressor):
    """Extra trees fit on dense values, since scikit-learn draws different trees from sparse values."""

    def fit(self, X, y, **kwargs):
        return super().fit(to_dense(X), y, **kwargs)

    def predict(self, X):
        return super().predict(to_dense(X))


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_miss_forest_impute_array_types(array_type, rng, monkeypatch):
    X = _continuous_data(rng, 2)
    edata = _numeric_edata(array_type(X))

    if array_type.flags & Flags.Dask:
        with pytest.raises(NotImplementedError, match="does not support dask arrays"):
            miss_forest_impute(edata, n_estimators=10)
        return

    monkeypatch.setattr("sklearn.ensemble.ExtraTreesRegressor", _DenseExtraTreesRegressor)
    expected = miss_forest_impute(_numeric_edata(X), n_estimators=10, copy=True).X
    result = miss_forest_impute(edata, n_estimators=10, copy=True).X

    _assert_same_array_type(result, edata.X)
    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, equal_nan=True)
    if array_type.flags & Flags.Sparse:
        assert _stored_positions(result) == _stored_positions(edata.X)


@pytest.mark.parametrize("edata_mini_3D_missing_values", [True], indirect=True)
def test_missforest_impute_3D_edata(edata_mini_3D_missing_values):
    edata = edata_mini_3D_missing_values.copy()
    edata_imputed = miss_forest_impute(edata, layer=DEFAULT_TEM_LAYER_NAME, copy=True)
    _base_check_imputation(
        edata_mini_3D_missing_values,
        edata_imputed,
        before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
    )


def test_missforest_impute_3d_var_names_subset(edata_mini_3D_missing_values):
    edata = edata_mini_3D_missing_values.copy()
    imputed = miss_forest_impute(edata, layer=DEFAULT_TEM_LAYER_NAME, var_names=["1", "2"], copy=True)
    edata_imputed = imputed[:, :2].copy()
    _base_check_imputation(
        edata_mini_3D_missing_values[:, :2],
        edata_imputed,
        before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
    )
    assert edata.shape == imputed.shape


def test_missforest_impute_3d_layer_none(edata_mini_3D_missing_values):
    with pytest.raises(ValueError, match="requires a layer"):
        miss_forest_impute(edata_mini_3D_missing_values, copy=True)


def test_missforest_impute_non_numerical_data(impute_edata):
    with pytest.raises(ValueError):
        miss_forest_impute(impute_edata, copy=True)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_missforest_impute_numerical_data(impute_num_edata, array_type):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    impute_num_edata.X = array_type(impute_num_edata.X)

    if array_type.flags & Flags.Dask:
        with pytest.raises(NotImplementedError):
            miss_forest_impute(impute_num_edata, copy=True)
        return

    edata_imputed = miss_forest_impute(impute_num_edata, copy=True)

    assert isinstance(edata_imputed.X, array_type.cls)
    _base_check_imputation(impute_num_edata, edata_imputed)


def test_missforest_impute_subset(impute_num_edata):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    var_names = ("col2", "col3")
    edata_imputed = miss_forest_impute(impute_num_edata, var_names=var_names, copy=True)

    _base_check_imputation(impute_num_edata, edata_imputed, imputed_var_names=var_names)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "replacement", [-1.0, {"0": 7.0, "1": -2.0}, [1.0, 2.0, 3.0]], ids=["scalar", "mapping", "per-timepoint"]
)
def test_explicit_impute_array_types(array_type, ndim, replacement, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    if ndim == 2 and isinstance(replacement, list):
        pytest.skip("per-timepoint replacement needs 3D data")
    X = _array_types_data(rng, ndim)
    expected = explicit_impute(_numeric_edata(X), replacement=replacement, copy=True).X
    edata = _numeric_edata(array_type(X))

    with forbid_dask_compute():
        result = explicit_impute(edata, replacement=replacement, copy=True).X

    assert isinstance(result, array_type.cls)
    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, equal_nan=True)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize(
    "impute",
    [
        pytest.param(partial(explicit_impute, replacement="REPLACED"), id="explicit"),
        pytest.param(partial(explicit_impute, replacement=["first", "second"]), id="explicit-per-timepoint"),
        pytest.param(partial(simple_impute, strategy="most_frequent"), id="most_frequent"),
    ],
)
def test_impute_object_array_types(array_type, impute, edata_mini_3D_missing_values):
    layer = edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME]
    expected = impute(edata_mini_3D_missing_values, layer=DEFAULT_TEM_LAYER_NAME, copy=True)
    edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME] = array_type(layer)

    with forbid_dask_compute():
        result = impute(edata_mini_3D_missing_values, layer=DEFAULT_TEM_LAYER_NAME, copy=True)

    result = result.layers[DEFAULT_TEM_LAYER_NAME]
    assert isinstance(result, array_type.cls)
    np.testing.assert_array_equal(to_dense(result, to_cpu_memory=True), expected.layers[DEFAULT_TEM_LAYER_NAME])


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_explicit_impute_3D_edata(edata_mini_3D_missing_values, array_type):
    edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME] = array_type(
        edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME]
    )
    with forbid_dask_compute():
        edata_imputed = explicit_impute(
            edata_mini_3D_missing_values, replacement=1011, layer=DEFAULT_TEM_LAYER_NAME, copy=True
        )
    assert isinstance(edata_imputed.layers[DEFAULT_TEM_LAYER_NAME], array_type.cls)
    layer_after = to_dense(edata_imputed.layers[DEFAULT_TEM_LAYER_NAME], to_cpu_memory=True)

    _base_check_imputation(
        edata_mini_3D_missing_values,
        edata_imputed,
        before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
    )

    # manually check if the NaNs are replaced with replacement value
    assert layer_after[0, 1, 1] == 1011
    assert layer_after[2, 2, 1] == 1011
    assert layer_after[3, 2, 1] == 1011
    assert layer_after[0, 5, 1] == 1011
    assert layer_after[2, 4, 0] == 1011


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_explicit_impute_3D_edata_cat(edata_mini_3D_missing_values, array_type):
    edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME] = array_type(
        edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME]
    )
    with forbid_dask_compute():
        edata_imputed = explicit_impute(
            edata_mini_3D_missing_values,
            replacement={"4": "REPLACED", "5": "REPLACED"},
            layer=DEFAULT_TEM_LAYER_NAME,
            copy=True,
        )
    assert isinstance(edata_imputed.layers[DEFAULT_TEM_LAYER_NAME], array_type.cls)
    layer_after = to_dense(edata_imputed.layers[DEFAULT_TEM_LAYER_NAME], to_cpu_memory=True)
    _base_check_imputation(
        edata_mini_3D_missing_values,
        edata_imputed,
        imputed_var_names=("4", "5"),
        before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
    )

    # manually check if the NaNs are replaced with replacement value in categorical columns
    assert layer_after[0, 5, 1] == "REPLACED"
    assert layer_after[2, 4, 0] == "REPLACED"


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_explicit_impute_all(array_type, impute_num_edata):
    warnings.filterwarnings("ignore", category=FutureWarning)
    impute_num_edata.X = array_type(impute_num_edata.X)

    with forbid_dask_compute():
        edata_imputed = explicit_impute(impute_num_edata, replacement=1011, copy=True)

    assert isinstance(edata_imputed.X, array_type.cls)
    _base_check_imputation(impute_num_edata, edata_imputed)
    assert np.sum([to_dense(edata_imputed.X, to_cpu_memory=True) == 1011]) == 3


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_explicit_impute_subset(impute_edata, array_type):
    impute_edata.X = array_type(impute_edata.X)
    with forbid_dask_compute():
        edata_imputed = explicit_impute(impute_edata, replacement={"strcol": "REPLACED", "intcol": 1011}, copy=True)

    assert isinstance(edata_imputed.X, array_type.cls)
    _base_check_imputation(impute_edata, edata_imputed, imputed_var_names=("strcol", "intcol"))
    X = to_dense(edata_imputed.X, to_cpu_memory=True)
    assert np.sum([X == 1011]) == 1
    assert np.sum([X == "REPLACED"]) == 1


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize(
    "column_name,row_idx,replacement_value",
    [
        ("intcol", 0, 0),
        ("intcol", 0, 0.0),
        ("strcol", 1, ""),
    ],
)
def test_explicit_impute_subset_accepts_falsy_replacement_value(
    impute_edata, array_type, column_name, row_idx, replacement_value
):
    if replacement_value == "":
        impute_edata.X[row_idx, impute_edata.var_names.get_loc(column_name)] = np.nan
    impute_edata.X = array_type(impute_edata.X)
    with forbid_dask_compute():
        edata_imputed = explicit_impute(impute_edata, replacement={column_name: replacement_value}, copy=True)

    assert isinstance(edata_imputed.X, array_type.cls)
    if replacement_value != "":
        _base_check_imputation(impute_edata, edata_imputed, imputed_var_names=(column_name,))
    col_idx = edata_imputed.var_names.get_loc(column_name)
    assert to_dense(edata_imputed.X, to_cpu_memory=True)[row_idx, col_idx] == replacement_value


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_explicit_impute_timepoints(edata_mini_3D_missing_values, array_type):
    edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME] = array_type(
        edata_mini_3D_missing_values.layers[DEFAULT_TEM_LAYER_NAME]
    )
    with forbid_dask_compute():
        edata_imputed = explicit_impute(
            edata_mini_3D_missing_values,
            replacement=[1, 2],
            layer=DEFAULT_TEM_LAYER_NAME,
            copy=True,
        )
    assert isinstance(edata_imputed.layers[DEFAULT_TEM_LAYER_NAME], array_type.cls)
    layer_after = to_dense(edata_imputed.layers[DEFAULT_TEM_LAYER_NAME], to_cpu_memory=True)
    _base_check_imputation(
        edata_mini_3D_missing_values,
        edata_imputed,
        before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
    )

    # manually check if the NaNs are replaced with different replacement values in different timepoints
    assert layer_after[0, 1, 1] == 2
    assert layer_after[2, 2, 1] == 2
    assert layer_after[3, 2, 1] == 2
    assert layer_after[0, 5, 1] == 2
    assert layer_after[2, 4, 0] == 1


def test_explicit_impute_error(impute_edata, edata_mini_3D_missing_values):
    with pytest.raises(ValueError, match="List replacement is only supported for 3D data"):
        explicit_impute(impute_edata, replacement=[1, 2])
    with pytest.raises(ValueError, match="must match number of timepoints"):
        explicit_impute(edata_mini_3D_missing_values, replacement=[1, 2, 3], layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_warning(impute_num_edata, array_type):
    impute_num_edata.X = array_type(impute_num_edata.X)
    with forbid_dask_compute():
        warning_results = _warn_imputation_threshold(impute_num_edata, threshold=20, var_names=None)
    assert warning_results == ({} if array_type.flags & Flags.Dask else {"col1": 25, "col3": 50})


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
def test_warn_imputation_threshold_array_types(array_type, ndim, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    X = _array_types_data(rng, ndim)
    expected = _warn_imputation_threshold(_numeric_edata(X), None, threshold=15)
    edata = _numeric_edata(array_type(X))

    with forbid_dask_compute():
        result = _warn_imputation_threshold(edata, None, threshold=15)

    assert str(ALL_NAN_VAR) in expected
    assert result == ({} if array_type.flags & Flags.Dask else expected)


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


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize("fallback_method", ["mean", "median", "most_frequent", "bfill", None])
@pytest.mark.parametrize("limit", [None, 1])
def test_locf_impute_array_types(array_type, fallback_method, limit, rng):
    X = _array_types_data(rng, 3)
    expected = locf_impute(_numeric_edata(X), fallback_method=fallback_method, limit=limit, copy=True).X
    edata = _numeric_edata(array_type(X))

    with forbid_dask_compute():
        result = locf_impute(edata, fallback_method=fallback_method, limit=limit, copy=True).X

    assert isinstance(result, array_type.cls)
    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, equal_nan=True)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_locf_impute_forward_fill(locf_edata_3d, array_type):
    original = locf_edata_3d.copy()
    locf_edata_3d.layers[DEFAULT_TEM_LAYER_NAME] = array_type(locf_edata_3d.layers[DEFAULT_TEM_LAYER_NAME])
    with forbid_dask_compute():
        result = locf_impute(locf_edata_3d, layer=DEFAULT_TEM_LAYER_NAME, copy=True)

    assert isinstance(result.layers[DEFAULT_TEM_LAYER_NAME], array_type.cls)
    arr = to_dense(result.layers[DEFAULT_TEM_LAYER_NAME], to_cpu_memory=True)

    assert not np.any(np.isnan(arr))

    # ffilled values
    assert arr[0, 0, 1] == 1.0
    assert arr[0, 0, 3] == 3.0
    assert arr[1, 1, 1] == 1.0
    assert arr[1, 1, 2] == 1.0
    assert arr[1, 1, 3] == 1.0

    # mean (default fallback method) values

    _base_check_imputation(
        original,
        result,
        before_imputation_layer=DEFAULT_TEM_LAYER_NAME,
        after_imputation_layer=DEFAULT_TEM_LAYER_NAME,
    )


@pytest.mark.parametrize("fallback_method", ["mean", "median", "most_frequent"])
def test_locf_impute_fallback(locf_edata_3d, fallback_method):
    result = locf_impute(locf_edata_3d, layer=DEFAULT_TEM_LAYER_NAME, fallback_method=fallback_method, copy=True)
    imputed = result.layers[DEFAULT_TEM_LAYER_NAME]

    assert not np.any(np.isnan(imputed))

    original = locf_edata_3d.layers[DEFAULT_TEM_LAYER_NAME].astype(float)
    if fallback_method == "mean":
        expected = np.nanmean(original, axis=(0, 2))
    elif fallback_method == "median":
        expected = np.nanmedian(original, axis=(0, 2))
    else:
        from scipy.stats import mode as scipy_mode

        n_vars = original.shape[1]
        expected = np.array([scipy_mode(original[:, j, :][~np.isnan(original[:, j, :])]).mode for j in range(n_vars)])

    assert np.isclose(imputed[0, 1, 0], expected[1])
    assert np.isclose(imputed[1, 0, 0], expected[0])
    assert np.isclose(imputed[1, 0, 1], expected[0])
    assert np.isclose(imputed[1, 2, 0], expected[2])


def test_locf_impute_bfill(locf_edata_3d):
    result = locf_impute(locf_edata_3d, layer=DEFAULT_TEM_LAYER_NAME, fallback_method="bfill", copy=True)
    imputed = result.layers[DEFAULT_TEM_LAYER_NAME]

    assert not np.any(np.isnan(imputed))

    assert np.isclose(imputed[0, 1, 0], 2.0)
    assert np.isclose(imputed[1, 0, 0], 3.0)
    assert np.isclose(imputed[1, 0, 1], 3.0)
    assert np.isclose(imputed[1, 2, 0], 2.0)


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


def test_locf_impute_no_fallback(locf_edata_3d):
    """fallback_method=None applies only ffill; leading NaNs remain."""
    result = locf_impute(locf_edata_3d, layer=DEFAULT_TEM_LAYER_NAME, fallback_method=None, copy=True)
    imputed = result.layers[DEFAULT_TEM_LAYER_NAME]

    assert imputed[0, 0, 1] == 1.0
    assert imputed[0, 0, 3] == 3.0
    assert imputed[1, 1, 1] == 1.0

    assert np.isnan(imputed[0, 1, 0])
    assert np.isnan(imputed[1, 0, 0])
    assert np.isnan(imputed[1, 0, 1])
    assert np.isnan(imputed[1, 2, 0])


@pytest.mark.parametrize(
    ("fallback_method", "expected"),
    [
        ("mean", [[2.0, 1.0, 1.0, np.nan, 3.0, 3.0]]),
        ("bfill", [[1.0, 1.0, 1.0, np.nan, 3.0, 3.0]]),
        (None, [[np.nan, 1.0, 1.0, np.nan, 3.0, 3.0]]),
    ],
)
def test_locf_impute_limit(fallback_method, expected):
    edata = _numeric_edata(np.array([[[np.nan, 1.0, np.nan, np.nan, 3.0, np.nan]]]))

    locf_impute(edata, fallback_method=fallback_method, limit=1)

    np.testing.assert_array_equal(edata.X[0], expected)


def test_locf_impute_invalid_fallback(locf_edata_3d):
    with pytest.raises(ValueError, match="Unsupported fallback method"):
        locf_impute(locf_edata_3d, layer=DEFAULT_TEM_LAYER_NAME, fallback_method="invalid")


def test_knn_impute_defaults_to_numeric_variables(mimic_2_encoded):
    numeric = mimic_2_encoded.var_names[mimic_2_encoded.var[FEATURE_TYPE_KEY] == NUMERIC_TAG]

    knn_impute(mimic_2_encoded, backend="scikit-learn")

    assert not np.isnan(mimic_2_encoded[:, numeric].X.astype(float)).any()


def test_missforest_impute_reproducible(impute_num_edata):
    first = miss_forest_impute(impute_num_edata, n_estimators=5, random_state=1, copy=True)
    second = miss_forest_impute(impute_num_edata, n_estimators=5, random_state=1, copy=True)

    np.testing.assert_array_equal(first.X, second.X)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
def test_gradient_boosting_impute_array_types(array_type, ndim, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    X = _array_types_data(rng, ndim)
    expected = gradient_boosting_impute(_numeric_edata(X), max_train_obs=15, copy=True).X
    edata = _numeric_edata(array_type(X))

    with forbid_dask_compute():
        result = gradient_boosting_impute(edata, max_train_obs=15, copy=True).X

    assert type(result) is type(edata.X)
    if array_type.flags & Flags.Dask:
        assert type(result._meta) is type(edata.X._meta)
    elif array_type.flags & Flags.Sparse:
        np.testing.assert_array_equal(result.indices, edata.X.indices)
    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, equal_nan=True)
    observed = ~np.isnan(X)
    np.testing.assert_array_equal(expected[observed], X[observed])
    assert np.isnan(expected[:, ALL_NAN_VAR]).all()
    assert not np.isnan(np.delete(expected, ALL_NAN_VAR, axis=1)).any()


def test_gradient_boosting_impute_ignores_missingness_seen_only_in_training(rng):
    values = rng.normal(100, 10, size=2000)
    X = np.column_stack([values, values + rng.normal(size=2000)])
    artifacts = rng.random(2000) < 0.01
    X[artifacts] = [0, np.nan]
    missing = ~artifacts & (rng.random(2000) < 0.3)
    X[missing] = np.nan

    imputed = gradient_boosting_impute(_numeric_edata(X), copy=True).X

    assert np.median(imputed[missing, 0]) > 90


def test_gradient_boosting_impute_stays_within_observed_range():
    predictors = np.random.default_rng(0).uniform(0, 10, size=(2000, 2))
    predictors = predictors[predictors.sum(axis=1) <= 10]
    X = np.vstack([np.column_stack([predictors.sum(axis=1), predictors]), [np.nan, 10, 10]])

    imputed = gradient_boosting_impute(_numeric_edata(X), copy=True).X

    assert imputed[-1, 0] <= np.nanmax(X[:, 0])


def test_gradient_boosting_impute_beats_locf(rng):
    slopes = rng.normal(size=(300, 1, 1))
    trend = slopes * np.arange(12)
    X = np.concatenate([trend, 2 * trend + rng.normal(scale=0.1, size=trend.shape)], axis=1)
    held_out = rng.random(X.shape) < 0.2
    X_missing = np.where(held_out, np.nan, X)

    def error(imputed):
        return np.sqrt(np.mean((imputed[held_out] - X[held_out]) ** 2))

    imputed = gradient_boosting_impute(_numeric_edata(X_missing), copy=True).X
    locf = locf_impute(_numeric_edata(X_missing), copy=True).X
    assert error(imputed) < error(locf) / 2


def test_gradient_boosting_impute_skips_rarely_observed_predictors(rng):
    X = rng.normal(size=(11_000, 6))
    X[:, 1:5] = np.nan
    X[np.arange(4), np.arange(1, 5)] = 1.0
    X[rng.random(11_000) < 0.1, 5] = np.nan

    imputed = gradient_boosting_impute(_numeric_edata(X), max_train_obs=None, random_state=3, copy=True).X

    assert not np.isnan(imputed[:, 5]).any()


def test_gradient_boosting_impute_warns_for_variables_without_observed_values(rng, monkeypatch):
    messages = []
    monkeypatch.setattr(logger, "warning", lambda msg, **kwargs: messages.append(msg))
    X = rng.normal(size=(50, 2))
    X[:, 1] = np.nan

    imputed = gradient_boosting_impute(_numeric_edata(X), copy=True).X

    assert np.isnan(imputed[:, 1]).all()
    assert any("Variable '1' has no observed values" in message for message in messages)


def test_gradient_boosting_impute_categorical(rng):
    X = rng.normal(size=(200, 2))
    X[:, 1] = X[:, 0] > 0
    X[rng.random(200) < 0.2, 1] = np.nan
    edata = EHRData(X=X)
    edata.var[FEATURE_TYPE_KEY] = [NUMERIC_TAG, CATEGORICAL_TAG]

    imputed = gradient_boosting_impute(edata, copy=True).X

    assert set(np.unique(imputed[:, 1])) == {0.0, 1.0}


def test_gradient_boosting_impute_subset(impute_num_edata):
    var_names = ("col2", "col3")
    edata_imputed = gradient_boosting_impute(impute_num_edata, var_names=var_names, copy=True)

    _base_check_imputation(impute_num_edata, edata_imputed, imputed_var_names=var_names)


def test_gradient_boosting_impute_lazy_needs_feature_types():
    edata = EHRData(X=da.from_array(np.array([[1.0, np.nan], [2.0, 3.0]])))

    with forbid_dask_compute(), pytest.raises(ValueError, match="needs feature types"):
        gradient_boosting_impute(edata)
