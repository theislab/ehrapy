from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import singledispatch
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Literal

import array_api_extra as xpx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from array_api_compat import array_namespace, is_lazy_array
from ehrdata._feature_types import _check_feature_types
from ehrdata._logger import logger
from ehrdata.core.constants import FEATURE_TYPE_KEY, NUMERIC_TAG
from fast_array_utils import stats
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase, DaskArray
from sklearn.experimental import enable_iterative_imputer
from sklearn.utils import safe_sqr

from ehrapy._compat import (
    _apply_over_time_axis,
    _broadcast_var_stat,
    _obs_axes,
    _raise_if_dask,
    _raise_if_dask_with_sparse_chunks,
    _set_columns,
    _sparse_columns,
    _sparse_rows,
    nanquantile,
    sparse_nan_moments,
    sparse_nanquantile,
)
from ehrapy._progress import spinner
from ehrapy._settings import settings
from ehrapy.preprocessing._missing_data import _missing_mask
from ehrapy.preprocessing._quality_control import _compute_missing_values

if TYPE_CHECKING:
    from ehrdata import EHRData
    from sklearn.impute import KNNImputer

    type Array = np.ndarray | DaskArray
    type Strategy = Literal["mean", "median", "most_frequent"]


@singledispatch
def _fill_missing(X: Array, values: Array, *, empty_strings: bool = False) -> Array:
    """Replace the missing values of every variable by its entry in `values`, which may be a numpy array also for dask `X`."""
    xp = array_namespace(X)
    return xp.where(
        _missing_mask(X, ("",) if empty_strings and X.dtype == object else ()), _broadcast_var_stat(values, X), X
    )


@_fill_missing.register(CSBase)
def _(X: CSBase, values: np.ndarray, *, empty_strings: bool = False) -> CSBase:
    X = X.copy()
    missing = np.isnan(X.data)
    X.data[missing] = np.broadcast_to(values, X.shape)[_sparse_rows(X)[missing], _sparse_columns(X)[missing]]
    return X


def explicit_impute(
    edata: EHRData,
    replacement: (str | int | float) | (Mapping[str, str | int | float]) | (Sequence[str | int | float]),
    *,
    layer: str | None = None,
    impute_empty_strings: bool = True,
    warning_threshold: int = 70,
    copy: bool = False,
) -> EHRData | None:
    """Replaces all missing values in all columns or a subset of columns specified by the user with the passed replacement value.

    There are three scenarios to cover:
    1. Replace all missing values with the specified value.
    2. Replace all missing values in a subset of columns with a specified value per column.
    3. Replace all missing values with a different value per timepoint.

    The replacement is elementwise, so 3D data is imputed at every timepoint, or with one value per timepoint if a sequence is passed.

    Args:
        edata: Central data object.
        replacement: The value to replace missing values with.
            If a dictionary is provided, the keys represent column
            names and the values represent replacement values for those columns.
            If a list with a length of timepoints is provided, the index of the list represent the timepoint
            and the values represent the replacement value for the respective timepoint.
        layer: The layer to impute.
        impute_empty_strings: If True, empty strings are also replaced.
        warning_threshold: Threshold of percentage of missing values to display a warning for.
        copy: If True, returns a modified copy of the original data object. If False, modifies the object in place.

    Returns:
        If copy is True, a modified copy of the original data object with imputed X.
        If copy is False, the original data object is modified in place, and None is returned.

    Examples:
        Replace all missing values in edata with the value 0:

        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.explicit_impute(edata, replacement=0)

        Replace all missing values in the first timepoint with 1 and all missing values in second timepoint with 2:

        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=10, n_observations=10, base_timepoints=2, missing_values=0.5)
        >>> ep.pp.explicit_impute(edata, replacement=[1, 2])

        Example Output:

        >>> edata.X[0, :, 0]
        [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        0.021176  , -5.25906637,  1.        ,  1.        ,  1.        ]
        >>> edata.X[0, :, 1]
        [ 2.        , 10.30041167, -3.6883699 ,  2.        ,  2.        ,
        0.09374899,  2.        , -3.77042107,  2.        ,  2.45151241]
    """
    if copy:
        edata = edata.copy()

    X = edata.X if layer is None else edata.layers[layer]
    _raise_if_dask_with_sparse_chunks(X, "explicit_impute")

    if isinstance(replacement, str | int | float):
        replacement = dict.fromkeys(edata.var_names, replacement)

    if isinstance(replacement, Mapping):
        values = {var: _extract_impute_value(replacement, var) for var in edata.var_names}
        for var in (var for var, value in values.items() if value is None):
            logger.warning(f"No replace value passed and found for var [not bold green]{var}.")
        values = {var: value for var, value in values.items() if value is not None}
        _warn_imputation_threshold(edata, list(values), threshold=warning_threshold, layer=layer)
        var_indices = edata.var_names.get_indexer(list(values))
        per_var = np.asarray(list(values.values()), dtype=X.dtype)
        X = _set_columns(X, var_indices, _fill_missing(X[:, var_indices], per_var, empty_strings=impute_empty_strings))

    elif isinstance(replacement, Sequence):
        if X.ndim != 3:
            raise ValueError("List replacement is only supported for 3D data.")
        if len(replacement) != X.shape[2]:
            raise ValueError(
                f"Length of replacement sequence ({len(replacement)}) must match number of timepoints ({X.shape[2]})."
            )
        _warn_imputation_threshold(edata, None, threshold=warning_threshold, layer=layer)
        per_timepoint = np.asarray(replacement, dtype=X.dtype).reshape(1, 1, -1)
        X = array_namespace(X).where(
            _missing_mask(X, ("",) if impute_empty_strings and X.dtype == object else ()), per_timepoint, X
        )

    else:
        raise ValueError(  # pragma: no cover
            f"Type {type(replacement)} is not a valid datatype for replacement parameter. Either use int, str or a dict!"
        )

    if layer is None:
        edata.X = X
    else:
        edata.layers[layer] = X

    return edata if copy else None


def _extract_impute_value(replacement: Mapping[str, str | int | float], column_name: str) -> str | int | float | None:
    """Extract the replacement value for a given column in the data object.

    Returns:
        The value to replace missing values
    """
    # try to get a value for the specific column
    if column_name in replacement:
        return replacement[column_name]
    # search for a default value in case no value was specified for that column
    if "default" in replacement:  # pragma: no cover
        return replacement["default"]
    else:
        return None


@singledispatch
def _most_frequent(X: np.ndarray) -> np.ndarray:
    """Most frequent value of every variable across observations (and timepoints) ignoring missing values, the smallest one on ties."""
    samples = pd.DataFrame(np.moveaxis(X, 1, -1).reshape(-1, X.shape[1]))
    return samples.mode(dropna=True).reindex([0]).to_numpy(dtype=X.dtype)[0]


@_most_frequent.register(DaskArray)
def _(X: DaskArray) -> DaskArray:
    axes = _obs_axes(X)
    # every block must hold all observations and timepoints of its variables
    return X.rechunk(dict.fromkeys(axes, -1)).map_blocks(
        _most_frequent.dispatch(np.ndarray), drop_axis=axes, dtype=X.dtype
    )


@_most_frequent.register(CSBase)
def _(X: CSBase) -> np.ndarray:
    columns = _sparse_columns(X)
    valid = ~np.isnan(X.data)
    n_implicit_zeros = X.shape[0] - np.bincount(columns, minlength=X.shape[1])
    zero_columns = np.flatnonzero(n_implicit_zeros)
    pairs, inverse = np.unique(
        np.column_stack(
            [
                np.concatenate([columns[valid], zero_columns]),
                np.concatenate([X.data[valid], np.zeros(len(zero_columns))]),
            ]
        ),
        axis=0,
        return_inverse=True,
    )
    counts = np.bincount(
        inverse.ravel(), weights=np.concatenate([np.ones(valid.sum()), n_implicit_zeros[zero_columns]])
    )
    # lexsort is stable, so values with equal counts keep the ascending order of np.unique
    order = np.lexsort((-counts, pairs[:, 0]))
    top = order[np.unique(pairs[order, 0], return_index=True)[1]]
    mode = np.full(X.shape[1], np.nan)
    mode[pairs[top, 0].astype(np.intp)] = pairs[top, 1]
    return mode


@singledispatch
def _impute_value(X: Array, strategy: Strategy) -> Array:
    """Per-variable statistic across observations (and timepoints) ignoring missing values; NaN for variables without values."""
    if strategy == "mean":
        return xpx.nanmean(X, axis=_obs_axes(X))
    if strategy == "median":
        return nanquantile(X, 0.5, axis=_obs_axes(X))
    return _most_frequent(X)


@_impute_value.register(CSBase)
def _(X: CSBase, strategy: Strategy) -> np.ndarray:
    if strategy == "mean":
        return sparse_nan_moments(X)[1]
    if strategy == "median":
        return sparse_nanquantile(X, 0.5)
    return _most_frequent(X)


def _simple_impute(X: Array | CSBase, strategy: Strategy) -> Array | CSBase:
    if strategy != "most_frequent" and not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float64)
    return _fill_missing(X, _impute_value(X, strategy))


def simple_impute(
    edata: EHRData,
    *,
    var_names: Iterable[str] | None = None,
    strategy: Literal["mean", "median", "most_frequent"] = "mean",
    warning_threshold: int = 70,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Impute missing values in numerical data using mean/median/most frequent imputation.

    If required and using mean or median strategy, the data needs to be properly encoded as this imputation requires
    numerical data only.
    For 3D data, the statistic of a variable is computed across observations and timepoints.
    Variables without any observed value stay missing.

    Args:
        edata: Central data object.
        var_names: A list of column names to apply imputation on (if None, impute all columns).
        strategy: Imputation strategy to use. One of {'mean', 'median', 'most_frequent'}.
        warning_threshold: Display a warning message if percentage of missing values exceeds this threshold.
        layer: The layer to impute.
        copy: Whether to return a copy of `edata` or modify it inplace.

    Returns:
        If copy is True, a modified copy of the original data object with imputed X.
        If copy is False, the original data object is modified in place, and None is returned.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.simple_impute(edata, strategy="median")
    """
    if strategy not in {"mean", "median", "most_frequent"}:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose one of 'mean', 'median' and 'most_frequent'.")

    if copy:
        edata = edata.copy()

    var_names = list(edata.var_names if var_names is None else var_names)
    X = edata.X if layer is None else edata.layers[layer]
    _raise_if_dask_with_sparse_chunks(X, "simple_impute")
    _warn_imputation_threshold(edata, var_names, threshold=warning_threshold, layer=layer)

    var_indices = edata.var_names.get_indexer(var_names)
    X = _set_columns(X, var_indices, _simple_impute(X[:, var_indices], strategy))

    if layer is None:
        edata.X = X
    else:
        edata.layers[layer] = X

    return edata if copy else None


@_check_feature_types
@spinner("Performing KNN impute")
def knn_impute(
    edata: EHRData,
    *,
    var_names: Iterable[str] | None = None,
    n_neighbors: int = 5,
    layer: str | None = None,
    copy: bool = False,
    backend: Literal["scikit-learn", "faiss"] = "faiss",
    warning_threshold: int = 70,
    backend_kwargs: Mapping[str, Any] | None = None,
) -> EHRData | None:
    """Imputes missing values in the input data object using K-nearest neighbor imputation.

    If required, the data needs to be properly encoded as this imputation requires numerical data only.
    For 2D data, if layer is `None`, `edata.X` is used directly.
    For 3D data, the layer is flattened along axis 0 before imputation and reshaped back to 3D afterwards, so every timepoint of an observation is imputed as an observation of its own.
    Variables without any observed value stay missing.

    Args:
        edata: Central data object.
        var_names: A list of variable names indicating which columns to impute.
                   If `None`, all columns are imputed. Default is `None`.
        n_neighbors: Number of neighbors to use when performing the imputation.
        layer: The layer to impute.
        copy: Whether to perform the imputation on a copy of the original data object.
              If `True`, the original object remains unmodified.
        backend: The implementation to use for the KNN imputation.
                 'scikit-learn' is very slow but uses an exact KNN algorithm, whereas 'faiss'
                 is drastically faster but uses an approximation for the KNN graph.
                 In practice, 'faiss' is close enough to the 'scikit-learn' results.
        warning_threshold: Percentage of missing values above which a warning is issued.
        backend_kwargs: Passed to the backend.
                  Pass "mean", "median", or "weighted" for 'strategy' to set the imputation strategy for faiss.
                  See `sklearn.impute.KNNImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html>`_ for more information on the 'scikit-learn' backend.
                  See `fknni.faiss.FaissImputer <https://fknni.readthedocs.io/en/latest/>`_ for more information on the 'faiss' backend.

    Returns:
        If copy is True, a modified copy of the original data object with imputed X.
        If copy is False, the original data object is modified in place, and None is returned.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata_3d = ed.dt.ehrdata_blobs(n_variables=3, n_observations=3, base_timepoints=2, missing_values=0.3)
        >>> edata_imputed = ep.pp.knn_impute(edata_3d, copy=True)

        Example Output:

        >>> edata_3d.X[0, :, :]
        [[-12.12732884, -18.37304373],
        [         nan,  -0.91339411],
        [         nan,  -7.88514984]]
        >>> edata_imputed.X[0, :, :]
        [[-12.12732884, -18.37304373],
        [ -0.07689509,  -0.91339411],
        [ -2.75584421,  -7.88514984]]

    """
    if edata.X is None and layer is None:  # if edata is 3D
        raise ValueError(
            "3D imputation requires a layer to be specified. Pass the layer containing the full temporal data."
        )
    X = edata.X if layer is None else edata.layers[layer]
    if not isinstance(X, np.ndarray) and (backend == "faiss" or backend_kwargs):
        raise NotImplementedError(
            f"knn_impute supports {type(X).__name__} only with backend='scikit-learn' and without backend_kwargs."
        )

    if copy:
        edata = edata.copy()

    _warn_imputation_threshold(edata, var_names, threshold=warning_threshold, layer=layer)

    if backend not in {"scikit-learn", "faiss"}:
        raise ValueError(f"Unknown backend '{backend}' for KNN imputation. Choose between 'scikit-learn' and 'faiss'.")

    if backend_kwargs is None:
        backend_kwargs = {}

    if find_spec("sklearnex") is not None:  # pragma: no cover
        from sklearnex import patch_sklearn, unpatch_sklearn

        patch_sklearn()

    _knn_impute(edata, var_names, n_neighbors, backend=backend, layer=layer, **backend_kwargs)

    if find_spec("sklearnex") is not None:  # pragma: no cover
        unpatch_sklearn()

    return edata if copy else None


@singledispatch
@_apply_over_time_axis
def _knn_impute_function(arr: np.ndarray, var_indices: list[int], numerical_indices: list[int], imputer) -> np.ndarray:
    input_dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float64
    missing = np.isnan(arr[:, numerical_indices].astype(input_dtype))
    # imputers drop variables without observed values, so those are left missing
    empty_columns = set(np.array(numerical_indices)[missing.all(axis=0)].tolist())

    # complete columns to be used as anchors
    complete_numerical_columns = np.array(numerical_indices)[~missing.any(axis=0)].tolist()

    imputer_data_indices = [column for column in var_indices if column not in empty_columns] + [
        column for column in complete_numerical_columns if column not in var_indices
    ]  # columns to impute

    result = arr.copy()
    if imputer_data_indices:
        imputer_x = arr[:, imputer_data_indices].astype(input_dtype, copy=True)
        result[:, imputer_data_indices] = imputer.fit_transform(imputer_x)
    return result


@_knn_impute_function.register(CSBase)
def _(arr: CSBase, var_indices: list[int], numerical_indices: list[int], imputer: KNNImputer) -> CSBase:
    from sklearn import get_config
    from sklearn.utils import gen_batches

    features, sums, n_observed = _knn_statistics(
        arr, _fill_missing(arr, np.zeros(arr.shape[1])), var_indices, numerical_indices
    )
    X, reference = arr.tocsr(), arr.tocsc()
    row_bytes = 8 * (X.shape[0] + X.shape[1] + 2 * imputer.n_neighbors * len(var_indices))
    batch_size = max(1, int(get_config()["working_memory"] * 2**20 // row_bytes))
    batches = [
        _impute_rows(
            X[rows],
            _knn_candidates(X[rows], reference, features, var_indices, imputer.n_neighbors),
            sums,
            n_observed,
            var_indices,
        )
        for rows in gen_batches(X.shape[0], batch_size)
    ]
    return sp.vstack(batches, format=arr.format)


@_knn_impute_function.register(DaskArray)
@_apply_over_time_axis
def _(arr: DaskArray, var_indices: list[int], numerical_indices: list[int], imputer: KNNImputer) -> DaskArray:
    import dask.array as da

    # every block must hold all variables of its rows for the distances
    X = arr.rechunk({1: -1})
    features, sums, n_observed = _knn_statistics(
        X, X.map_blocks(_fill_missing, np.zeros(X.shape[1]), meta=X._meta), var_indices, numerical_indices
    )
    n_neighbors = imputer.n_neighbors
    candidates = da.blockwise(
        _knn_candidates,
        "itcj",
        X,
        "iv",
        X,
        "jv",
        features,
        "v",
        targets=var_indices,
        n_neighbors=n_neighbors,
        new_axes={"t": len(var_indices), "c": 2},
        adjust_chunks={"j": n_neighbors},
        concatenate=True,
        meta=np.empty((0, 0, 0, 0)),
    )
    select = lambda candidates, axis, keepdims: _nearest(candidates, n_neighbors)
    nearest = da.reduction(candidates, select, select, axis=3, keepdims=True, dtype=np.float64, output_size=n_neighbors)
    return da.blockwise(
        _impute_rows,
        "iv",
        X,
        "iv",
        nearest,
        "itck",
        sums,
        "v",
        n_observed,
        "v",
        targets=var_indices,
        concatenate=True,
        meta=X._meta,
    )


def _knn_statistics(
    X: CSBase | DaskArray, filled: CSBase | DaskArray, var_indices: list[int], numerical_indices: list[int]
) -> tuple[Array, Array, Array]:
    """Variables to measure distances on, the imputed and the complete numerical ones, and the sum and number of observed values of every variable."""
    n_observed = X.shape[0] - stats.sum(_missing_mask(X), axis=0)
    columns = np.arange(X.shape[1])
    complete = np.isin(columns, numerical_indices) & (n_observed == X.shape[0])
    return np.isin(columns, var_indices) | complete, stats.sum(filled, axis=0), n_observed


def _mean_squared_differences(Q: np.ndarray | CSBase, R: np.ndarray | CSBase) -> np.ndarray:
    """Mean squared difference of all pairs of rows over their commonly observed variables, infinite if there is none.

    This is the squared nan-Euclidean distance divided by the number of variables.
    """
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.utils.extmath import safe_sparse_dot

    q, r = (_fill_missing(x, np.zeros(x.shape[1])) for x in (Q, R))
    q_missing, r_missing = (_missing_mask(x).astype(np.float64) for x in (Q, R))
    squared = (
        euclidean_distances(q, r, squared=True)
        - safe_sparse_dot(safe_sqr(q), r_missing.T, dense_output=True)
        - safe_sparse_dot(q_missing, safe_sqr(r).T, dense_output=True)
    )
    n_common = (
        Q.shape[1]
        - stats.sum(q_missing, axis=1)[:, None]
        - stats.sum(r_missing, axis=1)
        + safe_sparse_dot(q_missing, r_missing.T, dense_output=True)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n_common > 0, np.maximum(squared, 0) / n_common, np.inf)


def _nearest(candidates: np.ndarray, n_neighbors: int) -> np.ndarray:
    """The `n_neighbors` pairs of distance and value with the smallest distances along the last axis."""
    distances = candidates[..., 0, :]
    if distances.shape[-1] <= n_neighbors:
        return candidates
    kth = np.partition(distances, n_neighbors - 1, axis=-1)[..., n_neighbors - 1, None]
    # ties with the k-th distance go to the earliest candidates, so the result does not depend on the chunking
    rank = (distances >= kth).astype(np.int8) + (distances > kth)
    nearest = np.argsort(rank, axis=-1, kind="stable")[..., None, :n_neighbors]
    return np.take_along_axis(candidates, nearest, axis=-1)


def _knn_candidates(
    Q: np.ndarray | CSBase, R: np.ndarray | CSBase, features: np.ndarray, targets: list[int], n_neighbors: int
) -> np.ndarray:
    """Distances and values of the nearest rows of `R` that observe a target variable, for the rows of `Q` missing it.

    Returns an array of shape `(n_rows, n_targets, 2, n_neighbors)`, with infinite distances where donors are lacking.
    """
    distances = _mean_squared_differences(Q[:, features], R[:, features])
    candidates = np.full((Q.shape[0], len(targets), 2, n_neighbors), np.inf)
    for t, j in enumerate(targets):
        receivers = np.flatnonzero(np.isnan(to_dense(Q[:, [j]])))
        values = to_dense(R[:, [j]]).ravel()
        donors = np.flatnonzero(~np.isnan(values))
        if receivers.size and donors.size:
            pairs = np.broadcast_arrays(distances[np.ix_(receivers, donors)], values[donors])
            nearest = _nearest(np.stack(pairs, axis=-2), n_neighbors)
            candidates[receivers, t, :, : nearest.shape[-1]] = nearest
    return candidates


def _impute_rows(
    X: np.ndarray | CSBase, nearest: np.ndarray, sums: np.ndarray, n_observed: np.ndarray, targets: list[int]
) -> np.ndarray | CSBase:
    """Fill missing target values with the mean of their nearest donors, or of all observed values if no donor shares a variable."""
    found = np.isfinite(nearest[..., 0, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        donor_means = np.where(found, nearest[..., 1, :], 0).sum(axis=-1) / found.sum(axis=-1)
        means = sums[targets] / n_observed[targets]
    values = np.full(X.shape, np.nan)
    values[:, targets] = np.where(found.any(axis=-1), donor_means, means)
    return _fill_missing(X, values)


def _knn_impute(
    edata: EHRData,
    var_names: Iterable[str] | None,
    n_neighbors: int,
    layer: str | None,
    backend: Literal["scikit-learn", "faiss"],
    **kwargs,
) -> None:
    if backend == "scikit-learn":
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=n_neighbors, **kwargs)
    else:
        from fknni import FastKNNImputer

        imputer = FastKNNImputer(n_neighbors=n_neighbors, **kwargs)

    if var_names is None:
        var_names = edata.var_names
    var_indices = edata.var_names.get_indexer(var_names).tolist()

    numerical_var_names = edata.var_names[edata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG]
    numerical_indices = edata.var_names.get_indexer(numerical_var_names).tolist()
    if any(idx not in numerical_indices for idx in var_indices):
        raise ValueError(
            "Can only impute numerical data. Try to restrict imputation to certain columns using "
            "var_names parameter or perform an encoding of your data."
        )
    mtx = edata.X if layer is None else edata.layers[layer]

    X_imputed = _knn_impute_function(mtx, var_indices, numerical_indices, imputer)
    mtx = _set_columns(mtx, var_indices, X_imputed[:, var_indices])

    if layer is None:
        edata.X = mtx
    else:
        edata.layers[layer] = mtx


@singledispatch
@_apply_over_time_axis
def _miss_forest_impute_function(
    arr: np.ndarray, num_initial_strategy, n_estimators, max_iter, random_state
) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import IterativeImputer

    # IterativeImputer drops variables without observed values, so those are left missing
    observed = ~np.isnan(arr).all(axis=0)
    result = arr.copy()
    if observed.any():
        result[:, observed] = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=settings.n_jobs, random_state=random_state),
            initial_strategy=num_initial_strategy,
            max_iter=max_iter,
            random_state=random_state,
        ).fit_transform(arr[:, observed])
    return result


@_miss_forest_impute_function.register(CSBase)
def _(
    arr: CSBase,
    num_initial_strategy: Literal["mean", "median", "most_frequent", "constant"],
    n_estimators: int,
    max_iter: int,
    random_state: int,
) -> CSBase:
    from sklearn.ensemble import ExtraTreesRegressor

    X = arr.tocsc()
    missing = np.isnan(X.data)
    columns = _sparse_columns(X)
    n_missing = np.bincount(columns[missing], minlength=X.shape[1])
    # as in IterativeImputer, variables without observed values stay missing and the others are imputed by ascending missingness
    observed = np.flatnonzero(n_missing < X.shape[0])
    initial = np.zeros(X.shape[1]) if num_initial_strategy == "constant" else _impute_value(X, num_initial_strategy)
    X = _fill_missing(X, np.where(n_missing < X.shape[0], initial, np.nan))
    imputed = missing & np.isin(columns, observed)
    tolerance = 1e-3 * np.abs(X.data[~missing]).max(initial=0)
    for _ in range(max_iter if len(observed) > 1 else 0):
        previous = X.data[imputed]
        for j in observed[np.argsort(n_missing[observed], kind="mergesort")]:
            positions = X.indptr[j] + np.flatnonzero(missing[X.indptr[j] : X.indptr[j + 1]])
            if not positions.size:
                continue
            predictors = X[:, np.setdiff1d(observed, j)].tocsr()
            train = np.ones(X.shape[0], dtype=bool)
            train[X.indices[positions]] = False
            forest = ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=settings.n_jobs, random_state=random_state)
            forest.fit(predictors[train], to_dense(X[:, [j]])[train, 0])
            X.data[positions] = forest.predict(predictors[X.indices[positions]])
        change = np.bincount(_sparse_rows(X)[imputed], weights=np.abs(X.data[imputed] - previous))
        if change.max(initial=0) < tolerance:
            break
    return X.asformat(arr.format)


@spinner("Performing miss-forest impute")
def miss_forest_impute(
    edata: EHRData,
    *,
    var_names: Iterable[str] | None = None,
    num_initial_strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
    max_iter: int = 3,
    n_estimators: int = 100,
    random_state: int = 0,
    warning_threshold: int = 70,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Impute data using the MissForest strategy.

    This function uses the MissForest strategy to impute missing values in the data matrix of an data object.
    The strategy works by fitting a random forest model on each feature containing missing values,
    and using the trained model to predict the missing values.

    For 2D data, if layer is `None`, `edata.X` is used directly.
    For 3D data, the layer is flattened along axis 0 before imputation and reshaped back to 3D afterwards, so every timepoint of an observation is imputed as an observation of its own.
    Variables without any observed value stay missing.

    See https://academic.oup.com/bioinformatics/article/28/1/112/219101.

    If required, the data needs to be properly encoded as this imputation requires numerical data only.

    Args:
        edata: Central data object.
        var_names: Iterable of columns to impute
        num_initial_strategy: The initial strategy to replace all missing numerical values with.
        max_iter: The maximum number of iterations if the stop criterion has not been met yet.
        n_estimators: The number of trees to fit for every missing variable. Has a big effect on the run time.
                      Decrease for faster computations.
        random_state: The random seed for the initialization and the forests.
        warning_threshold: Threshold of percentage of missing values to display a warning for.
        layer: The layer to impute.
        copy: Whether to return a copy or act in place.

    Returns:
        If copy is True, a modified copy of the original data object with imputed X.
        If copy is False, the original data object is modified in place, and None is returned.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=3, n_observations=3, base_timepoints=2, missing_values=0.3)
        >>> edata_imputed = ep.pp.miss_forest_impute(edata, copy=True)

        Example Output:

        >>> edata.X[0, :, :]
        [[-12.12732884, -18.37304373],
        [         nan,  -0.91339411],
        [         nan,  -7.88514984]]
        >>> edata_imputed.X[0, :, :]
        [[-12.12732884, -18.37304373],
        [ -0.3278448 ,  -0.91339411],
        [ -4.39722201,  -7.88514984]]

    """
    if edata.X is None and layer is None:  # if edata is 3D
        raise ValueError(
            "3D imputation requires a layer to be specified. Pass the layer containing the full temporal data."
        )
    _raise_if_dask(
        edata.X if layer is None else edata.layers[layer],
        "miss_forest_impute",
        "the forests are refit on all observations in every iteration",
    )

    if copy:
        edata = edata.copy()

    mtx = edata.X if layer is None else edata.layers[layer]
    var_names = list(edata.var_names if var_names is None else var_names)
    _warn_imputation_threshold(edata, var_names, threshold=warning_threshold, layer=layer)

    var_indices = edata.var_names.get_indexer(var_names).tolist()
    if not var_indices:
        raise ValueError("Cannot find any feature to perform imputation")

    if find_spec("sklearnex") is not None:  # pragma: no cover
        from sklearnex import patch_sklearn, unpatch_sklearn

        patch_sklearn()

    # ensure floating point dtype before imputation, e.g. in case input layer dtype=object
    input_dtype = mtx.dtype if np.issubdtype(mtx.dtype, np.floating) else np.float64
    # this step is the most expensive one and might extremely slow down the impute process
    imputed = _miss_forest_impute_function(
        mtx[:, var_indices].astype(input_dtype, copy=True), num_initial_strategy, n_estimators, max_iter, random_state
    )
    mtx = _set_columns(mtx, var_indices, imputed)

    if find_spec("sklearnex") is not None:  # pragma: no cover
        unpatch_sklearn()

    if layer is None:
        edata.X = mtx
    else:
        edata.layers[layer] = mtx

    return edata if copy else None


def _warn_imputation_threshold(
    edata: EHRData, var_names: Iterable[str] | None, threshold: int = 75, layer: str | None = None
) -> dict[str, float]:
    """Warns the user if the more than $threshold percent had to be imputed.

    Use :func:`ehrdata.harmonize_missing_values` to convert other missing value
    symbols to `np.nan` before imputing.
    For 3D data, the percentage is computed across observations and timepoints.
    Unless `edata.var["missing_values_pct"]` exists, the check is skipped for lazy arrays such as dask arrays, so that it never triggers a computation.

    Args:
        edata: The data object to check
        var_names: The var names which were imputed.
        threshold: A percentage value from 0 to 100 used as minimum.
        layer: The layer to check.
    """
    if "missing_values_pct" not in edata.var:
        mtx = edata.X if layer is None else edata.layers[layer]
        if is_lazy_array(mtx):
            return {}
        n_values = math.prod(mtx.shape[axis] for axis in _obs_axes(mtx))
        edata.var["missing_values_pct"] = _compute_missing_values(mtx, axis=_obs_axes(mtx)) / n_values * 100

    used_var_names = set(edata.var_names) if var_names is None else set(var_names)

    thresholded_var_names = set(edata.var[edata.var["missing_values_pct"] > threshold].index) & set(used_var_names)

    var_name_to_pct: dict[str, float] = {}
    for var in thresholded_var_names:
        var_name_to_pct[var] = edata.var["missing_values_pct"].loc[var]
        logger.warning(f"Feature '{var}' had more than {var_name_to_pct[var]:.2f}% missing values!")

    return var_name_to_pct


def locf_impute(
    edata: EHRData,
    *,
    var_names: Iterable[str] | None = None,
    layer: str | None = None,
    fallback_method: Literal["mean", "median", "most_frequent", "bfill"] | None = "mean",
    copy: bool = False,
) -> EHRData | None:
    """Impute missing values by carrying forward the last observed value along the time axis.

    Implements Last Observation Carried Forward (LOCF) for longitudinal (3D) data.
    For each patient and feature, missing values are replaced with the most recent
    non-missing value. Missing values that occur before any observation for a given
    patient are filled using a fallback method.

    Args:
        edata: Central data object.
        var_names: A list of column names to apply imputation on (if ``None``, impute all columns).
        layer: The layer to impute. Must contain 3D data of shape ``(n_obs, n_vars, n_time)``.
        fallback_method: Method for imputing values before the first observation per patient.
                         ``'mean'`` fills with the per-feature mean, ``'median'`` fills with the
                         per-feature median, ``'most_frequent'`` fills with the per-feature most
                         frequent value (all computed from the original data, before forward
                         filling), ``'bfill'`` fills with each patient's first observed
                         value (backward fill), and ``None`` leaves remaining NaN values
                         untouched.
        copy: Whether to return a copy of ``edata`` or modify it inplace.

    Returns:
        If copy is True, a modified copy of the original data object with imputed data.
        If copy is False, the original data object is modified in place, and None is returned.

    Raises:
        ValueError: If the data is not 3D or if an unsupported ``fallback_method`` is specified.

    Examples:
        >>> import numpy as np
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> data = np.array(
        ...     [
        ...         [[1.0, np.nan, 3.0, np.nan], [np.nan, 2.0, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0]],
        ...         [[np.nan, np.nan, 3.0, np.nan], [1.0, np.nan, np.nan, np.nan], [np.nan, 2.0, np.nan, 4.0]],
        ...     ]
        ... )
        >>> edata = ed.EHRData(X=data)
        >>> ep.pp.locf_impute(edata)
        >>> edata.X
        array([[[1.        , 1.        , 3.        , 3.        ],
                [2.33, 2.        , 2.        , 4.        ],
                [5.        , 6.        , 7.        , 8.        ]],
        <BLANKLINE>
               [[2.33, 2.33, 3.        , 3.        ],
                [1.        , 1.        , 1.        , 1.        ],
                [5.33, 2.        , 2.        , 4.        ]]])
    """
    import xarray as xr

    if fallback_method not in ("mean", "median", "most_frequent", "bfill", None):
        raise ValueError(
            f"Unsupported fallback method '{fallback_method}'. Use 'mean', 'median', 'most_frequent', 'bfill', or None."
        )

    if copy:
        edata = edata.copy()

    X = edata.X if layer is None else edata.layers[layer]

    if X.ndim != 3:
        raise ValueError(
            f"locf_impute requires 3D data (n_obs, n_vars, n_time), got array with shape {X.shape}. "
            "Use the 'layer' parameter to specify a layer containing 3D data."
        )

    var_indices = edata.var_names.get_indexer(list(edata.var_names if var_names is None else var_names))
    original = X[:, var_indices]
    if not np.issubdtype(original.dtype, np.floating):
        original = original.astype(np.float64)

    filled = xr.DataArray(original, dims=["obs", "var", "time"]).ffill(dim="time")
    if fallback_method == "bfill":
        filled = filled.bfill(dim="time")
    filled = filled.data
    if fallback_method in ("mean", "median", "most_frequent"):
        filled = _fill_missing(filled, _impute_value(original, fallback_method))

    X = _set_columns(X, var_indices, filled)
    if layer is None:
        edata.X = X
    else:
        edata.layers[layer] = X

    return edata if copy else None
