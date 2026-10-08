from __future__ import annotations

import inspect
from collections.abc import Sequence
from functools import singledispatch, wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import array_api_extra as xpx
import numpy as np
import scipy.sparse as sp
from array_api_compat import array_namespace, is_lazy_array
from fast_array_utils.types import CSBase, DaskArray

P = ParamSpec("P")
R = TypeVar("R")

if TYPE_CHECKING:
    from collections.abc import Callable

    type Array = np.ndarray | DaskArray


def _raise_array_type_not_implemented(func: Callable, type_: type) -> NotImplementedError:
    supported = ", ".join(t.__name__ for t in func.registry if t is not object)  # type: ignore[attr-defined]
    raise NotImplementedError(f"{func.__name__} does not support array type {type_.__name__}. Supported: {supported}.")


def _apply_over_time_axis(f: Callable) -> Callable:
    """Decorator to allow functions to handle both 2D and 3D arrays.

    - If the input is 2D: pass it through unchanged.
    - If the input is 3D: reshape to 2D before calling the function, then reshape the result back to 3D.
    """

    @wraps(f)
    def wrapper(arr, *args, **kwargs):
        if arr.ndim == 2:
            return f(arr, *args, **kwargs)

        elif arr.ndim == 3:
            n_obs, n_vars, n_time = arr.shape
            arr_2d = np.moveaxis(arr, 1, 2).reshape(-1, n_vars)
            arr_modified_2d = f(arr_2d, *args, **kwargs)
            return np.moveaxis(arr_modified_2d.reshape(n_obs, n_time, n_vars), 1, 2)

        else:
            raise ValueError(f"Unsupported array dimensionality: {arr.ndim}. Please reshape the array to 2D or 3D.")

    return wrapper


def function_2D_only(*, allow_single_timepoint: bool = False):
    """Reject 3D input in functions that only operate on `(n_obs, n_vars)` data.

    The checked arrays are the ones the function reads: `edata.obsm[use_rep]`, the layers named by `layer` or `layers`, or `edata.X`.

    Args:
        allow_single_timepoint: Also accept 3D arrays with a single timepoint, for functions that squeeze it themselves.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                arguments = signature.bind_partial(*args, **kwargs).arguments
            except TypeError:
                return func(*args, **kwargs)
            data = arguments.get("edata")
            use_rep = arguments.get("use_rep")
            layers = arguments.get("layer", arguments.get("layers"))
            layers = [layers] if isinstance(layers, str) else [layer for layer in layers or () if layer is not None]

            if data is None or not hasattr(data, "X"):
                arrays = {"the input": data}
            elif use_rep is not None and use_rep in data.obsm:
                arrays = {f"edata.obsm[{use_rep!r}]": data.obsm[use_rep]}
            elif layers:
                arrays = {f"edata.layers[{layer!r}]": data.layers[layer] for layer in layers}
            else:
                arrays = {"edata.X": data.X}

            for name, array in arrays.items():
                if getattr(array, "ndim", 2) == 3 and not (allow_single_timepoint and array.shape[2] == 1):
                    raise ValueError(
                        f"{func.__name__}() only supports 2D data, but {name} has shape {array.shape}. "
                        "Aggregate the time axis first, e.g. with `ep.pp.summarize_measurements()`."
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _raise_if_dask_with_sparse_chunks(X, name: str) -> None:
    if isinstance(X, DaskArray) and isinstance(X._meta, CSBase):
        raise NotImplementedError(f"{name} does not support dask arrays with sparse chunks.")


def _obs_axes(X) -> tuple[int, ...]:
    """Axes that hold samples of a variable: observations, and timepoints for 3D data."""
    return (0,) if X.ndim == 2 else (0, 2)


def _var_axes(X: Array | CSBase) -> tuple[int, ...]:
    """Axes that hold the values of an observation: variables, and timepoints for 3D data."""
    return (1,) if X.ndim == 2 else (1, 2)


def _broadcast_var_stat(stat, X):
    """Reshape a per-variable statistic of shape `(n_vars,)` or per-row statistics `(n_obs, n_vars)` to broadcast against `X`."""
    xp = array_namespace(stat)
    if stat.ndim == 1:
        stat = xp.reshape(stat, (1, -1))
    return xp.reshape(stat, (*stat.shape, *(1,) * (X.ndim - 2)))


def nanvar(X, /, *, axis: int | tuple[int, ...]):
    """Population variance ignoring NaNs."""
    xp = array_namespace(X)
    axes = (axis,) if isinstance(axis, int) else axis
    mean = xp.reshape(xpx.nanmean(X, axis=axes, xp=xp), [1 if i in axes else n for i, n in enumerate(X.shape)])
    nan_mask = xp.isnan(X)
    squared = xp.where(nan_mask, xp.zeros_like(X), (X - mean) ** 2)
    return xp.sum(squared, axis=axes) / xp.sum(xp.astype(~nan_mask, X.dtype), axis=axes)


def nanstd(X, /, *, axis: int | tuple[int, ...]):
    """Population standard deviation ignoring NaNs."""
    return array_namespace(X).sqrt(nanvar(X, axis=axis))


def nanquantile(X, q: float | Sequence[float], /, *, axis: int | tuple[int, ...]):
    """Quantiles ignoring NaNs; lazy for dask, which rechunks only the reduced axes."""
    q = [float(x) for x in q] if isinstance(q, Sequence) else float(q)
    return array_namespace(X).nanquantile(X, q, axis=axis)


def _sparse_columns(X: CSBase) -> np.ndarray:
    """Column index of every stored element of a CSR or CSC matrix."""
    if X.format == "csr":
        return X.indices
    return np.repeat(np.arange(X.shape[1]), np.diff(X.indptr))


def _sparse_rows(X: CSBase) -> np.ndarray:
    """Row index of every stored element of a CSR or CSC matrix."""
    if X.format == "csc":
        return X.indices
    return np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))


def sparse_nan_moments(X: CSBase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column count of non-NaN values, mean and population variance, counting implicit zeros as values."""
    columns = _sparse_columns(X)
    valid = ~np.isnan(X.data)
    n_nan = np.bincount(columns[~valid], minlength=X.shape[1])
    count = X.shape[0] - n_nan
    n_implicit_zeros = X.shape[0] - np.bincount(columns, minlength=X.shape[1])
    values = X.data[valid].astype(np.float64)
    total = np.bincount(columns[valid], weights=values, minlength=X.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        squared_deviations = np.bincount(
            columns[valid], weights=(values - mean[columns[valid]]) ** 2, minlength=X.shape[1]
        )
        var = (squared_deviations + n_implicit_zeros * mean**2) / count
    return count, mean, var


def sparse_nan_min_max(X: CSBase) -> tuple[np.ndarray, np.ndarray]:
    """Per-column minimum and maximum ignoring NaNs, counting implicit zeros as values."""
    columns = _sparse_columns(X)
    valid = ~np.isnan(X.data)
    minimum = np.full(X.shape[1], np.inf)
    maximum = np.full(X.shape[1], -np.inf)
    np.minimum.at(minimum, columns[valid], X.data[valid])
    np.maximum.at(maximum, columns[valid], X.data[valid])
    has_implicit_zeros = np.bincount(columns, minlength=X.shape[1]) < X.shape[0]
    minimum[has_implicit_zeros] = np.minimum(minimum[has_implicit_zeros], 0)
    maximum[has_implicit_zeros] = np.maximum(maximum[has_implicit_zeros], 0)
    all_nan = np.isinf(minimum)
    minimum[all_nan] = maximum[all_nan] = np.nan
    return minimum, maximum


def _order_statistic(values: np.ndarray, n_negative: int, n_implicit_zeros: int, rank: np.ndarray) -> np.ndarray:
    """Order statistics of sorted stored `values` with `n_implicit_zeros` zeros inserted after the negative ones."""
    stat = np.zeros(rank.shape)
    negative = rank < n_negative
    stat[negative] = values[rank[negative]]
    positive = rank >= n_negative + n_implicit_zeros
    stat[positive] = values[rank[positive] - n_implicit_zeros]
    return stat


def sparse_nanquantile(X: CSBase, q: float | Sequence[float]) -> np.ndarray:
    """Per-column quantiles ignoring NaNs, counting implicit zeros as values, with numpy's linear interpolation.

    Returns an array of shape `(len(q), n_vars)`, or `(n_vars,)` for scalar `q`.
    """
    X = X.tocsc()
    qs = np.atleast_1d(np.asarray(q, dtype=np.float64))
    result = np.full((len(qs), X.shape[1]), np.nan)
    for j in range(X.shape[1]):
        stored = X.data[X.indptr[j] : X.indptr[j + 1]]
        n_implicit_zeros = X.shape[0] - len(stored)
        values = np.sort(stored[~np.isnan(stored)])
        n = len(values) + n_implicit_zeros
        if n == 0:
            continue
        n_negative = int(np.searchsorted(values, 0))
        position = (n - 1) * qs
        lower = np.floor(position).astype(int)
        upper = np.minimum(lower + 1, n - 1)
        low_value, high_value = (
            _order_statistic(values, n_negative, n_implicit_zeros, rank) for rank in (lower, upper)
        )
        result[:, j] = low_value + (position - lower) * (high_value - low_value)
    return result if np.ndim(q) else result[0]


def _by_group(X, groups: np.ndarray | None, stats: Callable[[Any], Sequence[Any | None]]) -> tuple[Any | None, ...]:
    """Per-variable statistics broadcastable against `X`, estimated per group if `groups` is given; `None` statistics stay `None`."""
    if groups is None:
        return tuple(None if stat is None else _broadcast_var_stat(stat, X) for stat in stats(X))
    xp = array_namespace(X)
    per_group = zip(*(stats(X[groups == group]) for group in range(groups.max() + 1)), strict=True)
    return tuple(None if stat[0] is None else _broadcast_var_stat(xp.stack(stat)[groups], X) for stat in per_group)


@singledispatch
def _set_columns(X: Array, indices: np.ndarray, values: Array) -> Array:
    X[:, indices] = values
    return X


@_set_columns.register(CSBase)
def _(X: CSBase, indices: np.ndarray, values: CSBase) -> CSBase:
    rest = np.setdiff1d(np.arange(X.shape[1]), indices)
    combined = sp.hstack([values, X[:, rest]], format=X.format)
    return combined[:, np.argsort(np.concatenate([indices, rest]))]


@singledispatch
def _columnwise(X: Array, groups: np.ndarray | None, kernel: Callable[[np.ndarray], np.ndarray]) -> Array:
    """Apply a scikit-learn transformer that treats every variable independently, fitted per group if `groups` is given."""
    kernel = _apply_over_time_axis(kernel)
    if groups is None:
        return kernel(X)
    result = np.empty(X.shape, dtype=np.float64)
    for group in np.unique(groups):
        result[groups == group] = kernel(X[groups == group])
    return result


@_columnwise.register(CSBase)
def _(X: CSBase, groups: np.ndarray | None, kernel: Callable[[np.ndarray], np.ndarray]) -> CSBase:
    result = X.astype(np.float64).tocsc()
    groups = np.zeros(X.shape[0], dtype=np.intp) if groups is None else groups
    group_sizes = np.bincount(groups)
    for column in range(X.shape[1]):
        stored = slice(result.indptr[column], result.indptr[column + 1])
        rows, values = result.indices[stored], result.data[stored]
        for group in np.flatnonzero(group_sizes):
            in_group = groups[rows] == group
            n_stored = np.count_nonzero(in_group)
            samples = np.zeros((group_sizes[group], 1))
            samples[:n_stored, 0] = values[in_group]
            transformed = kernel(samples)[:, 0]
            if np.any(transformed[n_stored:] != 0):
                _raise_densifying("This transformation", "it maps implicit zeros to nonzero values")
            values[in_group] = transformed[:n_stored]
    return result.asformat(X.format)


@_columnwise.register(DaskArray)
def _(X: DaskArray, groups: np.ndarray | None, kernel: Callable[[np.ndarray], np.ndarray]) -> DaskArray:
    return _map_variable_blocks(X, _columnwise, groups, kernel, meta=X._meta.astype(np.float64))


def _map_variable_blocks(X: DaskArray, func: Callable[..., Any], *args: Any, meta: Any) -> DaskArray:
    """Apply an in-memory `func` to blocks that hold all observations and timepoints of their variables."""
    return X.rechunk(dict.fromkeys(_obs_axes(X), -1)).map_blocks(func, *args, meta=meta)


def _map_reduction(X: DaskArray, func: Callable[..., Any], axis: tuple[int, ...], dtype: np.dtype | type) -> DaskArray:
    """Reduce over `axis` with an in-memory `func(block, axis=axis)`, applied to blocks that hold all values along `axis`."""
    return X.rechunk(dict.fromkeys(axis, -1)).map_blocks(
        func, axis=axis, drop_axis=axis, meta=np.array((), dtype=dtype)
    )


def _materialize(*arrays: Array) -> list[np.ndarray]:
    """Convert to numpy arrays, computing all lazy arrays with a single `dask.compute`."""
    if any(is_lazy_array(array) for array in arrays):
        import dask

        arrays = dask.compute(*arrays)
    return [np.asarray(array) for array in arrays]


def _raise_densifying(name: str, reason: str) -> None:
    raise NotImplementedError(f"{name} does not support sparse arrays because {reason}, which would densify them.")


def _raise_if_not_numpy(X: Array | CSBase, name: str, reason: str) -> None:
    if not isinstance(X, np.ndarray):
        raise NotImplementedError(f"{name} only supports numpy arrays because {reason}, got {type(X).__name__}.")


def _raise_if_sparse(X: Array | CSBase, name: str, reason: str) -> None:
    if isinstance(X, CSBase) or (isinstance(X, DaskArray) and isinstance(X._meta, CSBase)):
        _raise_densifying(name, reason)


@singledispatch
def _like_obs(X: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Per-observation `values` as an array of the same kind as `X`, chunked like its observations."""
    return values


@_like_obs.register(DaskArray)
def _(X: DaskArray, values: np.ndarray) -> DaskArray:
    import dask.array as da

    return da.from_array(values, chunks=(X.chunks[0], *(-1,) * (values.ndim - 1)))
