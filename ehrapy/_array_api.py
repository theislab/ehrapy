"""Helpers for array operations compatible across backends (NumPy, Dask, etc.)."""

from __future__ import annotations

from typing import Any


def ffill_along_axis(xp: Any, arr: Any, axis: int = -1) -> Any:
    """Forward-fill NaN values along an axis.

    Each NaN is replaced with the most recent non-NaN value in the same position
    along the specified axis. Leading NaNs (before any valid value) are left as
    NaN and can be filled by the caller (e.g. with mean/median).

    Uses a parallel-prefix doubling strategy: at pass *d*, the array is shifted
    right by 2^d positions and NaN slots are filled from the shifted copy. After
    ceil(log2(n)) passes every NaN reachable from a prior valid value has been
    filled. Each pass is a single vectorized ``xp.where`` over the full array,
    so the total work stays in compiled code regardless of backend.

    Args:
        xp: Array namespace (e.g. from ``array_api_compat.array_namespace()``).
        arr: Input array.
        axis: Axis along which to forward-fill (default: last axis).

    Returns:
        Array with NaNs forward-filled along the specified axis.
    """
    import math

    axis = axis if axis >= 0 else arr.ndim + axis
    n = arr.shape[axis]
    result = xp.asarray(arr, copy=True)

    def _slice_along(ax: int, start: int, stop: int | None = None) -> tuple:
        s = [slice(None)] * result.ndim
        s[ax] = slice(start, stop)
        return tuple(s)

    for d in range(math.ceil(math.log2(max(n, 2)))):
        shift = 2**d
        if shift >= n:
            break
        pad_shape = list(result.shape)
        pad_shape[axis] = shift
        pad = xp.full(pad_shape, float("nan"), dtype=result.dtype)
        shifted = xp.concat([pad, result[_slice_along(axis, 0, n - shift)]], axis=axis)
        result = xp.where(xp.isnan(result), shifted, result)

    return result


def nanmean_api(xp: Any, arr: Any, axes: int | tuple[int, ...]) -> Any:
    """Compute mean ignoring NaN values.

    Args:
        xp: Array namespace (e.g. from ``array_api_compat.array_namespace()``).
        arr: Input array.
        axes: Axis or axes along which to compute the mean.

    Returns:
        Array of means, with NaN values excluded from the computation.
    """
    mask = xp.isnan(arr)
    zero_filled = xp.where(mask, xp.zeros_like(arr), arr)
    count = xp.sum(xp.astype(~mask, arr.dtype), axis=axes)
    return xp.sum(zero_filled, axis=axes) / count


def nanmedian_api(xp: Any, arr: Any) -> Any:
    """Compute per-feature median ignoring NaN values.

    For a 3D array of shape ``(n_obs, n_vars, n_time)``, returns one median per
    feature (n_vars), computed across all observations and time steps.

    Args:
        xp: Array namespace (e.g. from ``array_api_compat.array_namespace()``).
        arr: Input array of shape ``(n_obs, n_vars, n_time)``.

    Returns:
        Array of shape ``(n_vars,)`` with the median per feature. Returns NaN
        for features where all values are NaN.
    """
    n_obs, n_vars, n_time = arr.shape
    arr_flat = xp.reshape(xp.permute_dims(arr, (1, 0, 2)), (n_vars, -1))

    medians = []
    for i in range(n_vars):
        row = arr_flat[i, :]
        not_nan = ~xp.isnan(row)
        n = int(xp.sum(xp.astype(not_nan, xp.float64)))
        if n == 0:
            medians.append(float("nan"))
            continue
        filled = xp.where(not_nan, row, xp.asarray(float("inf"), dtype=arr.dtype))
        sorted_row = xp.sort(filled)
        if n % 2 == 1:
            medians.append(float(sorted_row[n // 2]))
        else:
            medians.append(float((sorted_row[n // 2 - 1] + sorted_row[n // 2]) / 2))

    return xp.asarray(medians, dtype=arr.dtype)
