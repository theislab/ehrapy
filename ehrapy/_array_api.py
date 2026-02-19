from __future__ import annotations


def nanmean_api(xp, arr, axes):
    """Compute mean ignoring NaN values using Array API operations."""
    mask = xp.isnan(arr)
    zero_filled = xp.where(mask, xp.zeros_like(arr), arr)
    count = xp.sum(xp.astype(~mask, arr.dtype), axis=axes)
    return xp.sum(zero_filled, axis=axes) / count


def nanmedian_api(xp, arr):
    """Compute per-feature median ignoring NaN values using Array API operations.

    Computes the median for each feature across all patients and time steps
    for a 3D array of shape ``(n_obs, n_vars, n_time)``.
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
