from collections.abc import Callable
from functools import singledispatch
from typing import Any, Literal

import numpy as np
from scipy.sparse import coo_matrix


def _raise_array_type_not_implemented(function: Callable[..., Any], array_type: type) -> None:
    raise NotImplementedError(f"{function.__name__} not implemented for type {array_type}")


def timeseries_distance(
    obs_indices_x: np.ndarray,
    obs_indices_y: np.ndarray,
    R: np.ndarray | coo_matrix,
    metric: Literal["dtw", "soft_dtw", "gak"] = "dtw",
) -> float:
    """Calculate temporal distance between two patients across all variables.

    For each variable where both patients have >3 valid measurements, computes distance between their time series.
    Returns average distance across all valid variable pairs.

    Args:
        obs_indices_x: Array containing patient index [i]
        obs_indices_y: Array containing patient index [j]
        R: Array containing timeseries measurements
        metric: Temporal distance metric.

            - "dtw": Standard DTW with exact optimal alignment via dynamic programming
            - "soft_dtw": Differentiable approximation using soft-min, robust to noise/outliers
            - "gak": Global Alignment Kernel, similarity-based metric robust to irregular sampling

    Returns:
        Average temporal distance across valid variable pairs.
        Returns 0 if no valid variable pairs exist.
    """
    return _timeseries_distance_impl(R, obs_indices_x, obs_indices_y, metric)


@singledispatch
def _timeseries_distance_impl(
    R: np.ndarray | coo_matrix,
    obs_indices_x: np.ndarray,
    obs_indices_y: np.ndarray,
    metric: Literal["dtw", "soft_dtw", "gak"],
) -> float:
    _raise_array_type_not_implemented(timeseries_distance, type(R))

    return None


@_timeseries_distance_impl.register
def _(
    R: np.ndarray,
    obs_indices_x: np.ndarray,
    obs_indices_y: np.ndarray,
    metric: Literal["dtw", "soft_dtw", "gak"],
) -> float:
    match metric:
        case "dtw":
            from tslearn.metrics import dtw

            metric_func = dtw
        case "soft_dtw":
            from tslearn.metrics import soft_dtw

            metric_func = lambda x, y: abs(soft_dtw(x, y) - 0.5 * (soft_dtw(x, x) + soft_dtw(y, y)))
        case "gak":
            from tslearn.metrics import gak

            metric_func = lambda x, y: 1.0 - gak(x, y)
        case _:
            raise ValueError(f"Unknown time series metric {metric}. Must be one of 'dtw', 'soft_dtw', or 'gak'.")

    obs_i = int(np.asarray(obs_indices_x).flat[0])
    obs_j = int(np.asarray(obs_indices_y).flat[0])
    total_distance = 0
    valid_variable_count = 0

    for variable_idx in range(R.shape[1]):
        series_i = R[obs_i, variable_idx, :]
        series_j = R[obs_j, variable_idx, :]
        valid_measurements_i = ~np.isnan(series_i)
        valid_measurements_j = ~np.isnan(series_j)

        if np.sum(valid_measurements_i) > 3 and np.sum(valid_measurements_j) > 3:
            valid_series_i = series_i[valid_measurements_i].reshape(-1, 1)
            valid_series_j = series_j[valid_measurements_j].reshape(-1, 1)
            variable_distance = metric_func(valid_series_i, valid_series_j)
            total_distance += variable_distance
            valid_variable_count += 1

    return total_distance / max(valid_variable_count, 1)


@_timeseries_distance_impl.register
def _(
    R: coo_matrix,
    obs_indices_x: np.ndarray,
    obs_indices_y: np.ndarray,
    metric: Literal["dtw", "soft_dtw", "gak"],
) -> float:
    _raise_array_type_not_implemented(timeseries_distance, type(R))

    return None
