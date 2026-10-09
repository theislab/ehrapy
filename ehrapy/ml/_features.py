from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from array_api_compat import array_namespace
from fast_array_utils.conv import to_dense

from ehrapy._compat import _like_obs
from ehrapy.preprocessing._summarize_measurements import _tem_times, summarize_measurements

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData
    from fast_array_utils.types import DaskArray

    from ehrapy.ml._task import Task
    from ehrapy.preprocessing._summarize_measurements import Statistic

    type Array = np.ndarray | DaskArray


def _features(
    edata: EHRData,
    task: Task,
    var_names: Sequence[str],
    obs_keys: Sequence[str],
    layer: str | None,
    statistics: Sequence[Statistic],
    *,
    feature_names: Sequence[str] | None = None,
) -> tuple[Array, list[str]]:
    """Dense features of every observation and their names, aligning one-hot encoded `obs` columns with `feature_names`.

    The features of rolling tasks have a third axis with the features of every timepoint, which are missing where no timepoint precedes it by more than the gap.
    """
    X = edata.X if layer is None else edata.layers[layer]
    if task.rolling:
        windows = task._windows(X.shape[2])
        observed = {str(t): window for t, window in enumerate(windows) if window is not None}
        if not observed:
            raise ValueError(f"No timepoint lies more than {task.gap} timepoints after another.")
        summary = summarize_measurements(
            edata, layer=layer, var_names=var_names, statistics=statistics, tem_names=observed
        )
        xp = array_namespace(summary.X)
        names = [f"{var}_{statistic}" for var in var_names for statistic in statistics]
        values = xp.reshape(summary.X, (summary.shape[0], len(names), len(observed)))
        # Only the first timepoints lack a window.
        missing = xp.full((*values.shape[:2], len(windows) - len(observed)), np.nan, dtype=values.dtype)
        values = xp.concat([missing, values], axis=2)
    elif X.ndim == 3:
        summary = summarize_measurements(
            edata, layer=layer, var_names=var_names, statistics=statistics, tem_names=task._window(X.shape[2])
        )
        values, names = summary.X, list(summary.var_names)
    elif task.prediction_time is not None or task.observation_window is not None or task.gap:
        raise ValueError("The timepoints of a task only apply to longitudinal data.")
    else:
        view = edata[:, list(var_names)]
        values, names = (view.X if layer is None else view.layers[layer]), list(var_names)

    covariates = _covariates(edata, obs_keys, None if feature_names is None else feature_names[len(names) :])
    return _with_covariates(values, covariates), [*names, *covariates.columns]


def _sequences(
    edata: EHRData,
    task: Task,
    var_names: Sequence[str],
    obs_keys: Sequence[str],
    layer: str | None,
    *,
    feature_names: Sequence[str] | None = None,
) -> tuple[Array, list[str]]:
    """Time series of the variables over the observation window, followed by the `obs` covariates repeated over time."""
    X = edata.X if layer is None else edata.layers[layer]
    if X.ndim != 3:
        raise ValueError("Models of time series need longitudinal data.")
    window = edata[:, list(var_names), task._window(X.shape[2])]
    covariates = _covariates(edata, obs_keys, None if feature_names is None else feature_names[len(var_names) :])
    return _with_covariates(window.X if layer is None else window.layers[layer], covariates), [
        *var_names,
        *covariates.columns,
    ]


def _with_covariates(values: Array, covariates: pd.DataFrame) -> Array:
    """Dense `values` followed by the covariates, which are repeated over time for 3D values."""
    values = to_dense(values)
    xp = array_namespace(values)
    static = _like_obs(values, covariates.to_numpy(np.float64))
    if values.ndim == 3:
        static = xp.broadcast_to(static[:, :, None], (*static.shape, values.shape[2]))
    return xp.concat([xp.astype(values, xp.float64), static], axis=1)


def _times(edata: EHRData, task: Task, layer: str | None, time_key: str) -> np.ndarray:
    """Time of every timepoint of the observation window from `edata.tem[time_key]`, or its position if `tem` has no such column."""
    X = edata.X if layer is None else edata.layers[layer]
    return _tem_times(edata, time_key)[task._window(X.shape[2])]


def _covariates(edata: EHRData, obs_keys: Sequence[str], columns: Sequence[str] | None) -> pd.DataFrame:
    """`obs` columns with categorical columns one-hot encoded, aligned with `columns` if given."""
    covariates = pd.get_dummies(edata.obs[list(obs_keys)]) if obs_keys else edata.obs[[]]
    return covariates if columns is None else covariates.reindex(columns=columns, fill_value=0)
