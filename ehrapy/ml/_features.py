from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from array_api_compat import array_namespace
from fast_array_utils.conv import to_dense

from ehrapy._compat import _like_obs
from ehrapy.preprocessing._summarize_measurements import summarize_measurements

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData
    from fast_array_utils.types import DaskArray

    from ehrapy.ml._task import Task

    type Array = np.ndarray | DaskArray
    type Statistic = Literal["min", "max", "mean", "median", "first", "last"]


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
    """Dense features of every observation and their names, aligning one-hot encoded `obs` columns with `feature_names`."""
    X = edata.X if layer is None else edata.layers[layer]
    if X.ndim == 3:
        summary = summarize_measurements(
            edata[:, :, task._window(X.shape[2])], layer=layer, var_names=var_names, statistics=statistics
        )
        values, names = summary.X, list(summary.var_names)
    elif task.prediction_time is not None or task.observation_window is not None or task.gap:
        raise ValueError("The timepoints of a task only apply to longitudinal data.")
    else:
        view = edata[:, list(var_names)]
        values, names = (view.X if layer is None else view.layers[layer]), list(var_names)

    covariates = _covariates(edata, obs_keys, None if feature_names is None else feature_names[len(names) :])
    values = to_dense(values)
    xp = array_namespace(values)
    features = xp.concat([xp.astype(values, xp.float64), _like_obs(values, covariates.to_numpy(np.float64))], axis=1)
    return features, [*names, *covariates.columns]


def _covariates(edata: EHRData, obs_keys: Sequence[str], columns: Sequence[str] | None) -> pd.DataFrame:
    """`obs` columns with categorical columns one-hot encoded, aligned with `columns` if given."""
    covariates = pd.get_dummies(edata.obs[list(obs_keys)]) if obs_keys else edata.obs[[]]
    return covariates if columns is None else covariates.reindex(columns=columns, fill_value=0)
