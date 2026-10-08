from __future__ import annotations

import tokenize
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from ehrdata._logger import logger
from fast_array_utils.conv import to_dense
from pandas.core.computation.parsing import BACKTICK_QUOTED_STRING, clean_column_name, tokenize_string

from ehrapy._compat import _materialize, _resolve_axis, _var_indices
from ehrapy.core._constants import MISSING_VALUE_COUNT_KEY_2D, MISSING_VALUE_COUNT_KEY_3D
from ehrapy.preprocessing._quality_control import _compute_missing_values
from ehrapy.preprocessing._summarize_measurements import _aggregate_time

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ehrdata import EHRData
    from fast_array_utils.types import CSBase, DaskArray

    from ehrapy.preprocessing._summarize_measurements import Statistic
    from ehrapy.tools import CohortTracker

    type Array = np.ndarray | CSBase | DaskArray


def filter_features(
    edata: EHRData,
    *,
    layer: str | None = None,
    min_obs: int | None = None,
    max_obs: int | None = None,
    time_mode: Literal["all", "any", "proportion"] = "all",
    prop: float | None = None,
    copy: bool = False,
) -> EHRData | None:  # pragma: no cover
    """Filter features based on missing data thresholds.

    Keep only features which have at least `min_obs` observations and/or have at most `max_obs` observations.
    An observation is considered non-missing if it contains a valid (non-NaN / non-null) value.

    When a longitudinal `EHRData` is passed, filtering can be done across time points according to the specific `time_mode`.
    For 3D data, the non-missing values of a feature are counted over observations at every timepoint, and `time_mode` combines the timepoints.

    Only provide one of `min_obs` and/or `max_obs`.

    Args:
        edata: Central data object.
        layer: layer to use for filtering.
            If `None` (default), filtering is done on `.X`.
        min_obs: Minimum number of observations required for a feature to pass filtering.
        max_obs: Maximum number of observations allowed for a feature to pass filtering.
        time_mode: How to combine filtering criteria across the time axis. Use it only with 3 dimensional EHRData obejcts. Options are:

            * `'all'` (default): The feature must pass the filtering criteria in all time points.
            * `'any'`: The feature must pass the filtering criteria in at least one time point.
            * `'proportion'`: The feature must pass the filtering criteria in at least a proportion `prop` of time points.
                For example, with `prop=0.3`, the feature must pass the filtering criteria in at least 30% of the time points.

        prop: Proportion of time points in which the feature must pass the filtering criteria. Only relevant if `time_mode='proportion'`.
        copy: Determines whether a copy is returned.

    Returns:
        Depending on `copy`, subsets and annotates the passed data object and returns a filtered copy of the data object or acts in place

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6)
        >>> edata.X.shape
        (500, 45, 15)
        >>> ep.pp.filter_features(edata, min_obs=185, time_mode="all")
        >>> edata.X.shape
        (500, 18, 15)
    """
    data = edata.copy() if copy else edata

    if min_obs is None and max_obs is None:
        raise ValueError("You must provide at least one of 'min_obs' and 'max_obs'")

    if time_mode not in {"all", "any", "proportion"}:
        raise ValueError(f"time_mode must be one of 'all', 'any', 'proportion', got {time_mode}")

    if time_mode == "proportion" and (prop is None or not (0 < prop <= 1)):
        raise ValueError("prop must be set to a value between 0 and 1 when time_mode is 'proportion'")

    arr = edata.X if layer is None else edata.layers[layer]
    is_3d = arr.ndim == 3 and arr.shape[2] > 1

    (missing_counts,) = _materialize(_compute_missing_values(arr, axis=0))
    features_passing_filtering_mask, nonmissing_counts_per_feature = _compute_mask(
        missing_counts, arr.shape[0], min_count=min_obs, max_count=max_obs, time_mode=time_mode, prop=prop
    )

    n_features_filtered = int((~features_passing_filtering_mask).sum())
    if n_features_filtered > 0:
        msg = f"filtered out {n_features_filtered} features that are measured "
        if min_obs is not None:
            msg += f"less than {min_obs} counts"
        if max_obs is not None:
            msg += f"more than {max_obs} counts"

        if is_3d:
            if time_mode == "proportion":
                msg += f" in less than {prop * 100:.1f}% of time points"
            else:
                msg += f" in {time_mode} time points"
        logger.info(msg)

    label = MISSING_VALUE_COUNT_KEY_2D if not is_3d else MISSING_VALUE_COUNT_KEY_3D
    data.var[label] = nonmissing_counts_per_feature.astype(np.float64)
    data._inplace_subset_var(features_passing_filtering_mask)

    return data if copy else None


def filter_observations(
    edata: EHRData,
    *,
    layer: str | None = None,
    min_vars: int | None = None,
    max_vars: int | None = None,
    time_mode: Literal["all", "any", "proportion"] = "all",
    prop: float | None = None,
    query: str | Sequence[str] | None = None,
    tem_names: Any | Sequence[Any] | slice | None = None,
    agg: Statistic | None = None,
    tracker: CohortTracker | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Filter observations based on missing data thresholds (features/measurements) and inclusion criteria.

    Keep only observations which have at least `min_vars` variables and/or at most `max_vars` variables.
    An observation is considered non-missing if it contains a valid (non-NaN / non-null) value.
    When a longitudinal `EHRData` is passed, filtering can be done across time points.
    For 3D data, the non-missing values of an observation are counted over variables at every timepoint, and `time_mode` combines the timepoints.

    A cohort can be defined by inclusion criteria in `query`, pandas query expressions such as `"age >= 18 and glucose > 7"` on the columns of `obs` and on variables.
    The criteria are applied one after the other, and observations for which a criterion is missing are excluded.
    For 3D data, a variable in a criterion takes its value at the timepoint selected by `tem_names`, or `agg` reduces the selected timepoints.
    With a `tracker`, every missing data threshold and criterion is recorded as a step labeled by the criterion, after the unfiltered cohort if the tracker has no steps yet.

    Args:
        edata: Central data object.
        layer: layer to use for filtering.
            If `None` (default), filtering is done on `.X`.
        min_vars: Minimum number of variables required for an observation to pass filtering.
        max_vars: Maximum number of variables allowed for an observation to pass filtering.
        time_mode: How to combine filtering criteria across the time axis. Only relevant if an `EHRData` is passed. Options are:

            * `'all'` (default): The observation must pass the filtering criteria in all time points.
            * `'any'`: The observation must pass the filtering criteria in at least one time point.
            * `'proportion'`: The observation must pass the filtering criteria in at least a proportion `prop` of time points.
                For example, with `prop=0.3`, the observation must pass the filtering criteria in at least 30% of the time points.

        prop: Proportion of time points in which the observation must pass the filtering criteria. Only relevant if `time_mode='proportion'`.
        query: One or several inclusion criteria on `obs` columns and variables, which are applied in order.
            Names that are not valid Python identifiers are enclosed in backticks.
        tem_names: Labels of `edata.tem.index` or a positional slice that select the timepoints of 3D data that the missing data thresholds and criteria consider.
            If `None` (default), all timepoints are used.
        agg: How variables in `query` are reduced over the selected timepoints of 3D data, one of the statistics of :func:`~ehrapy.preprocessing.summarize_measurements`.
            Required if more than one timepoint is selected.
        tracker: Cohort tracker that records every filtering step.
        copy: Determines whether a copy is returned.

    Returns:
        Depending on `copy`, subsets and annotates the passed data object and returns a filtered copy of the data object or acts in place

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6)
        >>> edata.X.shape
        (500, 45, 15)
        >>> ep.pp.filter_observations(edata, min_vars=10, time_mode="all")
        >>> edata.X.shape
        (477, 45, 15)

        Define a cohort by inclusion criteria and track every step:

        >>> edata = ed.dt.ehrdata_blobs(n_variables=5, n_observations=500, base_timepoints=15)
        >>> tracker = ep.tl.CohortTracker(edata)
        >>> ep.pp.filter_observations(edata, query=["cluster != 0", "feature_0 > 0"], agg="max", tracker=tracker)
        >>> tracker.tracked_steps
        3
        >>> fig = tracker.plot_flowchart()
    """
    data = edata.copy() if copy else edata

    has_threshold = min_vars is not None or max_vars is not None
    if not has_threshold and query is None:
        raise ValueError("You must provide at least one of 'min_vars', 'max_vars' and 'query'")
    if time_mode not in {"all", "any", "proportion"}:
        raise ValueError(f"time_mode must be one of 'all', 'any', 'proportion', got {time_mode}")
    if time_mode == "proportion" and (prop is None or not (0 < prop <= 1)):
        raise ValueError("prop must be set to a value between 0 and 1 when time_mode is 'proportion'")

    arr = edata.X if layer is None else edata.layers[layer]
    is_3d = arr.ndim == 3 and arr.shape[2] > 1
    if arr.ndim == 3:
        tem_pos, _ = _resolve_axis(pd.Index(edata.tem.index), tem_names, "tem_names")
        arr = arr[:, :, tem_pos]
    elif tem_names is not None:
        raise ValueError("tem_names needs 3D data with a time axis.")

    criteria = [query] if isinstance(query, str) else list(query or ())
    var_names = _referenced_var_names(edata, criteria)
    lazy = [_compute_missing_values(arr, axis=1)] if has_threshold else []
    if var_names:
        lazy.append(_variable_values(arr, _var_indices(edata, var_names), agg))
    computed = _materialize(*lazy)

    if tracker is not None and tracker.tracked_steps == 0:
        tracker(data)
    keep = np.ones(data.n_obs, dtype=bool)

    if has_threshold:
        keep, nonmissing_counts_per_observation = _compute_mask(
            computed.pop(0), arr.shape[1], min_count=min_vars, max_count=max_vars, time_mode=time_mode, prop=prop
        )

        n_observations_filtered = int((~keep).sum())
        if n_observations_filtered > 0:
            msg = f"filtered out {n_observations_filtered} observations that have"
            if min_vars is not None:
                msg += f"less than {min_vars} " + "features"
            else:
                msg += f"more than {max_vars} " + "features"

            if is_3d:
                if time_mode == "proportion":
                    msg += f" in < {prop * 100:.1f}% of time points"
                else:
                    msg += f" in {time_mode} time points"

            logger.info(msg)

        label = MISSING_VALUE_COUNT_KEY_2D if not is_3d else MISSING_VALUE_COUNT_KEY_3D
        data.obs[label] = nonmissing_counts_per_observation.astype(np.float64)
        if tracker is not None:
            thresholds = {"min_vars": min_vars, "max_vars": max_vars}
            tracker(data[keep], label=", ".join(f"{k}={v}" for k, v in thresholds.items() if v is not None))

    variables = pd.DataFrame(
        computed[0] if var_names else None,
        index=data.obs_names,
        columns=[clean_column_name(var_name) for var_name in var_names],
    )
    for criterion in criteria:
        result = data.obs[keep].eval(criterion, resolvers=[dict(variables[keep].items())], engine="python")
        if not pd.api.types.is_bool_dtype(result):
            raise TypeError(f"The criterion {criterion!r} must evaluate to a boolean per observation.")
        passed = np.zeros_like(keep)
        passed[keep] = result.fillna(False).to_numpy(dtype=bool)
        logger.info(f"filtered out {int(keep.sum() - passed.sum())} observations that do not satisfy {criterion!r}")
        keep = passed
        if tracker is not None:
            tracker(data[keep], label=criterion)

    data._inplace_subset_obs(keep)

    return data if copy else None


def _referenced_var_names(edata: EHRData, criteria: Iterable[str]) -> list[str]:
    """Variables that the query expressions `criteria` refer to."""
    names = {
        token
        for criterion in criteria
        for kind, token in tokenize_string(criterion)
        if kind in {tokenize.NAME, BACKTICK_QUOTED_STRING}
    }
    var_names = [var_name for var_name in edata.var_names if var_name in names]
    if ambiguous := [var_name for var_name in var_names if var_name in edata.obs.columns]:
        raise ValueError(f"{ambiguous} are both obs columns and variables, rename one of them to use it in a query.")
    return var_names


def _variable_values(arr: Array, indices: np.ndarray, agg: Statistic | None) -> Array:
    """Values of the variables at `indices` per observation, reduced over time by `agg` for 3D data."""
    values = arr[:, indices]
    if values.ndim == 3:
        if agg is None and values.shape[2] > 1:
            raise ValueError("Variables of 3D data in a query need `agg` or `tem_names` that select one timepoint.")
        values = values[:, :, 0] if agg is None else _aggregate_time(values, agg)
    return to_dense(values)


def _compute_mask(
    missing_counts: np.ndarray,
    n: int,
    *,
    min_count: int | None,
    max_count: int | None,
    time_mode: str,
    prop: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mask for filtering based on missing data thresholds, given the number of missing values out of `n`.

    Returns:
        mask: boolean array indicating which features/observations pass the filtering criteria
        totals: total counts per feature/observation
    """
    present_counts = (n - missing_counts).reshape(missing_counts.shape[0], -1)
    if min_count is not None and max_count is not None:
        pass_threshold_mask = (present_counts >= float(min_count)) & (present_counts <= float(max_count))
    elif min_count is not None:
        pass_threshold_mask = present_counts >= float(min_count)
    else:
        pass_threshold_mask = present_counts <= float(max_count)

    if time_mode == "all":
        mask = pass_threshold_mask.all(axis=1)
    elif time_mode == "any":
        mask = pass_threshold_mask.any(axis=1)
    else:
        if prop is None:
            raise ValueError("prop must be set when time_mode is 'proportion'")
        mask = (pass_threshold_mask.sum(axis=1) / pass_threshold_mask.shape[1]) >= prop

    totals = present_counts.sum(axis=1).astype(np.float64)
    return mask, totals
