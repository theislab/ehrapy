from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ehrdata._logger import logger
from ehrdata.core.constants import MISSING_VALUES
from scipy import sparse

from ehrapy._compat import DaskArray, _raise_array_type_not_implemented, use_ehrdata
from ehrapy.core._constants import MISSING_VALUE_COUNT_KEY_2D, MISSING_VALUE_COUNT_KEY_3D

if TYPE_CHECKING:
    from anndata import AnnData
    from ehrdata import EHRData


@singledispatch
def _filtering_function(arr, *, function: Callable[..., Any]) -> None:
    _raise_array_type_not_implemented(function, type(arr))


@_filtering_function.register
def _(arr: np.ndarray, *, function: Callable[..., Any]) -> None:
    return None


@_filtering_function.register
def _(arr: DaskArray, *, function: Callable[..., Any]) -> None:
    _raise_array_type_not_implemented(function, type(arr))


@_filtering_function.register
def _(arr: sparse.coo_array, *, function: Callable[..., Any]) -> None:
    _raise_array_type_not_implemented(function, type(arr))


@use_ehrdata(deprecated_after="1.0.0")
def filter_features(
    edata: EHRData | AnnData,
    *,
    layer: str | None = None,
    min_obs: int | None = None,
    max_obs: int | None = None,
    time_mode: Literal["all", "any", "proportion"] = "all",
    prop: float | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:  # pragma: no cover
    """Filter features based on missing data thresholds.

    Keep only features which have at least `min_obs` observations and/or have at most `max_obs` observations.
    An observation is considered non-missing if it contains a valid (non-NaN / non-null) value.

    When a longitudinal `EHRData` is passed, filtering can be done across time points according to the specific `time_mode`.

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
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(
        ...     n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6, layer="tem_data"
        ... )
        >>> edata.layers["tem_data"].shape
        (500, 45, 15)
        >>> ep.pp.filter_features(edata, min_obs=185, time_mode="all", layer="tem_data")
        >>> edata.layers["tem_data"].shape
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

    features_passing_filtering_mask, nonmissing_counts_per_feature = _compute_mask(
        arr, axis=0, min_count=min_obs, max_count=max_obs, time_mode=time_mode, prop=prop, caller=filter_features
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


@use_ehrdata(deprecated_after="1.0.0")
def filter_observations(
    edata: EHRData | AnnData,
    *,
    layer: str | None = None,
    min_vars: int | None = None,
    max_vars: int | None = None,
    time_mode: Literal["all", "any", "proportion"] = "all",
    prop: float | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:
    """Filter observations based on missing data thresholds (features/measurements).

    Keep only observations which have at least `min_vars` variables and/or at most `max_vars` variables.
    An observation is considered non-missing if it contains a valid (non-NaN / non-null) value.
    When a longitudinal `EHRData` is passed, filtering can be done across time points.

    Only provide one of `min_vars` and/or `max_vars`.

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
        copy: Determines whether a copy is returned.

    Returns:
        Depending on `copy`, subsets and annotates the passed data object and returns a filtered copy of the data object or acts in place

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(
        ...     n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6, layer="tem_data"
        ... )
        >>> edata.layers["tem_data"].shape
        (500, 45, 15)
        >>> ep.pp.filter_observations(edata, min_vars=10, time_mode="all", layer="tem_data")
        >>> edata.layers["tem_data"].shape
        (477, 45, 15)
    """
    data = edata.copy() if copy else edata

    if min_vars is None and max_vars is None:
        raise ValueError("You must provide at least one of 'min_vars' and 'max_vars'")
    if time_mode not in {"all", "any", "proportion"}:
        raise ValueError(f"time_mode must be one of 'all', 'any', 'proportion', got {time_mode}")
    if time_mode == "proportion" and (prop is None or not (0 < prop <= 1)):
        raise ValueError("prop must be set to a value between 0 and 1 when time_mode is 'proportion'")

    arr = edata.X if layer is None else edata.layers[layer]

    is_3d = arr.ndim == 3 and arr.shape[2] > 1

    observations_passing_filtering_mask, nonmissing_counts_per_observation = _compute_mask(
        arr, axis=1, min_count=min_vars, max_count=max_vars, time_mode=time_mode, prop=prop, caller=filter_observations
    )

    n_observations_filtered = int((~observations_passing_filtering_mask).sum())
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
    data._inplace_subset_obs(observations_passing_filtering_mask)

    return data if copy else None


def _compute_mask(
    arr: np.ndarray, *, min_count: int, max_count: int, time_mode: str, prop: float, axis: int, caller
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mask for filtering based on missing data thresholds.

    Returns:
        mask: boolean array indicating which features/observations pass the filtering criteria
        totals: total counts per feature/observation
    """
    _filtering_function(arr, function=caller)
    if arr.ndim == 2:
        arr3 = arr[:, :, None]
    elif arr.ndim == 3:
        arr3 = arr
    else:
        raise ValueError(f"expected 2D or 3D array, got {arr.shape}")

    present_mask = ~(np.isin(arr3, MISSING_VALUES) | np.isnan(arr3))
    present_counts = present_mask.sum(axis=axis)

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
