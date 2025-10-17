from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ehrdata.core.constants import MISSING_VALUES
from lamin_utils import logger
from scipy import sparse

from ehrapy._compat import DaskArray, _raise_array_type_not_implemented, use_ehrdata

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
            If `None` (default), filtering is performed on `.R` for 3D EHRData objects and on `.X` for 2D EHRData objects.
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
        Depending on `copy`, subsets and annotates the passed data object and returns `None`

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6)
        >>> edata.R.shape
        (500, 45, 15)
        >>> ep.pp.filter_features(edata, min_obs=185, time_mode="all")
        >>> edata.R.shape
        (500, 18, 15)
    """
    data = edata.copy() if copy else edata

    if min_obs is None and max_obs is None:
        raise ValueError("You must provide at least one of 'min_obs' and 'max_obs'")

    if time_mode not in {"all", "any", "proportion"}:
        raise ValueError(f"time_mode must be one of 'all', 'any', 'proportion', got {time_mode}")

    if time_mode == "proportion" and (prop is None or not (0 < prop <= 1)):
        raise ValueError("prop must be set to a value between 0 and 1 when time_mode is 'proportion'")

    arrs = _arrays_for_filtering(data, layer)

    layer_masks = []
    first_counts = None
    is_2d_ref = False
    for arr in arrs:
        feature_mask, totals, is_2d = _compute_mask(
            arr, axis=0, min_count=min_obs, max_count=max_obs, time_mode=time_mode, prop=prop, caller=filter_features
        )
        if first_counts is None:
            first_counts = totals
            is_2d_ref = is_2d

        layer_masks.append(feature_mask)

    final_feature_mask = np.logical_and.reduce(layer_masks)

    number_per_feature = first_counts.astype(np.float64)

    n_filtered = int((~final_feature_mask).sum())

    if n_filtered > 0:
        msg = f"filtered out {n_filtered} features that are measured "
        if min_obs is not None:
            msg += f"less than {min_obs} counts"
        else:
            msg += f"more than {max_obs} counts"

        if not is_2d_ref:
            if time_mode == "proportion":
                msg += f" in less than {prop * 100:.1f}% of time points"
            else:
                msg += f" in {time_mode} time points"
        logger.info(msg)

    label = "n_obs" if is_2d_ref else "n_obs_over_time"
    data.var[label] = number_per_feature
    data._inplace_subset_var(final_feature_mask)

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
            If `None` (default), filtering is performed on `.R` for 3D EHRData objects and on `.X` for 2D EHRData objects.
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
        Depending on `copy`, subsets and annotates the passed data object and returns `None`

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6)
        >>> edata.R.shape
        (500, 45, 15)
        >>> ep.pp.filter_observations(edata, min_vars=10, time_mode="all")
        >>> edata.R.shape
        (477, 45, 15)
    """
    data = edata.copy() if copy else edata

    if min_vars is None and max_vars is None:
        raise ValueError("You must provide at least one of 'min_vars' and 'max_vars'")
    if time_mode not in {"all", "any", "proportion"}:
        raise ValueError(f"time_mode must be one of 'all', 'any', 'proportion', got {time_mode}")
    if time_mode == "proportion" and (prop is None or not (0 < prop <= 1)):
        raise ValueError("prop must be set to a value between 0 and 1 when time_mode is 'proportion'")

    arrs = _arrays_for_filtering(data, layer)

    layers_obs_masks: list[np.ndarray] = []
    first_number_per_obs: np.ndarray | None = None
    is_2d_ref = False
    for arr in arrs:
        obs_mask, totals, is_2d = _compute_mask(
            arr,
            axis=1,
            min_count=min_vars,
            max_count=max_vars,
            time_mode=time_mode,
            prop=prop,
            caller=filter_observations,
        )
        if first_number_per_obs is None:
            first_number_per_obs = totals
            is_2d_ref = is_2d
        layers_obs_masks.append(obs_mask)

    final_obs_mask = np.logical_and.reduce(layers_obs_masks)

    n_filtered = int((~final_obs_mask).sum())
    if n_filtered > 0:
        msg = f"filtered out {n_filtered} observations that have"
        if min_vars is not None:
            msg += f"less than {min_vars} " + "features"
        else:
            msg += f"more than {max_vars} " + "features"

        if not is_2d_ref and (arrs[0].ndim == 3 and arrs[0].shape[2] > 1):
            if time_mode == "proportion":
                msg += f" in < {prop * 100:.1f}% of time points"
            else:
                msg += f" in {time_mode} time points"

        logger.info(msg)

    label = "n_vars" if is_2d_ref else "n_vars_over_time"
    data.obs[label] = first_number_per_obs
    data._inplace_subset_obs(final_obs_mask)

    return data if copy else None


def _arrays_for_filtering(data: EHRData | AnnData, layer: str | None) -> list[np.ndarray]:
    """Get arrays to be used for filtering based on the provided layer."""
    if layer is None:
        arr = data.R if data.R is not None else data.X
        if arr is None:
            raise ValueError("Both X and R are None, no data to filter")
        arrs = [arr]
    elif isinstance(layer, str):
        if layer not in data.layers:
            raise ValueError(f"Invalid layer provided. Available layers are: {list(data.layers.keys())}")
        arrs = [data.layers[layer]]
    else:
        raise ValueError("layer must be a string or None")

    return arrs


def _compute_mask(arr: np.ndarray, *, min_count: int, max_count: int, time_mode: str, prop: float, axis: int, caller):
    """Compute mask for filtering based on missing data thresholds.

    Returns:
        mask: boolean array indicating which features/observations pass the filtering criteria
        totals: total counts per feature/observation
        is_2d: whether the input array was 2D.
    """
    _filtering_function(arr, function=caller)
    if arr.ndim == 2:
        arr3 = arr[:, :, None]
        is_2d = True
    elif arr.ndim == 3:
        arr3 = arr
        is_2d = False
    else:
        raise ValueError(f"expected 2D or 3D array, got {arr.shape}")

    missing = np.isin(arr3, MISSING_VALUES) | np.isnan(arr3)
    present = ~missing
    counts = present.sum(axis=axis)

    if min_count is not None and max_count is not None:
        pass_threshold = (counts >= float(min_count)) & (counts <= float(max_count))
    elif min_count is not None:
        pass_threshold = counts >= float(min_count)
    else:
        pass_threshold = counts <= float(max_count)

    if time_mode == "all":
        mask = pass_threshold.all(axis=1)
    elif time_mode == "any":
        mask = pass_threshold.any(axis=1)
    else:
        if prop is None:
            raise ValueError("prop must be set when time_mode is 'proportion'")
        mask = (pass_threshold.sum(axis=1) / pass_threshold.shape[1]) >= prop

    totals = counts.sum(axis=1).astype(np.float64)
    return mask, totals, is_2d
