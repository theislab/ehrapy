from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from anndata import AnnData
from ehrdata import EHRData
from ehrdata.core.constants import MISSING_VALUES
from lamin_utils import logger
from scipy.sparse import sparray

from ehrapy._compat import DaskArray, _raise_array_type_not_implemented, use_ehrdata


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
def _(arr: sparray, *, function: Callable[..., Any]) -> None:
    _raise_array_type_not_implemented(function, type(arr))


@use_ehrdata(deprecated_after="1.0.0")
def filter_features(
    edata: EHRData | AnnData,
    *,
    layers: str | Sequence[str] | None = None,
    min_obs: int | None = None,
    max_obs: int | None = None,
    time_mode: Literal["all", "any", "proportion"] = "all",
    prop: float | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:  # pragma: no cover
    """Filter features based on missing data thresholds.

    Keep only features which have at least `min_obs` observations
    and/or have at most `max_obs` observations. An observation is considered non-missing if it contains a valid (non-NaN / non-null) value.

    When a longitudinal `EHRData` is passed, filtering can be done across time points according to the specific `time_mode`.

    Only provide `min_obs` and/or `max_obs` per call.

    Args:
        edata: Central data object.
        layers: layer(s) to use for filtering. If `None` (default), filtering is performed on `.R` for 3D EHRData objects and on `.X` for 2D EHRData objects.
                When multiple layers are provided, a feature passes the filtering only if it satisifies the criteria in every layer.
        min_obs: Minimum number of observations required for a feature to pass filtering.
        max_obs: Maximum number of observations allowed for a feature to pass filtering.
        time_mode: How to combine filtering criteria across the time axis. Use it only with 3 dimensional EHRData obejcts. Options are:

                    * `'all'` (default): The feature must pass the filtering criteria in all time points.
                    * `'any'`: The feature must pass the filtering criteria in at least one time point.
                    * `'proportion'`: The feature must pass the filtering criteria in at least a proportion `prop` of time points. For example, with `prop=0.3`,
                        the feature must pass the filtering criteria in at least 30% of the time points.

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
    if not isinstance(edata, EHRData) and not isinstance(edata, AnnData):
        raise TypeError("Data object must be an EHRData or AnnData object")

    if isinstance(edata, AnnData) and not isinstance(edata, EHRData) and edata.R is not None:
        raise ValueError("When passing an AnnData object, it must be 2-dimensional")

    data = edata.copy() if copy else edata

    if min_obs is None and max_obs is None:
        raise ValueError("You must provide at least one of 'min_obs' and 'max_obs'")

    if time_mode not in {"all", "any", "proportion"}:
        raise ValueError(f"time_mode must be one of 'all', 'any', 'proportion', got {time_mode}")

    if time_mode == "proportion" and (prop is None or not (0 < prop <= 1)):
        raise ValueError("prop must be set to a value between 0 and 1 when time_mode is 'proportion'")

    if layers is None:
        arr = data.R if data.R is not None else data.X
        if arr is None:
            raise ValueError("Both X and R are None, no data to filter")
        arrs = [arr]
    elif isinstance(layers, str):
        if layers not in edata.layers:
            raise ValueError(f"Invalid layer provided. Available layers are: {list(edata.layers.keys())}")
        arrs = [data.layers[layers]]
    else:  # when filtering is done across multiple layers
        arrs = [data.layers[layer] for layer in layers]

    layer_masks = []
    first_counts = None
    is_2d_ref = False
    for arr in arrs:
        _filtering_function(arr, function=filter_features)
        if arr.ndim == 2:
            arr = arr[:, :, None]
            if first_counts is None:
                is_2d_ref = True
        elif arr.ndim != 3:
            raise ValueError(f"expected a 2D or 3D array, got {arr.shape}")

        missing_mask = np.isin(arr, MISSING_VALUES) | np.isnan(arr)

        present = ~missing_mask
        counts = present.sum(axis=0)

        if first_counts is None:
            first_counts = counts

        if max_obs is not None and min_obs is not None:
            pass_threshold = (min_obs <= counts) & (counts <= max_obs)
        elif min_obs is not None:
            pass_threshold = counts >= min_obs
        else:
            pass_threshold = counts <= max_obs

        if time_mode == "all":
            feature_mask = pass_threshold.all(axis=1)
        elif time_mode == "any":
            feature_mask = pass_threshold.any(axis=1)
        elif time_mode == "proportion":
            if prop is None:
                raise ValueError("prop must be set when time_mode is 'proportion'")
            feature_mask = (pass_threshold.sum(axis=1) / pass_threshold.shape[1]) >= prop
        else:
            raise ValueError(f"Unknown time_mode: {time_mode}")

        layer_masks.append(feature_mask)

    final_feature_mask = np.logical_and.reduce(layer_masks)

    number_per_feature = first_counts.sum(axis=1).astype(np.float64)

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
    layers: str | Sequence[str] | None = None,
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

    Only provide `min_vars` and/or `max_vars` per call.

    Args:
        edata: Central data object.
        layers: layer(s) to use for filtering. If `None` (default), filtering is performed on `.R` for 3D EHRData objects and on `.X` for 2D EHRData objects.
                When multiple layers are provided, a feature passes the filtering only if it satisifies the criteria in every layer.
        min_vars: Minimum number of variables required for an observation to pass filtering.
        max_vars: Maximum number of variables allowed for an observation to pass filtering.
        time_mode: How to combine filtering criteria across the time axis. Only relevant if an `EHRData` is passed. Options are:

                    * `'all'` (default): The observation must pass the filtering criteria in all time points.
                    * `'any'`: The observation must pass the filtering criteria in at least one time point.
                    * `'proportion'`: The observation must pass the filtering criteria in at least a proportion `prop` of time points. For example, with `prop=0.3`,
                        the observation must pass the filtering criteria in at least 30% of the time points.

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
    if not isinstance(edata, EHRData) and not isinstance(edata, AnnData):
        raise TypeError("Data object must be an EHRData or an AnnData object")

    if isinstance(edata, AnnData) and not isinstance(edata, EHRData) and edata.R is not None:
        raise ValueError("When passing an AnnData object, it must be 2-dimensional")

    data = edata.copy() if copy else edata

    if min_vars is None and max_vars is None:
        raise ValueError("You must provide at least one of 'min_vars' and 'max_vars'")
    if time_mode not in {"all", "any", "proportion"}:
        raise ValueError(f"time_mode must be one of 'all', 'any', 'proportion', got {time_mode}")
    if time_mode == "proportion" and (prop is None or not (0 < prop <= 1)):
        raise ValueError("prop must be set to a value between 0 and 1 when time_mode is 'proportion'")

    if layers is None:
        arr = data.R if data.R is not None else data.X
        if arr is None:
            raise ValueError("Both R and X are None, no data to filter")
        arrs = [arr]
    elif isinstance(layers, str):
        if layers not in edata.layers:
            raise ValueError(f"Invalid layer provided. Available layers are: {list(edata.layers.keys())}")
        arrs = [data.layers[layers]]
    else:
        arrs = [data.layers[layer] for layer in layers]

    layers_obs_masks: list[np.ndarray] = []
    first_number_per_obs: np.ndarray | None = None
    is_2d_ref = False

    for arr in arrs:
        _filtering_function(arr, function=filter_observations)
        if arr.ndim == 2:
            arr = arr[:, :, None]
            is_2d = True
        elif arr.ndim == 3:
            is_2d = False
        else:
            raise ValueError(f"expected 2D or 3D array, got {arr.shape}")

        missing_mask = np.isin(arr, MISSING_VALUES) | np.isnan(arr)
        present = ~missing_mask

        per_time_vals = present.sum(axis=1).astype(float)

        if first_number_per_obs is None:
            first_number_per_obs = per_time_vals.sum(axis=1).astype(np.float64)
            is_2d_ref = is_2d

        if min_vars is not None and max_vars is not None:
            masks_t = (per_time_vals >= float(min_vars)) & (per_time_vals <= float(max_vars))
        elif min_vars is not None:
            masks_t = per_time_vals >= float(min_vars)
        elif max_vars is not None:
            masks_t = per_time_vals <= float(max_vars)

        if time_mode == "all":
            obs_mask = masks_t.all(axis=1)
        elif time_mode == "any":
            obs_mask = masks_t.any(axis=1)
        else:
            obs_mask = masks_t.mean(axis=1) >= float(prop)

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
