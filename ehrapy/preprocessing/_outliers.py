from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats.mstats

from ehrapy._compat import function_2D_only, use_ehrdata

if TYPE_CHECKING:
    from collections.abc import Collection

    from anndata import AnnData
    from ehrdata import EHRData


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
def winsorize(
    edata: EHRData | AnnData,
    vars: Collection[str] = None,
    obs_cols: Collection[str] = None,
    *,
    limits: tuple[float, float] = (0.01, 0.99),
    layer: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | AnnData | None:
    """Returns a Winsorized version of the input array.

    The implementation is based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html

    Args:
        edata: Central data object.
        vars: The features to winsorize.
        obs_cols: Columns in obs with features to winsorize.
        limits: Tuple of the percentages to cut on each side of the array as floats between 0. and 1.
        layer: The layer to operate on.
        copy: Whether to return a copy.
        **kwargs: Keywords arguments get passed to scipy.stats.mstats.winsorize.

    Returns:
        Winsorized data object if copy is True.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.winsorize(edata, vars=["bmi"])
    """
    if copy:  # pragma: no cover
        edata = edata.copy()

    obs_cols_set, vars_set = _validate_outlier_input(edata, obs_cols, vars)

    if vars_set:
        for var in vars_set:
            edata_view = edata[:, var]
            X = edata_view.X if layer is None else edata_view.layers[layer]
            data_array = np.array(X, dtype=float)
            winsorized_data = scipy.stats.mstats.winsorize(data_array, limits=limits, nan_policy="omit", **kwargs)
            if layer is None:
                edata[:, var].X = winsorized_data
            else:
                edata[:, var].layers[layer] = winsorized_data

    if obs_cols_set:
        for col in obs_cols_set:
            obs_array = edata.obs[col].to_numpy(dtype=float)
            winsorized_obs = scipy.stats.mstats.winsorize(obs_array, limits=limits, nan_policy="omit", **kwargs)
            edata.obs[col] = pd.Series(winsorized_obs).values

    return edata if copy else None


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
def clip_quantile(
    edata: EHRData | AnnData,
    limits: tuple[float, float],
    vars: Collection[str] = None,
    obs_cols: Collection[str] = None,
    *,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:
    """Clips (limits) features.

    Given an interval, values outside the interval are clipped to the interval edges.

    Args:
        edata: Central data object.
        limits: Values outside the interval are clipped to the interval edges.
        vars: Columns in var with features to clip.
        obs_cols: Columns in obs with features to clip
        layer: The layer to operate on.
        copy: Whether to return a copy of data or not

    Returns:
        A copy of original data object with clipped features.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.clip_quantile(edata, limits=(0, 75), vars=["bmi"])
    """
    obs_cols, vars = _validate_outlier_input(edata, obs_cols, vars)  # type: ignore

    if vars:
        for var in vars:
            edata_view = edata[:, var]
            X = edata_view.X if layer is None else edata_view.layers[layer]
            X = np.clip(X, limits[0], limits[1])
            if layer is None:
                edata[:, var].X = X
            else:
                edata[:, var].layers[layer] = X

    if obs_cols:
        for col in obs_cols:
            obs_array = edata.obs[col].to_numpy()
            clipped_array = np.clip(obs_array, limits[0], limits[1])
            edata.obs[col] = pd.Series(clipped_array).values

    if copy:  # pragma: no cover
        edata = edata.copy()

    return edata if copy else None


def _validate_outlier_input(edata, obs_cols: Collection[str], vars: Collection[str]) -> tuple[set[str], set[str]]:
    """Validates the obs/var columns for outlier preprocessing."""
    vars = set(vars) if vars else set()
    obs_cols = set(obs_cols) if obs_cols else set()

    if vars is not None:
        diff = vars - set(edata.var_names)
        if len(diff) != 0:
            raise ValueError(f"Columns {','.join(var for var in diff)} are not in var_names.")
    if obs_cols is not None:
        diff = obs_cols - set(edata.obs.columns.values)
        if len(diff) != 0:
            raise ValueError(f"Columns {','.join(var for var in diff)} are not in obs.")

    return obs_cols, vars
