from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats.mstats

if TYPE_CHECKING:
    from collections.abc import Collection

    from anndata import AnnData


def winsorize(
    adata: AnnData,
    vars: Collection[str] = None,
    obs_cols: Collection[str] = None,
    *,
    limits: tuple[float, float] = (0.01, 0.99),
    copy: bool = False,
    **kwargs,
) -> AnnData:
    """Returns a Winsorized version of the input array.

    The implementation is based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html

    Args:
        adata: AnnData object to winsorize.
        vars: The features to winsorize.
        obs_cols: Columns in obs with features to winsorize.
        limits: Tuple of the percentages to cut on each side of the array as floats between 0. and 1.
        copy: Whether to return a copy.
        **kwargs: Keywords arguments get passed to scipy.stats.mstats.winsorize

    Returns:
        Winsorized AnnData object if copy is True.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.winsorize(adata, vars=["bmi"])
    """
    if copy:  # pragma: no cover
        adata = adata.copy()

    obs_cols_set, vars_set = _validate_outlier_input(adata, obs_cols, vars)

    if vars_set:
        for var in vars_set:
            data_array = np.array(adata[:, var].X, dtype=float)
            winsorized_data = scipy.stats.mstats.winsorize(data_array, limits=limits, nan_policy="omit", **kwargs)
            adata[:, var].X = winsorized_data

    if obs_cols_set:
        for col in obs_cols_set:
            obs_array = adata.obs[col].to_numpy(dtype=float)
            winsorized_obs = scipy.stats.mstats.winsorize(obs_array, limits=limits, nan_policy="omit", **kwargs)
            adata.obs[col] = pd.Series(winsorized_obs).values

    return adata if copy else None


def clip_quantile(
    adata: AnnData,
    limits: tuple[float, float],
    vars: Collection[str] = None,
    obs_cols: Collection[str] = None,
    *,
    copy: bool = False,
) -> AnnData:
    """Clips (limits) features.

    Given an interval, values outside the interval are clipped to the interval edges.

    Args:
        adata: The AnnData object to clip.
        limits: Values outside the interval are clipped to the interval edges.
        vars: Columns in var with features to clip.
        obs_cols: Columns in obs with features to clip
        copy: Whether to return a copy of AnnData or not

    Returns:
        A copy of original AnnData object with clipped features.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.clip_quantile(adata, vars=["bmi"])
    """
    obs_cols, vars = _validate_outlier_input(adata, obs_cols, vars)  # type: ignore

    if vars:
        for var in vars:
            adata[:, var].X = np.clip(adata[:, var].X, limits[0], limits[1])

    if obs_cols:
        for col in obs_cols:
            obs_array = adata.obs[col].to_numpy()
            clipped_array = np.clip(obs_array, limits[0], limits[1])
            adata.obs[col] = pd.Series(clipped_array).values

    if copy:  # pragma: no cover
        adata = adata.copy()

    if copy:
        return adata


def _validate_outlier_input(adata, obs_cols: Collection[str], vars: Collection[str]) -> tuple[set[str], set[str]]:
    """Validates the obs/var columns for outlier preprocessing."""
    vars = set(vars) if vars else set()
    obs_cols = set(obs_cols) if obs_cols else set()

    if vars is not None:
        diff = vars - set(adata.var_names)
        if len(diff) != 0:
            raise ValueError(f"Columns {','.join(var for var in diff)} are not in var_names.")
    if obs_cols is not None:
        diff = obs_cols - set(adata.obs.columns.values)
        if len(diff) != 0:
            raise ValueError(f"Columns {','.join(var for var in diff)} are not in obs.")

    return obs_cols, vars
