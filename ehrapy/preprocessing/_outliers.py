from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats.mstats

if TYPE_CHECKING:
    from anndata import AnnData


def winsorize(
    adata: AnnData,
    vars: str | list[str] | set[str] = None,
    obs_cols: str | list[str] | set[str] = None,
    limits: list[float] = None,
    copy: bool = False,
    **kwargs,
) -> AnnData:
    """Returns a Winsorized version of the input array.

    The implementation is based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html

    Args:
        adata: AnnData object to winsorize
        vars: The features to winsorize.
        obs_cols: Columns in obs with features to winsorize.
        limits: Tuple of the percentages to cut on each side of the array as floats between 0. and 1.
                Defaults to (0.01, 0.99)
        copy: Whether to return a copy or not
        **kwargs: Keywords arguments get passed to scipy.stats.mstats.winsorize

    Returns:
        Winsorized AnnData object if copy is True.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.winsorize(adata, ["bmi"])
    """
    _validate_outlier_input(adata, obs_cols, vars)

    if copy:  # pragma: no cover
        adata = adata.copy()

    if limits is None:
        limits = [0.01, 0.99]

    if vars:
        for var in vars:
            adata[:, var].X = scipy.stats.mstats.winsorize(
                np.array(adata[:, var].X), limits=limits, nan_policy="omit", **kwargs
            )

    if obs_cols:
        for col in obs_cols:
            winsorized_array = scipy.stats.mstats.winsorize(adata.obs[col], limits=limits, nan_policy="omit", **kwargs)
            adata.obs[col] = pd.Series(winsorized_array).values

    if copy:
        return adata


def clip_quantile(
    adata: AnnData,
    limits: list[float],
    vars: str | list[str] | set[str] = None,
    obs_cols: str | list[str] | set[str] = None,
    copy: bool = False,
) -> AnnData:
    """Clips (limits) features.

    Given an interval, values outside the interval are clipped to the interval edges.

    The implementation is based on https://numpy.org/doc/stable/reference/generated/numpy.clip.html

    Args:
        adata: The AnnData object
        vars: Columns in var with features to clip
        obs_cols: Columns in obs with features to clip
        limits: Interval, values outside of which are clipped to the interval edges
        copy: Whether to return a copy of AnnData or not

    Returns:
        A copy of original AnnData object with clipped features.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.clip_quantile(adata, ["bmi"])
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


def _validate_outlier_input(
    adata, obs_cols: str | list[str] | set[str], vars: str | list[str] | set[str]
) -> tuple[set[str], set[str]]:
    """Validates the obs/var columns for outlier preprocessing.

    Args:
        adata: AnnData object
        obs_cols: str or list of obs columns
        vars: str or list of var names

    Returns:
        A tuple of lists of obs/var columns
    """
    if isinstance(vars, str) or isinstance(vars, list):  # pragma: no cover
        vars = set(vars)
    if isinstance(obs_cols, str) or isinstance(obs_cols, list):  # pragma: no cover
        obs_cols = set(obs_cols)

    if vars is not None:
        diff = vars - set(adata.var_names)
        if len(diff) != 0:
            raise ValueError(f"Columns {','.join(var for var in diff)} are not in var_names.")
    if obs_cols is not None:
        diff = obs_cols - set(adata.obs.columns.values)
        if len(diff) != 0:
            raise ValueError(f"Columns {','.join(var for var in diff)} are not in obs.")

    return obs_cols, vars
