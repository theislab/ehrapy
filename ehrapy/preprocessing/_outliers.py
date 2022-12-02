from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats.mstats
from anndata import AnnData


def winsorize(
    adata: AnnData,
    vars: str | list[str] = None,
    obs_cols: str | list[str] = None,
    limits: list[float] = None,
    copy: bool = False,
    **kwargs,
) -> AnnData:
    """Returns a Winsorized version of the input array.

    The implementation is based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html

    Args:
        adata: AnnData object to winsorize
        vars: The features to winsorize
        obs_cols: Columns in obs with features to winsorize
        limits: Tuple of the percentages to cut on each side of the array as floats between 0. and 1. (default: 0.01 and 0.99)
        copy: Whether to return a copy or not
        **kwargs: Keywords arguments get passed to scipy.stats.mstats.winsorize

    Returns:
        Winsorized AnnData object if copy is True
    """
    if vars is not None and not all(elem in adata.var_names.values for elem in vars):
        raise ValueError(
            f"Columns `{[col for col in vars if col not in adata.var_names.values]}` are not in var_names."
        )

    if obs_cols is not None and not all(elem in adata.obs.columns.values for elem in obs_cols):
        raise ValueError(
            f"Columns `{[col for col in obs_cols if col not in adata.obs.columns.values]}` are not in obs."
        )

    if isinstance(vars, str):  # pragma: no cover
        vars = [vars]

    if isinstance(obs_cols, str):  # pragma: no cover
        obs_cols = [obs_cols]

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
            obs_array = adata.obs[col].to_numpy()
            winsorized_array = scipy.stats.mstats.winsorize(obs_array, limits=limits, nan_policy="omit", **kwargs)
            adata.obs[col] = pd.Series(winsorized_array).values

    if copy:
        return adata


def clip_quantile(
    adata: AnnData,
    limits: list[float],
    vars: str | list[str] = None,
    obs_cols: str | list[str] = None,
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
    """
    if vars is not None and not all(elem in adata.var_names.values for elem in vars):
        raise ValueError(
            f"Columns `{[col for col in vars if col not in adata.var_names.values]}` are not in var_names."
        )

    if obs_cols is not None and not all(elem in adata.obs.columns.values for elem in obs_cols):
        raise ValueError(
            f"Columns `{[col for col in obs_cols if col not in adata.obs.columns.values]}` are not in obs."
        )

    if isinstance(vars, str):  # pragma: no cover
        vars = [vars]

    if isinstance(obs_cols, str):  # pragma: no cover
        obs_cols = [obs_cols]

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


def filter_quantiles(
    adata: AnnData,
    vars: str | list[str],
    obs_cols: list[str],
    quantile_top: int | None,
    quantile_bottom: int | None,
    copy: bool = False,
) -> AnnData:
    """Filter numeric features by top/bottom quantiles

    Args:
        adata: The AnnData object
        vars: Columns with features to filter
        obs_cols: Columns in obs with features to filter
        quantile_top: Remove the top % largest values.
        quantile_bottom: Remove the bottom % lowest values.
        copy: Whether to return a copy of AnnData or not

    Returns:
        A copy of original AnnData object with filtered features.
    """
    if vars is not None and not all(elem in adata.var_names.values for elem in vars):
        raise ValueError(
            f"Columns `{[col for col in vars if col not in adata.var_names.values]}` are not in var_names."
        )

    if obs_cols is not None and not all(elem in adata.obs.columns.values for elem in obs_cols):
        raise ValueError(
            f"Columns `{[col for col in obs_cols if col not in adata.obs.columns.values]}` are not in obs."
        )

    if isinstance(vars, str):  # pragma: no cover
        vars = [vars]

    if isinstance(obs_cols, str):  # pragma: no cover
        obs_cols = [obs_cols]

    if copy:  # pragma: no cover
        adata = adata.copy()

    if copy:
        return adata
