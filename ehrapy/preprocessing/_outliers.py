from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats.mstats
from anndata import AnnData


def winsorize(
    adata: AnnData,
    vars: list[str] = None,
    obs_cols: list[str] = None,
    limits: list[float] = [0.01, 0.99],
    copy: bool = False,
    **kwargs,
) -> AnnData:
    """Returns a Winsorized version of the input array.

    The implementation is based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html

    Args:
        adata: AnnData object to winsorize
        vars: The features to winsorize
        limits: Tuple of the percentages to cut on each side of the array as floats between 0. and 1. (default: 0.01 and 0.99).
        copy: Whether to return a copy or not
        **kwargs: Keywords arguments get passed to scipy.stats.mstats.winsorize

    Returns:
        Winsorized AnnData object if copy is True
    """
    if copy:  # pragma: no cover
        adata = adata.copy()

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
    vars: list[str],
    obs_cols: list[str],
):
    # https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    pass
