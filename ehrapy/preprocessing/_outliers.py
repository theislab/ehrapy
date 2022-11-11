from __future__ import annotations

from anndata import AnnData


def winsorize(
    adata: AnnData, vars: list[str], obs_cols: list[str], limits: list[float], copy: bool = False, **kwargs
) -> AnnData:
    """Returns a Winsorized version of the input array.

    The implementation is based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html

    Args:
        adata: AnnData object to winsorize
        vars: The features to winsorize
        limits: Tuple of the percentages to cut on each side of the array as floats between 0. and 1.
        copy: Whether to return a copy or not
        **kwargs: Keywords arguments get passed to scipy.stats.mstats.winsorize

    Returns:
        Winsorized AnnData object if copy is True
    """
    if copy:  # pragma: no cover
        adata = adata.copy()

    if copy:
        return adata


def clip_quantile(
    adata: AnnData,
    vars: list[str],
    obs_cols: list[str],
):
    # https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    pass
