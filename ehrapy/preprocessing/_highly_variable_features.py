from __future__ import annotations

from typing import TYPE_CHECKING

import scanpy as sc

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def highly_variable_features(
    adata: AnnData,
    layer: str | None = None,
    top_features_percentage: float = 0.2,
    span: float | None = 0.3,
    n_bins: int = 20,
    subset: bool = False,
    inplace: bool = True,
    check_values: bool = True,
) -> pd.DataFrame | None:
    """Annotate highly variable features.

    Expects count data. A normalized variance for each feature is computed. First, the data
    are standardized (i.e., z-score normalization per feature) with a regularized
    standard deviation. Next, the normalized variance is computed as the variance
    of each feature after the transformation. Features are ranked by the normalized variance.

    Args:
        adata: The annotated data matrix of shape `n_obs` × `n_vars`.
        layer: If provided, use `adata.layers[layer]` for expression values instead of `adata.X`. Defaults to None .
        top_features_percentage: Percentage of highly-variable features to keep. Defaults to 0.2 .
        span: The fraction of the data used when estimating the variance in the loess model fit. Defaults to 0.3 .
        n_bins: Number of bins for binning. Normalization is done with respect to each bin.
                If just a single observation falls into a bin, the normalized dispersion is artificially set to 1.
                You'll be informed about this if you set `settings.verbosity = 4`. Defaults to 20 .
        subset: Inplace subset to highly-variable features if `True` otherwise merely indicate highly variable features.
                Defaults to False .
        inplace: Whether to place calculated metrics in `.var` or return them. Defaults to True .
        check_values: Check if counts in selected layer are integers. A Warning is returned if set to True.
                      Defaults to True .

    Returns:
        Depending on `inplace` returns calculated metrics (:class:`~pandas.DataFrame`) or
        updates `.var` with the following fields

    highly_variable : bool
        boolean indicator of highly-variable features
    **means**
        means per feature
    **variances**
        variance per feature
    **variances_norm**
        normalized variance per feature, averaged in the case of multiple batches
    highly_variable_rank : float
        rank of the feature according to normalized variance, median rank in the case of multiple batches
    """
    n_top_features = int(top_features_percentage * len(adata.var))

    return sc.pp.highly_variable_genes(
        adata=adata,
        layer=layer,
        n_top_genes=n_top_features,
        span=span,
        n_bins=n_bins,
        flavor="seurat_v3",
        subset=subset,
        inplace=inplace,
        check_values=check_values,
    )
