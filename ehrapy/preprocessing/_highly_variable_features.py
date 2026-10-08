from __future__ import annotations

from typing import TYPE_CHECKING

import scanpy as sc

from ehrapy._compat import function_2D_only

if TYPE_CHECKING:
    from ehrdata import EHRData


@function_2D_only()
def highly_variable_features(
    edata: EHRData,
    *,
    layer: str | None = None,
    top_features_percentage: float = 0.2,
    span: float = 0.3,
    batch_key: str | None = None,
    subset: bool = False,
    check_values: bool = True,
    copy: bool = False,
) -> EHRData | None:
    """Annotate highly variable features :cite:p:`Stuart2019`.

    Expects count data.
    A normalized variance for each feature is computed.
    First, the data are standardized (i.e., z-score normalization per feature) with a regularized standard deviation.
    Next, the normalized variance is computed as the variance of each feature after the transformation.
    Features are ranked by the normalized variance.

    Args:
        edata: Central data object.
        layer: If provided, use `edata.layers[layer]` for expression values instead of `edata.X`.
        top_features_percentage: Percentage of highly-variable features to keep.
        span: The fraction of the data (observations) used when estimating the variance in the loess model fit.
        batch_key: If specified, highly-variable features are selected within each batch separately and merged.
                   Features are first sorted by the median (across batches) rank, with ties broken by the number of batches a feature is highly variable in.
        subset: Inplace subset to highly-variable features if `True` otherwise merely indicate highly variable features.
        check_values: Check if counts in selected layer are integers. A Warning is returned if set to True.
        copy: Whether to return a copy of `edata` or modify it in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object.
        Updates `.var` with the following fields

    **highly_variable**
        boolean indicator of highly-variable features
    **means**
        means per feature
    **variances**
        variance per feature
    **variances_norm**
        normalized variance per feature, averaged in the case of multiple batches
    **highly_variable_rank**
        rank of the feature according to normalized variance, median rank in the case of multiple batches
    **highly_variable_nbatches**
        if `batch_key` is given, in how many batches the feature is highly variable
    """
    edata = edata.copy() if copy else edata
    n_top_features = int(top_features_percentage * len(edata.var))

    sc.pp.highly_variable_genes(
        adata=edata,
        layer=layer,
        n_top_genes=n_top_features,
        span=span,
        flavor="seurat_v3",
        subset=subset,
        inplace=True,
        batch_key=batch_key,
        check_values=check_values,
    )

    return edata if copy else None
