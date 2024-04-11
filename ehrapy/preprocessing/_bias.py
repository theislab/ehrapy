from collections.abc import Iterable
from typing import Literal

from anndata import AnnData

from ehrapy import logging as logg
from ehrapy.anndata import anndata_to_df


def bias_detection(adata: AnnData, sensitive_features: Iterable[str], corr_threshold: float = 0.5):
    """Detects bias in the data.

    Args:
        adata: An annotated data matrix containing patient data.
        sensitive_features: A list of sensitive features to check for bias.

    Returns:
        #TODO
    """
    correlations = _feature_correlations(adata)
    adata.varp["correlation"] = correlations


def _feature_correlations(adata: AnnData, method: Literal["pearson", "spearman"] = "pearson"):
    """Computes pairwise correlations between features in the AnnData object.

    Args:
        adata: An annotated data matrix containing patient data.

    Returns:
        A pandas DataFrame containing the correlation matrix.
    """
    corr_matrix = anndata_to_df(adata).corr(method=method)
    return corr_matrix
