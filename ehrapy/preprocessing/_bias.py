from collections.abc import Iterable
from typing import Literal

import pandas as pd
from anndata import AnnData

from ehrapy import logging as logg
from ehrapy.anndata import anndata_to_df
from ehrapy.tools.feature_ranking._feature_importances import rank_features_supervised


def bias_detection(
    adata: AnnData,
    sensitive_features: Iterable[str],
    corr_threshold: float = 0.5,
    smd_threshold: float = 0.5,
    feature_importance_threshold: float = 0.25,
):
    """Detects bias in the data.

    Args:
        adata: An annotated data matrix containing patient data.
        sensitive_features: A list of sensitive features to check for bias.

    Returns:
        #TODO
    """
    correlations = _feature_correlations(adata)
    adata.varp["correlation"] = correlations

    for feature in sensitive_features:
        if correlations.loc[feature, :].abs().max() > corr_threshold:
            logg.warning(
                f"Feature {feature} is highly correlated with another feature."
            )  # TODO: How do we print results?

    smd = _standardized_mean_differences(adata)
    adata.varp["smd"] = smd
    for feature in sensitive_features:
        for comp_feature in adata.var_names:
            if smd.loc[_standardized_mean_differences, feature] > smd_threshold:
                logg.warning(f"Feature {comp_feature} has a high standardized mean difference with {feature}.")

    # feature importances
    # TODO


def _feature_correlations(adata: AnnData, method: Literal["pearson", "spearman"] = "pearson"):
    """Computes pairwise correlations between features in the AnnData object.

    Args:
        adata: An annotated data matrix containing patient data.

    Returns:
        A pandas DataFrame containing the correlation matrix.
    """
    corr_matrix = anndata_to_df(adata).corr(method=method)
    return corr_matrix


def _standardized_mean_differences(adata: AnnData) -> pd.DataFrame:
    """Computes the standardized mean differences between sensitive features.

    Args:
        adata: An annotated data matrix containing patient data.
        sensitive_features: A list of sensitive features to check for bias.

    Returns:
        A pandas DataFrame containing the standardized mean differences.
    """
    df = anndata_to_df(adata)
    smd_results = {}  # type: ignore

    for feature1 in df.columns:
        smd_results[feature1] = {}
        comparison_features = [feature for feature in df.columns if feature != feature1]

        overall_mean = df[comparison_features].mean()
        overall_std = df[comparison_features].std()

        group_mean = df.groupby(feature1)[comparison_features].mean()
        for feature2 in comparison_features:
            smd = (group_mean[feature2] - overall_mean[feature2]) / overall_std[feature2]
            smd_results[feature1][feature2] = smd.to_dict()

    smd_results = pd.DataFrame(smd_results).reindex(adata.var_names)
    smd_results = smd_results[adata.var_names]
    return smd_results
