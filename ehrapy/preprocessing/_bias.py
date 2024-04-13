from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ehrapy import logging as logg
from ehrapy.anndata import anndata_to_df


def bias_detection(
    adata: AnnData,
    sensitive_features: Iterable[str] | Literal["all"],
    corr_threshold: float = 0.5,
    smd_threshold: float = 0.5,
    feature_importance_threshold: float = 0.1,
    prediction_confidence_threshold: float = 0.5,
):
    """Detects bias in the data.

    Args:
        adata: An annotated data matrix containing patient data.
        sensitive_features: A list of sensitive features to check for bias.
        corr_threshold: The threshold for the correlation coefficient between two features to be considered of interest. Defaults to 0.5.
        smd_threshold: The threshold for the standardized mean difference between two features to be considered of interest. Defaults to 0.5.
        feature_importance_threshold: The threshold for the feature importance of a sensitive feature for predicting another feature to be considered
            of interest. Defaults to 0.1.
        prediction_confidence_threshold: The threshold for the prediction confidence (R2 or accuracy) of a sensitive feature for predicting another
            feature to be considered of interest. Defaults to 0.5.
    """
    from ehrapy.tools import rank_features_supervised

    if sensitive_features == "all":
        sensitive_features = adata.var_names

    correlations = _feature_correlations(adata)
    adata.varp["correlation"] = correlations

    for feature in sensitive_features:
        for comp_feature in adata.var_names:
            if correlations.loc[feature, comp_feature] > corr_threshold:
                logg.warning(
                    f"Feature {feature} is highly correlated with {comp_feature} (correlation coefficient ≈{correlations.loc[feature, comp_feature]:.3f})."
                )  # TODO: How do we print results?

    smd_dict = _standardized_mean_differences(adata, sensitive_features)
    for feature in sensitive_features:
        abs_smd = smd_dict[feature].abs()
        for comp_feature in adata.var_names:
            if abs_smd[comp_feature].max() > smd_threshold:
                logg.warning(
                    f"Feature {comp_feature} has a high standardized mean difference with {feature}."
                )  # TODO: Do we look at / print groups individually?

    for prediction_feature in adata.var_names:
        prediction_score = rank_features_supervised(
            adata,
            prediction_feature,
            input_features="all",
            model="rf",
            key_added=f"{prediction_feature}_feature_importances",
            percent_output=True,
            logging=False,
            return_score=True,
        )
        for feature in sensitive_features:
            feature_importance = adata.var[f"{prediction_feature}_feature_importances"][feature] / 100
            if feature_importance > feature_importance_threshold and prediction_score > prediction_confidence_threshold:
                logg.warning(
                    f"Feature {feature} has a high feature importance for predicting {prediction_feature} (importance in %: {feature_importance:.3f}, prediction score: {prediction_score:.3f})."
                )


def _feature_correlations(adata: AnnData, method: Literal["pearson", "spearman"] = "spearman"):
    """Computes pairwise correlations between features in the AnnData object.

    Args:
        adata: An annotated data matrix containing patient data.
        method: The correlation method to use. Choose between "pearson" and "spearman". Defaults to "spearman".

    Returns:
        A pandas DataFrame containing the correlation matrix.
    """
    corr_matrix = anndata_to_df(adata).corr(method=method)
    return corr_matrix


def _standardized_mean_differences(adata: AnnData, features: Iterable[str]) -> dict:
    """Computes the standardized mean differences between sensitive features.

    Args:
        adata: An annotated data matrix containing patient data.
        features: A list of features to compute the standardized mean differences (SMD) for. For each listed feature, the SMD is computed for each
            feature, comparing one group to the rest. Thus, we obtain a n_groups_in_feature x n_features matrix of SMDs for each listed feature.

    Returns:
        A dictionary mapping each feature to a pandas DataFrame containing the standardized mean differences.
    """
    df = anndata_to_df(adata)
    smd_results = {}  # type: ignore

    for group_feature in features:  # TODO: Restrict to categorical features (wait for other PR)
        smd_results[group_feature] = {}
        for group in df[group_feature].unique():
            group_mean = df[df[group_feature] == group].mean()
            group_std = df[df[group_feature] == group].std()

            comparison_mean = df[df[group_feature] != group].mean()
            comparison_std = df[df[group_feature] != group].std()

            smd = (group_mean - comparison_mean) / np.sqrt((group_std**2 + comparison_std**2) / 2)
            smd_results[group_feature][group] = smd

        smd_results[group_feature] = pd.DataFrame(smd_results[group_feature]).T[adata.var_names]

    return smd_results
