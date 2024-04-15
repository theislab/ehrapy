import itertools
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ehrapy.anndata import anndata_to_df


def detect_bias(
    adata: AnnData,
    sensitive_features: Iterable[str] | Literal["all"],
    *,
    run_feature_importances: bool | None = None,
    corr_threshold: float = 0.5,
    smd_threshold: float = 0.5,
    feature_importance_threshold: float = 0.1,
    prediction_confidence_threshold: float = 0.5,
    corr_method: Literal["pearson", "spearman"] = "spearman",
) -> dict[str, pd.DataFrame]:
    """Detects biases in the data using feature correlations, standardized mean differences, and feature importances.

    Detects biases with respect to sensitive features, which can be either a specified subset of features or all features in adata.var.
    The method computes pairwise correlations between features, standardized mean differences between groups of sensitive features, and
    feature importances for predicting one feature with another. The results are stored in adata.varp and adata.varm.
    Values that exceed the specified thresholds are considered of interest and returned in the results.

    Args:
        adata: An annotated data matrix containing EHR data.
        sensitive_features: Sensitive features to consider for bias detection. If set to "all", all features in adata.var will be considered.
            If only a subset of features should be considered, provide as an iterable.
        run_feature_importances: Whether to run feature importances for detecting bias. If set to None, the function will run feature importances if
            sensitive_features is not set to "all", as this can be computationally expensive. Defaults to None.
        corr_threshold: The threshold for the correlation coefficient between two features to be considered of interest. Defaults to 0.5.
        smd_threshold: The threshold for the standardized mean difference between two features to be considered of interest. Defaults to 0.5.
        feature_importance_threshold: The threshold for the feature importance of a sensitive feature for predicting another feature to be considered
            of interest. Defaults to 0.1.
        prediction_confidence_threshold: The threshold for the prediction confidence (R2 or accuracy) of a sensitive feature for predicting another
            feature to be considered of interest. Defaults to 0.5.
        corr_method: The correlation method to use. Choose between "pearson" and "spearman". Defaults to "spearman".

    Returns:
        A dictionary containing the results of the bias detection. The keys are:
        - "feature_correlations": Pairwise correlations between features that exceed the correlation threshold.
        - "standardized_mean_differences": Standardized mean differences between groups of sensitive features that exceed the SMD threshold.
        - "feature_importances": Feature importances for predicting one feature with another that exceed the feature importance and prediction
            confidence thresholds.
    """
    from ehrapy.tools import rank_features_supervised

    bias_results = {}

    if run_feature_importances is None:
        run_feature_importances = sensitive_features != "all"

    if sensitive_features == "all":
        sensitive_features = adata.var_names

    adata_df = anndata_to_df(adata)

    # Feature correlations
    correlations = adata_df.corr(method=corr_method)
    adata.varp["feature_correlations"] = correlations

    corr_results = {"Sensitive Feature": [], "Compared Feature": [], "Correlation Coefficient": []}  # type: ignore
    for sens_feature, comp_feature in itertools.product(sensitive_features, adata.var_names):
        if sens_feature == comp_feature:
            continue
        if abs(correlations.loc[sens_feature, comp_feature]) > corr_threshold:
            corr_results["Sensitive Feature"].append(sens_feature)
            corr_results["Compared Feature"].append(comp_feature)
            corr_results["Correlation Coefficient"].append(correlations.loc[sens_feature, comp_feature])
    bias_results["feature_correlations"] = pd.DataFrame(corr_results)

    # Standardized mean differences
    for groupby_feature in sensitive_features:  # TODO: Restrict to categorical features (wait for other PR)
        smd_results = {}
        for group in adata_df[groupby_feature].unique():
            group_mean = adata_df[adata_df[groupby_feature] == group].mean()
            group_std = adata_df[adata_df[groupby_feature] == group].std()

            comparison_mean = adata_df[adata_df[groupby_feature] != group].mean()
            comparison_std = adata_df[adata_df[groupby_feature] != group].std()

            smd = (group_mean - comparison_mean) / np.sqrt((group_std**2 + comparison_std**2) / 2)
            smd_results[group] = smd

        adata.varm[f"smd_{groupby_feature}"] = pd.DataFrame(smd_results).T[adata.var_names]

    smd_results = {"Sensitive Feature": [], "Compared Feature": [], "Group": [], "Standardized Mean Difference": []}
    for sens_feature in sensitive_features:
        abs_smd = adata.varm[f"smd_{sens_feature}"].abs()
        for comp_feature in adata.var_names:
            if sens_feature == comp_feature:
                continue
            if abs_smd[comp_feature].max() > smd_threshold:
                smd_results["Sensitive Feature"].append([sens_feature] * len(abs_smd[comp_feature]))
                smd_results["Compared Feature"].append([comp_feature] * len(abs_smd[comp_feature]))
                smd_results["Group"].append(abs_smd[comp_feature].index.values)
                smd_results["Standardized Mean Difference"] = adata.varm[f"smd_{sens_feature}"].values
    bias_results["standardized_mean_differences"] = pd.DataFrame(smd_results)

    # Feature importances
    if run_feature_importances:
        feature_importances_results = {
            "Sensitive Feature": [],
            "Predicted Feature": [],
            "Feature Importance": [],
            "Prediction Score": [],
        }  # type: ignore
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

            for sens_feature in sensitive_features:
                if prediction_feature == sens_feature:
                    continue
                feature_importance = adata.var[f"{prediction_feature}_feature_importances"][sens_feature] / 100
                if (
                    feature_importance > feature_importance_threshold
                    and prediction_score > prediction_confidence_threshold
                ):
                    feature_importances_results["Sensitive Feature"].append(sens_feature)
                    feature_importances_results["Predicted Feature"].append(prediction_feature)
                    feature_importances_results["Feature Importance"].append(feature_importance)
                    feature_importances_results["Prediction Score"].append(prediction_score)
        bias_results["feature_importances"] = pd.DataFrame(feature_importances_results)

    return bias_results
