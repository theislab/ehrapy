import itertools
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ehrapy.anndata import anndata_to_df


def detect_bias(
    adata: AnnData,
    sensitive_features: Iterable[str] | np.ndarray | Literal["all"],
    *,
    run_feature_importances: bool | None = None,
    corr_threshold: float = 0.5,
    smd_threshold: float = 0.5,
    categorical_factor_threshold: float = 2,
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
        categorical_factor_threshold: The threshold for the factor between the value counts (as percentages) of a feature compared between two
            groups of a sensitive feature. Defaults to 2.
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
        sens_features_list = adata.var_names.values.tolist()
        categorical_sensitive_features = adata.var_names.values[
            adata.var["feature_type"] == "categorical"
        ]  # TODO: Double-check that named correctly
    else:
        for feat in sensitive_features:
            if feat not in adata.var_names:
                raise ValueError(f"Feature {feat} not found in adata.var.")
        sens_features_list = sensitive_features
        categorical_sensitive_features = [
            feat for feat in sensitive_features if adata.var["feature_type"][feat] == "categorical"
        ]

    adata_df = anndata_to_df(adata)
    categorical_var_names = adata.var_names[adata.var["feature_type"] == "categorical"]

    # Feature correlations
    correlations = adata_df.corr(method=corr_method)
    adata.varp["feature_correlations"] = correlations
    print(type(correlations))

    corr_results: dict[str, list] = {"Feature 1": [], "Feature 2": [], "Correlation Coefficient": []}
    if sensitive_features == "all":
        feature_tuples = list(itertools.combinations(sens_features_list, 2))
    else:
        feature_tuples = list(itertools.product(sens_features_list, adata.var_names))
    for sens_feature, comp_feature in feature_tuples:
        if sens_feature == comp_feature:
            continue
        if abs(correlations.loc[sens_feature, comp_feature]) > corr_threshold:
            corr_results["Feature 1"].append(sens_feature)
            corr_results["Feature 2"].append(comp_feature)
            corr_results["Correlation Coefficient"].append(correlations.loc[sens_feature, comp_feature])
    bias_results["feature_correlations"] = pd.DataFrame(corr_results)

    # Standardized mean differences
    smd_results: dict[str, list] = {
        "Sensitive Feature": [],
        "Compared Feature": [],
        "Group": [],
        "Standardized Mean Difference": [],
    }  # type: ignore
    for sens_feature in categorical_sensitive_features:  # TODO: Restrict to categorical features (wait for other PR)
        alphabetic_groups = sorted(adata_df[sens_feature].unique())
        smd_nparray = np.zeros((len(alphabetic_groups), len(adata.var_names)))

        for group_nr, group in enumerate(alphabetic_groups):
            group_mean = adata_df[adata_df[sens_feature] == group].mean()
            group_std = adata_df[adata_df[sens_feature] == group].std()

            comparison_mean = adata_df[adata_df[sens_feature] != group].mean()
            comparison_std = adata_df[adata_df[sens_feature] != group].std()

            smd = (group_mean - comparison_mean) / np.sqrt((group_std**2 + comparison_std**2) / 2)
            smd_nparray[group_nr] = smd

            abs_smd = smd.abs()
            for i, comp_feature in enumerate(adata.var_names):  # TODO: Restrict to continuous features
                if sens_feature == comp_feature:
                    continue
                if abs_smd[i] > smd_threshold:
                    smd_results["Sensitive Feature"].append(sens_feature)
                    smd_results["Compared Feature"].append(comp_feature)
                    smd_results["Group"].append(group)
                    smd_results["Standardized Mean Difference"] = smd[i]

        adata.varm[f"smd_{sens_feature}"] = smd_nparray.T  # TODO: Double check

    bias_results["standardized_mean_differences"] = pd.DataFrame(smd_results)

    # Categorical value counts
    cat_value_count_results: dict[str, list] = {
        "Sensitive Feature": [],
        "Sensitive Group": [],
        "Compared Feature": [],
        "Group 1": [],
        "Group 2": [],
        "Group 1 Percentage": [],
        "Group 2 Percentage": [],
    }
    for sens_feature in categorical_sensitive_features:  # TODO: Restrict to categorical features (wait for other PR)
        for comp_feature in categorical_var_names:  # TODO: Restrict to categorical features (wait for other PR)
            if sens_feature == comp_feature:
                continue
            value_counts = adata_df.groupby([sens_feature, comp_feature]).size().unstack(fill_value=0)
            value_counts = value_counts.div(value_counts.sum(axis=1), axis=0)

            for sens_group in value_counts.index:
                for comp_group1, comp_group2 in itertools.combinations(
                    value_counts.columns, 2
                ):  # TODO: Try to find computationally more efficient way
                    value_count_diff = (
                        value_counts.loc[sens_group, comp_group1] - value_counts.loc[sens_group, comp_group2]
                    )
                    if (
                        value_count_diff > categorical_factor_threshold
                        or value_count_diff < 1 / categorical_factor_threshold
                    ):
                        cat_value_count_results["Sensitive Feature"].append(sens_feature)
                        cat_value_count_results["Sensitive Group"].append(sens_group)
                        cat_value_count_results["Compared Feature"].append(comp_feature)
                        cat_value_count_results["Group 1"].append(comp_group1)
                        cat_value_count_results["Group 2"].append(comp_group2)
                        cat_value_count_results["Group 1 Percentage"].append(value_counts.loc[sens_group, comp_group1])
                        cat_value_count_results["Group 2 Percentage"].append(value_counts.loc[sens_group, comp_group2])
    bias_results["categorical_value_counts"] = pd.DataFrame(cat_value_count_results)

    # Feature importances
    if run_feature_importances:
        feature_importances_results: dict[str, list] = {
            "Sensitive Feature": [],
            "Predicted Feature": [],
            "Feature Importance": [],
            "Prediction Score": [],
        }
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

            for sens_feature in sens_features_list:
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
