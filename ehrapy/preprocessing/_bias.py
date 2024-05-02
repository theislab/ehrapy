import itertools
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ehrapy.anndata import anndata_to_df, check_feature_types
from ehrapy.anndata._constants import CATEGORICAL_TAG, CONTINUOUS_TAG, DATE_TAG, FEATURE_TYPE_KEY


@check_feature_types
def detect_bias(
    adata: AnnData,
    sensitive_features: Iterable[str] | Literal["all"],
    *,
    run_feature_importances: bool | None = None,
    corr_threshold: float = 0.5,
    smd_threshold: float = 0.5,
    categorical_factor_threshold: float = 2,
    feature_importance_threshold: float = 0.1,
    prediction_confidence_threshold: float = 0.5,
    corr_method: Literal["pearson", "spearman"] = "spearman",
    copy: bool = False,
) -> dict[str, pd.DataFrame] | tuple[dict[str, pd.DataFrame], AnnData]:
    """Detects biases in the data using feature correlations, standardized mean differences, and feature importances.

    Detects biases with respect to sensitive features, which can be either a specified subset of features or all features in adata.var.
    The method detects biases by computing:
    - pairwise correlations between features
    - standardized mean differences for numeric features between groups of sensitive features
    - value counts of categorical features between groups of sensitive features
    - feature importances for predicting one feature with another.
    Results of the computations are stored in var, varp, and uns of the adata object.
    Values that exceed the specified thresholds are considered of interest and returned in the results dictionary.

    Args:
        adata: An annotated data matrix containing EHR data.
        sensitive_features: Sensitive features to consider for bias detection. If set to "all", all features in adata.var will be considered.
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
        corr_method: The correlation method to use. Defaults to "spearman".
        copy: If set to False, adata is updated in place. If set to True, the adata is copied and the results are stored in the copied adata, which
            is then returned. Defaults to False.

    Returns:
        A dictionary containing the results of the bias detection. The keys are:
        - "feature_correlations": Pairwise correlations between features that exceed the correlation threshold.
        - "standardized_mean_differences": Standardized mean differences between groups of sensitive features that exceed the SMD threshold.
        - "categorical_value_counts": Value counts of categorical features between groups of sensitive features that exceed the categorical factor
            threshold.
        - "feature_importances": Feature importances for predicting one feature with another that exceed the feature importance and prediction
            confidence thresholds.

        If copy is set to True, the function returns a tuple with the results dictionary and the updated adata.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.ad.infer_feature_types(adata, output=None)
        >>> results_dict = ep.pp.detect_bias(adata, "all")
    """
    from ehrapy.tools import rank_features_supervised

    bias_results = {}

    if run_feature_importances is None:
        run_feature_importances = sensitive_features != "all"

    if sensitive_features == "all":
        sens_features_list = adata.var_names.values.tolist()
        cat_sens_features = adata.var_names.values[adata.var[FEATURE_TYPE_KEY] == CATEGORICAL_TAG]
    else:
        for feat in sensitive_features:
            if feat not in adata.var_names:
                raise ValueError(f"Feature {feat} not found in adata.var.")
        sens_features_list = sensitive_features
        cat_sens_features = [
            feat for feat in sensitive_features if adata.var[FEATURE_TYPE_KEY][feat] == CATEGORICAL_TAG
        ]

    if copy:
        adata = adata.copy()

    adata_df = anndata_to_df(adata)

    for feature in adata.var_names:
        if not np.all(adata_df[feature].dropna().apply(type).isin([int, float, complex])):
            raise ValueError(
                f"Feature {feature} is not encoded numerically. Please encode the data (ep.pp.encode) before running bias detection."
            )

    # --------------------
    # Feature correlations
    # --------------------
    correlations = adata_df.corr(method=corr_method)
    adata.varp["feature_correlations"] = correlations

    corr_results: dict[str, list] = {"Feature 1": [], "Feature 2": [], f"{corr_method.capitalize()} CC": []}
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
            corr_results[f"{corr_method.capitalize()} CC"].append(correlations.loc[sens_feature, comp_feature])
    bias_results["feature_correlations"] = pd.DataFrame(corr_results).sort_values(
        by=f"{corr_method.capitalize()} CC", key=abs
    )

    # -----------------------------
    # Standardized mean differences
    # -----------------------------
    smd_results: dict[str, list] = {
        "Sensitive Feature": [],
        "Sensitive Group": [],
        "Compared Feature": [],
        "Standardized Mean Difference": [],
    }
    adata.uns["smd"] = {}
    continuous_var_names = adata.var_names[adata.var[FEATURE_TYPE_KEY] == CONTINUOUS_TAG]
    for sens_feature in cat_sens_features:
        sens_feature_groups = sorted(adata_df[sens_feature].unique())
        if len(sens_feature_groups) == 1:
            continue
        smd_df = pd.DataFrame(index=continuous_var_names, columns=sens_feature_groups)

        for _group_nr, group in enumerate(sens_feature_groups):
            # Compute SMD for all continuous features between the sensitive group and all other observations
            group_mean = adata_df[continuous_var_names][adata_df[sens_feature] == group].mean()
            group_std = adata_df[continuous_var_names][adata_df[sens_feature] == group].std()

            comparison_mean = adata_df[continuous_var_names][adata_df[sens_feature] != group].mean()
            comparison_std = adata_df[continuous_var_names][adata_df[sens_feature] != group].std()

            smd = (group_mean - comparison_mean) / np.sqrt((group_std**2 + comparison_std**2) / 2)
            smd_df[group] = smd

            abs_smd = smd.abs()
            for comp_feature in continuous_var_names:
                if abs_smd[comp_feature] > smd_threshold:
                    smd_results["Sensitive Feature"].append(sens_feature)
                    smd_results["Sensitive Group"].append(group)
                    smd_results["Compared Feature"].append(comp_feature)
                    smd_results["Standardized Mean Difference"].append(smd[comp_feature])
        adata.uns["smd"][sens_feature] = smd_df

    bias_results["standardized_mean_differences"] = pd.DataFrame(smd_results).sort_values(
        by="Standardized Mean Difference", key=abs
    )

    # ------------------------
    # Categorical value counts
    # ------------------------
    cat_value_count_results: dict[str, list] = {
        "Sensitive Feature": [],
        "Sensitive Group": [],
        "Compared Feature": [],
        "Group 1": [],
        "Group 2": [],
        "Group 1 Percentage": [],
        "Group 2 Percentage": [],
    }
    cat_var_names = adata.var_names[adata.var[FEATURE_TYPE_KEY] == CATEGORICAL_TAG]
    for sens_feature in cat_sens_features:
        for comp_feature in cat_var_names:
            if sens_feature == comp_feature:
                continue
            value_counts = adata_df.groupby([sens_feature, comp_feature]).size().unstack(fill_value=0)
            value_counts = value_counts.div(value_counts.sum(axis=1), axis=0)

            for sens_group in value_counts.index:
                for comp_group1, comp_group2 in itertools.combinations(value_counts.columns, 2):
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

    # -------------------
    # Feature importances
    # -------------------
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
        bias_results["feature_importances"] = pd.DataFrame(feature_importances_results).sort_values(
            by="Feature Importance", key=abs
        )

    if copy:
        return bias_results, adata
    return bias_results
