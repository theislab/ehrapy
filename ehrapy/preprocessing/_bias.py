from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal

import ehrdata as ed
import numpy as np
import pandas as pd
from ehrdata.core.constants import CATEGORICAL_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG

from ehrapy._compat import function_2D_only, use_ehrdata
from ehrapy.anndata import _check_feature_types

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from ehrdata import EHRData


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
@_check_feature_types
def detect_bias(
    edata: EHRData | AnnData,
    sensitive_features: Iterable[str] | Literal["all"],
    *,
    run_feature_importances: bool | None = None,
    corr_threshold: float = 0.5,
    smd_threshold: float = 0.5,
    categorical_factor_threshold: float = 2,
    feature_importance_threshold: float = 0.1,
    prediction_confidence_threshold: float = 0.5,
    corr_method: Literal["pearson", "spearman"] = "spearman",
    layer: str | None = None,
    copy: bool = False,
) -> dict[str, pd.DataFrame] | tuple[dict[str, pd.DataFrame], EHRData | AnnData]:
    """Detects biases in the data using feature correlations, standardized mean differences, and feature importances.

    Detects biases with respect to sensitive features, which can be either a specified subset of features or all features in `.var`.
    The method detects biases by computing:

    - pairwise correlations between features
    - standardized mean differences for numeric features between groups of sensitive features
    - value counts of categorical features between groups of sensitive features
    - feature importances for predicting one feature with another

    Results of the computations are stored in `.var`, `.varp`, and `.uns` of the edata object.
    Values that exceed the specified thresholds are considered of interest and returned in the results dictionary.
    Be aware that the results depend on the encoding of the data. E.g. when using one-hot encoding, each group of a categorical feature will
    be treated as a separate feature, which can lead to an increased number of detected biases. Please take this into consideration when
    interpreting the results.

    Args:
        edata: Central data object. Encoded features are required for bias detection.
        sensitive_features: Sensitive features to consider for bias detection. If set to "all", all features in `.var` will be considered.
        run_feature_importances: Whether to run feature importances for detecting bias. If set to None, the function will run feature importances if
            sensitive_features is not set to "all", as this can be computationally expensive.
        corr_threshold: The threshold for the correlation coefficient between two features to be considered of interest.
        smd_threshold: The threshold for the standardized mean difference between two features to be considered of interest.
        categorical_factor_threshold: The threshold for the factor between the value counts (as percentages) of a feature compared between two
            groups of a sensitive feature.
        feature_importance_threshold: The threshold for the feature importance of a sensitive feature for predicting another feature to be considered
            of interest.
        prediction_confidence_threshold: The threshold for the prediction confidence (R2 or accuracy) of a sensitive feature for predicting another
            feature to be considered of interest.
        corr_method: The correlation method to use.
        layer: The layer in `.layers` to use for computation. If None, `.X` will be used.
        copy: If set to `False`, `edata` is updated in place. If set to `True`, the `edata` is copied and the results are stored in the copied `edata`, which
            is then returned.

    Returns:
        A dictionary containing the results of the bias detection. The keys are

        - "feature_correlations": Pairwise correlations between features that exceed the correlation threshold.
        - "standardized_mean_differences": Standardized mean differences between groups of sensitive features that exceed the SMD threshold.
        - "categorical_value_counts": Value counts of categorical features between groups of sensitive features that exceed the categorical factor
          threshold.
        - "feature_importances": Feature importances for predicting one feature with another that exceed the feature importance and prediction
          confidence thresholds.

        If `copy` is set to `True`, the function returns a tuple with the results dictionary and the updated `edata`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ed.infer_feature_types(edata)
        >>> edata = ep.pp.encode(edata, autodetect=True, encodings="label")
        >>> results_dict = ep.pp.detect_bias(edata, "all")

        >>> # Example with specified sensitive features
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.diabetes_130_fairlearn()
        >>> ed.infer_feature_types(edata)
        >>> edata = ep.pp.encode(edata, autodetect=True, encodings="label")
        >>> results_dict = ep.pp.detect_bias(edata, sensitive_features=["race", "gender"])
    """
    from ehrapy.tools import rank_features_supervised

    bias_results = {}

    if run_feature_importances is None:
        run_feature_importances = sensitive_features != "all"

    if sensitive_features == "all":
        sens_features_list = edata.var_names.values.tolist()
        cat_sens_features = edata.var_names.values[edata.var[FEATURE_TYPE_KEY] == CATEGORICAL_TAG]
    else:
        sens_features_list = []
        for variable in sensitive_features:
            if variable not in edata.var_names:
                # check if feature has been encodeds
                encoded_categorical_features = [
                    feature for feature in edata.var_names if feature.startswith(f"ehrapycat_{variable}")
                ]

                if len(encoded_categorical_features) == 0:
                    raise ValueError(f"Feature {variable} not found in edata.var.")

                sens_features_list.extend(encoded_categorical_features)
            else:
                sens_features_list.append(variable)

        cat_sens_features = [
            variable for variable in sens_features_list if edata.var[FEATURE_TYPE_KEY][variable] == CATEGORICAL_TAG
        ]

    if copy:
        edata = edata.copy()

    edata_df = ed.io.to_pandas(edata, layer=layer)

    for feature in edata.var_names:
        if not np.all(edata_df[feature].dropna().apply(type).isin([int, float, complex])):
            raise ValueError(
                f"Feature {feature} is not encoded numerically. Please encode the data (ep.pp.encode) before running bias detection."
            )

    def _get_group_name(encoded_feature: str, group_val: int) -> str | int:
        try:
            feature_name = encoded_feature.split("_")[1]
            # Get the original group name stored in edata.obs by filtering the data for the encoded group value
            return edata.obs[feature_name][list(edata[:, encoded_feature].X.squeeze() == group_val)].unique()[0]
        except KeyError:
            return group_val

    # --------------------
    # Feature correlations
    # --------------------
    correlations = edata_df.corr(method=corr_method)
    edata.varp["feature_correlations"] = correlations

    corr_results: dict[str, list] = {"Feature 1": [], "Feature 2": [], f"{corr_method.capitalize()} CC": []}
    if sensitive_features == "all":
        feature_tuples = list(itertools.combinations(sens_features_list, 2))
    else:
        feature_tuples = list(itertools.product(sens_features_list, edata.var_names))
    for sens_feature, comp_feature in feature_tuples:
        if sens_feature == comp_feature:
            continue
        if abs(correlations.loc[sens_feature, comp_feature]) > corr_threshold:
            corr_results["Feature 1"].append(sens_feature)
            corr_results["Feature 2"].append(comp_feature)
            corr_results[f"{corr_method.capitalize()} CC"].append(correlations.loc[sens_feature, comp_feature])
    bias_results["feature_correlations"] = pd.DataFrame(corr_results).sort_values(
        by=f"{corr_method.capitalize()} CC", key=abs, ascending=False
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
    edata.uns["smd"] = {}
    continuous_var_names = edata.var_names[edata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG]
    for sens_feature in cat_sens_features:
        sens_feature_groups = sorted(edata_df[sens_feature].unique())
        if len(sens_feature_groups) == 1:
            continue
        smd_df = pd.DataFrame(index=continuous_var_names, columns=sens_feature_groups)

        for _group_nr, group in enumerate(sens_feature_groups):
            # Compute SMD for all continuous features between the sensitive group and all other observations
            group_mean = edata_df[continuous_var_names][edata_df[sens_feature] == group].mean()
            group_std = edata_df[continuous_var_names][edata_df[sens_feature] == group].std()

            comparison_mean = edata_df[continuous_var_names][edata_df[sens_feature] != group].mean()
            comparison_std = edata_df[continuous_var_names][edata_df[sens_feature] != group].std()

            smd = (group_mean - comparison_mean) / np.sqrt((group_std**2 + comparison_std**2) / 2)
            smd_df[group] = smd

            abs_smd = smd.abs()
            for comp_feature in continuous_var_names:
                if abs_smd[comp_feature] > smd_threshold:
                    smd_results["Sensitive Feature"].append(sens_feature)
                    group_name = (
                        _get_group_name(sens_feature, group) if sens_feature.startswith("ehrapycat_") else group
                    )
                    smd_results["Sensitive Group"].append(group_name)
                    smd_results["Compared Feature"].append(comp_feature)
                    smd_results["Standardized Mean Difference"].append(smd[comp_feature])
        edata.uns["smd"][sens_feature] = smd_df

    bias_results["standardized_mean_differences"] = pd.DataFrame(smd_results).sort_values(
        by="Standardized Mean Difference", key=abs, ascending=False
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
    cat_var_names = edata.var_names[edata.var[FEATURE_TYPE_KEY] == CATEGORICAL_TAG]
    for sens_feature in cat_sens_features:
        for comp_feature in cat_var_names:
            if sens_feature == comp_feature:
                continue
            value_counts = edata_df.groupby([sens_feature, comp_feature]).size().unstack(fill_value=0)
            value_counts = value_counts.div(value_counts.sum(axis=1), axis=0)

            for sens_group in value_counts.index:
                for comp_group1, comp_group2 in itertools.combinations(value_counts.columns, 2):
                    value_count_diff = (
                        value_counts.loc[sens_group, comp_group1] / value_counts.loc[sens_group, comp_group2]
                    )
                    if (
                        value_count_diff > categorical_factor_threshold
                        or value_count_diff < 1 / categorical_factor_threshold
                    ):
                        cat_value_count_results["Sensitive Feature"].append(sens_feature)

                        sens_group_name = (
                            _get_group_name(sens_feature, sens_group)
                            if sens_feature.startswith("ehrapycat_")
                            else sens_group
                        )
                        cat_value_count_results["Sensitive Group"].append(sens_group_name)

                        cat_value_count_results["Compared Feature"].append(comp_feature)
                        comp_group1_name = (
                            _get_group_name(comp_feature, comp_group1)
                            if comp_feature.startswith("ehrapycat_")
                            else comp_group1
                        )
                        cat_value_count_results["Group 1"].append(comp_group1_name)
                        comp_group2_name = (
                            _get_group_name(comp_feature, comp_group2)
                            if comp_feature.startswith("ehrapycat_")
                            else comp_group2
                        )
                        cat_value_count_results["Group 2"].append(comp_group2_name)

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
        for prediction_feature in edata.var_names:
            try:
                prediction_score = rank_features_supervised(
                    edata,
                    prediction_feature,
                    input_features="all",
                    model="rf",
                    key_added=f"{prediction_feature}_feature_importances",
                    percent_output=True,
                    verbose=False,
                    return_score=True,
                )
            except ValueError as e:
                if "Input y contains NaN" in str(e):
                    raise ValueError(
                        f"During feature importance computation, input feature y ({prediction_feature}) was found to contain NaNs."
                    ) from e
                else:
                    raise e

            for sens_feature in sens_features_list:
                if prediction_feature == sens_feature:
                    continue
                feature_importance = edata.var[f"{prediction_feature}_feature_importances"][sens_feature] / 100
                if (
                    feature_importance > feature_importance_threshold
                    and prediction_score > prediction_confidence_threshold
                ):
                    feature_importances_results["Sensitive Feature"].append(sens_feature)
                    feature_importances_results["Predicted Feature"].append(prediction_feature)
                    feature_importances_results["Feature Importance"].append(feature_importance)
                    feature_importances_results["Prediction Score"].append(prediction_score)
        bias_results["feature_importances"] = pd.DataFrame(feature_importances_results).sort_values(
            by="Feature Importance", key=abs, ascending=False
        )

    if copy:
        return bias_results, edata
    return bias_results
