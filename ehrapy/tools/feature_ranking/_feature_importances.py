from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from ehrdata._logger import logger
from ehrdata.core.constants import CATEGORICAL_TAG, DATE_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR

from ehrapy._compat import function_2D_only, use_ehrdata
from ehrapy.anndata import _check_feature_types, anndata_to_df

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from ehrdata import EHRData


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
@_check_feature_types
def rank_features_supervised(
    edata: EHRData | AnnData,
    predicted_feature: str,
    *,
    model: Literal["regression", "svm", "rf"] = "rf",
    input_features: Iterable[str] | Literal["all"] = "all",
    layer: str | None = None,
    test_split_size: float = 0.2,
    key_added: str = "feature_importances",
    feature_scaling: Literal["standard", "minmax"] | None = "standard",
    percent_output: bool = False,
    verbose: bool = True,
    return_score: bool = False,
    **kwargs,
) -> float | None:
    """Calculate feature importances for predicting a specified feature in adata.var.

    Args:
        edata: Central data object.
        predicted_feature: The feature to predict by the model. Must be present in edata.var_names.
        model: The model to use for prediction.
            Choose between 'regression', 'svm', or 'rf'.
            Multi-class classification is only possible with 'rf'.
        input_features: The features in edata.var to use for prediction.
            Should be a list of feature names.
            If 'all', all features in edata.var will be used.
            Non-numeric input features will error.
        layer: The layer in edata.layers to use for prediction. If None, edata.X will be used.
        test_split_size: The split of data used for testing the model. Should be a float between 0 and 1, representing the proportion.
        key_added: The key in edata.var to store the feature importances.
        feature_scaling: The type of feature scaling to use for the input.
            Choose between 'standard', 'minmax', or None.
            'standard' uses sklearn's StandardScaler, 'minmax' uses MinMaxScaler.
            Scaler will be fit and transformed for each feature individually.
        percent_output: Set to True to output the feature importances as percentages.
            Note that information about positive or negative coefficients for regression models will be lost.
        verbose: Set to False to disable logging.
        return_score: Set to True to return the R2 score / the accuracy of the model.
        **kwargs: Additional keyword arguments to pass to the model. See the documentation of the respective model in scikit-learn for details.

    Returns:
        If return_score is True, the R2 score / accuracy of the model on the test set. Otherwise, None.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.ad.infer_feature_types(edata)
        >>> ep.pp.knn_impute(edata, n_neighbors=5)
        >>> input_features = [
        ...     feat for feat in edata.var_names if feat not in {"service_unit", "day_icu_intime", "tco2_first"}
        ... ]
        >>> ep.tl.rank_features_supervised(edata, "tco2_first", model="rf", input_features=input_features)
    """
    if predicted_feature not in edata.var_names:
        raise ValueError(f"Feature {predicted_feature} not found in edata.var.")

    if input_features != "all":
        for feature in input_features:
            if feature not in edata.var_names:
                raise ValueError(f"Feature {feature} not found in edata.var.")

    if model not in ["regression", "svm", "rf"]:
        raise ValueError(f"Model {model} not recognized. Please choose either 'regression', 'svm', or 'rf'.")

    if feature_scaling not in ["standard", "minmax", None]:
        raise ValueError(
            f"Feature scaling type {feature_scaling} not recognized. Please choose either 'standard', 'minmax', or None."
        )

    data = anndata_to_df(edata, layer=layer)

    prediction_type = edata.var[FEATURE_TYPE_KEY].loc[predicted_feature]

    if prediction_type == DATE_TAG:
        raise ValueError(
            f"Feature {predicted_feature} is of type 'date' and cannot be used for prediction. Please choose a continuous or categorical feature."
        )

    if prediction_type == NUMERIC_TAG:
        if model == "regression":
            predictor = LinearRegression(**kwargs)
        elif model == "svm":
            predictor = SVR(kernel="linear", **kwargs)
        elif model == "rf":
            predictor = RandomForestRegressor(**kwargs)

    elif prediction_type == CATEGORICAL_TAG:
        if data[predicted_feature].nunique() > 2 and model in ["regression", "svm"]:
            raise ValueError(
                f"Feature {predicted_feature} has more than two categories. Please choose 'rf' as model for multi-class classification."
            )

        if model == "regression":
            predictor = LogisticRegression(**kwargs)
        elif model == "svm":
            predictor = SVC(kernel="linear", **kwargs)
        elif model == "rf":
            predictor = RandomForestClassifier(**kwargs)

    if input_features == "all":
        input_features = list(edata.var_names)
        input_features.remove(predicted_feature)

    input_data = data[input_features]
    labels = data[predicted_feature]

    x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size=test_split_size, random_state=42)

    for feature in input_data.columns:
        try:
            x_train.loc[:, feature] = x_train[feature].astype(np.float32)
            x_test.loc[:, feature] = x_test[feature].astype(np.float32)

            if feature_scaling is not None:
                scaler = StandardScaler() if feature_scaling == "standard" else MinMaxScaler()
                scaled_data = scaler.fit_transform(x_train[[feature]].values.astype(np.float32))
                x_train.loc[:, feature] = scaled_data.flatten()

                scaled_data = scaler.transform(x_test[[feature]].values.astype(np.float32))
                x_test.loc[:, feature] = scaled_data.flatten()
        except ValueError as e:
            raise ValueError(
                f"Feature {feature} is not numeric. Please encode non-numeric features before calculating "
                f"feature importances or drop them from the input_features list."
            ) from e

    predictor.fit(x_train, y_train)

    score = predictor.score(x_test, y_test)
    evaluation_metric = "R2 score" if prediction_type == "continuous" else "accuracy"

    if verbose:
        logger.info(
            f"Training completed. The model achieved an {evaluation_metric} of {score:.2f} on the test set, consisting of {len(y_test)} samples."
        )

    if model == "regression" or model == "svm":
        feature_importances = pd.Series(predictor.coef_.squeeze(), index=input_data.columns)
    else:
        feature_importances = pd.Series(predictor.feature_importances_.squeeze(), index=input_data.columns)

    if percent_output:
        feature_importances = feature_importances.abs() / feature_importances.abs().sum() * 100

    # Reorder feature importances to match edata.var order and save importances in edata.var
    feature_importances = feature_importances.reindex(edata.var_names)
    edata.var[key_added] = feature_importances

    return score if return_score else None
