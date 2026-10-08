from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from ehrdata._feature_types import _check_feature_types
from ehrdata._logger import logger
from ehrdata.core.constants import CATEGORICAL_TAG, DATE_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR

from ehrapy._compat import function_2D_only
from ehrapy.get import obs_df

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData


@function_2D_only()
@_check_feature_types
def rank_features_supervised(
    edata: EHRData,
    predicted_feature: str,
    *,
    model: Literal["regression", "svm", "rf"] = "rf",
    var_names: Sequence[str] | Literal["all"] = "all",
    layer: str | None = None,
    test_split_size: float = 0.2,
    key_added: str = "feature_importances",
    feature_scaling: Literal["standard", "minmax"] | None = "standard",
    percent_output: bool = False,
    verbose: bool = True,
    copy: bool = False,
    **kwargs,
) -> EHRData | None:
    """Calculate feature importances for predicting a specified feature in adata.var.

    Args:
        edata: Central data object.
        predicted_feature: The feature to predict by the model. Must be present in edata.var_names.
        model: The model to use for prediction.
            Choose between 'regression', 'svm', or 'rf'.
            Multi-class classification is only possible with 'rf'.
        var_names: The features in edata.var to use for prediction.
            Should be a list of feature names.
            If 'all', all features in edata.var will be used.
            Non-numeric input features will error.
        layer: The layer in edata.layers to use for prediction. If None, edata.X will be used.
        test_split_size: The split of data used for testing the model. Should be a float between 0 and 1, representing the proportion.
        key_added: The key in `edata.var` to store the feature importances and in `edata.uns` to store the model's test set performance.
        feature_scaling: The type of feature scaling to use for the input.
            Choose between 'standard', 'minmax', or None.
            'standard' uses sklearn's StandardScaler, 'minmax' uses MinMaxScaler.
            Scaler will be fit and transformed for each feature individually.
        percent_output: Set to True to output the feature importances as percentages.
            Note that information about positive or negative coefficients for regression models will be lost.
        verbose: Set to False to disable logging.
        copy: Copy `edata` before computation and return a copy. Otherwise, perform computation in place.
        **kwargs: Additional keyword arguments to pass to the model. See the documentation of the respective model in scikit-learn for details.

    Returns:
        Depending on `copy`, returns or updates `edata` with the feature importances in `edata.var[key_added]`
        and the model's R2 score (numeric target) or accuracy (categorical target) on the test set in `edata.uns[key_added]`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ed.infer_feature_types(edata)
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.tl.rank_features_supervised(edata, predicted_feature="tco2_first", model="rf")
    """
    if predicted_feature not in edata.var_names:
        raise ValueError(f"Feature {predicted_feature} not found in edata.var.")

    if var_names != "all":
        for feature in var_names:
            if feature not in edata.var_names:
                raise ValueError(f"Feature {feature} not found in edata.var.")

    if model not in ["regression", "svm", "rf"]:
        raise ValueError(f"Model {model} not recognized. Please choose either 'regression', 'svm', or 'rf'.")

    if feature_scaling not in ["standard", "minmax", None]:
        raise ValueError(
            f"Feature scaling type {feature_scaling} not recognized. Please choose either 'standard', 'minmax', or None."
        )

    edata = edata.copy() if copy else edata
    if var_names == "all":
        var_names = [var_name for var_name in edata.var_names if var_name != predicted_feature]
    columns = list(dict.fromkeys([*var_names, predicted_feature]))
    data = obs_df(edata, keys=columns, layer=layer)

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

    input_data = data[list(var_names)]
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
                f"feature importances or drop them from the var_names list."
            ) from e

    predictor.fit(x_train, y_train)

    score = predictor.score(x_test, y_test)
    evaluation_metric = "r2" if prediction_type == NUMERIC_TAG else "accuracy"

    if verbose:
        logger.info(f"Training completed. Test set {evaluation_metric}: {score:.2f} ({len(y_test)} samples).")

    if model == "regression" or model == "svm":
        feature_importances = pd.Series(predictor.coef_.squeeze(), index=input_data.columns)
    else:
        feature_importances = pd.Series(predictor.feature_importances_.squeeze(), index=input_data.columns)

    if percent_output:
        feature_importances = feature_importances.abs() / feature_importances.abs().sum() * 100

    # Reorder feature importances to match edata.var order and save importances in edata.var
    feature_importances = feature_importances.reindex(edata.var_names)
    edata.var[key_added] = feature_importances
    edata.uns[key_added] = {
        "predicted_feature": predicted_feature,
        "model": model,
        "metric": evaluation_metric,
        "score": score,
    }

    return edata if copy else None
