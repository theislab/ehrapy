from collections.abc import Iterable
from typing import Literal

import pandas as pd
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR

from ehrapy import logging as logg
from ehrapy.anndata import anndata_to_df
from ehrapy.anndata._constants import EHRAPY_TYPE_KEY, NON_NUMERIC_ENCODED_TAG, NON_NUMERIC_TAG, NUMERIC_TAG


def rank_features_supervised(
    adata: AnnData,
    predicted_feature: str,
    prediction_type: Literal["continuous", "categorical", "auto"] = "auto",
    model: Literal["regression", "svm", "rf"] = "regression",
    input_features: Iterable[str] | Literal["all"] = "all",
    layer: str | None = None,
    test_split_size: float = 0.2,
    key_added: str = "feature_importances",
    feature_scaling: Literal["standard", "minmax"] | None = "standard",
    percent_output: bool = False,
    **kwargs,
):
    """Calculate feature importances for predicting a specified feature in adata.var.

    Args:
        adata: :class:`~anndata.AnnData` object storing the data.
        predicted_feature: The feature to predict by the model. Must be present in adata.var_names.
        prediction_type: Whether the predicted feature is continuous or categorical. If the data type of the predicted feature
            is not correct, conversion will be attempted. If set to 'auto', the function will try to infer the data type from the data.
            Defaults to 'auto'.
        model: The model to use for prediction. Choose between 'regression', 'svm', or 'rf'. Note that multi-class classification
            is only possible with 'rf'. Defaults to 'regression'.
        input_features: The features in adata.var to use for prediction. Should be a list of feature names. If 'all', all features
            in adata.var will be used. Note that non-numeric input features will cause an error, so make sure to encode them properly
            before. Defaults to 'all'.
        layer: The layer in adata.layers to use for prediction. If None, adata.X will be used. Defaults to None.
        test_split_size: The split of data used for testing the model. Should be a float between 0 and 1, representing the proportion.
            Defaults to 0.2.
        key_added: The key in adata.var to store the feature importances. Defaults to 'feature_importances'.
        feature_scaling: The type of feature scaling to use for the input. Choose between 'standard', 'minmax', or None.
            'standard' uses sklearn's StandardScaler, 'minmax' uses MinMaxScaler. Scaler will be fit and transformed
            for each feature individually. Defaults to 'standard'.
        percent_output: Set to True to output the feature importances as percentages. Note that information about positive or negative
            coefficients for regression models will be lost. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the model. See the documentation of the respective model in scikit-learn for details.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> ep.pp.knn_impute(adata, n_neighbours=5)
        >>> input_features = [
        ...     feat for feat in adata.var_names if feat not in {"service_unit", "day_icu_intime", "tco2_first"}
        ... ]
        >>> ep.tl.rank_features_supervised(
        ...     adata, "tco2_first", prediction_type="continuous", model="rf", input_features=input_features
        ... )
    """
    if predicted_feature not in adata.var_names:
        raise ValueError(f"Feature {predicted_feature} not found in adata.var.")

    if input_features != "all":
        for feature in input_features:
            if feature not in adata.var_names:
                raise ValueError(f"Feature {feature} not found in adata.var.")

    if model not in ["regression", "svm", "rf"]:
        raise ValueError(f"Model {model} not recognized. Please choose either 'regression', 'svm', or 'rf'.")

    if feature_scaling not in ["standard", "minmax", None]:
        raise ValueError(
            f"Feature scaling type {feature_scaling} not recognized. Please choose either 'standard', 'minmax', or None."
        )

    data = anndata_to_df(adata, layer=layer)

    if prediction_type == "auto":
        if EHRAPY_TYPE_KEY in adata.var:
            prediction_encoding_type = adata.var[EHRAPY_TYPE_KEY][predicted_feature]
            if prediction_encoding_type == NON_NUMERIC_TAG or prediction_encoding_type == NON_NUMERIC_ENCODED_TAG:
                prediction_type = "categorical"
            else:
                prediction_type = "continuous"
        else:
            if pd.api.types.is_categorical_dtype(data[predicted_feature].dtype):
                prediction_type = "categorical"
            else:
                prediction_type = "continuous"
        logg.info(
            f"Predicted feature {predicted_feature} was detected as {prediction_type}. If this is incorrect, please specify in the prediction_type argument."
        )

    elif prediction_type == "continuous":
        if pd.api.types.is_categorical_dtype(data[predicted_feature].dtype):
            try:
                data[predicted_feature] = data[predicted_feature].astype(float)
            except ValueError as e:
                raise ValueError(
                    f"Feature {predicted_feature} is not continuous and conversion to float failed. Either change the prediction "
                    f"type to 'categorical' or change the feature data type to a continuous type."
                ) from e

    elif prediction_type == "categorical":
        if not pd.api.types.is_categorical_dtype(data[predicted_feature].dtype):
            try:
                data[predicted_feature] = data[predicted_feature].astype("category")
            except ValueError as e:
                raise ValueError(
                    f"Feature {predicted_feature} is not categorical and conversion to category failed. Either change the prediction "
                    f"type to 'continuous' or change the feature data type to a categorical type."
                ) from e
    else:
        raise ValueError(
            f"Prediction type {prediction_type} not recognized. Please choose 'continuous', 'categorical', or 'auto'."
        )

    if prediction_type == "continuous":
        if model == "regression":
            predictor = LinearRegression(**kwargs)
        elif model == "svm":
            predictor = SVR(kernel="linear", **kwargs)
        elif model == "rf":
            predictor = RandomForestRegressor(**kwargs)

    elif prediction_type == "categorical":
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
        input_features = list(adata.var_names)
        input_features.remove(predicted_feature)

    input_data = data[input_features]
    labels = data[predicted_feature]

    for feature in input_data.columns:
        try:
            input_data[feature] = input_data[feature].astype(float)

            if feature_scaling is not None:
                scaler = StandardScaler() if feature_scaling == "standard" else MinMaxScaler()
                input_data[feature] = scaler.fit_transform(input_data[[feature]])
        except ValueError as e:
            raise ValueError(
                f"Feature {feature} is not numeric. Please encode non-numeric features before calculating "
                f"feature importances or drop them from the input_features list."
            ) from e

    x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size=test_split_size)

    predictor.fit(x_train, y_train)

    score = predictor.score(x_test, y_test)
    evaluation_metric = "R2 score" if prediction_type == "continuous" else "accuracy"
    logg.info(
        f"Training completed. The model achieved an {evaluation_metric} of {score:.2f} on the test set, consisting of {len(y_test)} samples."
    )

    if model == "regression" or model == "svm":
        feature_importances = pd.Series(predictor.coef_.squeeze(), index=input_data.columns)
    else:
        feature_importances = pd.Series(predictor.feature_importances_.squeeze(), index=input_data.columns)

    if percent_output:
        feature_importances = feature_importances.abs() / feature_importances.abs().sum() * 100

    # Reorder feature importances to match adata.var order and save importances in adata.var
    feature_importances = feature_importances.reindex(adata.var_names)
    adata.var[key_added] = feature_importances
