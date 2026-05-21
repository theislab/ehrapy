"""Internal helpers for resolving propensity / outcome model specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


_PROPENSITY_STRINGS = ("logistic", "gradient_boosting", "random_forest")
_OUTCOME_STRINGS = ("auto", "linear", "logistic", "gradient_boosting", "random_forest")


def resolve_propensity_model(spec: str | BaseEstimator) -> BaseEstimator:
    """Return a cloned sklearn classifier from a string shortcut or a user-supplied estimator."""
    if isinstance(spec, str):
        if spec == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000)
        if spec == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier()
        if spec == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=200, n_jobs=-1)
        raise ValueError(f"Unknown propensity_model {spec!r}; expected one of {_PROPENSITY_STRINGS}.")

    from sklearn.base import clone

    return clone(spec)


def resolve_outcome_model(spec: str | BaseEstimator, *, y: np.ndarray) -> BaseEstimator:
    """Return a cloned sklearn estimator (regressor or classifier) for the outcome model.

    For ``spec='auto'`` the type is inferred from ``y``: binary 0/1 → logistic regression, else linear regression.
    """
    if isinstance(spec, str):
        if spec == "auto":
            unique = np.unique(y[~np.isnan(y)])
            spec = "logistic" if set(unique.tolist()).issubset({0.0, 1.0}) and len(unique) <= 2 else "linear"
        if spec == "linear":
            from sklearn.linear_model import LinearRegression

            return LinearRegression()
        if spec == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000)
        if spec == "gradient_boosting":
            unique = np.unique(y[~np.isnan(y)])
            if set(unique.tolist()).issubset({0.0, 1.0}) and len(unique) <= 2:
                from sklearn.ensemble import GradientBoostingClassifier

                return GradientBoostingClassifier()
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor()
        if spec == "random_forest":
            unique = np.unique(y[~np.isnan(y)])
            if set(unique.tolist()).issubset({0.0, 1.0}) and len(unique) <= 2:
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(n_estimators=200, n_jobs=-1)
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(n_estimators=200, n_jobs=-1)
        raise ValueError(f"Unknown outcome_model {spec!r}; expected one of {_OUTCOME_STRINGS}.")

    from sklearn.base import clone

    return clone(spec)


def predict_mean(model, X: np.ndarray) -> np.ndarray:
    """Return mean predictions: ``predict_proba(...)[:, 1]`` for classifiers, ``predict(...)`` for regressors."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def fit_propensity(
    model_spec, X: np.ndarray, T: np.ndarray, *, clip: tuple[float, float] | None
) -> tuple[np.ndarray, object]:
    """Fit a propensity model and return (clipped scores, fitted model)."""
    model = resolve_propensity_model(model_spec)
    model.fit(X, T.astype(int))
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Propensity model {type(model).__name__} must implement predict_proba; got a regressor.")
    ps = model.predict_proba(X)[:, 1]
    if clip is not None:
        lo, hi = clip
        ps = np.clip(ps, lo, hi)
    return ps, model
