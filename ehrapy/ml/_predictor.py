from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from fast_array_utils.types import DaskArray
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ehrapy._compat import _map_observation_blocks, _materialize
from ehrapy._settings import settings
from ehrapy.ml._features import _features
from ehrapy.ml._task import Kind, Task, _targets

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

    from ehrdata import EHRData


@dataclass(frozen=True)
class Predictor:
    """A model fitted by :func:`~ehrapy.ml.fit` with everything needed to compute its features."""

    #: The prediction task.
    task: Task
    #: The fitted imputation, scaling and model steps.
    model: Pipeline
    #: Variables the features are computed from.
    var_names: list[str]
    #: Columns of `obs` used as features.
    obs_keys: list[str]
    #: Layer the variables are read from, or `None` for `.X`.
    layer: str | None
    #: Statistics that summarize longitudinal variables.
    statistics: list[Literal["min", "max", "mean", "median", "first", "last"]]
    #: Names of the features in the order the model receives them.
    feature_names: list[str]
    #: Classes of classification tasks or labels of multilabel tasks, in the order of the predicted probabilities.
    classes: list[Hashable]


def fit(
    edata: EHRData,
    task: Task,
    *,
    model: Literal["logistic", "linear", "gradient_boosting", "random_forest", "cox"] | BaseEstimator | None = None,
    var_names: Iterable[str] | None = None,
    obs_keys: Iterable[str] = (),
    layer: str | None = None,
    statistics: Iterable[Literal["min", "max", "mean", "median", "first", "last"]] = ("min", "max", "mean"),
    split_key: str = "split",
    max_train_obs: int | None = 10_000,
    random_state: int = 0,
) -> Predictor:
    """Fit a model that predicts the targets of a task from variables and `obs` columns.

    Longitudinal variables are summarized over the observation window of the task with :func:`~ehrapy.preprocessing.summarize_measurements`.
    Missing values are imputed with the median and features are standardized, with both steps fit on the training set only, like the model.

    The models are

    - `"logistic"`: a logistic regression for binary, multiclass and multilabel tasks,
    - `"linear"`: a ridge regression for regression tasks,
    - `"gradient_boosting"`: gradient boosted trees for every task except survival,
    - `"random_forest"`: a random forest for every task except survival,
    - `"cox"`: a Cox proportional hazards model for survival tasks.

    Args:
        edata: Central data object.
        task: The prediction task.
        model: Name of the model or a scikit-learn estimator.
            An estimator for survival tasks is fit on the time and whether the event occurred and predicts a risk score.
            If `None`, gradient boosting, or the Cox model for survival tasks.
        var_names: Variables to compute features from.
            If `None`, all variables except the targets are used.
        obs_keys: Columns of `obs` to use as features, with categorical columns one-hot encoded.
        layer: Layer to read the variables from.
            If `None`, `.X` is used.
        statistics: Statistics that summarize every longitudinal variable over the observation window.
        split_key: Column of `obs` with the sets from :func:`~ehrapy.ml.split`.
            The model is fit on the observations in `"train"` with all targets.
        max_train_obs: Maximum number of randomly chosen training observations the model is fit on.
            If `None`, all training observations are used.
        random_state: Seed for choosing the training observations and for the built-in models.

    Returns:
        The fitted model, to pass to :func:`~ehrapy.ml.predict`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_observations=200, n_centers=2, base_timepoints=10)
        >>> ep.ml.split(edata, stratify="cluster")
        >>> task = ep.ml.Task("cluster", prediction_time=6)
        >>> predictor = ep.ml.fit(edata, task, model="logistic")
    """
    if split_key not in edata.obs:
        raise KeyError(f"`edata.obs` has no column {split_key!r}. Split the observations first with `ep.ml.split`.")
    var_names = [var for var in edata.var_names if var not in task._columns] if var_names is None else list(var_names)
    obs_keys = list(obs_keys)
    if leaked := set(task._columns) & {*var_names, *obs_keys}:
        raise ValueError(f"The targets {sorted(leaked)} cannot be features.")
    targets, classes = _targets(edata.obs, task)
    if model is None:
        model = "cox" if task.kind == "survival" else "gradient_boosting"
    estimator = _estimator(model, task.kind, random_state)

    labeled = ~np.isnan(targets.reshape(len(targets), -1)).any(axis=1)
    train = np.flatnonzero((edata.obs[split_key] == "train").to_numpy() & labeled)
    if max_train_obs is not None and len(train) > max_train_obs:
        train = np.sort(np.random.default_rng(random_state).choice(train, max_train_obs, replace=False))
    statistics = list(statistics)
    features, feature_names = _features(edata, task, var_names, obs_keys, layer, statistics)
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
    pipeline.fit(_materialize(features[train])[0], targets[train])
    return Predictor(task, pipeline, var_names, obs_keys, layer, statistics, feature_names, classes)


def predict(
    edata: EHRData,
    predictor: Predictor,
    *,
    key_added: str = "prediction",
    copy: bool = False,
) -> EHRData | None:
    """Predict the targets of every observation with a fitted model.

    Args:
        edata: Central data object with the variables and `obs` columns the model was fit on.
        predictor: The model fitted by :func:`~ehrapy.ml.fit`.
        key_added: Key to store the predictions under.
        copy: Whether to return a copy of `edata` instead of modifying it in place.

    Returns:
        ``None`` if ``copy=False``, otherwise the updated data object.
        The predictions are stored in `edata.obs[key_added]`.
        They are the probability of the larger of the two label values, such as `1` or `True`, for binary tasks, the most probable class for multiclass tasks, the predicted value for regression tasks and a risk score, which is higher for earlier events, for survival tasks.
        The probability of every class of multiclass tasks and of every label of multilabel tasks is stored in `edata.obsm[key_added]`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_observations=200, n_centers=2, base_timepoints=10)
        >>> ep.ml.split(edata, stratify="cluster")
        >>> predictor = ep.ml.fit(edata, ep.ml.Task("cluster"))
        >>> ep.ml.predict(edata, predictor)
    """
    if copy:
        edata = edata.copy()
    outputs = _predictions(edata, predictor)
    kind = predictor.task.kind
    if kind in {"multiclass", "multilabel"}:
        columns = [str(c) for c in predictor.classes]
        edata.obsm[key_added] = pd.DataFrame(outputs, index=edata.obs_names, columns=columns)
    if kind == "multiclass":
        classes = np.asarray(predictor.classes, dtype=object)
        edata.obs[key_added] = pd.Categorical(classes[outputs.argmax(axis=1)], categories=predictor.classes)
    elif kind != "multilabel":
        edata.obs[key_added] = outputs[:, 0]
    return edata if copy else None


def _predictions(edata: EHRData, predictor: Predictor) -> np.ndarray:
    """Predictions of every observation, with a column for every class of multiclass and every label of multilabel tasks."""
    features, _ = _features(
        edata,
        predictor.task,
        predictor.var_names,
        predictor.obs_keys,
        predictor.layer,
        predictor.statistics,
        feature_names=predictor.feature_names,
    )
    n_outputs = len(predictor.classes) if predictor.task.kind in {"multiclass", "multilabel"} else 1
    return _materialize(_outputs(features, predictor.model, predictor.task.kind, n_outputs))[0]


def _estimator(model: str | BaseEstimator, kind: Kind, random_state: int) -> BaseEstimator:
    if not isinstance(model, str):
        return clone(model)
    match "binary" if kind in {"multiclass", "multilabel"} else kind, model:
        case "binary", "logistic":
            estimator = LogisticRegression(max_iter=1000)
        case "binary", "gradient_boosting":
            estimator = HistGradientBoostingClassifier(random_state=random_state)
        case "binary", "random_forest":
            estimator = RandomForestClassifier(n_jobs=settings.n_jobs, random_state=random_state)
        case "regression", "linear":
            estimator = Ridge()
        case "regression", "gradient_boosting":
            estimator = HistGradientBoostingRegressor(random_state=random_state)
        case "regression", "random_forest":
            estimator = RandomForestRegressor(n_jobs=settings.n_jobs, random_state=random_state)
        case "survival", "cox":
            estimator = _CoxPH()
        case _:
            raise ValueError(f"Unknown `model` {model!r} for {kind} tasks.")
    return MultiOutputClassifier(estimator) if kind == "multilabel" else estimator


@singledispatch
def _outputs(features: np.ndarray, model: Pipeline, kind: Kind, n_outputs: int) -> np.ndarray:
    match kind:
        case "binary":
            return model.predict_proba(features)[:, 1:]
        case "multiclass":
            outputs = np.zeros((len(features), n_outputs))
            outputs[:, model.classes_.astype(int)] = model.predict_proba(features)
            return outputs
        case "multilabel":
            probabilities = model.predict_proba(features)
            return (
                np.column_stack([p[:, 1] for p in probabilities]) if isinstance(probabilities, list) else probabilities
            )
    return model.predict(features).reshape(len(features), -1)


@_outputs.register(DaskArray)
def _(features: DaskArray, model: Pipeline, kind: Kind, n_outputs: int) -> DaskArray:
    return _map_observation_blocks(
        features,
        _outputs,
        model,
        kind,
        n_outputs,
        chunks=(features.chunks[0], (n_outputs,)),
        meta=np.empty((0, 0), dtype=np.float64),
    )


class _CoxPH(BaseEstimator):
    """Penalized Cox proportional hazards model fit on the time and whether the event occurred."""

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer

    def fit(self, X: np.ndarray, y: np.ndarray) -> _CoxPH:
        from lifelines import CoxPHFitter

        data = self._frame(X).assign(duration=y[:, 0], event=y[:, 1])
        self.model_ = CoxPHFitter(penalizer=self.penalizer).fit(data, duration_col="duration", event_col="event")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_log_partial_hazard(self._frame(X)).to_numpy()

    @staticmethod
    def _frame(X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
