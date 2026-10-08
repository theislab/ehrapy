from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from array_api_compat import array_namespace
from fast_array_utils.conv import to_dense
from fast_array_utils.types import DaskArray
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ehrapy._compat import _like_obs, _map_observation_blocks, _materialize
from ehrapy.preprocessing._summarize_measurements import summarize_measurements

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ehrdata import EHRData

    type Array = np.ndarray | DaskArray
    type Statistic = Literal["min", "max", "mean", "median", "first", "last"]


@dataclass(frozen=True)
class Task:
    """A prediction target and the timepoints its features are computed from.

    For longitudinal data, the features summarize the `observation_window` timepoints that end `gap` timepoints before `prediction_time`.

    Examples:
        >>> import ehrapy as ep
        >>> task = ep.ml.Task("In-hospital_death", prediction_time=24, gap=2)
    """

    #: Column of `obs` with the label.
    label: str
    _: KW_ONLY
    #: Whether the label is binary or a number to regress.
    kind: Literal["binary", "regression"] = "binary"
    #: Position along `tem` at which the prediction is made, or `None` for the end of the recorded timepoints.
    prediction_time: int | None = None
    #: Number of timepoints the features summarize, or `None` for all timepoints until the gap.
    observation_window: int | None = None
    #: Number of timepoints right before the prediction time that the features leave out.
    gap: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"binary", "regression"}:
            raise ValueError(f"`kind` must be 'binary' or 'regression', got {self.kind!r}.")

    def _window(self, n_timepoints: int) -> slice:
        end = (n_timepoints if self.prediction_time is None else self.prediction_time) - self.gap
        start = 0 if self.observation_window is None else end - self.observation_window
        if not 0 <= start < end <= n_timepoints:
            raise ValueError(f"The timepoints {start} to {end} of {self} are not within the {n_timepoints} timepoints.")
        return slice(start, end)


@dataclass(frozen=True)
class Predictor:
    """A model fitted by :func:`~ehrapy.ml.fit` with everything needed to compute its features."""

    #: The prediction task.
    task: Task
    #: The fitted imputation, scaling and model steps.
    pipeline: Pipeline
    #: Variables the features are computed from.
    var_names: list[str]
    #: Columns of `obs` used as features.
    obs_keys: list[str]
    #: Layer the variables are read from, or `None` for `.X`.
    layer: str | None
    #: Statistics that summarize longitudinal variables.
    statistics: list[Literal["min", "max", "mean", "median", "first", "last"]]
    #: Names of the features in the order the pipeline receives them.
    feature_names: list[str]


def fit(
    edata: EHRData,
    task: Task,
    *,
    model: Literal["logistic", "linear", "gradient_boosting"] | BaseEstimator = "gradient_boosting",
    var_names: Iterable[str] | None = None,
    obs_keys: Iterable[str] = (),
    layer: str | None = None,
    statistics: Iterable[Literal["min", "max", "mean", "median", "first", "last"]] = ("min", "max", "mean"),
    split_key: str = "split",
    max_train_obs: int | None = 10_000,
    random_state: int = 0,
) -> Predictor:
    """Fit a model that predicts the label of a task from variables and `obs` columns.

    Longitudinal variables are summarized over the observation window of the task with :func:`~ehrapy.preprocessing.summarize_measurements`.
    Missing values are imputed with the median and features are standardized, with both steps fit on the training set only, like the model.

    Args:
        edata: Central data object.
        task: The prediction task.
        model: `"logistic"` for a logistic regression of binary labels, `"linear"` for a ridge regression of numeric labels, `"gradient_boosting"` for gradient boosted trees, or any scikit-learn estimator.
        var_names: Variables to compute features from.
            If `None`, all variables except the label are used.
        obs_keys: Columns of `obs` to use as features, with categorical columns one-hot encoded.
        layer: Layer to read the variables from.
            If `None`, `.X` is used.
        statistics: Statistics that summarize every longitudinal variable over the observation window.
        split_key: Column of `obs` with the sets from :func:`~ehrapy.ml.split`.
            The model is fit on the observations in `"train"` with a label.
        max_train_obs: Maximum number of randomly chosen training observations the model is fit on.
            If `None`, all training observations are used.
        random_state: Seed for choosing the training observations and for the built-in models.

    Returns:
        The fitted model, to pass to :func:`~ehrapy.ml.predict`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.physionet2012()
        >>> ep.ml.split(edata, stratify="In-hospital_death")
        >>> task = ep.ml.Task("In-hospital_death", prediction_time=24)
        >>> predictor = ep.ml.fit(edata, task, obs_keys=["Age", "Gender", "ICUType"])
    """
    if split_key not in edata.obs:
        raise KeyError(f"`edata.obs` has no column {split_key!r}. Split the observations first with `ep.ml.split`.")
    var_names = [var for var in edata.var_names if var != task.label] if var_names is None else list(var_names)
    obs_keys = list(obs_keys)
    if task.label in var_names or task.label in obs_keys:
        raise ValueError(f"The label {task.label!r} cannot be a feature.")
    labels = edata.obs[task.label]
    if task.kind == "binary" and labels.nunique() != 2:
        raise ValueError(f"A binary task needs exactly two values of {task.label!r}, got {labels.nunique()}.")

    estimator = _estimator(model, task.kind, random_state)
    if is_classifier(estimator) != (task.kind == "binary"):
        raise ValueError(f"A {task.kind} task needs a {'classifier' if task.kind == 'binary' else 'regressor'}.")

    train = np.flatnonzero((edata.obs[split_key] == "train").to_numpy() & labels.notna().to_numpy())
    if max_train_obs is not None and len(train) > max_train_obs:
        train = np.sort(np.random.default_rng(random_state).choice(train, max_train_obs, replace=False))
    statistics = list(statistics)
    features, feature_names = _features(edata, task, var_names, obs_keys, layer, statistics)
    train_features = _materialize(features[train])[0]
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
    pipeline.fit(train_features, labels.to_numpy()[train])
    return Predictor(task, pipeline, var_names, obs_keys, layer, statistics, feature_names)


def predict(
    edata: EHRData,
    predictor: Predictor,
    *,
    key_added: str = "prediction",
    copy: bool = False,
) -> EHRData | None:
    """Predict the label of every observation with a fitted model.

    Args:
        edata: Central data object with the variables and `obs` columns the model was fit on.
        predictor: The model fitted by :func:`~ehrapy.ml.fit`.
        key_added: Column of `obs` to store the predictions in.
        copy: Whether to return a copy of `edata` instead of modifying it in place.

    Returns:
        ``None`` if ``copy=False``, otherwise the updated data object.
        The predictions are stored in `edata.obs[key_added]`, as the probability of the larger of the two label values, such as `1` or `True`, for binary tasks.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.physionet2012()
        >>> ep.ml.split(edata, stratify="In-hospital_death")
        >>> predictor = ep.ml.fit(edata, ep.ml.Task("In-hospital_death"))
        >>> ep.ml.predict(edata, predictor)
    """
    if copy:
        edata = edata.copy()
    features, _ = _features(
        edata,
        predictor.task,
        predictor.var_names,
        predictor.obs_keys,
        predictor.layer,
        predictor.statistics,
        feature_names=predictor.feature_names,
    )
    edata.obs[key_added] = _materialize(_predict(features, predictor.pipeline))[0]
    return edata if copy else None


def _estimator(model: str | BaseEstimator, kind: str, random_state: int) -> BaseEstimator:
    match kind, model:
        case "binary", "logistic":
            return LogisticRegression(max_iter=1000)
        case "binary", "gradient_boosting":
            return HistGradientBoostingClassifier(random_state=random_state)
        case "regression", "linear":
            return Ridge()
        case "regression", "gradient_boosting":
            return HistGradientBoostingRegressor(random_state=random_state)
        case _, str():
            raise ValueError(f"Unknown `model` {model!r} for {kind} tasks.")
    return clone(model)


def _features(
    edata: EHRData,
    task: Task,
    var_names: Sequence[str],
    obs_keys: Sequence[str],
    layer: str | None,
    statistics: Sequence[Statistic],
    *,
    feature_names: Sequence[str] | None = None,
) -> tuple[Array, list[str]]:
    """Dense features of every observation and their names, aligning one-hot encoded `obs` columns with `feature_names`."""
    X = edata.X if layer is None else edata.layers[layer]
    if X.ndim == 3:
        summary = summarize_measurements(
            edata[:, :, task._window(X.shape[2])], layer=layer, var_names=var_names, statistics=statistics
        )
        values, names = summary.X, list(summary.var_names)
    elif task.prediction_time is not None or task.observation_window is not None or task.gap:
        raise ValueError("The timepoints of a task only apply to longitudinal data.")
    else:
        view = edata[:, list(var_names)]
        values, names = (view.X if layer is None else view.layers[layer]), list(var_names)

    covariates = pd.get_dummies(edata.obs[list(obs_keys)]) if obs_keys else edata.obs[[]]
    if feature_names is not None:
        covariates = covariates.reindex(columns=feature_names[len(names) :], fill_value=0)
    values = to_dense(values)
    xp = array_namespace(values)
    features = xp.concat([xp.astype(values, xp.float64), _like_obs(values, covariates.to_numpy(np.float64))], axis=1)
    return features, [*names, *covariates.columns]


@singledispatch
def _predict(features: np.ndarray, pipeline: Pipeline) -> np.ndarray:
    return pipeline.predict_proba(features)[:, 1] if is_classifier(pipeline) else pipeline.predict(features)


@_predict.register(DaskArray)
def _(features: DaskArray, pipeline: Pipeline) -> DaskArray:
    return _map_observation_blocks(features, _predict, pipeline, drop_axis=1, meta=np.array((), dtype=np.float64))
