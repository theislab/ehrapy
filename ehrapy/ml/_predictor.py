from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Literal

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
from ehrapy.ml._deep import DeepModel, _deep_model, _FittedModel
from ehrapy.ml._features import _features, _sequences, _times
from ehrapy.ml._task import Kind, Task, _targets

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable

    from ehrdata import EHRData
    from torch import nn

    from ehrapy.preprocessing._summarize_measurements import Statistic


@dataclass(frozen=True)
class Predictor:
    """A model fitted by :func:`~ehrapy.ml.fit` with everything needed to compute its features."""

    #: The prediction task.
    task: Task
    #: The fitted imputation and scaling of summarized features, or `None` for models of time series.
    preprocessing: Pipeline | None
    #: The fitted scikit-learn estimator or deep learning model.
    model: Any
    #: Variables the features are computed from.
    var_names: list[str]
    #: Columns of `obs` used as features.
    obs_keys: list[str]
    #: Layer the variables are read from, or `None` for `.X`.
    layer: str | None
    #: Statistics that summarize longitudinal variables.
    statistics: list[Statistic]
    #: Names of the features in the order the model receives them.
    feature_names: list[str]
    #: Classes of classification tasks or labels of multilabel tasks, in the order of the predicted probabilities.
    classes: list[Hashable]
    #: Calibration of the predicted probabilities from :func:`~ehrapy.ml.calibrate`, or `None`.
    calibrator: Callable[[np.ndarray], np.ndarray] | None = None
    #: Quantile of the nonconformity scores from :func:`~ehrapy.ml.conformalize`, or `None`.
    conformal: float | None = None
    #: Column of `tem` with the time of every timepoint.
    time_key: str = "interval_start_offset"


def fit(
    edata: EHRData,
    task: Task,
    *,
    model: Literal[
        "logistic",
        "linear",
        "gradient_boosting",
        "random_forest",
        "cox",
        "mlp",
        "gru",
        "lstm",
        "grud",
        "tcn",
        "transformer",
        "retain",
    ]
    | BaseEstimator
    | DeepModel
    | nn.Module
    | None = None,
    var_names: Iterable[str] | None = None,
    obs_keys: Iterable[str] = (),
    layer: str | None = None,
    statistics: Iterable[Statistic] = ("min", "max", "mean"),
    split_key: str = "split",
    time_key: str = "interval_start_offset",
    max_train_obs: int | None = 10_000,
    random_state: int = 0,
) -> Predictor:
    """Fit a model that predicts the targets of a task from variables and `obs` columns.

    Most models read the longitudinal variables summarized over the observation window of the task with :func:`~ehrapy.preprocessing.summarize_measurements`.
    For them, missing values are imputed with the median and features are standardized, with both steps fit on the training set only, like the model.
    Models of time series read the variables at every timepoint of the observation window, whether they were observed and the time since their last observation, standardized with statistics of the training set.
    Deep learning models stop training once their loss on the `"tuning"` set stops improving.
    For rolling tasks, models of summaries are fit on every labeled timepoint, and models of time series except RETAIN predict every timepoint from all earlier ones.

    The models are

    - `"logistic"`: a logistic regression for binary, multiclass and multilabel tasks,
    - `"linear"`: a ridge regression for regression tasks,
    - `"gradient_boosting"`: gradient boosted trees for every task except survival,
    - `"random_forest"`: a random forest for every task except survival,
    - `"cox"`: a Cox proportional hazards model for survival tasks,
    - `"mlp"`: a multilayer perceptron as in :class:`~ehrapy.ml.MLP`,
    - `"gru"`, `"lstm"`, `"grud"`, `"tcn"`, `"transformer"` and `"retain"`: models of time series as in :class:`~ehrapy.ml.GRU`, :class:`~ehrapy.ml.LSTM`, :class:`~ehrapy.ml.GRUD`, :class:`~ehrapy.ml.TCN`, :class:`~ehrapy.ml.Transformer` and :class:`~ehrapy.ml.RETAIN`.

    Deep learning models fit every kind of task and need PyTorch, which `pip install 'ehrapy[ml]'` installs.

    Args:
        edata: Central data object.
        task: The prediction task.
        model: Name of the model, a scikit-learn estimator, a configured deep learning model, or a torch module.
            An estimator for survival tasks is fit on the time and whether the event occurred and predicts a risk score.
            A torch module is a model of time series that maps the `values`, whether they were observed (`mask`) and the time since their last observation (`time_since_observed`), each of shape `(observations, timepoints, variables)`, and the `static` covariates of shape `(observations, covariates)` to an embedding of shape `(observations, features)`, or to an embedding and the attention to every timepoint.
            For rolling tasks, it returns an embedding of every timepoint of shape `(observations, timepoints, features)` that only reads earlier timepoints.
            If `None`, gradient boosting, or the Cox model for survival tasks.
        var_names: Variables to compute features from.
            If `None`, all variables except the targets are used.
        obs_keys: Columns of `obs` to use as features, with categorical columns one-hot encoded.
        layer: Layer to read the variables from.
            If `None`, `.X` is used.
        statistics: Statistics that summarize every longitudinal variable over the observation window, see :func:`~ehrapy.preprocessing.summarize_measurements`.
        split_key: Column of `obs` with the sets from :func:`~ehrapy.ml.split`.
            The model is fit on the observations in `"train"` with all targets.
        time_key: Column of `tem` with the time of every timepoint, as numbers, time differences or dates, from which models of time series compute the time since the last observation.
            If `tem` has no such column, the timepoints are evenly spaced.
        max_train_obs: Maximum number of randomly chosen training and tuning observations the model is fit on.
            If `None`, all are used.
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
    targets, classes = _targets(edata, task)
    if model is None:
        model = "cox" if task.kind == "survival" else "gradient_boosting"
    deep = _deep_model(model)
    sequential = deep is not None and deep.sequential
    n_outputs = len(classes) if task.kind in {"multiclass", "multilabel"} else 1

    rng = np.random.default_rng(random_state)
    labeled = np.ones(edata.n_obs, dtype=bool)
    if not task.rolling:
        labeled = ~np.isnan(targets.reshape(len(targets), -1)).any(axis=1)
    train, tuning = (
        _sample(np.flatnonzero((edata.obs[split_key] == split).to_numpy() & labeled), max_train_obs, rng)
        for split in ("train", "tuning")
    )
    statistics = list(statistics)
    if sequential:
        features, feature_names = _sequences(edata, task, var_names, obs_keys, layer)
    else:
        features, feature_names = _features(edata, task, var_names, obs_keys, layer, statistics)
    train_features, tuning_features, train_targets, tuning_targets = _materialize(
        features[train], features[tuning], targets[train], targets[tuning]
    )
    if task.rolling and task.kind == "binary" and not np.isin(train_targets[~np.isnan(train_targets)], (0, 1)).all():
        raise ValueError(f"A rolling binary task marks positive timepoints of {task.label!r} with 1 and others with 0.")
    if task.rolling and not sequential:
        train_features, train_targets = _samples(train_features, train_targets, task.gap)
        tuning_features, tuning_targets = _samples(tuning_features, tuning_targets, task.gap)

    preprocessing = None
    if not sequential:
        preprocessing = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(train_features)
        train_features, tuning_features = (
            preprocessing.transform(train_features),
            preprocessing.transform(tuning_features),
        )
    if deep is None:
        fitted = _estimator(model, task.kind, random_state).fit(train_features, train_targets)
    else:
        fitted = deep._fit(
            train_features,
            train_targets,
            task=task,
            n_outputs=n_outputs,
            n_static=len(feature_names) - len(var_names) if sequential else train_features.shape[1],
            tuning=(tuning_features, tuning_targets),
            times=_times(edata, task, layer, time_key) if sequential else np.empty(0),
            random_state=random_state,
        )
    return Predictor(
        task, preprocessing, fitted, var_names, obs_keys, layer, statistics, feature_names, classes, time_key=time_key
    )


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
        Models from :func:`~ehrapy.ml.conformalize` store prediction sets of classes in `edata.obsm[f"{key_added}_set"]` and prediction intervals in `edata.obs[f"{key_added}_lower"]` and `edata.obs[f"{key_added}_upper"]`.
        Rolling tasks store the prediction for every timepoint in `edata.obsm[key_added]`, missing at timepoints without an observation window, and their prediction sets and intervals in `edata.obsm` as well.
        Deep learning models store the embedding of every observation in `edata.obsm[f"X_{key_added}"]`, and the transformer and RETAIN the attention to every timepoint in `edata.obsm[f"{key_added}_attention"]`.

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
    outputs, embedding, attention = _outputs_of(edata, predictor)
    outputs = _calibrated(predictor, outputs)
    kind = predictor.task.kind
    if predictor.task.rolling:
        edata.obsm[key_added] = outputs
        if predictor.conformal is not None and kind == "regression":
            edata.obsm[f"{key_added}_lower"] = outputs - predictor.conformal
            edata.obsm[f"{key_added}_upper"] = outputs + predictor.conformal
        elif predictor.conformal is not None:
            edata.obsm[f"{key_added}_set"] = np.stack([1 - outputs, outputs], axis=2) >= 1 - predictor.conformal
        return edata if copy else None
    if kind in {"multiclass", "multilabel"}:
        columns = [str(c) for c in predictor.classes]
        edata.obsm[key_added] = pd.DataFrame(outputs, index=edata.obs_names, columns=columns)
    if kind == "multiclass":
        classes = np.asarray(predictor.classes, dtype=object)
        edata.obs[key_added] = pd.Categorical(classes[outputs.argmax(axis=1)], categories=predictor.classes)
    elif kind != "multilabel":
        edata.obs[key_added] = outputs[:, 0]
    if predictor.conformal is not None and kind == "regression":
        edata.obs[f"{key_added}_lower"] = outputs[:, 0] - predictor.conformal
        edata.obs[f"{key_added}_upper"] = outputs[:, 0] + predictor.conformal
    elif predictor.conformal is not None:
        columns = [str(c) for c in predictor.classes]
        edata.obsm[f"{key_added}_set"] = pd.DataFrame(
            _class_probabilities(outputs) >= 1 - predictor.conformal, index=edata.obs_names, columns=columns
        )
    if embedding.shape[1]:
        edata.obsm[f"X_{key_added}"] = embedding
    if attention.shape[1]:
        X = edata.X if predictor.layer is None else edata.layers[predictor.layer]
        timepoints = edata.tem.index[predictor.task._window(X.shape[2])].astype(str)
        edata.obsm[f"{key_added}_attention"] = pd.DataFrame(attention, index=edata.obs_names, columns=timepoints)
    return edata if copy else None


def _features_of(edata: EHRData, predictor: Predictor) -> Any:
    """Features of every observation as the model of `predictor` receives them before preprocessing."""
    if predictor.preprocessing is None:
        return _sequences(
            edata,
            predictor.task,
            predictor.var_names,
            predictor.obs_keys,
            predictor.layer,
            feature_names=predictor.feature_names,
        )[0]
    return _features(
        edata,
        predictor.task,
        predictor.var_names,
        predictor.obs_keys,
        predictor.layer,
        predictor.statistics,
        feature_names=predictor.feature_names,
    )[0]


def _outputs_of(edata: EHRData, predictor: Predictor, features: Any = None) -> list[np.ndarray]:
    """Predictions, with a column for every class of multiclass and every label of multilabel tasks, embeddings and attention of every observation."""
    features = _features_of(edata, predictor) if features is None else features
    kind = predictor.task.kind
    n_outputs = len(predictor.classes) if kind in {"multiclass", "multilabel"} else 1
    widths = [n_outputs, 0, 0]
    if predictor.task.rolling:
        widths[0] = features.shape[2]
    elif isinstance(predictor.model, _FittedModel):
        widths[1:] = predictor.model.n_embedding, predictor.model.n_attention
    times = np.empty(0)
    if predictor.preprocessing is None:
        times = _times(edata, predictor.task, predictor.layer, predictor.time_key)
    outputs = _outputs(features, predictor.preprocessing, predictor.model, kind, n_outputs, sum(widths), times)
    outputs = _materialize(outputs)[0]
    if predictor.task.rolling:
        outputs[:, : predictor.task.gap + 1] = np.nan
    return np.split(outputs, np.cumsum(widths)[:-1], axis=1)


def _calibrated(predictor: Predictor, outputs: np.ndarray) -> np.ndarray:
    if predictor.calibrator is None:
        return outputs
    if not predictor.task.rolling:
        return predictor.calibrator(outputs)
    predicted = ~np.isnan(outputs)
    calibrated = outputs.copy()
    calibrated[predicted] = predictor.calibrator(outputs[predicted][:, None])[:, 0]
    return calibrated


def _held_out(edata: EHRData, predictor: Predictor, *, split_key: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the observations in `split` with all targets, or with any target of rolling tasks, and their targets."""
    targets, _ = _targets(edata, predictor.task)
    in_split = (edata.obs[split_key] == split).to_numpy()
    if predictor.task.rolling:
        rows = np.flatnonzero(in_split)
        targets = _materialize(targets[rows])[0]
    else:
        rows = np.flatnonzero(in_split & ~np.isnan(targets.reshape(len(targets), -1)).any(axis=1))
        targets = targets[rows]
    if not len(rows):
        raise ValueError(f"No observations with targets in the {split!r} set of `edata.obs[{split_key!r}]`.")
    return rows, targets


def _paired(task: Task, targets: np.ndarray, outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Targets and predictions of every sample, which are the labeled and predicted timepoints of rolling tasks."""
    if not task.rolling:
        return targets, outputs
    paired = ~np.isnan(targets) & ~np.isnan(outputs)
    return targets[paired], outputs[paired][:, None]


def _samples(features: np.ndarray, targets: np.ndarray, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Features and targets of the labeled timepoints of a rolling task that lie more than `gap` timepoints after the first."""
    labeled = ~np.isnan(targets) & (np.arange(targets.shape[1]) > gap)
    return np.moveaxis(features, 2, 1)[labeled], targets[labeled]


def _class_probabilities(outputs: np.ndarray) -> np.ndarray:
    """Probabilities of every class, with both classes of binary predictions."""
    return np.column_stack([1 - outputs, outputs]) if outputs.shape[1] == 1 else outputs


def _sample(rows: np.ndarray, max_obs: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_obs is None or len(rows) <= max_obs:
        return rows
    return np.sort(rng.choice(rows, max_obs, replace=False))


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
def _outputs(
    features: np.ndarray,
    preprocessing: Pipeline | None,
    model: Any,
    kind: Kind,
    n_outputs: int,
    n_columns: int,
    times: np.ndarray,
) -> np.ndarray:
    if preprocessing is not None and features.ndim == 3:
        samples = np.moveaxis(features, 2, 1).reshape(-1, features.shape[1])
        return _outputs(samples, preprocessing, model, kind, n_outputs, 1, times)[:, 0].reshape(len(features), -1)
    if preprocessing is not None:
        features = preprocessing.transform(features)
    if isinstance(model, _FittedModel):
        return model.outputs(features, times)
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
def _(
    features: DaskArray,
    preprocessing: Pipeline | None,
    model: Any,
    kind: Kind,
    n_outputs: int,
    n_columns: int,
    times: np.ndarray,
) -> DaskArray:
    return _map_observation_blocks(
        features,
        _outputs,
        preprocessing,
        model,
        kind,
        n_outputs,
        n_columns,
        times,
        chunks=(features.chunks[0], (n_columns,)),
        drop_axis=2 if features.ndim == 3 else [],
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
