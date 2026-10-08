from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from ehrapy.ml._evaluate import _log_odds
from ehrapy.ml._predictor import _calibrated, _class_probabilities, _held_out, _outputs_of, _paired

if TYPE_CHECKING:
    from collections.abc import Callable

    from ehrdata import EHRData

    from ehrapy.ml._predictor import Predictor


def calibrate(
    edata: EHRData,
    predictor: Predictor,
    *,
    method: Literal["platt", "isotonic", "temperature"] = "isotonic",
    split_key: str = "split",
    split: str = "tuning",
) -> Predictor:
    """Calibrate the predicted probabilities of a fitted model on held-out observations.

    Platt scaling fits a logistic regression to the log-odds of binary predictions, isotonic regression fits a monotonic function to them, and temperature scaling divides the log-odds or log-probabilities by a single temperature, which keeps the order of predictions and also applies to multiclass tasks.

    Args:
        edata: Central data object.
        predictor: The model fitted by :func:`~ehrapy.ml.fit`.
        method: How to calibrate.
        split_key: Column of `obs` with the sets from :func:`~ehrapy.ml.split`.
        split: The set to calibrate on, which the model must not have been fit on.

    Returns:
        The model with calibrated predictions and without conformal prediction, which :func:`~ehrapy.ml.conformalize` adds again.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_observations=200, n_centers=2, base_timepoints=10)
        >>> ep.ml.split(edata, stratify="cluster")
        >>> predictor = ep.ml.fit(edata, ep.ml.Task("cluster"))
        >>> predictor = ep.ml.calibrate(edata, predictor, method="temperature")
        >>> ep.ml.predict(edata, predictor)
    """
    kind = predictor.task.kind
    if kind not in {"binary", "multiclass"} or (kind == "multiclass" and method != "temperature"):
        raise ValueError(f"{method!r} calibration does not apply to {kind} tasks.")
    rows, y = _held_out(edata, predictor, split_key=split_key, split=split)
    y, outputs = _paired(predictor.task, y, _outputs_of(edata[rows], predictor)[0])
    calibrator: Callable[[np.ndarray], np.ndarray]
    match method:
        case "platt":
            calibrator = _Platt(LogisticRegression().fit(_log_odds(outputs), y))
        case "isotonic":
            calibrator = _Isotonic(IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(outputs[:, 0], y))
        case "temperature":
            labels = np.arange(max(outputs.shape[1], 2))

            def negative_log_likelihood(temperature: float) -> float:
                return log_loss(y, _Temperature(temperature)(outputs), labels=labels)

            calibrator = _Temperature(minimize_scalar(negative_log_likelihood, bounds=(0.05, 20), method="bounded").x)
        case _:
            raise ValueError(f"Unknown `method` {method!r}.")
    return replace(predictor, calibrator=calibrator, conformal=None)


def conformalize(
    edata: EHRData,
    predictor: Predictor,
    *,
    alpha: float = 0.1,
    split_key: str = "split",
    split: str = "tuning",
) -> Predictor:
    """Add split conformal prediction sets for classification and intervals for regression to a fitted model.

    The sets and intervals contain the true class or value of a new observation with a probability of at least `1 - alpha`, if the observations are exchangeable with the ones they are computed on.
    Sets contain every class whose probability is at least 1 minus the `1 - alpha` quantile of 1 minus the probability of the true class, and intervals reach as far as the quantile of the absolute errors.

    Args:
        edata: Central data object.
        predictor: The model fitted by :func:`~ehrapy.ml.fit` and possibly calibrated by :func:`~ehrapy.ml.calibrate`.
        alpha: Probability that the true class or value lies outside the set or interval.
        split_key: Column of `obs` with the sets from :func:`~ehrapy.ml.split`.
        split: The set to compute the quantile on, which the model must not have been fit on.

    Returns:
        The model, which :func:`~ehrapy.ml.predict` now also stores prediction sets or intervals of.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_observations=200, n_centers=3, base_timepoints=10)
        >>> ep.ml.split(edata, stratify="cluster")
        >>> predictor = ep.ml.fit(edata, ep.ml.Task("cluster", kind="multiclass"))
        >>> predictor = ep.ml.conformalize(edata, predictor, alpha=0.1)
        >>> ep.ml.predict(edata, predictor)
        >>> edata.obsm["prediction_set"].head()
    """
    kind = predictor.task.kind
    if kind not in {"binary", "multiclass", "regression"}:
        raise ValueError(f"Conformal prediction does not apply to {kind} tasks.")
    rows, y = _held_out(edata, predictor, split_key=split_key, split=split)
    y, outputs = _paired(predictor.task, y, _calibrated(predictor, _outputs_of(edata[rows], predictor)[0]))
    if kind == "regression":
        scores = np.abs(y - outputs[:, 0])
    else:
        scores = 1 - np.take_along_axis(_class_probabilities(outputs), y.astype(int)[:, None], axis=1)[:, 0]
    level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    return replace(predictor, conformal=float(np.quantile(scores, level, method="higher")))


@dataclass(frozen=True)
class _Platt:
    model: LogisticRegression

    def __call__(self, outputs: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_log_odds(outputs))[:, 1:]


@dataclass(frozen=True)
class _Isotonic:
    model: IsotonicRegression

    def __call__(self, outputs: np.ndarray) -> np.ndarray:
        return self.model.predict(outputs[:, 0])[:, None]


@dataclass(frozen=True)
class _Temperature:
    temperature: float

    def __call__(self, outputs: np.ndarray) -> np.ndarray:
        if outputs.shape[1] == 1:
            return expit(_log_odds(outputs) / self.temperature)
        return softmax(np.log(np.clip(outputs, 1e-12, None)) / self.temperature, axis=1)
