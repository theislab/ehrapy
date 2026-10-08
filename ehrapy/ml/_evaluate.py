from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ehrdata import EHRData

    from ehrapy.ml._predictor import Task

    type Metric = Callable[[np.ndarray, np.ndarray], float]


def evaluate(
    edata: EHRData,
    task: Task,
    *,
    key: str = "prediction",
    split_key: str = "split",
    split: str | None = "held_out",
    groupby: str | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> pd.DataFrame:
    """Evaluate predictions with bootstrap confidence intervals, overall or per subgroup.

    Binary tasks are evaluated with the area under the ROC curve (`auroc`), the average precision (`auprc`), the Brier score (`brier`), and the calibration intercept and slope of a logistic recalibration (`calibration_intercept`, `calibration_slope`), which are 0 and 1 for calibrated predictions.
    Regression tasks are evaluated with the mean absolute error (`mae`), the root mean squared error (`rmse`) and the coefficient of determination (`r2`).
    The confidence intervals are percentiles of the metrics on observations resampled with replacement.

    Args:
        edata: Central data object.
        task: The prediction task.
        key: Column of `obs` with the predictions from :func:`~ehrapy.ml.predict`.
        split_key: Column of `obs` with the sets from :func:`~ehrapy.ml.split`.
        split: The set to evaluate.
            If `None`, all observations are evaluated.
        groupby: Column of `obs` with subgroups, such as sex, to evaluate separately.
        n_bootstrap: Number of bootstrap samples for the confidence intervals.
            If 0, no confidence intervals are computed.
        alpha: The confidence intervals cover `1 - alpha` of the bootstrap samples.
        random_state: Seed for the bootstrap samples.

    Returns:
        The metrics with columns `value`, `ci_lower` and `ci_upper`, indexed by metric, and by subgroup first if `groupby` is given.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.physionet2012()
        >>> ep.ml.split(edata, stratify="In-hospital_death")
        >>> task = ep.ml.Task("In-hospital_death")
        >>> ep.ml.predict(edata, ep.ml.fit(edata, task))
        >>> ep.ml.evaluate(edata, task, groupby="Gender")
    """
    y, prediction, obs = _evaluation_data(edata, task, key=key, split_key=split_key, split=split)
    metrics = _BINARY_METRICS if task.kind == "binary" else _REGRESSION_METRICS
    rng = np.random.default_rng(random_state)
    if groupby is None:
        return _metrics(y, prediction, metrics, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
    return pd.concat(
        {
            group: _metrics(y[rows], prediction[rows], metrics, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
            for group, rows in obs.groupby(groupby, observed=True).indices.items()
        },
        names=[groupby],
    )


def _evaluation_data(
    edata: EHRData, task: Task, *, key: str, split_key: str, split: str | None
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Labels, as 0 and 1 for binary tasks, predictions and `obs` of the evaluated observations with a label and a prediction."""
    if key not in edata.obs:
        raise KeyError(f"`edata.obs` has no column {key!r}. Predict first with `ep.ml.predict`.")
    obs = edata.obs
    rows = obs[task.label].notna() & obs[key].notna()
    if split is not None:
        rows &= obs[split_key] == split
    obs = obs[rows]
    y = obs[task.label].to_numpy()
    if task.kind == "binary":
        y = (y == np.unique(edata.obs[task.label].dropna())[-1]).astype(np.int64)
    return y, obs[key].to_numpy(np.float64), obs


def _metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    metrics: Mapping[str, Metric],
    *,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    lower = upper = np.full(len(metrics), np.nan)
    if n_bootstrap:
        samples = (rng.integers(0, len(y), len(y)) for _ in range(n_bootstrap))
        bootstrap = [_values(y[sample], prediction[sample], metrics) for sample in samples]
        lower, upper = np.nanquantile(bootstrap, [alpha / 2, 1 - alpha / 2], axis=0)
    return pd.DataFrame(
        {"value": _values(y, prediction, metrics), "ci_lower": lower, "ci_upper": upper},
        index=pd.Index(list(metrics), name="metric"),
    )


def _values(y: np.ndarray, prediction: np.ndarray, metrics: Mapping[str, Metric]) -> list[float]:
    """The metrics, or NaN if `y` has a single value for which they are undefined."""
    if len(np.unique(y)) < 2:
        return [np.nan] * len(metrics)
    return [float(metric(y, prediction)) for metric in metrics.values()]


def _calibration_intercept(y: np.ndarray, probability: np.ndarray) -> float:
    log_odds = _log_odds(probability)
    return sm.GLM(y, np.ones_like(log_odds), family=sm.families.Binomial(), offset=log_odds).fit().params[0]


def _calibration_slope(y: np.ndarray, probability: np.ndarray) -> float:
    log_odds = _log_odds(probability)
    return sm.GLM(y, np.column_stack([np.ones_like(log_odds), log_odds]), family=sm.families.Binomial()).fit().params[1]


def _log_odds(probability: np.ndarray) -> np.ndarray:
    return logit(np.clip(probability, 1e-12, 1 - 1e-12))


_BINARY_METRICS: dict[str, Metric] = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
    "brier": brier_score_loss,
    "calibration_intercept": _calibration_intercept,
    "calibration_slope": _calibration_slope,
}
_REGRESSION_METRICS: dict[str, Metric] = {"mae": mean_absolute_error, "rmse": root_mean_squared_error, "r2": r2_score}
