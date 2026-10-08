from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import label_binarize

from ehrapy._compat import _materialize
from ehrapy.ml._task import _targets

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ehrdata import EHRData

    from ehrapy.ml._task import Kind, Task

    type Metric = Callable[[np.ndarray, np.ndarray], float]


def evaluate(
    edata: EHRData,
    task: Task,
    *,
    key: str = "prediction",
    split_key: str = "split",
    split: str | None = "held_out",
    groupby: str | None = None,
    patient_key: str | None = None,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> pd.DataFrame:
    """Evaluate predictions with bootstrap confidence intervals, overall or per subgroup.

    The metrics depend on the kind of task:

    - binary: the area under the ROC curve (`auroc`), the average precision (`auprc`), the Brier score (`brier`), the expected calibration error over ten bins (`ece`), the calibration intercept and slope of a logistic recalibration (`calibration_intercept`, `calibration_slope`), which are 0 and 1 for calibrated predictions, and the `sensitivity`, `specificity`, `f1` score and fraction of positive predictions (`positive_rate`) when predictions at or above `threshold` are positive,
    - multiclass and multilabel: the area under the ROC curve averaged over classes or labels (`auroc_macro`) or over all predictions (`auroc_micro`), the `accuracy`, and the F1 score averaged over classes or labels (`f1_macro`) or over all predictions (`f1_micro`),
    - regression: the mean absolute error (`mae`), the root mean squared error (`rmse`) and the coefficient of determination (`r2`),
    - survival: Harrell's concordance index (`c_index`).

    With `groupby`, the subgroup `"difference"` holds the largest difference of every metric between subgroups.
    The difference of `positive_rate` is the demographic parity difference, and `equalized_odds` is the larger difference of `sensitivity` and `specificity`.
    The confidence intervals are percentiles of the metrics on patients resampled with replacement.

    Args:
        edata: Central data object.
        task: The prediction task.
        key: Key of the predictions from :func:`~ehrapy.ml.predict`.
        split_key: Column of `obs` with the sets from :func:`~ehrapy.ml.split`.
        split: The set to evaluate.
            If `None`, all observations are evaluated.
        groupby: Column of `obs` with subgroups, such as sex, to evaluate separately.
        patient_key: Column of `obs` with the patient of every observation, whose observations are resampled together.
            If `None`, every observation is its own patient.
        threshold: Probability at or above which a binary or multilabel prediction is positive.
        n_bootstrap: Number of bootstrap samples for the confidence intervals.
            If 0, no confidence intervals are computed.
        alpha: The confidence intervals cover `1 - alpha` of the bootstrap samples.
        random_state: Seed for the bootstrap samples.

    Returns:
        The metrics with columns `value`, `ci_lower` and `ci_upper`, indexed by metric, and by subgroup first if `groupby` is given.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_observations=200, n_centers=2, base_timepoints=10)
        >>> ep.ml.split(edata, stratify="cluster")
        >>> task = ep.ml.Task("cluster")
        >>> ep.ml.predict(edata, ep.ml.fit(edata, task))
        >>> ep.ml.evaluate(edata, task, n_bootstrap=100)
    """
    y, prediction, obs = _evaluation_data(edata, task, key=key, split_key=split_key, split=split)
    metrics = _metrics(task.kind, threshold)
    groups = None if groupby is None else obs[groupby].to_numpy()
    patients = pd.factorize(obs.index if patient_key is None else obs[patient_key])[0]

    def table(rows: np.ndarray) -> pd.Series:
        return _table(y[rows], prediction[rows], None if groups is None else groups[rows], metrics, task.kind)

    value = table(np.arange(len(y)))
    rng = np.random.default_rng(random_state)
    samples = pd.DataFrame([table(_resample(patients, rng)) for _ in range(n_bootstrap)], columns=value.index)
    result = pd.DataFrame(
        {
            "value": value,
            "ci_lower": samples.quantile(alpha / 2) if n_bootstrap else np.nan,
            "ci_upper": samples.quantile(1 - alpha / 2) if n_bootstrap else np.nan,
        }
    )
    return result if groupby is None else result.rename_axis([groupby, "metric"])


def _evaluation_data(
    edata: EHRData, task: Task, *, key: str, split_key: str, split: str | None
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Targets, predictions and `obs` of the evaluated samples with all targets and a prediction.

    The samples of rolling tasks are the labeled and predicted timepoints, each with the `obs` row of its observation.
    """
    predictions = edata.obsm if task.kind in {"multiclass", "multilabel"} or task.rolling else edata.obs
    if key not in predictions:
        raise KeyError(f"No predictions under {key!r}. Predict first with `ep.ml.predict`.")
    prediction = np.asarray(predictions[key], dtype=np.float64)
    y = _materialize(_targets(edata, task)[0])[0]
    in_split = np.ones(edata.n_obs, dtype=bool) if split is None else (edata.obs[split_key] == split).to_numpy()
    if task.rolling:
        samples = ~np.isnan(y) & ~np.isnan(prediction) & in_split[:, None]
        y, prediction, obs = y[samples], prediction[samples], edata.obs.iloc[np.nonzero(samples)[0]]
    else:
        rows = ~np.isnan(np.column_stack([y, prediction])).any(axis=1) & in_split
        y, prediction, obs = y[rows], prediction[rows], edata.obs[rows]
    return (y if task.kind in {"regression", "survival"} else y.astype(np.int64)), prediction, obs


def _table(
    y: np.ndarray, prediction: np.ndarray, groups: np.ndarray | None, metrics: Mapping[str, Metric], kind: Kind
) -> pd.Series:
    """The metrics, per subgroup with their largest differences if `groups` is given."""
    if groups is None:
        return pd.Series(_values(y, prediction, metrics, kind), index=pd.Index(list(metrics), name="metric"))
    per_group = pd.DataFrame(
        {
            group: _values(y[groups == group], prediction[groups == group], metrics, kind)
            for group in np.unique(groups[pd.notna(groups)])
        },
        index=pd.Index(list(metrics), name="metric"),
    )
    difference = per_group.max(axis=1) - per_group.min(axis=1)
    if kind == "binary":
        difference["equalized_odds"] = difference[["sensitivity", "specificity"]].max()
    return pd.concat([per_group.unstack(), pd.concat({"difference": difference})])


def _values(y: np.ndarray, prediction: np.ndarray, metrics: Mapping[str, Metric], kind: Kind) -> list[float]:
    """The metrics, or NaN where a single class, single label values or no events leave them undefined."""
    match kind:
        case "survival":
            undefined = not y[:, 1].any()
        case "multilabel":
            undefined = bool((y.min(axis=0) == y.max(axis=0)).all())
        case _:
            undefined = len(np.unique(y)) < 2
    return [np.nan if undefined else float(metric(y, prediction)) for metric in metrics.values()]


def _resample(patients: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rows of patients drawn with replacement."""
    n_patients = patients.max() + 1
    sizes = np.bincount(patients, minlength=n_patients)
    drawn = rng.integers(0, n_patients, n_patients)
    ends = np.cumsum(sizes[drawn])
    offsets = np.arange(ends[-1]) - np.repeat(ends - sizes[drawn], sizes[drawn])
    return np.argsort(patients, kind="stable")[
        np.repeat(np.cumsum(sizes)[drawn] - sizes[drawn], sizes[drawn]) + offsets
    ]


def _metrics(kind: Kind, threshold: float) -> dict[str, Metric]:
    match kind:
        case "binary":
            return {
                "auroc": roc_auc_score,
                "auprc": average_precision_score,
                "brier": brier_score_loss,
                "ece": _expected_calibration_error,
                "calibration_intercept": _calibration_intercept,
                "calibration_slope": _calibration_slope,
                "sensitivity": lambda y, p: recall_score(y, p >= threshold),
                "specificity": lambda y, p: recall_score(y, p >= threshold, pos_label=0),
                "f1": lambda y, p: f1_score(y, p >= threshold),
                "positive_rate": lambda y, p: np.mean(p >= threshold),
            }
        case "multiclass" | "multilabel":

            def indicators(y: np.ndarray, p: np.ndarray) -> np.ndarray:
                return y if kind == "multilabel" else label_binarize(y, classes=np.arange(p.shape[1]))

            def decisions(p: np.ndarray) -> np.ndarray:
                return p >= threshold if kind == "multilabel" else p.argmax(axis=1)

            return {
                "auroc_macro": lambda y, p: _auroc(indicators(y, p), p, "macro"),
                "auroc_micro": lambda y, p: _auroc(indicators(y, p), p, "micro"),
                "accuracy": lambda y, p: accuracy_score(y, decisions(p)),
                "f1_macro": lambda y, p: f1_score(y, decisions(p), average="macro", zero_division=0),
                "f1_micro": lambda y, p: f1_score(y, decisions(p), average="micro", zero_division=0),
            }
        case "regression":
            return {"mae": mean_absolute_error, "rmse": root_mean_squared_error, "r2": r2_score}
    return {"c_index": _concordance_index}


def _auroc(y: np.ndarray, prediction: np.ndarray, average: str) -> float:
    """AUROC of 0/1 columns, leaving columns with a single value out of the macro average."""
    varying = (y.min(axis=0) != y.max(axis=0)) if average == "macro" else slice(None)
    return roc_auc_score(y[:, varying], prediction[:, varying], average=average)


def _expected_calibration_error(y: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> float:
    bins = np.minimum((probability * n_bins).astype(int), n_bins - 1)
    return np.abs(np.bincount(bins, weights=y - probability, minlength=n_bins)).sum() / len(y)


def _calibration_intercept(y: np.ndarray, probability: np.ndarray) -> float:
    log_odds = _log_odds(probability)
    return sm.GLM(y, np.ones_like(log_odds), family=sm.families.Binomial(), offset=log_odds).fit().params[0]


def _calibration_slope(y: np.ndarray, probability: np.ndarray) -> float:
    log_odds = _log_odds(probability)
    return sm.GLM(y, np.column_stack([np.ones_like(log_odds), log_odds]), family=sm.families.Binomial()).fit().params[1]


def _log_odds(probability: np.ndarray) -> np.ndarray:
    return logit(np.clip(probability, 1e-12, 1 - 1e-12))


def _concordance_index(y: np.ndarray, risk: np.ndarray) -> float:
    from lifelines.utils import concordance_index

    return concordance_index(y[:, 0], -risk, y[:, 1])
