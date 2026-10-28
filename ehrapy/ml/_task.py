from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from ehrdata import EHRData
    from fast_array_utils.types import DaskArray

Kind = Literal["binary", "multiclass", "multilabel", "regression", "survival"]


@dataclass(frozen=True)
class Task:
    """A prediction target and the timepoints its features are computed from.

    For longitudinal data, the features summarize the `observation_window` timepoints that end `gap` timepoints before `prediction_time`.
    A rolling task predicts a longitudinal variable at every timepoint, each as if `prediction_time` were that timepoint, with the observation window cut off at the first timepoint.

    Examples:
        >>> import ehrapy as ep
        >>> mortality = ep.ml.Task("In-hospital_death", prediction_time=24, gap=2)
        >>> survival = ep.ml.Task("mort_day_censored", kind="survival", event="death")
        >>> sepsis = ep.ml.Task("SepsisLabel", rolling=True, observation_window=6)
    """

    #: Column of `obs` with the label, the columns of a multilabel task, the time to the event or censoring of a survival task, or the longitudinal variable in `.X` of a rolling task.
    label: str | Sequence[str]
    _: KW_ONLY
    #: The kind of label.
    kind: Kind = "binary"
    #: Column of `obs` that marks with `True` or `1` whether the event of a survival task occurred.
    event: str | None = None
    #: Whether to predict the label at every timepoint, which marks positive binary labels with 1.
    rolling: bool = False
    #: Position along `tem` at which the prediction is made, or `None` for the end of the recorded timepoints.
    prediction_time: int | None = None
    #: Number of timepoints the features summarize, or `None` for all timepoints until the gap.
    observation_window: int | None = None
    #: Number of timepoints right before the prediction time that the features leave out.
    gap: int = 0

    def __post_init__(self) -> None:
        if self.kind not in get_args(Kind):
            raise ValueError(f"`kind` must be one of {get_args(Kind)}, got {self.kind!r}.")
        if (self.kind == "multilabel") == isinstance(self.label, str):
            raise ValueError("A multilabel task needs several label columns, every other task one.")
        if (self.kind == "survival") != (self.event is not None):
            raise ValueError("A survival task needs an `event` column, every other task none.")
        if self.rolling and (self.kind not in {"binary", "regression"} or self.prediction_time is not None):
            raise ValueError(
                "A rolling task is binary or a regression and predicts every timepoint without `prediction_time`."
            )

    @property
    def _columns(self) -> list[str]:
        """Names of the targets, which must not be features."""
        labels = [self.label] if isinstance(self.label, str) else list(self.label)
        return labels if self.event is None else [*labels, self.event]

    def _window(self, n_timepoints: int) -> slice:
        if self.rolling:
            return slice(0, n_timepoints)
        end = (n_timepoints if self.prediction_time is None else self.prediction_time) - self.gap
        start = 0 if self.observation_window is None else end - self.observation_window
        if not 0 <= start < end <= n_timepoints:
            raise ValueError(f"The timepoints {start} to {end} of {self} are not within the {n_timepoints} timepoints.")
        return slice(start, end)

    def _windows(self, n_timepoints: int) -> list[slice | None]:
        """Observation window of every timepoint of a rolling task, or `None` if no timepoint lies more than `gap` before it."""
        windows: list[slice | None] = []
        for end in range(-self.gap, n_timepoints - self.gap):
            start = 0 if self.observation_window is None else max(0, end - self.observation_window)
            windows.append(slice(start, end) if end > 0 else None)
        return windows


def _targets(edata: EHRData, task: Task) -> tuple[np.ndarray | DaskArray, list[Hashable]]:
    """Targets as models are fit on them, NaN where a label is missing, and the classes or labels they stand for.

    Binary labels are 1 for the larger of their two values, multiclass labels are the positions of their sorted classes, and survival targets hold the time and whether the event occurred.
    The targets of rolling tasks are the label variable at every timepoint.
    """
    if task.rolling:
        if edata.X is None or edata.X.ndim != 3:
            raise ValueError("A rolling task needs longitudinal data in `.X`.")
        labels = edata.X[:, edata.var_names.get_loc(task.label), :]
        return labels.astype(np.float64), [0, 1] if task.kind == "binary" else [task.label]
    obs = edata.obs
    match task.kind:
        case "binary" | "multiclass":
            classes = sorted(obs[task.label].dropna().unique())
            if task.kind == "binary" and len(classes) != 2:
                raise ValueError(f"A binary task needs exactly two values of {task.label!r}, got {len(classes)}.")
            codes = pd.Categorical(obs[task.label], categories=classes).codes
            return np.where(codes < 0, np.nan, codes), classes
        case "multilabel":
            return np.column_stack([_targets(edata, Task(label))[0] for label in task.label]), list(task.label)
        case "regression":
            return obs[task.label].to_numpy(np.float64), [task.label]
        case "survival":
            return obs[task._columns].to_numpy(np.float64), [task.label]
