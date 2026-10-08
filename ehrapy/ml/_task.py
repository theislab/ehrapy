from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

Kind = Literal["binary", "multiclass", "multilabel", "regression", "survival"]


@dataclass(frozen=True)
class Task:
    """A prediction target and the timepoints its features are computed from.

    For longitudinal data, the features summarize the `observation_window` timepoints that end `gap` timepoints before `prediction_time`.

    Examples:
        >>> import ehrapy as ep
        >>> mortality = ep.ml.Task("In-hospital_death", prediction_time=24, gap=2)
        >>> survival = ep.ml.Task("mort_day_censored", kind="survival", event="death")
    """

    #: Column of `obs` with the label, the columns of a multilabel task, or the time to the event or censoring of a survival task.
    label: str | Sequence[str]
    _: KW_ONLY
    #: The kind of label.
    kind: Kind = "binary"
    #: Column of `obs` that marks with `True` or `1` whether the event of a survival task occurred.
    event: str | None = None
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

    @property
    def _columns(self) -> list[str]:
        """Columns of `obs` with the targets."""
        labels = [self.label] if isinstance(self.label, str) else list(self.label)
        return labels if self.event is None else [*labels, self.event]

    def _window(self, n_timepoints: int) -> slice:
        end = (n_timepoints if self.prediction_time is None else self.prediction_time) - self.gap
        start = 0 if self.observation_window is None else end - self.observation_window
        if not 0 <= start < end <= n_timepoints:
            raise ValueError(f"The timepoints {start} to {end} of {self} are not within the {n_timepoints} timepoints.")
        return slice(start, end)


def _targets(obs: pd.DataFrame, task: Task) -> tuple[np.ndarray, list[Hashable]]:
    """Targets as models are fit on them, NaN where a label is missing, and the classes or labels they stand for.

    Binary labels are 1 for the larger of their two values, multiclass labels are the positions of their sorted classes, and survival targets hold the time and whether the event occurred.
    """
    match task.kind:
        case "binary" | "multiclass":
            classes = sorted(obs[task.label].dropna().unique())
            if task.kind == "binary" and len(classes) != 2:
                raise ValueError(f"A binary task needs exactly two values of {task.label!r}, got {len(classes)}.")
            codes = pd.Categorical(obs[task.label], categories=classes).codes
            return np.where(codes < 0, np.nan, codes), classes
        case "multilabel":
            return np.column_stack([_targets(obs, Task(label))[0] for label in task.label]), list(task.label)
        case "regression":
            return obs[task.label].to_numpy(np.float64), [task.label]
        case "survival":
            return obs[task._columns].to_numpy(np.float64), [task.label]
