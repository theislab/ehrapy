from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData

SPLITS = ("train", "tuning", "held_out")


def split(
    edata: EHRData,
    *,
    groupby: str | None = None,
    stratify: str | None = None,
    time_key: str | None = None,
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    random_state: int = 0,
    key_added: str = "split",
    copy: bool = False,
) -> EHRData | None:
    """Split observations into a training, a tuning and a held-out set.

    All observations of a group, such as a patient, end up in the same set.
    Models are fit on `train`, tuned on `tuning` and evaluated on `held_out`.

    Args:
        edata: Central data object.
        groupby: Column of `obs` with the group of every observation, such as a patient id.
            If `None`, every observation is its own group.
        stratify: Column of `obs`, such as the label, whose proportions every set keeps.
            It must be the same for all observations of a group.
        time_key: Column of `obs` with a time, such as the admission time, to split by time instead of at random.
            The groups with the earliest times are used for `train` and the ones with the latest for `held_out`.
        fractions: Fractions of the groups in `train`, `tuning` and `held_out`.
        random_state: Seed for the random assignment.
        key_added: Column of `obs` to store the set of every observation in.
        copy: Whether to return a copy of `edata` instead of modifying it in place.

    Returns:
        ``None`` if ``copy=False``, otherwise the updated data object.
        The set of every observation is stored as `"train"`, `"tuning"` or `"held_out"` in `edata.obs[key_added]`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.physionet2012()
        >>> ep.ml.split(edata, stratify="In-hospital_death")
        >>> edata.obs["split"].value_counts()
    """
    if len(fractions) != len(SPLITS) or not np.isclose(sum(fractions), 1):
        raise ValueError(f"`fractions` must be {len(SPLITS)} fractions that sum to 1, got {fractions}.")
    if stratify is not None and time_key is not None:
        raise ValueError("A split by time cannot be stratified, pass either `stratify` or `time_key`.")
    if copy:
        edata = edata.copy()

    groups, unique_groups = pd.factorize(edata.obs_names if groupby is None else edata.obs[groupby])
    if (groups < 0).any():
        raise ValueError(f"`edata.obs[{groupby!r}]` has missing values.")
    assignment = np.empty(len(unique_groups), dtype=np.intp)

    if time_key is not None:
        times = edata.obs[time_key]
        if times.isna().any():
            raise ValueError(f"`edata.obs[{time_key!r}]` has missing values.")
        order = np.argsort(times.groupby(groups).min().to_numpy(), kind="stable")
        for i, members in enumerate(_partition(order, fractions)):
            assignment[members] = i
    else:
        strata = np.zeros(len(unique_groups), dtype=np.intp)
        if stratify is not None:
            row_strata, _ = pd.factorize(edata.obs[stratify], use_na_sentinel=False)
            strata[groups] = row_strata
            if (strata[groups] != row_strata).any():
                raise ValueError(f"`edata.obs[{stratify!r}]` differs within groups of `edata.obs[{groupby!r}]`.")
        rng = np.random.default_rng(random_state)
        for stratum in np.unique(strata):
            for i, members in enumerate(_partition(rng.permutation(np.flatnonzero(strata == stratum)), fractions)):
                assignment[members] = i

    edata.obs[key_added] = pd.Categorical.from_codes(assignment[groups], categories=SPLITS)
    return edata if copy else None


def _partition(groups: np.ndarray, fractions: Sequence[float]) -> list[np.ndarray]:
    """Consecutive parts of `groups` with the given fractions of its length."""
    return np.split(groups, np.round(np.cumsum(fractions)[:-1] * len(groups)).astype(int))
