from __future__ import annotations

from functools import singledispatch
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from fast_array_utils.types import DaskArray

from ehrapy._compat import _like_obs, _map_observation_blocks, _materialize, _raise_array_type_not_implemented
from ehrapy.tools._timeseries import TOLERANCE, _closed, _count, _obs_positions, _rolling, _timeseries, _tolerance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData

STAGES = ("creatinine", "urine")
# The creatinine criteria of :cite:p:`KDIGO2012` in mg/dL: the rise within the rise window, the ratios to the baseline
# of stages 1, 2 and 3, and the creatinine of stage 3.
CREATININE_RISE = 0.3
CREATININE_RATIOS = (1.5, 2.0, 3.0)
CREATININE_STAGE_3 = 4.0
# The urine criteria of :cite:p:`KDIGO2012` on hourly timepoints as (stage, timepoints, mL/kg per timepoint): a mean urine
# output below the rate over the timepoints.
URINE_CRITERIA = ((1, 6, 0.5), (2, 12, 0.5), (3, 24, 0.3))


def _creatinine_stage(
    creatinine: np.ndarray, rise_window: int, baseline_window: int, closed: Literal["right", "both"], tolerance: float
) -> np.ndarray:
    """The stage by creatinine at every timepoint with creatinine, else NaN."""
    lowest = _rolling(creatinine, rise_window, "min", closed=closed)
    baseline = _rolling(creatinine, baseline_window, "min", closed=closed)
    stage_1, stage_2, stage_3 = CREATININE_RATIOS
    # creatinine reaches a threshold within `tolerance` of it
    level = creatinine + tolerance
    with np.errstate(invalid="ignore"):
        # the rise is computed as `ricu` computes it, by subtraction for stage 3 and by addition for stage 1, which can
        # differ by a rounding error without a tolerance
        stage = np.select(
            [
                level >= stage_3 * baseline,
                (level >= CREATININE_STAGE_3)
                & ((creatinine - lowest + tolerance >= CREATININE_RISE) | (level >= stage_1 * baseline)),
                level >= stage_2 * baseline,
                level >= lowest + CREATININE_RISE,
                level >= stage_1 * baseline,
            ],
            [3.0, 3.0, 2.0, 1.0, 1.0],
            default=0.0,
        )
    return np.where(np.isnan(creatinine), np.nan, stage)


def _urine_stage(
    urine: np.ndarray,
    weight: np.ndarray,
    start: np.ndarray,
    criteria: tuple[tuple[float, int, float], ...],
    anuria_window: int | None,
    min_observed: int | None,
    closed: Literal["right", "both"],
    tolerance: float,
) -> np.ndarray:
    """The stage by urine output at every timepoint with urine of an observation with a weight, else NaN."""
    windows = {window for _, window, _ in criteria} | ({anuria_window} if anuria_window is not None else set())
    totals = {window: _rolling(urine, window, "sum", closed=closed) for window in windows}
    covered = dict.fromkeys(windows, True)
    if min_observed is not None:
        recorded = np.where(np.isnan(urine), np.nan, 1.0)
        covered = {window: _rolling(recorded, window, "count", closed=closed) >= min_observed for window in windows}
    elapsed = np.arange(urine.shape[1])[None, :] - start[:, None]
    stage = np.zeros(urine.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        for level, window, rate in criteria:
            # the mean rate is below `rate` if the urine per kg is below `rate` times the timepoints of the window;
            # compared this way, the rates 0.5 / 6, 0.5 / 12 and 0.3 / 24 give exactly the thresholds 0.5, 0.5 and 0.3
            # of `ricu`
            size = window + 1 if closed == "both" else window
            below = totals[window] / weight[:, None] + tolerance < rate * size
            met = (elapsed >= window) & covered[window] & below
            stage = np.where(met, np.fmax(stage, level), stage)
        if anuria_window is not None:
            none = np.abs(totals[anuria_window]) <= tolerance
            stage = np.where((elapsed >= anuria_window) & covered[anuria_window] & none, 3.0, stage)
        staged = ~np.isnan(urine) & (weight > 0)[:, None]
    return np.where(staged, stage, np.nan)


@singledispatch
def _kdigo_stages(X, per_obs: np.ndarray, **settings):
    """The KDIGO stage and the stages by creatinine and urine output (observations x 3 x timepoints) of the `inputs` in `X`."""
    _raise_array_type_not_implemented(_kdigo_stages, type(X))


@_kdigo_stages.register(np.ndarray)
def _(
    X: np.ndarray,
    per_obs: np.ndarray,
    *,
    inputs: tuple[str, ...],
    rise_window: int,
    baseline_window: int,
    urine_criteria: tuple[tuple[float, int, float], ...],
    anuria_window: int | None,
    urine_min_observed: int | None,
    closed: Literal["right", "both"],
    tolerance: float,
) -> np.ndarray:
    # `per_obs` holds the weight of every observation and the position of its urine start
    missing = np.full(X.shape[::2], np.nan)
    columns = {name: np.asarray(X[:, i, :], dtype=np.float64) for i, name in enumerate(inputs)}
    creatinine = missing
    if "creatinine" in columns:
        creatinine = _creatinine_stage(columns["creatinine"], rise_window, baseline_window, closed, tolerance)
    urine = missing
    if "urine" in columns:
        weight, start = per_obs[:, 0, 0], per_obs[:, 1, 0]
        urine = _urine_stage(
            columns["urine"], weight, start, urine_criteria, anuria_window, urine_min_observed, closed, tolerance
        )
    return np.stack([np.fmax(creatinine, urine), creatinine, urine], axis=1)


@_kdigo_stages.register(DaskArray)
def _(X: DaskArray, per_obs: np.ndarray, **settings) -> DaskArray:
    return _map_observation_blocks(
        X,
        _kdigo_stages,
        _like_obs(X, per_obs),
        **settings,
        chunks=(X.chunks[0], (1 + len(STAGES),), (X.shape[2],)),
        meta=np.empty((0, 0, 0), dtype=np.float64),
    )


def _urine_criteria(urine_criteria: Sequence[tuple[int, int, float]]) -> tuple[tuple[float, int, float], ...]:
    """The urine criteria as (stage, window in timepoints, rate per timepoint)."""
    criteria: list[tuple[float, int, float]] = []
    for criterion in urine_criteria:
        try:
            stage, window, rate = criterion
        except (TypeError, ValueError):
            raise ValueError(
                f"A urine criterion must be a tuple of a stage, a window and a rate, not {criterion!r}."
            ) from None
        if isinstance(stage, bool) or not isinstance(stage, Integral) or stage not in (1, 2, 3):
            raise ValueError(f"The stage of a urine criterion must be the integer 1, 2 or 3, not {stage!r}.")
        if isinstance(rate, bool) or not isinstance(rate, Real) or not np.isfinite(rate) or rate < 0:
            raise ValueError(
                f"The rate of a urine criterion must be a number of at least 0 mL/kg per timepoint, not {rate!r}."
            )
        criteria.append((float(stage), _count(window, "The window of a urine criterion", positive=True), float(rate)))
    return tuple(criteria)


def kdigo(
    edata: EHRData,
    *,
    creatinine: str | None = None,
    urine: str | None = None,
    weight_key: str | None = None,
    layer: str | None = None,
    rise_window: int = 48,
    baseline_window: int = 168,
    urine_criteria: Sequence[tuple[int, int, float]] = URINE_CRITERIA,
    anuria_window: int | None = 12,
    urine_start_key: str | None = None,
    urine_min_observed: int | None = None,
    closed: Literal["right", "both"] = "right",
    tolerance: float = TOLERANCE,
    key_added: str = "kdigo",
    copy: bool = False,
) -> EHRData | None:
    """Stage acute kidney injury at every timepoint of longitudinal data by the KDIGO criteria :cite:p:`KDIGO2012`.

    The stage from 0 to 3 is the higher of the stage by creatinine and the stage by urine output.
    By creatinine, a timepoint compares its creatinine with the lowest creatinine of the `rise_window` timepoints that end at it and with the lowest of the `baseline_window` timepoints that end at it as baseline:

    * stage 1 at a rise of at least 0.3 mg/dL over the lowest creatinine of `rise_window`, or at least 1.5 times the baseline,
    * stage 2 at least 2 times the baseline,
    * stage 3 at least 3 times the baseline, or at least 4 mg/dL that also meets a criterion of stage 1.

    By urine output, a timepoint compares the mean urine output per kg and timepoint of the windows that end at it with the rates of `urine_criteria`, by default on hourly data

    * stage 1 below 0.5 mL/kg/h over 6 hours,
    * stage 2 below 0.5 mL/kg/h over 12 hours,
    * stage 3 below 0.3 mL/kg/h over 24 hours, or no urine over `anuria_window`.

    The mean urine output of a window is its urine, timepoints without urine counting as none, divided by the weight and the number of timepoints of the window.
    A window counts only at timepoints at least its length after the urine start of the observation, so that it lies after the urine start, and with `urine_min_observed` only if at least that many of its timepoints have urine.
    Renal replacement therapy and the estimated glomerular filtration rate of children, criteria of stage 3, are not covered.

    The creatinine is staged at the timepoints with creatinine, and the urine output at the timepoints with urine of observations with a weight, since the urine of a window is known only once the urine of its last timepoint is recorded.
    To stage the urine output also at timepoints without urine, fill these with 0 where no urine was produced.
    Every input is a variable of `layer`, or else a key of `obsm` with a value per observation and timepoint, such as the result of another tool.
    Windows count timepoints, and the defaults are for hourly data.
    A value within `tolerance` of a threshold counts as reaching it, so that records in decimals reach the thresholds they meet despite the rounding of floating point numbers.

    The defaults follow :cite:p:`KDIGO2012`.
    With `urine_criteria=((1, 6, 0.5 / 6), (2, 12, 0.5 / 12), (3, 24, 0.3 / 24))`, the position of the ICU admission as `urine_start_key` and `tolerance=0`, this gives the KDIGO stage of YAIB's extensions of the `ricu` R package :cite:p:`vandeWater2024,Bennett2023` on hourly data prepared as YAIB prepares it, including the creatinine from before the ICU admission.
    YAIB compares the urine per kg of a window, rather than its mean rate, with the rates of :cite:p:`KDIGO2012` and takes its urine rate of every hour as the urine output.
    Of an observation whose weight is recorded as missing, it divides the urine by 75 kg, and of one without any weight record, it keeps the urine in mL, which a weight of 1 kg gives.

    Args:
        edata: Central data object with longitudinal data.
        creatinine: Variable of the serum creatinine in mg/dL.
        urine: Variable of the urine output in mL per timepoint.
        weight_key: Column of `obs` with the weight of every observation in kg, needed with `urine`.
            Observations with a missing weight or a weight of at most 0 are not staged by urine output.
        layer: The 3D layer to read the variables from.
            If `None`, `edata.X` is used.
        rise_window: Timepoints, ending at every timepoint, whose lowest creatinine the rise is taken from.
            This and the other windows must be more than 0.
        baseline_window: Timepoints, ending at every timepoint, whose lowest creatinine is the baseline, by default 7 days of hours.
        urine_criteria: Criteria of the urine output as tuples of a stage, 1, 2 or 3, a window in timepoints and a rate in mL/kg per timepoint, which is mL/kg/h on hourly data.
            A timepoint reaches the stage if the mean urine output of the window that ends at it is below the rate.
        anuria_window: Timepoints without urine, ending at every timepoint, that give stage 3.
            If `None`, anuria is not a criterion.
        urine_start_key: Column of `obs` with the position from which the urine output of every observation is recorded, 0 for the first timepoint and rounded down, such as the position of the ICU admission.
            A position before the first timepoint counts as the first timepoint.
            If `None`, and for observations with a missing position, the first timepoint is used.
        urine_min_observed: Fewest timepoints with urine of a urine window, or of the anuria window, to stage it.
            If `None`, every window is staged, timepoints without urine counting as none.
        closed: Whether a window holds the timepoint and the `window` - 1 before it (`"right"`), as in YAIB, or the timepoint and the `window` before it (`"both"`).
        tolerance: Margin of the comparisons with the thresholds: a value within `tolerance` of a threshold counts as reaching it, and a urine output of at most `tolerance` over `anuria_window` counts as none.
            The default is far below the precision of the records; with it, a rise from 1.1 to 1.4 mg/dL, 0.29999999999999982 in floating point, is a rise of 0.3.
            `0` compares exactly, as `ricu` does.
        key_added: The KDIGO stage of every timepoint is stored in `edata.obsm[key_added]`, and the stages by creatinine and by urine output in `edata.obsm[f"{key_added}_creatinine"]` and `edata.obsm[f"{key_added}_urine"]`.
            The stage by creatinine is missing at timepoints without creatinine, the stage by urine output at timepoints without urine and for observations without a weight, and the KDIGO stage where both are missing.
        copy: Whether to return a copy of `edata` rather than modifying it in place.

    Returns:
        A copy of `edata` with the results if `copy` is `True`, else `None`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> import pandas as pd
        >>> creatinine = [1.0, np.nan, 1.4, 2.1, np.nan, np.nan, np.nan, 3.2]
        >>> urine = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, np.nan, 10.0]
        >>> X = np.array([[creatinine, urine]])
        >>> obs = pd.DataFrame({"weight": [50.0]}, index=["a"])
        >>> var = pd.DataFrame(index=["creatinine", "urine"])
        >>> edata = ed.EHRData(X=X, obs=obs, var=var)
        >>> ep.tl.kdigo(edata, creatinine="creatinine", urine="urine", weight_key="weight")
        >>> edata.obsm["kdigo"]
        array([[ 0.,  0.,  1.,  2.,  0.,  0., nan,  3.]])
        >>> edata.obsm["kdigo_urine"]
        array([[ 0.,  0.,  0.,  0.,  0.,  0., nan,  1.]])
    """
    names = {name: var for name, var in {"creatinine": creatinine, "urine": urine}.items() if var is not None}
    if not names:
        raise ValueError("Name the variable of creatinine, of the urine output or of both.")
    if urine is None and (weight_key is not None or urine_start_key is not None or urine_min_observed is not None):
        raise ValueError(
            "weight_key, urine_start_key and urine_min_observed need the variable of the urine output, `urine`."
        )
    if urine is not None and weight_key is None:
        raise ValueError("urine needs the weight of every observation in kg, `weight_key`.")
    settings = {
        "urine_criteria": _urine_criteria(urine_criteria),
        "rise_window": _count(rise_window, "rise_window", positive=True),
        "baseline_window": _count(baseline_window, "baseline_window", positive=True),
        "anuria_window": None if anuria_window is None else _count(anuria_window, "anuria_window", positive=True),
        "urine_min_observed": None
        if urine_min_observed is None
        else _count(urine_min_observed, "urine_min_observed", positive=True),
        "closed": _closed(closed),
        "tolerance": _tolerance(tolerance),
    }
    selected = _timeseries(edata, names, layer, "The KDIGO stage")

    per_obs = np.full((edata.n_obs, 2, 1), np.nan)
    if urine is not None:
        if weight_key not in edata.obs:
            raise KeyError(f"edata.obs has no column {weight_key!r}.")
        per_obs[:, 0, 0] = pd.to_numeric(edata.obs[weight_key]).to_numpy(np.float64, na_value=np.nan)
        start = np.full(edata.n_obs, np.nan) if urine_start_key is None else _obs_positions(edata, urine_start_key)
        # a start before the data counts as the first timepoint, as the windows hold no urine before it
        per_obs[:, 1, 0] = np.where(np.isnan(start), 0.0, np.maximum(start, 0.0))

    (stages,) = _materialize(_kdigo_stages(selected, per_obs, inputs=tuple(names), **settings))

    edata = edata.copy() if copy else edata
    edata.obsm[key_added] = stages[:, 0, :]
    for i, name in enumerate(STAGES, start=1):
        edata.obsm[f"{key_added}_{name}"] = stages[:, i, :]
    return edata if copy else None
