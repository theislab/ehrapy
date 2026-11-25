from __future__ import annotations

import warnings
from functools import singledispatch
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from fast_array_utils.types import DaskArray

from ehrapy._compat import _like_obs, _map_observation_blocks, _materialize, _raise_array_type_not_implemented
from ehrapy.tools._timeseries import (
    TOLERANCE,
    _closed,
    _count,
    _locf,
    _obs_positions,
    _obsm_values,
    _period,
    _rolling,
    _timeseries,
    _tolerance,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ehrdata import EHRData

INPUTS = (
    "pao2",
    "fio2",
    "ventilation",
    "platelets",
    "bilirubin",
    "mean_arterial_pressure",
    "dopamine",
    "dobutamine",
    "epinephrine",
    "norepinephrine",
    "gcs",
    "creatinine",
    "urine",
)
ORGANS = ("respiration", "coagulation", "liver", "cardiovascular", "cns", "renal")
# The thresholds of :cite:p:`Vincent1996` in increasing order. Where a higher value is worse, a value scores the number of
# thresholds it reaches; where a lower value is worse, 4 minus that number.
RESPIRATION_BOUNDS = (100.0, 200.0, 300.0, 400.0)  # PaO2/FiO2 in mmHg
COAGULATION_BOUNDS = (20.0, 50.0, 100.0, 150.0)  # platelets in 10^3/µL
LIVER_BOUNDS = (1.2, 2.0, 6.0, 12.0)  # bilirubin in mg/dL
CNS_BOUNDS = (6.0, 10.0, 13.0, 15.0)  # Glasgow Coma Scale
CREATININE_BOUNDS = (1.2, 2.0, 3.5, 5.0)  # creatinine in mg/dL
URINE_BOUNDS = (200.0, 500.0)  # urine output in mL per urine window, which give 4 and 3 below them
ROOM_AIR = 21.0  # FiO2 in percent


def _count_reached(x: np.ndarray, bounds: tuple[float, ...], tolerance: float) -> np.ndarray:
    """Number of `bounds` that `x` reaches within `tolerance`, NaN where `x` is missing."""
    with np.errstate(invalid="ignore"):
        count = sum((x + tolerance >= bound).astype(np.float64) for bound in bounds)
    return np.where(np.isnan(x), np.nan, count)


def _score(raw: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """An organ score where the organ has an input, a missing grade counting as 0, else NaN."""
    return np.where(seen, np.nan_to_num(raw), np.nan)


def _pafi(pao2: np.ndarray, fio2: np.ndarray, within: int, default: float | None) -> np.ndarray:
    """The lowest PaO2/FiO2 ratio of every timepoint, FiO2 in percent.

    Each PaO2 takes the last FiO2 at most `within` timepoints before it, or `default` without one, and each FiO2 the last PaO2 at most `within` timepoints before it.
    """
    fio2_for_pao2 = _locf(fio2, within)
    if default is not None:
        fio2_for_pao2 = np.where(np.isnan(fio2_for_pao2), default, fio2_for_pao2)
    pao2_for_fio2 = _locf(pao2, within)
    with np.errstate(invalid="ignore", divide="ignore"):
        by_pao2 = np.where(~np.isnan(pao2) & (fio2_for_pao2 != 0), 100.0 * pao2 / fio2_for_pao2, np.nan)
        by_fio2 = np.where(
            ~np.isnan(fio2) & ~np.isnan(pao2_for_fio2) & (fio2 != 0), 100.0 * pao2_for_fio2 / fio2, np.nan
        )
    return np.fmin(by_pao2, by_fio2)


def _organ_scores(
    get: Callable[[str], np.ndarray],
    urine_first: np.ndarray,
    urine_last: np.ndarray,
    *,
    pafi_window: int,
    fio2_default: float | None,
    urine_window: int,
    urine_min_observed: int,
    closed: Literal["right", "both"],
    tolerance: float,
) -> list[np.ndarray]:
    """The scores of the organs of every observation and timepoint, in the order of ORGANS, NaN where an organ has no input."""
    tol = tolerance
    ventilated = np.nan_to_num(get("ventilation")) != 0
    pafi = _pafi(get("pao2"), get("fio2"), pafi_window, fio2_default)
    with np.errstate(invalid="ignore"):
        # without ventilation, a ratio below 200 scores as 200
        pafi_scored = np.where(np.isfinite(pafi) & (pafi + tol < 200) & ~ventilated, 200.0, pafi)
    respiration = _score(4 - _count_reached(pafi_scored, RESPIRATION_BOUNDS, tol), ~np.isnan(pafi) | ventilated)

    platelets, bilirubin, gcs = get("platelets"), get("bilirubin"), get("gcs")
    coagulation = _score(4 - _count_reached(platelets, COAGULATION_BOUNDS, tol), ~np.isnan(platelets))
    liver = _score(_count_reached(bilirubin, LIVER_BOUNDS, tol), ~np.isnan(bilirubin))
    cns = _score(4 - _count_reached(gcs, CNS_BOUNDS, tol), ~np.isnan(gcs))

    pressure = get("mean_arterial_pressure")
    dopamine, dobutamine, epinephrine, norepinephrine = (
        np.nan_to_num(get(name)) for name in ("dopamine", "dobutamine", "epinephrine", "norepinephrine")
    )
    with np.errstate(invalid="ignore"):
        cardiovascular = np.select(
            [
                (dopamine > 15 + tol) | (epinephrine > 0.1 + tol) | (norepinephrine > 0.1 + tol),
                (dopamine > 5 + tol)
                | ((epinephrine > tol) & (epinephrine <= 0.1 + tol))
                | ((norepinephrine > tol) & (norepinephrine <= 0.1 + tol)),
                ((dopamine > tol) & (dopamine <= 5 + tol)) | (dobutamine > tol),
                pressure + tol < 70,
            ],
            [4.0, 3.0, 2.0, 1.0],
            default=0.0,
        )
    vasopressors = (dopamine > 0) | (dobutamine > 0) | (epinephrine > 0) | (norepinephrine > 0)
    cardiovascular = _score(cardiovascular, ~np.isnan(pressure) | vasopressors)

    urine, creatinine = get("urine"), get("creatinine")
    in_period = _period(~np.isnan(urine), urine_first, urine_last)
    total = _rolling(np.where(in_period, np.nan_to_num(urine), np.nan), urine_window, "sum", closed=closed)
    count = _rolling(np.where(in_period, 1.0, np.nan), urine_window, "count", closed=closed)
    with np.errstate(invalid="ignore", divide="ignore"):
        covered = in_period & (count >= urine_min_observed)
        urine_per_window = np.where(covered, total * urine_window / count, np.nan)
        lowest, low = URINE_BOUNDS
        urine_score = np.select(
            [urine_per_window + tol < lowest, urine_per_window + tol < low], [4.0, 3.0], default=0.0
        )
    creatinine_score = np.nan_to_num(_count_reached(creatinine, CREATININE_BOUNDS, tol))
    renal = _score(np.fmax(creatinine_score, urine_score), ~np.isnan(creatinine) | in_period)

    return [respiration, coagulation, liver, cardiovascular, cns, renal]


@singledispatch
def _sofa_scores(X, periods: np.ndarray, **settings: float | str | tuple[str, ...] | None):
    """The SOFA score and the scores of the organs (observations x 1 + organs x timepoints) of the `inputs` in `X`."""
    _raise_array_type_not_implemented(_sofa_scores, type(X))


@_sofa_scores.register(np.ndarray)
def _(
    X: np.ndarray,
    periods: np.ndarray,
    *,
    inputs: tuple[str, ...],
    window: int,
    fio2_factor: float,
    closed: Literal["right", "both"],
    **settings: Any,
) -> np.ndarray:
    # `periods` holds the positions of the timepoint from which every observation is scored and of its first and last
    # urine, NaN if not given; FiO2 times `fio2_factor` is in percent
    missing = np.full(X.shape[::2], np.nan)
    columns = {name: np.asarray(X[:, i, :], dtype=np.float64) for i, name in enumerate(inputs)}
    if "fio2" in columns:
        columns["fio2"] = fio2_factor * columns["fio2"]
    start, urine_first, urine_last = (periods[:, i, 0] for i in range(3))
    organs = _organ_scores(lambda name: columns.get(name, missing), urine_first, urine_last, closed=closed, **settings)

    seen = np.zeros(missing.shape, dtype=bool)
    total = np.zeros(missing.shape)
    for score in organs:
        seen |= ~np.isnan(score)
        total += np.nan_to_num(_rolling(score, window, "max", closed=closed))
    total = np.where(_period(seen, start, np.full(len(start), np.nan)), total, np.nan)
    return np.stack([total, *organs], axis=1)


@_sofa_scores.register(DaskArray)
def _(X: DaskArray, periods: np.ndarray, **settings) -> DaskArray:
    return _map_observation_blocks(
        X,
        _sofa_scores,
        _like_obs(X, periods),
        **settings,
        chunks=(X.chunks[0], (1 + len(ORGANS),), (X.shape[2],)),
        meta=np.empty((0, 0, 0), dtype=np.float64),
    )


def _fio2_default(value: float | Literal["room air"] | None, unit: str) -> float | None:
    """The FiO2 of a PaO2 without one in percent, from `value` in `unit`."""
    if value is None:
        return None
    if isinstance(value, str):
        if value != "room air":
            raise ValueError(f"fio2_default must be a number, 'room air' or None, not {value!r}.")
        return ROOM_AIR
    low, high = (0.21, 1.0) if unit == "fraction" else (21.0, 100.0)
    if isinstance(value, bool) or not isinstance(value, Real) or not low <= value <= high:
        raise ValueError(
            f"fio2_default must be a FiO2 from {low:g} to {high:g} in the fio2_unit {unit!r}, not {value!r}."
        )
    return float(value) * (100.0 if unit == "fraction" else 1.0)


def sofa(
    edata: EHRData,
    *,
    pao2: str | None = None,
    fio2: str | None = None,
    ventilation: str | None = None,
    platelets: str | None = None,
    bilirubin: str | None = None,
    mean_arterial_pressure: str | None = None,
    dopamine: str | None = None,
    dobutamine: str | None = None,
    epinephrine: str | None = None,
    norepinephrine: str | None = None,
    gcs: str | None = None,
    creatinine: str | None = None,
    urine: str | None = None,
    layer: str | None = None,
    window: int = 24,
    pafi_window: int = 2,
    fio2_unit: Literal["percent", "fraction"] = "percent",
    fio2_default: float | Literal["room air"] | None = "room air",
    urine_window: int = 24,
    urine_min_observed: int = 12,
    urine_period_keys: tuple[str, str] | None = None,
    start_key: str | None = None,
    closed: Literal["right", "both"] = "right",
    tolerance: float = TOLERANCE,
    key_added: str = "sofa",
    copy: bool = False,
) -> EHRData | None:
    """Score organ failure at every timepoint of longitudinal data with the Sequential Organ Failure Assessment (SOFA).

    The six organs are scored from 0 to 4 at every timepoint by the thresholds of :cite:p:`Vincent1996`:

    * respiration by the PaO2/FiO2 ratio below 400, 300, 200 and 100 mmHg, where the scores 3 and 4 need ventilation: without it, a lower ratio scores 2,
    * coagulation by platelets below 150, 100, 50 and 20 x 10^3/µL,
    * the liver by bilirubin of at least 1.2, 2, 6 and 12 mg/dL,
    * the cardiovascular system by a mean arterial pressure below 70 mmHg (1), dopamine up to 5 µg/kg/min or any dobutamine (2), dopamine above 5 or epinephrine or norepinephrine up to 0.1 µg/kg/min (3), and dopamine above 15 or epinephrine or norepinephrine above 0.1 µg/kg/min (4), where :cite:p:`Vincent1996` counts rates given for at least an hour,
    * the central nervous system by a Glasgow Coma Scale below 15, 13, 10 and 6,
    * the kidneys by creatinine of at least 1.2, 2, 3.5 and 5 mg/dL, or a urine output below 500 (3) and 200 mL (4) per `urine_window`.

    A value within `tolerance` of a threshold counts as the threshold, so that values that miss a threshold by a rounding error, as values converted from other units or rates computed from doses, weights and durations can, are scored by the threshold they meet.

    The SOFA score at a timepoint is the sum of the worst score of every organ over the `window` timepoints that end at it, an organ without an input in this window counting as 0.
    It is scored from the first to the last timepoint at which any organ has an input, and missing at the other timepoints.

    The ratio pairs every PaO2 with the last FiO2 at most `pafi_window` timepoints before it, or `fio2_default` without one, and every FiO2 with the last PaO2 at most `pafi_window` timepoints before it, and keeps the lowest ratio of a timepoint.
    The urine output per `urine_window` is the urine of the `urine_window` timepoints that end at a timepoint, counting timepoints without urine as 0 from the first to the last timepoint with urine, extrapolated to `urine_window` timepoints from the timepoints between them in the window, if there are at least `urine_min_observed` of these.

    Every input is a variable of `layer`, or else a key of `obsm` with a value per observation and timepoint, such as the result of another tool.
    Windows count timepoints, and the defaults are for hourly data, on which 24 timepoints are a day.

    With `closed="both"` and `tolerance=0`, which compares as `ricu` does, the defaults give the SOFA score of the `ricu` R package :cite:p:`Bennett2023` on hourly data prepared as `ricu` prepares it:
    the lowest PaO2, platelets, mean arterial pressure and Glasgow Coma Scale, the highest FiO2, bilirubin, creatinine and vasopressor rates, and the urine output of every hour,
    ventilation from its windows, the Glasgow Coma Scale of sedated patients and vasopressor rates given for at least an hour as `ricu` defines them,
    and data of the whole stay, or `start_key` and `urine_period_keys` where the data hold only part of it.

    Args:
        edata: Central data object with longitudinal data.
        pao2: Variable of the arterial partial pressure of oxygen in mmHg.
        fio2: Variable of the fraction of inspired oxygen in the unit `fio2_unit`.
        ventilation: Variable that is non-zero at the timepoints of mechanical ventilation.
        platelets: Variable of the platelet count in 10^3/µL.
        bilirubin: Variable of the total bilirubin in mg/dL.
        mean_arterial_pressure: Variable of the mean arterial pressure in mmHg.
        dopamine: Variable of the dopamine rate in µg/kg/min, 0 or missing where none is given.
        dobutamine: Variable of the dobutamine rate in µg/kg/min.
        epinephrine: Variable of the epinephrine rate in µg/kg/min.
        norepinephrine: Variable of the norepinephrine rate in µg/kg/min.
        gcs: Variable of the Glasgow Coma Scale.
        creatinine: Variable of the serum creatinine in mg/dL.
        urine: Variable of the urine output in mL per timepoint.
            An input without a variable counts as missing at every timepoint.
        layer: The 3D layer to read the variables from.
            If `None`, `edata.X` is used.
        window: Timepoints, ending at every timepoint, over which every organ takes its worst score.
        pafi_window: Most timepoints between a PaO2 and the FiO2 it is divided by.
        fio2_unit: Unit of `fio2`, percent or a fraction of 1.
            A warning is raised if every FiO2 is at most 1 with `"percent"`.
        fio2_default: FiO2 of a PaO2 without one in `fio2_unit`, by default room air, 21% or 0.21.
            If `None`, a PaO2 without FiO2 gives no ratio.
        urine_window: Timepoints of the urine output, ending at every timepoint, to whose number the urine is extrapolated.
        urine_min_observed: Fewest timepoints of the urine window between the first and the last timepoint with urine to score it.
        urine_period_keys: Columns of `obs` with the position of the first and the last urine output of every observation, 0 for the first timepoint and rounded down.
            If the data hold only part of the urine outputs of an observation, these positions widen the period in which timepoints without urine count as 0.
        start_key: Column of `obs` with a position, 0 for the first timepoint and rounded down, from which an observation with an input is scored if that is before its first input, such as the position of its first input if the data start later.
        closed: Whether the windows of `window` and `urine_window` hold the timepoint and the `window` - 1 before it (`"right"`), or the timepoint and the `window` before it (`"both"`), as in `ricu`, whose window of a day includes both its ends.
        tolerance: Margin of the comparisons with the thresholds: a value within `tolerance` of a threshold counts as the threshold.
            The default is far below the precision of the records; `0` compares exactly.
        key_added: The SOFA score of every timepoint is stored in `edata.obsm[key_added]`, and the score of every organ before taking the worst of the window in `edata.obsm[f"{key_added}_{organ}"]` with the organs `respiration`, `coagulation`, `liver`, `cardiovascular`, `cns` and `renal`, missing at timepoints without an input of the organ.
        copy: Whether to return a copy of `edata` rather than modifying it in place.

    Returns:
        A copy of `edata` with the results if `copy` is `True`, else `None`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> import pandas as pd
        >>> X = np.array([[[200.0, 90.0, np.nan], [1.0, np.nan, 2.5]]])
        >>> var = pd.DataFrame(index=["platelets", "creatinine"])
        >>> edata = ed.EHRData(X=X, var=var)
        >>> ep.tl.sofa(edata, platelets="platelets", creatinine="creatinine")
        >>> edata.obsm["sofa"]
        array([[0., 2., 4.]])
    """
    arguments = {
        "pao2": pao2,
        "fio2": fio2,
        "ventilation": ventilation,
        "platelets": platelets,
        "bilirubin": bilirubin,
        "mean_arterial_pressure": mean_arterial_pressure,
        "dopamine": dopamine,
        "dobutamine": dobutamine,
        "epinephrine": epinephrine,
        "norepinephrine": norepinephrine,
        "gcs": gcs,
        "creatinine": creatinine,
        "urine": urine,
    }
    names = {name: var for name, var in arguments.items() if var is not None}
    if not names:
        raise ValueError(f"Name the variable of at least one input: {', '.join(INPUTS)}.")
    if fio2_unit not in ("percent", "fraction"):
        raise ValueError(f"fio2_unit must be 'percent' or 'fraction', not {fio2_unit!r}.")
    if urine_period_keys is not None:
        if isinstance(urine_period_keys, str) or len(urine_period_keys) != 2:
            raise ValueError(
                f"urine_period_keys must be two columns of obs, of the first and the last urine output, not {urine_period_keys!r}."
            )
        if urine is None:
            raise ValueError("urine_period_keys needs the variable of the urine output, `urine`.")
    settings = {
        "window": _count(window, "window", positive=True),
        "pafi_window": _count(pafi_window, "pafi_window"),
        "urine_window": _count(urine_window, "urine_window", positive=True),
        "urine_min_observed": _count(urine_min_observed, "urine_min_observed"),
        "closed": _closed(closed),
        "tolerance": _tolerance(tolerance),
        "fio2_default": _fio2_default(fio2_default, fio2_unit),
    }
    selected = _timeseries(edata, names, layer, "The SOFA score")

    # the positions of the timepoint from which every observation is scored and of its first and last urine
    periods = np.full((edata.n_obs, 3, 1), np.nan)
    if start_key is not None:
        periods[:, 0, 0] = _obs_positions(edata, start_key)
    if urine_period_keys is not None:
        for i, key in enumerate(urine_period_keys, start=1):
            periods[:, i, 0] = _obs_positions(edata, key)

    if edata.n_t == 0:
        scores = np.full((edata.n_obs, 1 + len(ORGANS), 0), np.nan)
    else:
        scores = _sofa_scores(
            selected,
            periods,
            inputs=tuple(names),
            fio2_factor=100.0 if fio2_unit == "fraction" else 1.0,
            **settings,
        )
        # the largest FiO2 of every observation, computed with the scores, to warn of fractions read as percent
        fio2_max = np.full(edata.n_obs, -np.inf)
        if fio2 is not None and fio2_unit == "percent":
            values = selected[:, list(names).index("fio2"), :]
            fio2_max = np.where(np.isnan(values), -np.inf, values).max(axis=1)
        scores, fio2_max = _materialize(scores, fio2_max)
        recorded = fio2_max[np.isfinite(fio2_max)]
        if recorded.size and recorded.max() <= 1:
            warnings.warn(
                "Every FiO2 is at most 1, which looks like a fraction; with fio2_unit='percent' it gives ratios 100 "
                "times too high. Pass fio2_unit='fraction' for fractions.",
                UserWarning,
                stacklevel=2,
            )

    edata = edata.copy() if copy else edata
    edata.obsm[key_added] = scores[:, 0, :]
    for i, organ in enumerate(ORGANS, start=1):
        edata.obsm[f"{key_added}_{organ}"] = scores[:, i, :]
    return edata if copy else None


BASELINES = ("minimum", "start", "zero")


def sepsis3(
    edata: EHRData,
    *,
    infection_key: str,
    sofa: str = "sofa",
    before: int = 48,
    after: int = 24,
    rise: float = 2,
    baseline: Literal["minimum", "start", "zero"] = "minimum",
    key_added: str = "sepsis3",
    copy: bool = False,
) -> EHRData | None:
    """Find the onset of sepsis by the Sepsis-3 definition, an acute rise of the SOFA score by at least 2 points around a suspected infection :cite:p:`Singer2016`.

    The onset is the first timepoint from `before` timepoints before the suspected infection to `after` timepoints after it at which the SOFA score has risen by at least `rise` points from its `baseline`, as in :cite:p:`Seymour2016`.
    The baseline is the lowest SOFA score of the window up to the timepoint, the first SOFA score of the window, or 0.
    The default, the lowest score, is the choice of the `ricu` R package :cite:p:`Bennett2023`; :cite:p:`Singer2016` assume a baseline of 0 in patients not known to have organ dysfunction before the infection, which `baseline="zero"` gives.
    Windows count timepoints, and the defaults are for hourly data.

    The defaults give the Sepsis-3 onset of `ricu` at the first suspected infection, from its SOFA score.

    Args:
        edata: Central data object with the SOFA score of every timepoint, as written by :func:`~ehrapy.tools.sofa`.
        infection_key: Column of `obs` with the position of the suspected infection of every observation, 0 for the first timepoint and rounded down, missing for observations without one.
        sofa: The SOFA score of every observation and timepoint is read from `edata.obsm[sofa]`, missing where there is none.
        before: Timepoints before the suspected infection in which the SOFA score is followed.
        after: Timepoints after the suspected infection in which the SOFA score is followed.
        rise: Smallest rise of the SOFA score from the baseline, more than 0.
        baseline: The lowest SOFA score of the window up to every timepoint (`"minimum"`), the first SOFA score of the window (`"start"`), or 0 (`"zero"`).
        key_added: Whether an observation has sepsis is stored in `edata.obs[key_added]` and the position of its onset in `edata.obs[f"{key_added}_onset"]`, missing for observations without one.
        copy: Whether to return a copy of `edata` rather than modifying it in place.

    Returns:
        A copy of `edata` with the results if `copy` is `True`, else `None`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> import pandas as pd
        >>> obs = pd.DataFrame({"infection": [3.0, np.nan]}, index=["a", "b"])
        >>> edata = ed.EHRData(shape=(2, 0), obs=obs, tem=pd.DataFrame(index=[f"t{i}" for i in range(6)]))
        >>> edata.obsm["sofa"] = np.array([[1, 0, 1, 2, 3, 3], [0, 0, 4, 4, 4, 4]], dtype=float)
        >>> ep.tl.sepsis3(edata, infection_key="infection")
        >>> edata.obs[["sepsis3", "sepsis3_onset"]]
           sepsis3  sepsis3_onset
        a     True            3.0
        b    False            NaN
    """
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {', '.join(map(repr, BASELINES))}, not {baseline!r}.")
    if isinstance(rise, bool) or not isinstance(rise, Real) or not np.isfinite(rise) or rise <= 0:
        raise ValueError(f"rise must be a number of more than 0 points, not {rise!r}.")
    if sofa not in edata.obsm:
        raise KeyError(f"edata.obsm has no {sofa!r}; score the data with `ep.tl.sofa` first.")
    (scores,) = _materialize(_obsm_values(edata, sofa))
    scores = np.asarray(scores, dtype=np.float64)
    before, after = _count(before, "before"), _count(after, "after")
    infection = _obs_positions(edata, infection_key)
    onset = np.full(edata.n_obs, np.nan)
    with np.errstate(invalid="ignore"):
        lo = np.maximum(infection - before, 0)
        hi = np.minimum(infection + after, edata.n_t - 1)
    for i in np.flatnonzero(np.isfinite(infection) & (lo <= hi)):
        window = scores[i, int(lo[i]) : int(hi[i]) + 1]
        followed = np.where(np.isnan(window), np.inf, window)
        if baseline == "minimum":
            reference = np.minimum.accumulate(followed)
        elif baseline == "start":
            reference = followed[np.argmax(np.isfinite(followed))]
        else:
            reference = 0.0
        with np.errstate(invalid="ignore"):
            (risen,) = np.nonzero(window - reference >= rise)
        if risen.size:
            onset[i] = lo[i] + risen[0]

    edata = edata.copy() if copy else edata
    edata.obs[key_added] = ~np.isnan(onset)
    edata.obs[f"{key_added}_onset"] = onset
    return edata if copy else None
