from __future__ import annotations

from fractions import Fraction

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute

INPUTS = ["creatinine", "urine"]
RICU_CRITERIA = ((1, 6, 0.5 / 6), (2, 12, 0.5 / 12), (3, 24, 0.3 / 24))
nan = np.nan


def _edata(n_t: int, *, weight=10.0, obs: pd.DataFrame | None = None, **inputs):
    """Observations x timepoints of every given input, the other inputs missing, with the weight in `obs`."""
    n_obs = np.atleast_2d(next(iter(inputs.values()))).shape[0]
    X = np.full((n_obs, len(INPUTS), n_t), nan)
    for name, values in inputs.items():
        X[:, INPUTS.index(name), :] = np.atleast_2d(values)
    obs = pd.DataFrame(index=[f"o{i}" for i in range(n_obs)]) if obs is None else obs
    obs["weight"] = weight
    return ed.EHRData(
        shape=(n_obs, len(INPUTS)), obs=obs, var=pd.DataFrame(index=INPUTS), layers={DEFAULT_TEM_LAYER_NAME: X}
    )


def _kdigo(edata: ed.EHRData, **kwargs) -> ed.EHRData:
    """`edata` staged with both inputs named like their variables, unless `kwargs` names them otherwise."""
    arguments = {"creatinine": "creatinine", "urine": "urine", "weight_key": "weight"}
    ep.tl.kdigo(edata, layer=DEFAULT_TEM_LAYER_NAME, **{**arguments, **kwargs})
    return edata


@pytest.mark.parametrize(
    ("lowest", "value", "expected"),
    [
        (1.0, 1.29, 0),
        (1.0, 1.3, 1),  # a rise of 0.3
        (1.0, 1.49, 1),
        (1.0, 1.5, 1),
        (1.0, 1.99, 1),
        (1.0, 2.0, 2),  # 2 times the baseline
        (1.0, 2.99, 2),
        (1.0, 3.0, 3),  # 3 times the baseline
        (3.0, 3.29, 0),
        (3.0, 3.3, 1),
        (2.0, 3.99, 1),
        (2.0, 4.0, 3),  # 4 mg/dL with a rise
        (3.8, 4.0, 0),  # 4 mg/dL without a rise of 0.3 or 1.5 times the baseline
        (1.0, 0.5, 0),
    ],
)
def test_kdigo_creatinine_thresholds(lowest, value, expected):
    edata = _kdigo(_edata(2, creatinine=[lowest, value]), urine=None, weight_key=None)

    np.testing.assert_array_equal(edata.obsm["kdigo_creatinine"], [[0, expected]])
    np.testing.assert_array_equal(edata.obsm["kdigo"], [[0, expected]])
    assert np.isnan(edata.obsm["kdigo_urine"]).all()


@pytest.mark.parametrize(
    ("values", "rise_window", "exact", "default"),
    [
        ([1.1, 1.4], 48, 0, 1),  # 1.4 - 1.1 is 0.29999999999999982
        ([0.8, 1.2], 1, 0, 1),  # 1.5 x 0.8 is 1.2000000000000002; the rise window holds 1.2 alone
        ([3.7, 4.0], 48, 1, 3),  # 4.0 - 3.7 is 0.2999999999999998, which stage 3 needs; 3.7 + 0.3 is 4.0
    ],
)
def test_kdigo_tolerance_reaches_decimal_thresholds(values, rise_window, exact, default):
    arguments = {"urine": None, "weight_key": None, "rise_window": rise_window}
    staged = _kdigo(_edata(2, creatinine=values), **arguments)
    exactly = _kdigo(_edata(2, creatinine=values), **arguments, tolerance=0)

    np.testing.assert_array_equal(staged.obsm["kdigo_creatinine"], [[0, default]])
    np.testing.assert_array_equal(exactly.obsm["kdigo_creatinine"], [[0, exact]])


def test_kdigo_tolerance_of_the_urine():
    # 10 kg: 6 timepoints of 4.99 mL is 2.994 mL/kg, below the 3.0 mL/kg of 0.5 mL/kg per timepoint, but within a
    # tolerance of 0.01
    edata = _kdigo(_edata(7, urine=np.full(7, 4.99)), creatinine=None, tolerance=0)
    wide = _kdigo(_edata(7, urine=np.full(7, 4.99)), creatinine=None, tolerance=0.01)

    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], [[0] * 6 + [1]])
    np.testing.assert_array_equal(wide.obsm["kdigo_urine"], [[0] * 7])


@pytest.mark.parametrize(
    ("rise_window", "baseline_window", "closed", "expected"),
    [
        (2, 5, "right", 0),  # both windows leave out the timepoint exactly their length before
        (3, 5, "right", 1),  # a rise of 0.35 over 1.25 at t3
        (2, 6, "right", 1),  # 1.6 times the baseline 1.0 at t0
        (np.int64(3), np.int64(5), "right", 1),
        (2, 4, "both", 1),  # the rise window of 2 holds t3
        (1, 5, "both", 1),  # the baseline window of 5 holds t0
        (1, 4, "both", 0),
    ],
)
def test_kdigo_creatinine_windows(rise_window, baseline_window, closed, expected):
    creatinine = [1.0, nan, nan, 1.25, nan, 1.6]
    edata = _kdigo(
        _edata(6, creatinine=creatinine),
        urine=None,
        weight_key=None,
        rise_window=rise_window,
        baseline_window=baseline_window,
        closed=closed,
    )

    np.testing.assert_array_equal(edata.obsm["kdigo_creatinine"], [[0, nan, nan, 0, nan, expected]])


def test_kdigo_creatinine_baseline_is_the_lowest_of_the_window():
    creatinine = [2.0, 1.0, nan, 2.1, 2.4, 3.0]
    edata = _kdigo(_edata(6, creatinine=creatinine), urine=None, weight_key=None, rise_window=1)

    # a rise window of 1 holds only the timepoint itself, so the baseline 1.0 of t1 gives the stages
    np.testing.assert_array_equal(edata.obsm["kdigo_creatinine"], [[0, 0, nan, 2, 2, 3]])


@pytest.mark.parametrize(
    ("volume", "expected"),
    [
        (5.0, [0] * 30),  # 0.5 mL/kg per timepoint is not below 0.5
        (4.99, [0] * 6 + [1] * 6 + [2] * 18),
        (3.0, [0] * 6 + [1] * 6 + [2] * 18),  # 0.3 mL/kg per timepoint is not below 0.3
        (2.99, [0] * 6 + [1] * 6 + [2] * 12 + [3] * 6),
        (0.0, [0] * 6 + [1] * 6 + [3] * 18),  # anuria for 12 timepoints
    ],
)
def test_kdigo_urine_thresholds(volume, expected):
    # 10 kg: a volume of 5 mL per timepoint is 0.5 mL/kg per timepoint
    edata = _kdigo(_edata(30, urine=np.full(30, volume)), creatinine=None)

    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], [expected])
    np.testing.assert_array_equal(edata.obsm["kdigo"], [expected])
    assert np.isnan(edata.obsm["kdigo_creatinine"]).all()


def test_kdigo_urine_without_anuria():
    edata = _kdigo(_edata(30, urine=np.zeros(30)), creatinine=None, anuria_window=None)

    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], [[0] * 6 + [1] * 6 + [2] * 12 + [3] * 6])


def test_kdigo_urine_criteria():
    edata = _kdigo(
        _edata(30, urine=np.full(30, 2.0)), creatinine=None, urine_criteria=[(2, 3, 0.25)], anuria_window=None
    )

    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], [[0] * 3 + [2] * 27])


def test_kdigo_ricu_criteria_compare_the_urine_per_kg_of_a_window():
    urine = np.full(30, 0.08)  # 0.48 mL/kg over 6 timepoints, 0.96 over 12
    guideline = _kdigo(_edata(30, weight=1.0, urine=urine), creatinine=None)
    ricu = _kdigo(_edata(30, weight=1.0, urine=urine), creatinine=None, urine_criteria=RICU_CRITERIA)

    np.testing.assert_array_equal(guideline.obsm["kdigo_urine"], [[0] * 6 + [1] * 6 + [2] * 12 + [3] * 6])
    np.testing.assert_array_equal(ricu.obsm["kdigo_urine"], [[0] * 6 + [1] * 24])


def test_kdigo_anuria_window_is_left_open():
    urine = np.zeros(16)
    urine[2] = 5.0
    edata = _kdigo(_edata(16, urine=urine), creatinine=None)

    # at t13 the window of 12 still holds the 5 mL of t2; at t14 it no longer does
    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], [[0] * 6 + [1] * 6 + [2] * 2 + [3] * 2])


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("right", [0] * 6 + [1] * 6 + [3] * 2),  # the windows of 6 at t6 and of 12 at t12 leave out t0
        ("both", [0] * 7 + [1] * 6 + [3]),  # they hold t0 and its 100 mL
    ],
)
def test_kdigo_closed_urine_windows(closed, expected):
    edata = _kdigo(_edata(14, urine=[100.0] + [0.0] * 13), creatinine=None, closed=closed)

    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], [expected])


def test_kdigo_urine_start():
    obs = pd.DataFrame({"start": [3.0, 3.9, nan, -100.0, -0.5, 25.0]}, index=list("abcdef"))
    obs["integer_start"] = pd.array([3, 3, None, -100, -1, 25], dtype="Int64")
    edata = _kdigo(_edata(20, obs=obs, urine=np.zeros((6, 20))), creatinine=None, urine_start_key="start")

    from_3 = [0] * 9 + [1] * 6 + [3] * 5
    from_0 = [0] * 6 + [1] * 6 + [3] * 8
    # positions are rounded down, and a missing start or one before the first timepoint is the first timepoint
    expected = [from_3, from_3, from_0, from_0, from_0, [0] * 20]
    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], expected)
    _kdigo(edata, creatinine=None, urine_start_key="integer_start")
    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], expected)


def test_kdigo_urine_start_before_the_first_timepoint():
    # 10 kg: 10 mL per timepoint is 1 mL/kg per timepoint, above every rate
    obs = pd.DataFrame({"start": [-100.0, 0.0]}, index=["a", "b"])
    edata = _kdigo(_edata(30, obs=obs, urine=np.full((2, 30), 10.0)), creatinine=None, urine_start_key="start")

    # the windows hold no unrecorded timepoints before the data, which would count as no urine
    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], np.zeros((2, 30)))


@pytest.mark.parametrize(
    "start",
    [["3"], [True], pd.to_timedelta([3], unit="h"), pd.to_datetime(["2020-01-01"])],
    ids=["string", "bool", "timedelta", "datetime"],
)
def test_kdigo_urine_start_must_be_positions(start):
    obs = pd.DataFrame({"start": start}, index=["a"])
    with pytest.raises(TypeError, match="must hold the positions of timepoints as numbers"):
        _kdigo(_edata(2, obs=obs, urine=[1.0, 1.0]), creatinine=None, urine_start_key="start")


def test_kdigo_urine_staged_at_timepoints_with_urine():
    urine = np.full(21, nan)
    urine[[2, 13, 20]] = [100.0, 0.0, 0.0]
    edata = _kdigo(_edata(21, urine=urine), creatinine=None)

    # timepoints without urine count as none: at t13 the window of 12 holds the 100 mL of t2, at t20 no urine
    expected = np.full(21, nan)
    expected[[2, 13, 20]] = [0, 1, 3]
    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], [expected])


def test_kdigo_urine_min_observed_leaves_sparse_windows_unstaged():
    # 10 kg: 2 mL at every timepoint and 8 mL at every fourth are both 0.2 mL/kg per timepoint
    sparse = np.full(24, nan)
    sparse[::4] = 8.0
    urine = np.stack([np.full(24, 2.0), sparse])
    every = _kdigo(_edata(24, urine=urine), creatinine=None).obsm["kdigo_urine"]
    observed = _kdigo(_edata(24, urine=urine), creatinine=None, urine_min_observed=3).obsm["kdigo_urine"]

    dense = [0] * 6 + [1] * 6 + [2] * 12
    np.testing.assert_array_equal(every[0], dense)
    np.testing.assert_array_equal(observed[0], dense)
    np.testing.assert_array_equal(every[1, ::4], [0, 0, 1, 2, 2, 2])
    # a window of 6 holds at most 2 timepoints with urine, one of 12 holds 3
    np.testing.assert_array_equal(observed[1, ::4], [0, 0, 0, 2, 2, 2])
    assert np.isnan(observed[1, np.isnan(sparse)]).all()


def test_kdigo_urine_min_observed_anuria_needs_recorded_zeros():
    urine = np.zeros((2, 14))
    urine[1, 1:13] = nan
    every = _kdigo(_edata(14, urine=urine), creatinine=None).obsm["kdigo_urine"]
    observed = _kdigo(_edata(14, urine=urine), creatinine=None, urine_min_observed=6).obsm["kdigo_urine"]

    recorded = [0] * 6 + [1] * 6 + [3] * 2
    np.testing.assert_array_equal(every[0], recorded)
    np.testing.assert_array_equal(observed[0], recorded)
    # zeros at t0 and t13 alone: unrecorded timepoints count as no urine only without urine_min_observed
    np.testing.assert_array_equal(every[1, [0, 13]], [0, 3])
    np.testing.assert_array_equal(observed[1, [0, 13]], [0, 0])


def test_kdigo_missing_weight():
    creatinine = np.tile([1.0, nan, 2.0], (4, 1))
    edata = _kdigo(_edata(3, weight=[10.0, nan, 0.0, -1.0], creatinine=creatinine, urine=np.zeros((4, 3))))

    np.testing.assert_array_equal(edata.obsm["kdigo_urine"][0], [0, 0, 0])
    assert np.isnan(edata.obsm["kdigo_urine"][1:]).all()
    np.testing.assert_array_equal(edata.obsm["kdigo"], [[0, 0, 2]] + [[0, nan, 2]] * 3)


def test_kdigo_stage_is_the_higher_of_both():
    creatinine = [1.0, nan, nan, nan, nan, nan, nan, 2.5]
    urine = [0.0, 0.0, 0.0, nan, 0.0, 0.0, 0.0, nan]
    edata = _kdigo(_edata(8, creatinine=creatinine, urine=urine))

    np.testing.assert_array_equal(edata.obsm["kdigo_creatinine"], [[0] + [nan] * 6 + [2]])
    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], [[0, 0, 0, nan, 0, 0, 1, nan]])
    np.testing.assert_array_equal(edata.obsm["kdigo"], [[0, 0, 0, nan, 0, 0, 1, 2]])


def _mean_urine_stage(urine, weight, criteria, *, start=0, min_observed=None):
    """The stage by urine output of `urine` at every timepoint, window by window, without anuria."""
    stage = np.full(urine.shape, nan)
    for t in np.flatnonzero(~np.isnan(urine)):
        stage[t] = 0
        for level, window, rate in criteria:
            values = urine[max(t - window + 1, 0) : t + 1]
            observed = np.count_nonzero(~np.isnan(values))
            # exactly, so that no rounding decides a window whose urine lies at the threshold
            mean = Fraction(np.nansum(values)) / Fraction(weight) / window
            if t - start >= window and observed >= (min_observed or 0) and mean < Fraction(str(rate)):
                stage[t] = max(stage[t], level)
    return stage


@pytest.mark.parametrize("urine_min_observed", [None, 4])
def test_kdigo_urine_stage_is_the_mean_urine_output_of_every_window(rng, urine_min_observed):
    urine = rng.choice([0.0, 2.0, 5.0, 20.0], size=(4, 80))
    urine[rng.random(urine.shape) < 0.2] = nan
    weight = [10.0, 8.0, 25.0, 4.0]
    obs = pd.DataFrame({"start": [0.0, 5.0, -3.0, nan]}, index=list("abcd"))
    first = [0, 5, 0, 0]  # a start before the first timepoint or missing is the first timepoint
    criteria = [(1, 6, 0.5), (2, 12, 0.5), (3, 24, 0.3)]
    edata = _kdigo(
        _edata(80, weight=weight, obs=obs, urine=urine),
        creatinine=None,
        urine_criteria=criteria,
        anuria_window=None,
        urine_start_key="start",
        urine_min_observed=urine_min_observed,
    )

    expected = np.stack(
        [
            _mean_urine_stage(urine[i], weight[i], criteria, start=first[i], min_observed=urine_min_observed)
            for i in range(4)
        ]
    )
    np.testing.assert_array_equal(edata.obsm["kdigo_urine"], expected)
    assert set(np.unique(expected[~np.isnan(expected)])) == {0, 1, 2, 3}


def test_kdigo_windows_count_timepoints_whatever_their_times(rng):
    creatinine = rng.uniform(0.5, 5, size=(3, 200))
    creatinine[rng.random(creatinine.shape) < 0.8] = nan
    urine = rng.choice([0.0, 1.0, 3.0, 5.0, 50.0], size=(3, 200))
    urine[rng.random(urine.shape) < 0.3] = nan
    expected = _kdigo(_edata(200, creatinine=creatinine, urine=urine)).obsm["kdigo"]
    edata = _edata(200, creatinine=creatinine, urine=urine)
    edata.tem = pd.DataFrame({"time_value": np.cumsum(rng.choice([0.5, 1.0, 3.0], size=200))})

    np.testing.assert_array_equal(_kdigo(edata).obsm["kdigo"], expected)
    assert set(np.unique(expected[~np.isnan(expected)])) == {0, 1, 2, 3}


def test_kdigo_without_timepoints():
    edata = _kdigo(_edata(0, creatinine=np.empty((2, 0))))

    for key in ("kdigo", "kdigo_creatinine", "kdigo_urine"):
        assert edata.obsm[key].shape == (2, 0)


def test_kdigo_inputs_from_obsm(rng):
    creatinine = rng.uniform(0.5, 5, size=(4, 40))
    urine = rng.choice([0.0, 2.0, 50.0], size=(4, 40))
    edata = _edata(40, creatinine=creatinine, urine=urine)
    expected = _kdigo(edata.copy()).obsm["kdigo"]
    edata.obsm["crea_series"] = creatinine.copy()
    _kdigo(edata, creatinine="crea_series")

    np.testing.assert_array_equal(edata.obsm["kdigo"], expected)


def test_kdigo_inputs_from_obsm_only():
    obs = pd.DataFrame({"weight": [10.0]}, index=["a"])
    # `tem` gives the number of timepoints
    edata = ed.EHRData(shape=(1, 0), obs=obs, tem=pd.DataFrame(index=range(7)))
    edata.obsm["crea"] = np.array([[1.0, nan, 2.0, nan, nan, nan, nan]])
    edata.obsm["uo"] = np.zeros((1, 7))
    ep.tl.kdigo(edata, creatinine="crea", urine="uo", weight_key="weight")

    np.testing.assert_array_equal(edata.obsm["kdigo"], [[0, 0, 2, 0, 0, 0, 1]])


def test_kdigo_dask_layer_with_obsm_input(rng):
    da = pytest.importorskip("dask.array")
    creatinine = rng.uniform(0.5, 5, size=(4, 30))
    urine = rng.choice([0.0, 2.0, 50.0], size=(4, 30))
    edata = _edata(30, creatinine=creatinine, urine=urine)
    expected = _kdigo(edata.copy()).obsm["kdigo"]
    edata.obsm["crea_series"] = creatinine.copy()
    edata.layers[DEFAULT_TEM_LAYER_NAME] = da.from_array(edata.layers[DEFAULT_TEM_LAYER_NAME], chunks=(2, -1, -1))
    _kdigo(edata, creatinine="crea_series")

    np.testing.assert_array_equal(np.asarray(edata.obsm["kdigo"]), expected)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize(
    "settings",
    [{}, {"closed": "both", "urine_min_observed": 3, "urine_start_key": "start"}],
    ids=["defaults", "settings"],
)
def test_kdigo_array_types(array_type, rng, settings):
    shape = (6, len(INPUTS), 60)
    X = np.stack([rng.uniform(0.5, 5, size=shape[::2]), rng.choice([0.0, 2.0, 50.0], size=shape[::2])], axis=1)
    X[rng.random(shape) < 0.5] = nan

    def edata():
        obs = pd.DataFrame({"start": [0.0, 4.0, nan, -2.0, 10.5, 70.0]}, index=list("abcdef"))
        weight = [10.0, nan, 70.0, 50.0, 0.0, 80.0]
        return _edata(60, weight=weight, obs=obs, **{name: X[:, i] for i, name in enumerate(INPUTS)})

    expected = _kdigo(edata(), **settings)
    lazy = edata()
    lazy.layers[DEFAULT_TEM_LAYER_NAME] = array_type(lazy.layers[DEFAULT_TEM_LAYER_NAME])

    with forbid_dask_compute(allowed=1):
        _kdigo(lazy, **settings)
    for key in ("kdigo", "kdigo_creatinine", "kdigo_urine"):
        np.testing.assert_array_equal(lazy.obsm[key], expected.obsm[key])


def test_kdigo_sparse_3D_layer_not_implemented():
    sparse = pytest.importorskip("sparse")
    edata = _edata(2, creatinine=[1.0, 2.0])
    edata.layers[DEFAULT_TEM_LAYER_NAME] = sparse.COO.from_numpy(np.nan_to_num(edata.layers[DEFAULT_TEM_LAYER_NAME]))
    with pytest.raises(NotImplementedError, match="does not support array type COO"):
        _kdigo(edata)


def test_kdigo_copy():
    edata = _edata(2, creatinine=[1.0, 2.0])
    result = ep.tl.kdigo(edata, creatinine="creatinine", layer=DEFAULT_TEM_LAYER_NAME, key_added="aki", copy=True)

    assert "aki" not in edata.obsm
    np.testing.assert_array_equal(result.obsm["aki"], [[0, 2]])
    assert set(result.obsm) == {"aki", "aki_creatinine", "aki_urine"}


@pytest.mark.parametrize(
    "window", ["rise_window", "baseline_window", "anuria_window", "urine_criteria", "urine_min_observed"]
)
@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (0, ValueError, "more than 0 timepoints"),
        (-6, ValueError, "more than 0 timepoints"),
        (6.0, TypeError, "whole number of timepoints"),
        (True, TypeError, "whole number of timepoints"),
        ("6h", TypeError, "whole number of timepoints"),
    ],
)
def test_kdigo_windows_are_numbers_of_timepoints(window, value, error, message):
    edata = _edata(2, creatinine=[1.0, 2.0], urine=[1.0, 1.0])
    arguments = {"urine_criteria": [(1, value, 0.5)]} if window == "urine_criteria" else {window: value}
    with pytest.raises(error, match=message):
        _kdigo(edata, **arguments)


def test_kdigo_errors():
    edata = _edata(2, creatinine=[1.0, 2.0], urine=[1.0, 1.0])
    with pytest.raises(ValueError, match="Name the variable"):
        _kdigo(edata, creatinine=None, urine=None, weight_key=None)
    with pytest.raises(ValueError, match="urine needs the weight"):
        _kdigo(edata, weight_key=None)
    with pytest.raises(ValueError, match="need the variable of the urine output"):
        _kdigo(edata, urine=None)
    for argument in ({"urine_start_key": "start"}, {"urine_min_observed": 3}):
        with pytest.raises(ValueError, match="need the variable of the urine output"):
            _kdigo(edata, urine=None, weight_key=None, **argument)
    with pytest.raises(KeyError, match="obs has no column 'kg'"):
        _kdigo(edata, weight_key="kg")
    with pytest.raises(KeyError, match="obs has no column 'start'"):
        _kdigo(edata, urine_start_key="start")
    with pytest.raises(KeyError, match="Not found in edata.var_names or edata.obsm"):
        _kdigo(edata, creatinine="crea")
    for tolerance in (-1e-9, True, np.inf, nan, "0"):
        with pytest.raises(ValueError, match="tolerance must be a number of at least 0"):
            _kdigo(edata, tolerance=tolerance)
    for closed in ("left", "neither", None):
        with pytest.raises(ValueError, match="closed must be 'right' or 'both'"):
            _kdigo(edata, closed=closed)
    with pytest.raises(ValueError, match="3D"):
        ep.tl.kdigo(ed.EHRData(np.zeros((1, 2)), var=pd.DataFrame(index=INPUTS)), creatinine="creatinine")
    for stage in (4, 0, True, 1.0, "1"):
        with pytest.raises(ValueError, match="stage of a urine criterion must be the integer 1, 2 or 3"):
            _kdigo(edata, urine_criteria=[(stage, 6, 0.5)])
    for rate in (-0.5, np.nan, np.inf, True, "0.5"):
        with pytest.raises(ValueError, match="rate of a urine criterion"):
            _kdigo(edata, urine_criteria=[(1, 6, rate)])
    with pytest.raises(ValueError, match="tuple of a stage, a window and a rate"):
        _kdigo(edata, urine_criteria=[(1, 6)])


@pytest.mark.parametrize("closed", ["right", "both"])
def test_kdigo_urine_rate_is_per_timepoint_of_the_window(closed):
    """A constant output just below 0.5 mL/kg per timepoint reaches stage 1 over 6 and stage 2 over 12 timepoints,
    whether the windows hold their length in timepoints or one more."""
    edata = ed.EHRData(
        shape=(1, 1),
        obs=pd.DataFrame({"weight": [10.0]}, index=["a"]),
        var=pd.DataFrame(index=["urine"]),
        layers={DEFAULT_TEM_LAYER_NAME: np.full((1, 1, 14), 4.5)},
    )
    ep.tl.kdigo(edata, urine="urine", weight_key="weight", layer=DEFAULT_TEM_LAYER_NAME, closed=closed)
    np.testing.assert_array_equal(edata.obsm["kdigo_urine"][0], [0] * 6 + [1] * 6 + [2, 2])
