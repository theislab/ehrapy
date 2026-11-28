from __future__ import annotations

import warnings

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute

INPUTS = [
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
]
ORGANS = ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]
# obs columns that do not hold positions
NOT_POSITIONS = [
    ["1", "2"],
    [True, False],
    pd.to_timedelta([1, 2], unit="h"),
    pd.to_datetime(["2020-01-01", "2020-01-02"]),
]
nan = np.nan


def _timepoints(n_t: int) -> pd.DataFrame:
    return pd.DataFrame(index=[f"t{i}" for i in range(n_t)])


def _edata(n_t: int, *, obs: pd.DataFrame | None = None, **inputs) -> ed.EHRData:
    """Observations x timepoints of every given input, the other inputs missing."""
    n_obs = np.atleast_2d(next(iter(inputs.values()))).shape[0]
    X = np.full((n_obs, len(INPUTS), n_t), nan)
    for name, values in inputs.items():
        X[:, INPUTS.index(name), :] = np.atleast_2d(values)
    obs = pd.DataFrame(index=[f"o{i}" for i in range(n_obs)]) if obs is None else obs
    return ed.EHRData(
        shape=(n_obs, len(INPUTS)), obs=obs, var=pd.DataFrame(index=INPUTS), layers={DEFAULT_TEM_LAYER_NAME: X}
    )


def _sofa(edata: ed.EHRData, **kwargs) -> ed.EHRData:
    """`edata` scored with every input named like its variable, unless `kwargs` names it otherwise."""
    ep.tl.sofa(edata, layer=DEFAULT_TEM_LAYER_NAME, **{**{name: name for name in INPUTS}, **kwargs})
    return edata


@pytest.mark.parametrize(
    ("variable", "organ", "values", "expected"),
    [
        ("platelets", "coagulation", [150, 149.9, 100, 99, 50, 49, 20, 19.9], [0, 1, 1, 2, 2, 3, 3, 4]),
        ("bilirubin", "liver", [1.19, 1.2, 1.99, 2, 6, 11.9, 12, 20], [0, 1, 1, 2, 3, 3, 4, 4]),
        ("gcs", "cns", [15, 14, 13, 12, 10, 9, 6, 5], [0, 1, 1, 2, 2, 3, 3, 4]),
        ("creatinine", "renal", [1.19, 1.2, 1.99, 2, 3.5, 4.99, 5, 9], [0, 1, 1, 2, 3, 3, 4, 4]),
        ("mean_arterial_pressure", "cardiovascular", [70, 69.9, 120, 40, 70, 70, 70, 70], [0, 1, 0, 1, 0, 0, 0, 0]),
    ],
)
def test_sofa_organ_thresholds(variable, organ, values, expected):
    edata = _sofa(_edata(len(values), **{variable: values}))

    np.testing.assert_array_equal(edata.obsm[f"sofa_{organ}"], [expected])
    for other in set(ORGANS) - {organ}:
        assert np.isnan(edata.obsm[f"sofa_{other}"]).all()


def test_sofa_cardiovascular_vasopressors():
    pressure = [80, 80, 80, 80, 80, 80, 80, 60, nan]
    dopamine = [5, 5.1, 15, 15.1, 0, 0, 0, 0, 3]
    dobutamine = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    epinephrine = [0, 0, 0, 0, 0, 0.1, 0.11, 0, 0]
    norepinephrine = [0, 0, 0, 0, 0, 0, 0, 0.05, 0]
    edata = _sofa(
        _edata(
            9,
            mean_arterial_pressure=pressure,
            dopamine=dopamine,
            dobutamine=dobutamine,
            epinephrine=epinephrine,
            norepinephrine=norepinephrine,
        )
    )

    np.testing.assert_array_equal(edata.obsm["sofa_cardiovascular"], [[2, 3, 3, 4, 2, 3, 4, 3, 2]])


@pytest.mark.parametrize(
    ("inputs", "kwargs", "organ", "scored", "exact"),
    [
        # 3.5 mg/dL creatinine and 6 mg/dL bilirubin converted from µmol/L are 3.4999999999999996 and 5.999999999999999
        ({"creatinine": 309.4 / 88.4}, {}, "renal", 3, 2),
        ({"bilirubin": 102.6 / 17.1}, {}, "liver", 3, 2),
        # a rate of 0.1 µg/kg/min computed as 0.10000000000000002
        ({"mean_arterial_pressure": 80, "norepinephrine": 0.1 * 3 / 3}, {}, "cardiovascular", 3, 4),
        # 84 mmHg on 0.28 is 299.99999999999994
        ({"pao2": 84, "fio2": 0.28}, {"fio2_unit": "fraction"}, "respiration", 1, 2),
        ({"platelets": 150 - 1e-12}, {}, "coagulation", 0, 1),
        ({"mean_arterial_pressure": 70 - 1e-12}, {}, "cardiovascular", 0, 1),
        ({"gcs": 15 - 1e-12}, {}, "cns", 0, 1),
        ({"urine": 500 - 1e-12}, {"urine_window": 1, "urine_min_observed": 1}, "renal", 0, 3),
    ],
)
def test_sofa_value_a_rounding_error_from_a_threshold(inputs, kwargs, organ, scored, exact):
    def score(**tolerance) -> np.ndarray:
        edata = _edata(1, **{name: [[value]] for name, value in inputs.items()})
        return _sofa(edata, **kwargs, **tolerance).obsm[f"sofa_{organ}"]

    np.testing.assert_array_equal(score(), [[scored]])
    np.testing.assert_array_equal(score(tolerance=0), [[exact]])


def test_sofa_tolerance_is_the_margin_of_the_thresholds():
    edata = _edata(2, mean_arterial_pressure=[80, 80], norepinephrine=[0.1 + 1e-12, 0.1 + 1e-6])

    np.testing.assert_array_equal(_sofa(edata.copy()).obsm["sofa_cardiovascular"], [[3, 4]])
    np.testing.assert_array_equal(_sofa(edata, tolerance=1e-5).obsm["sofa_cardiovascular"], [[3, 3]])


@pytest.mark.parametrize("tolerance", [-1e-9, np.inf, nan, True, "0"])
def test_sofa_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance must be a number of at least 0"):
        _sofa(_edata(1, platelets=[[90]]), tolerance=tolerance)


def test_sofa_respiration_needs_ventilation():
    pao2 = [80, 80, 45, 45, 300, nan]
    fio2 = [50, 50, 50, 50, 50, nan]
    ventilation = [0, 1, 0, 1, 0, 1]
    edata = _sofa(_edata(6, pao2=pao2, fio2=fio2, ventilation=ventilation), pafi_window=0)

    # ratios 160, 160, 90, 90, 600, none; ventilation alone scores 0
    np.testing.assert_array_equal(edata.obsm["sofa_respiration"], [[2, 3, 2, 4, 0, 0]])


def test_sofa_pafi_pairs_values_within_pafi_window():
    pao2 = [nan, nan, nan, 63, nan, nan, 60]
    fio2 = [30, nan, nan, nan, nan, 40, nan]
    edata = _sofa(_edata(7, pao2=pao2, fio2=fio2, ventilation=np.ones(7)))

    # t3: the FiO2 of t0 is 3 timepoints old, so room air: 300; t5: the PaO2 of t3 gives 157.5; t6: 60 / 40 = 150.
    # Ventilation without a ratio scores 0.
    respiration = edata.obsm["sofa_respiration"]
    np.testing.assert_array_equal(respiration, [[0, 0, 0, 1, 0, 3, 3]])

    without_default = _sofa(_edata(7, pao2=pao2, fio2=fio2, ventilation=np.ones(7)), fio2_default=None)
    np.testing.assert_array_equal(without_default.obsm["sofa_respiration"], [[0, 0, 0, 0, 0, 3, 3]])


def test_sofa_fio2_as_fraction():
    percent = _sofa(_edata(3, pao2=[80, 63, 90], fio2=[50, nan, 30], ventilation=np.ones(3)))
    fraction = _sofa(_edata(3, pao2=[80, 63, 90], fio2=[0.5, nan, 0.3], ventilation=np.ones(3)), fio2_unit="fraction")

    np.testing.assert_array_equal(fraction.obsm["sofa_respiration"], percent.obsm["sofa_respiration"])


@pytest.mark.parametrize(
    ("fio2_unit", "fio2_default", "expected"),
    [
        ("percent", "room air", 2),
        ("fraction", "room air", 2),
        ("percent", 21, 2),
        ("fraction", 0.21, 2),
        ("percent", 50, 3),
        ("fraction", 0.5, 3),
        ("percent", 100, 4),
        ("fraction", 1, 4),
        ("percent", None, 0),
    ],
)
def test_sofa_fio2_default_in_the_fio2_unit(fio2_unit, fio2_default, expected):
    edata = _sofa(_edata(1, pao2=[[60]], ventilation=[[1]]), fio2_unit=fio2_unit, fio2_default=fio2_default)

    # 60 mmHg on 21% is 286, on 50% 120 and on 100% 60; without a FiO2, ventilation alone scores 0
    np.testing.assert_array_equal(edata.obsm["sofa_respiration"], [[expected]])


@pytest.mark.parametrize(
    ("fio2_unit", "fio2_default"),
    [
        ("percent", 20),
        ("percent", 101),
        ("percent", 0.5),
        ("fraction", 0.2),
        ("fraction", 1.01),
        ("fraction", 50),
        ("percent", nan),
        ("percent", True),
        ("percent", "ambient"),
    ],
)
def test_sofa_invalid_fio2_default(fio2_unit, fio2_default):
    with pytest.raises(ValueError, match="fio2_default must be"):
        _sofa(_edata(1, pao2=[[60]]), fio2_unit=fio2_unit, fio2_default=fio2_default)


def test_sofa_warns_of_fio2_fractions_in_percent():
    fractions = _edata(2, pao2=[80, 80], fio2=[0.5, 0.3])
    with pytest.warns(UserWarning, match="Every FiO2 is at most 1"):
        _sofa(fractions)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Every FiO2")
        _sofa(fractions, fio2_unit="fraction")
        _sofa(_edata(2, pao2=[80, 80], fio2=[0.5, 30]))
        _sofa(_edata(2, pao2=[80, 80], fio2=[nan, nan]))


def test_sofa_worst_score_of_the_window():
    platelets = [90, nan, nan, nan, 40, nan, nan]
    creatinine = [nan, nan, nan, nan, nan, nan, 1.0]
    for window in (3, np.int64(3)):
        edata = _sofa(_edata(7, platelets=platelets, creatinine=creatinine), window=window)

        np.testing.assert_array_equal(edata.obsm["sofa"], [[2, 2, 2, 0, 3, 3, 3]])
    np.testing.assert_array_equal(edata.obsm["sofa_coagulation"], [[2, nan, nan, nan, 3, nan, nan]])

    both = _sofa(_edata(7, platelets=platelets, creatinine=creatinine), window=2, closed="both")
    np.testing.assert_array_equal(both.obsm["sofa"], edata.obsm["sofa"])


def test_sofa_window_closed_right_or_both():
    platelets = np.full(30, nan)
    platelets[[0, 29]] = [90, 200]
    right = _sofa(_edata(30, platelets=platelets)).obsm["sofa"]
    both = _sofa(_edata(30, platelets=platelets), closed="both").obsm["sofa"]

    # the 90 at t0 is the worst of a day of hourly timepoints for 24 timepoints, or for 25 with both ends of [t - 24, t]
    np.testing.assert_array_equal(right, [[2] * 24 + [0] * 6])
    np.testing.assert_array_equal(both, [[2] * 25 + [0] * 5])


def test_sofa_sums_the_organs():
    edata = _sofa(_edata(2, platelets=[90, 90], bilirubin=[2.5, nan], gcs=[nan, 9], creatinine=[1.5, 1.5]), window=1)

    np.testing.assert_array_equal(edata.obsm["sofa"], [[2 + 2 + 1, 2 + 3 + 1]])


def test_sofa_scored_from_the_first_to_the_last_input():
    edata = _sofa(_edata(6, platelets=[[nan, 200, nan, 200, nan, nan], [nan] * 6]))

    np.testing.assert_array_equal(edata.obsm["sofa"], [[nan, 0, 0, 0, nan, nan], [nan] * 6])


def test_sofa_start_key_scores_before_the_first_input():
    platelets = np.tile([nan, nan, nan, 90], (6, 1))
    platelets[5] = nan
    # before the data, at a timepoint, between two timepoints, none, after the first input, without inputs
    starts = {
        "float": [-5.0, 2.0, 1.7, nan, 10.0, 0.0],
        "integer": pd.array([-5, 2, 1, pd.NA, 10, 0], dtype="Int64"),
    }
    obs = pd.DataFrame(starts, index=list("abcdef"))
    expected = [[0, 0, 0, 2], [nan, nan, 0, 2], [nan, 0, 0, 2], [nan, nan, nan, 2], [nan, nan, nan, 2], [nan] * 4]
    for start_key in starts:
        edata = _sofa(_edata(4, obs=obs.copy(), platelets=platelets), start_key=start_key, window=1)

        np.testing.assert_array_equal(edata.obsm["sofa"], expected)


def test_sofa_urine_extrapolated_to_the_window():
    urine = np.full((3, 30), nan)
    urine[0, :12] = 10  # 120 mL in 12 h: 240 per 24 h
    urine[1, :12] = 20  # 240 mL in 12 h: 480 per 24 h
    urine[2, [0, 11]] = 40  # 80 mL in the 12 h between the first and the last urine: 160 per 24 h
    edata = _sofa(_edata(30, urine=urine))

    renal = edata.obsm["sofa_renal"]
    np.testing.assert_array_equal(renal[:, :11], 0)  # fewer than 12 h of urine
    np.testing.assert_array_equal(renal[:, 11], [3, 3, 4])
    assert np.isnan(renal[:, 12:]).all()  # after the last urine

    # from the first hour: the 40 mL of the third at t0 are 960 per 24 h, at t1 480 and at t4 192
    from_the_first_hour = _sofa(_edata(30, urine=urine), urine_min_observed=1).obsm["sofa_renal"]
    np.testing.assert_array_equal(from_the_first_hour[:, :12], [[3] * 12, [3] * 12, [0, 3, 3, 3] + [4] * 8])


def test_sofa_urine_window_closed_right_or_both():
    urine = np.full(30, 10.0)
    urine[0] = 300.0
    right = _sofa(_edata(30, urine=urine)).obsm["sofa_renal"]
    both = _sofa(_edata(30, urine=urine), closed="both").obsm["sofa_renal"]

    # at t24, 24 timepoints hold 240 mL, while 25 also hold the 300 mL of t0: 540 mL in 25 h is 518.4 per 24 h
    np.testing.assert_array_equal(right, [[0] * 24 + [3] * 6])
    np.testing.assert_array_equal(both, [[0] * 25 + [3] * 5])

    # 20 mL in every hour is 480 mL per 24 h, extrapolated from the 25 hours of a window with both ends
    for closed in ("right", "both"):
        renal = _sofa(_edata(30, urine=np.full(30, 20.0)), closed=closed).obsm["sofa_renal"]

        np.testing.assert_array_equal(renal, [[0] * 11 + [3] * 19])


def test_sofa_urine_period_widens_the_hours_without_urine():
    obs = pd.DataFrame({"first_urine": [-12.0, nan, 0.0], "last_urine": [30.0, nan, 17.9]}, index=["a", "b", "c"])
    urine = np.full((3, 24), nan)
    urine[:, :12] = 30  # 360 mL in 12 h: 720 per 24 h
    edata = _sofa(_edata(24, obs=obs, urine=urine), urine_period_keys=("first_urine", "last_urine"))

    renal = edata.obsm["sofa_renal"]
    # a: urine after the data, so the hours after t11 count as no urine: 360 mL in 18 h at t17 is 480 per 24 h
    np.testing.assert_array_equal(renal[0], [0] * 17 + [3] * 7)
    # b: no positions, so the period ends with the last urine of the data
    np.testing.assert_array_equal(renal[1], [0] * 12 + [nan] * 12)
    # c: the last urine at 17.9 belongs to t17
    np.testing.assert_array_equal(renal[2], [0] * 17 + [3] + [nan] * 6)


@pytest.mark.parametrize("missing", [nan, None, pd.NaT])
def test_sofa_obs_positions_all_missing(missing):
    obs = pd.DataFrame({"start": [missing] * 2, "first": [missing] * 2}, index=["a", "b"])
    edata = _sofa(
        _edata(5, obs=obs, platelets=[[nan, nan, nan, 90, nan], [nan] * 5], urine=np.full((2, 5), nan)),
        start_key="start",
        urine_period_keys=("first", "first"),
    )

    np.testing.assert_array_equal(edata.obsm["sofa"], [[nan, nan, nan, 2, nan], [nan] * 5])
    assert np.isnan(edata.obsm["sofa_renal"]).all()


@pytest.mark.parametrize("positions", NOT_POSITIONS)
def test_sofa_obs_positions_must_be_numbers(positions):
    obs = pd.DataFrame({"position": positions}, index=["a", "b"])
    edata = _edata(2, obs=obs, platelets=np.full((2, 2), 90.0))
    with pytest.raises(TypeError, match="positions of timepoints as numbers"):
        _sofa(edata, start_key="position")
    with pytest.raises(TypeError, match="positions of timepoints as numbers"):
        _sofa(edata, urine_period_keys=("position", "position"))


def test_sofa_reads_positions_not_the_times_of_tem():
    edata = _edata(4, platelets=[90, nan, nan, 200])
    edata.tem = pd.DataFrame({"time_value": [0.0, 10.0, 5.0, 100.0]})
    _sofa(edata, window=2)

    np.testing.assert_array_equal(edata.obsm["sofa"], [[2, 2, 0, 0]])


def test_sofa_input_errors():
    edata = _edata(2, platelets=[1, 2])
    with pytest.raises(KeyError, match="Not found in edata.var_names or edata.obsm"):
        _sofa(edata, gcs="glasgow")
    with pytest.raises(ValueError, match="at least one input"):
        _sofa(edata, **dict.fromkeys(INPUTS, None))
    with pytest.raises(ValueError, match="3D"):
        ep.tl.sofa(ed.EHRData(np.zeros((1, len(INPUTS))), var=pd.DataFrame(index=INPUTS)), platelets="platelets")
    with pytest.raises(ValueError, match="urine_period_keys needs"):
        _sofa(edata, urine=None, urine_period_keys=("a", "b"))
    with pytest.raises(KeyError, match="obs has no column 'start'"):
        _sofa(edata, start_key="start")
    with pytest.raises(ValueError, match="fio2_unit must be"):
        _sofa(edata, fio2_unit="percentage")
    with pytest.raises(ValueError, match="closed must be"):
        _sofa(edata, closed="left")


@pytest.mark.parametrize(
    ("argument", "value", "error", "match"),
    [
        ("window", 0, ValueError, "window must be a number of more than 0 timepoints"),
        ("urine_window", 0, ValueError, "urine_window must be a number of more than 0 timepoints"),
        ("window", -1, ValueError, "window must be a number of more than 0 timepoints"),
        ("pafi_window", -1, ValueError, "pafi_window must be a number of at least 0 timepoints"),
        ("urine_min_observed", -1, ValueError, "urine_min_observed must be a number of at least 0 timepoints"),
        ("window", 24.0, TypeError, "window must be a whole number of timepoints"),
        ("pafi_window", 1.5, TypeError, "pafi_window must be a whole number of timepoints"),
        ("urine_window", True, TypeError, "urine_window must be a whole number of timepoints"),
        ("urine_min_observed", "12", TypeError, "urine_min_observed must be a whole number of timepoints"),
        ("window", "24h", TypeError, "window must be a whole number of timepoints"),
        ("window", None, TypeError, "window must be a whole number of timepoints"),
    ],
)
def test_sofa_invalid_counts(argument, value, error, match):
    with pytest.raises(error, match=match):
        _sofa(_edata(2, platelets=[1, 2]), **{argument: value})


def test_sofa_without_timepoints():
    obs = pd.DataFrame({"start": [0.0, nan], "first": [1.0, nan]}, index=["a", "b"])
    edata = _sofa(
        _edata(0, obs=obs, platelets=np.empty((2, 0))), start_key="start", urine_period_keys=("first", "first")
    )

    for key in ("sofa", "sofa_respiration", "sofa_renal"):
        assert edata.obsm[key].shape == (2, 0)


def test_sofa_duplicated_and_ambiguous_names():
    edata = _edata(2, platelets=[90, 90])
    edata.obsm["gcs"] = np.ones((1, 2))
    with pytest.raises(ValueError, match=r"\['gcs'\] are both variables and keys of edata.obsm"):
        _sofa(edata)
    duplicated = _edata(2, platelets=[90, 90])
    duplicated.var_names = ["platelets"] * len(INPUTS)
    with pytest.raises(ValueError, match="holds \\['platelets'\\] more than once"):
        ep.tl.sofa(duplicated, platelets="platelets", layer=DEFAULT_TEM_LAYER_NAME)


def test_sofa_inputs_from_obsm(rng):
    X = rng.uniform(0, 300, size=(3, len(INPUTS), 8))
    X[rng.random(X.shape) < 0.3] = nan
    edata = _edata(8, **{name: X[:, i] for i, name in enumerate(INPUTS)})
    expected = _sofa(edata.copy()).obsm["sofa"]
    edata.obsm["vent_windows"] = edata.layers[DEFAULT_TEM_LAYER_NAME][:, INPUTS.index("ventilation"), :].copy()
    edata.obsm["plt_series"] = edata.layers[DEFAULT_TEM_LAYER_NAME][:, INPUTS.index("platelets"), :].copy()
    _sofa(edata, ventilation="vent_windows", platelets="plt_series")

    np.testing.assert_array_equal(edata.obsm["sofa"], expected)


def test_sofa_inputs_from_obsm_only():
    edata = ed.EHRData(shape=(1, 0), tem=_timepoints(3))
    edata.obsm["plt"] = np.array([[200.0, 90.0, nan]])
    ep.tl.sofa(edata, platelets="plt")

    np.testing.assert_array_equal(edata.obsm["sofa"], [[0, 2, nan]])


@pytest.mark.parametrize(
    "as_obsm", [lambda x, edata: pd.DataFrame(x, index=edata.obs_names), lambda x, edata: sp.csr_matrix(x)]
)
def test_sofa_obsm_input_as_data_frame_or_sparse(as_obsm):
    platelets = np.array([[200.0, 90.0, 0.0], [0.0, 40.0, 0.0]])
    edata = ed.EHRData(shape=(2, 0), tem=_timepoints(3))
    edata.obsm["plt"] = as_obsm(platelets, edata)
    ep.tl.sofa(edata, platelets="plt", window=1)

    np.testing.assert_array_equal(edata.obsm["sofa"], [[0, 2, 4], [4, 3, 4]])


def test_sofa_obsm_input_of_another_shape():
    edata = _edata(2, platelets=[1, 2])
    edata.obsm["per_obs"] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="a value per observation and timepoint"):
        _sofa(edata, gcs="per_obs")


def test_sofa_sparse_3D_layer_not_implemented():
    sparse = pytest.importorskip("sparse")
    edata = _edata(2, platelets=[90, 90])
    edata.layers[DEFAULT_TEM_LAYER_NAME] = sparse.COO.from_numpy(np.nan_to_num(edata.layers[DEFAULT_TEM_LAYER_NAME]))
    with pytest.raises(NotImplementedError, match="does not support array type COO"):
        _sofa(edata)


def test_sofa_without_variables_of_some_inputs():
    edata = _edata(2, platelets=[90, 90])[:, ["platelets"]].copy()
    ep.tl.sofa(edata, platelets="platelets", layer=DEFAULT_TEM_LAYER_NAME)

    np.testing.assert_array_equal(edata.obsm["sofa"], [[2, 2]])


def test_sofa_dask_layer_with_obsm_input(rng):
    da = pytest.importorskip("dask.array")
    X = rng.uniform(0, 300, size=(4, len(INPUTS), 6))
    edata = _edata(6, **{name: X[:, i] for i, name in enumerate(INPUTS)})
    expected = _sofa(edata.copy()).obsm["sofa"]
    edata.obsm["plt_series"] = edata.layers[DEFAULT_TEM_LAYER_NAME][:, INPUTS.index("platelets"), :].copy()
    edata.layers[DEFAULT_TEM_LAYER_NAME] = da.from_array(edata.layers[DEFAULT_TEM_LAYER_NAME], chunks=(2, -1, -1))
    _sofa(edata, platelets="plt_series")

    np.testing.assert_array_equal(np.asarray(edata.obsm["sofa"]), expected)


def test_sofa_copy():
    edata = _edata(2, platelets=[90, 90])
    result = ep.tl.sofa(edata, platelets="platelets", layer=DEFAULT_TEM_LAYER_NAME, key_added="score", copy=True)

    assert "score" not in edata.obsm
    np.testing.assert_array_equal(result.obsm["score"], [[2, 2]])


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_sofa_array_types(array_type, rng):
    shape = (6, len(INPUTS), 30)
    X = rng.uniform(0, 200, size=shape)
    X[rng.random(shape) < 0.8] = nan
    X[1, :, :10] = nan  # scored from its start before its first input
    obs = pd.DataFrame(
        {
            "start": [nan, 2.5, nan, nan, -4.0, 40.0],
            "first": [nan, nan, -5.0, nan, 3.0, nan],
            "last": [nan, nan, 35.0, 12.0, nan, nan],
        },
        index=[f"o{i}" for i in range(6)],
    )
    settings = {
        "start_key": "start",
        "urine_period_keys": ("first", "last"),
        "window": 6,
        "urine_window": 6,
        "urine_min_observed": 3,
        "closed": "both",
    }
    inputs = {name: X[:, i] for i, name in enumerate(INPUTS)}
    expected = _sofa(_edata(30, obs=obs.copy(), **inputs), **settings)
    edata = _edata(30, obs=obs.copy(), **inputs)
    edata.layers[DEFAULT_TEM_LAYER_NAME] = array_type(edata.layers[DEFAULT_TEM_LAYER_NAME])

    with forbid_dask_compute(allowed=1):
        _sofa(edata, **settings)
    for key in ("sofa", *(f"sofa_{organ}" for organ in ORGANS)):
        np.testing.assert_array_equal(edata.obsm[key], expected.obsm[key])


def _scored(scores, infection) -> ed.EHRData:
    scores = np.asarray(scores, dtype=np.float64)
    obs = pd.DataFrame({"infection": infection}, index=[f"o{i}" for i in range(len(scores))])
    edata = ed.EHRData(shape=(scores.shape[0], 1), obs=obs, tem=_timepoints(scores.shape[1]))
    edata.obsm["sofa"] = scores
    return edata


def test_sepsis3_rise_from_the_lowest_score_of_the_window():
    scores = [
        [3, 1, 2, 3, 3, 3],  # lowest 1 at t1, rise of 2 at t3
        [3, 3, 4, 4, 4, 4],  # no rise of 2 from 3
        [nan, nan, 0, nan, 2, 2],  # missing scores are skipped
        [0, 0, 0, 0, 0, 5],  # rise after the window
    ]
    edata = _scored(scores, [2.0, 2.0, 2.0, 1.0])
    ep.tl.sepsis3(edata, infection_key="infection", before=2, after=2)

    np.testing.assert_array_equal(edata.obs["sepsis3"], [True, False, True, False])
    np.testing.assert_array_equal(edata.obs["sepsis3_onset"], [3.0, nan, 4.0, nan])
    assert edata.obs["sepsis3_onset"].dtype == np.float64


def test_sepsis3_window_around_the_infection():
    scores = [[0, 2, 2, 2, 2, 2]]
    late = _scored(scores, [5])
    ep.tl.sepsis3(late, infection_key="infection", before=3, after=0)
    early = _scored(scores, [5])
    ep.tl.sepsis3(early, infection_key="infection", before=5, after=0)

    assert not late.obs["sepsis3"].iloc[0]  # the window starts at t2, after the rise
    assert early.obs["sepsis3_onset"].iloc[0] == 1.0


def test_sepsis3_infection_belongs_to_its_timepoint():
    edata = _scored([[3, 0, 1, 1, 2, 3]], [3.5])
    ep.tl.sepsis3(edata, infection_key="infection", before=2, after=1)

    # the infection belongs to t3: the window is t1 to t4, so the rise from 0 reaches 2 at t4
    assert edata.obs["sepsis3_onset"].iloc[0] == 4.0


def test_sepsis3_infection_outside_the_data():
    scores = [[0, 3, 3, 3], [0, 0, 3, 3], [0, 0, 0, 3], [0, 3, 3, 3], [0, 3, 3, 3]]
    edata = _scored(scores, [-1.0, -0.5, 4.0, 30.0, nan])
    ep.tl.sepsis3(edata, infection_key="infection", before=2, after=2)

    # -1 is followed to t1; -0.5 belongs to -1, so the rise at t2 comes too late; 4 is followed from t2
    np.testing.assert_array_equal(edata.obs["sepsis3"], [True, False, True, False, False])
    np.testing.assert_array_equal(edata.obs["sepsis3_onset"], [1.0, nan, 3.0, nan, nan])


@pytest.mark.parametrize("infection", NOT_POSITIONS)
def test_sepsis3_infection_positions_must_be_numbers(infection):
    with pytest.raises(TypeError, match="positions of timepoints as numbers"):
        ep.tl.sepsis3(_scored([[0, 3], [0, 3]], infection), infection_key="infection")


@pytest.mark.parametrize(("baseline", "onset"), [("minimum", 3.0), ("start", 4.0), ("zero", 1.0)])
def test_sepsis3_baseline(baseline, onset):
    edata = _scored([[nan, 2, 1, 3, 4]], [1.0])
    ep.tl.sepsis3(edata, infection_key="infection", before=1, after=10, baseline=baseline)

    # 3 is 2 above the lowest 1, 4 is 2 above the first score 2, and 2 is 2 above 0
    assert edata.obs["sepsis3_onset"].iloc[0] == onset


@pytest.mark.parametrize(("rise", "onset"), [(0.5, 2.0), (2, 3.0), (3, 4.0)])
def test_sepsis3_rise(rise, onset):
    edata = _scored([[2, 0, 1, 2, 3, 4]], [0.0])
    ep.tl.sepsis3(edata, infection_key="infection", rise=rise, before=0, after=10, key_added="sep")

    assert edata.obs["sep_onset"].iloc[0] == onset


@pytest.mark.parametrize("rise", [0, -1, np.inf, nan, True, "2"])
def test_sepsis3_invalid_rise(rise):
    with pytest.raises(ValueError, match="rise must be a number of more than 0 points"):
        ep.tl.sepsis3(_scored([[0, 3]], [0.0]), infection_key="infection", rise=rise)


@pytest.mark.parametrize(
    ("argument", "value", "error", "match"),
    [
        ("before", -1, ValueError, "before must be a number of at least 0 timepoints"),
        ("after", -1, ValueError, "after must be a number of at least 0 timepoints"),
        ("before", 1.5, TypeError, "before must be a whole number of timepoints"),
        ("after", True, TypeError, "after must be a whole number of timepoints"),
        ("before", "48h", TypeError, "before must be a whole number of timepoints"),
    ],
)
def test_sepsis3_invalid_counts(argument, value, error, match):
    with pytest.raises(error, match=match):
        ep.tl.sepsis3(_scored([[0, 3]], [0.0]), infection_key="infection", **{argument: value})


def test_sepsis3_of_the_sofa_score_under_another_key():
    obs = pd.DataFrame({"infection": [2.0]}, index=["a"])
    edata = _sofa(_edata(5, obs=obs, platelets=[200, 200, 90, 40, 40]), window=1, key_added="score")
    ep.tl.sepsis3(edata, infection_key="infection", sofa="score")

    # coagulation scores 0, 0, 2, 3, 3
    assert edata.obs["sepsis3_onset"].iloc[0] == 2.0


def test_sepsis3_without_timepoints():
    edata = _scored(np.empty((2, 0)), [1.0, nan])
    ep.tl.sepsis3(edata, infection_key="infection")

    assert not edata.obs["sepsis3"].any()
    assert edata.obs["sepsis3_onset"].isna().all()


def test_sepsis3_sofa_as_dask_array():
    da = pytest.importorskip("dask.array")
    scores = [[3, 1, 2, 3, 3, 3], [0, 0, 0, 0, 0, 0]]
    expected = _scored(scores, [2.0, 2.0])
    ep.tl.sepsis3(expected, infection_key="infection")
    edata = _scored(scores, [2.0, 2.0])
    edata.obsm["sofa"] = da.from_array(edata.obsm["sofa"], chunks=1)
    with forbid_dask_compute(allowed=1):
        ep.tl.sepsis3(edata, infection_key="infection")

    pd.testing.assert_frame_equal(edata.obs, expected.obs)


@pytest.mark.parametrize(
    "as_obsm", [lambda x, edata: pd.DataFrame(x, index=edata.obs_names), lambda x, edata: sp.csr_matrix(x)]
)
def test_sepsis3_sofa_as_data_frame_or_sparse(as_obsm):
    scores = [[3, 1, 2, 3, 3, 3], [0, 0, 0, 0, 0, 0], [0, 0, 4, 4, 4, 4]]
    edata = _scored(scores, [2.0, 2.0, 2.0])
    edata.obsm["sofa"] = as_obsm(edata.obsm["sofa"], edata)
    ep.tl.sepsis3(edata, infection_key="infection")

    np.testing.assert_array_equal(edata.obs["sepsis3_onset"], [3.0, nan, 2.0])


def test_sepsis3_copy():
    edata = _scored([[0, 3]], [0.0])
    result = ep.tl.sepsis3(edata, infection_key="infection", copy=True)

    assert "sepsis3" not in edata.obs
    assert result.obs["sepsis3_onset"].iloc[0] == 1.0


def test_sepsis3_errors():
    edata = _scored([[0, 3]], [0.0])
    with pytest.raises(KeyError, match="obsm has no 'other'"):
        ep.tl.sepsis3(edata, infection_key="infection", sofa="other")
    with pytest.raises(KeyError, match="obs has no"):
        ep.tl.sepsis3(edata, infection_key="other")
    with pytest.raises(ValueError, match="baseline"):
        ep.tl.sepsis3(edata, infection_key="infection", baseline="first")
    edata.obsm["per_obs"] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="a value per observation and timepoint"):
        ep.tl.sepsis3(edata, infection_key="infection", sofa="per_obs")


def _fio2_edata(fio2, chunks=None):
    """PaO2 80 and the FiO2 `fio2` of every observation at one timepoint, as numpy or as dask with `chunks`."""
    X = np.full((len(fio2), 2, 1), np.nan)
    X[:, 0, 0], X[:, 1, 0] = 80.0, fio2
    if chunks is not None:
        import dask.array as da

        X = da.from_array(X, chunks=(chunks, -1, -1))
    return ed.EHRData(
        shape=(len(fio2), 2), var=pd.DataFrame(index=["pao2", "fio2"]), layers={DEFAULT_TEM_LAYER_NAME: X}
    )


@pytest.mark.parametrize("chunks", [None, 1, 2])
def test_sofa_fio2_warning_does_not_depend_on_the_blocks(chunks, recwarn):
    """The warning looks at every FiO2 of the data, also when the observations come in blocks."""
    ep.tl.sofa(_fio2_edata([1.0, 50.0], chunks), pao2="pao2", fio2="fio2", layer=DEFAULT_TEM_LAYER_NAME)
    assert not [w for w in recwarn if "looks like a fraction" in str(w.message)]
    with pytest.warns(UserWarning, match="looks like a fraction"):
        ep.tl.sofa(_fio2_edata([1.0, 0.5], chunks), pao2="pao2", fio2="fio2", layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("keys", ["f", "first", ("a", "b", "c"), ("a",)])
def test_sofa_urine_period_keys_must_be_two_columns(keys):
    X = np.full((1, 1, 2), 10.0)
    obs = pd.DataFrame({"a": [0.0], "b": [1.0], "c": [1.0], "f": [0.0]}, index=["o"])
    edata = ed.EHRData(shape=(1, 1), obs=obs, var=pd.DataFrame(index=["urine"]), layers={DEFAULT_TEM_LAYER_NAME: X})
    with pytest.raises(ValueError, match="two columns of obs"):
        ep.tl.sofa(edata, urine="urine", layer=DEFAULT_TEM_LAYER_NAME, urine_period_keys=keys)
