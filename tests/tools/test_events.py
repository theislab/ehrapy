from __future__ import annotations

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute

nan = np.nan


def _obs(n_obs: int, **columns) -> pd.DataFrame:
    """`obs` of `n_obs` observations with the given columns."""
    return pd.DataFrame(columns, index=[f"o{i}" for i in range(n_obs)])


def _edata(values, *, names=None, obs: pd.DataFrame | None = None, dtype=np.float64) -> ed.EHRData:
    """Observations x variables x timepoints `values` in the layer `DEFAULT_TEM_LAYER_NAME`."""
    X = np.asarray(values, dtype=dtype)
    names = [f"v{i}" for i in range(X.shape[1])] if names is None else names
    obs = _obs(X.shape[0]) if obs is None else obs
    return ed.EHRData(shape=X.shape[:2], obs=obs, var=pd.DataFrame(index=names), layers={DEFAULT_TEM_LAYER_NAME: X})


def _marks(marked, obs: pd.DataFrame | None = None, **series) -> ed.EHRData:
    """`edata` without variables, with the time series `marked` and `series` in obsm."""
    marked = np.atleast_2d(np.asarray(marked, dtype=np.float64))
    n_obs, n_t = marked.shape
    obs = _obs(n_obs) if obs is None else obs
    # the tools read no times: tem only sets the number of timepoints
    edata = ed.EHRData(shape=(n_obs, 0), obs=obs, tem=pd.DataFrame(index=pd.RangeIndex(n_t)))
    edata.obsm["marked"] = marked
    for key, values in series.items():
        edata.obsm[key] = np.atleast_2d(np.asarray(values, dtype=np.float64))
    return edata


def _sustained_reference(marked, lengths, end, duration, max_gap):
    """Whether every marked timepoint starts sustained events, mark by mark."""
    positions = np.arange(marked.shape[1])
    qualified = np.zeros(marked.shape, dtype=bool)
    for i in range(len(marked)):
        last = np.floor(end[i])
        kept = marked[i] & ~(positions > last)
        for j in np.flatnonzero(kept):
            stop = j + duration if np.isnan(last) else min(j + duration, last)
            window = np.flatnonzero(kept & (positions >= j) & (positions <= j + duration))
            covered = np.maximum.accumulate(window + lengths[i, window])
            qualified[i, j] = (np.append(window[1:], stop) - covered).max() <= max_gap
    return qualified


def _paired_reference(a, b, a_to_b, b_to_a):
    """Whether every marked timepoint of `a` or `b` is followed by the other kind within its count, mark by mark."""
    positions = np.arange(a.shape[1])
    paired = np.zeros(a.shape, dtype=bool)
    for i, j in zip(*np.nonzero(a), strict=True):
        paired[i, j] = (b[i] & (positions >= j) & (positions <= j + a_to_b)).any()
    for i, j in zip(*np.nonzero(b), strict=True):
        paired[i, j] |= (a[i] & (positions >= j) & (positions <= j + b_to_a)).any()
    return paired


VALUES = [[[nan, 0, 2, nan, -1, nan], [nan, nan, 3, 0, nan, 0]]]


@pytest.mark.parametrize(
    ("how", "ignore_zero", "expected", "first_and_last"),
    [
        ("any", False, [False, True, True, True, True, True], (1, 5)),
        ("any", True, [False, False, True, False, True, False], (2, 4)),
        ("count", False, [0, 1, 2, 1, 1, 1], (1, 5)),
        ("count", True, [0, 0, 2, 0, 1, 0], (2, 4)),
        # a sum or largest value of 0 is a marker of 0, which marks no timepoint
        ("sum", False, [nan, 0, 5, 0, -1, 0], (2, 4)),
        ("sum", True, [nan, nan, 5, nan, -1, nan], (2, 4)),
        ("max", False, [nan, 0, 3, 0, -1, 0], (2, 4)),
        ("max", True, [nan, nan, 3, nan, -1, nan], (2, 4)),
    ],
)
def test_mark_events_how(how, ignore_zero, expected, first_and_last):
    edata = _edata(VALUES)
    ep.tl.mark_events(edata, ["v0", "v1"], how=how, ignore_zero=ignore_zero, layer=DEFAULT_TEM_LAYER_NAME)

    np.testing.assert_array_equal(edata.obsm["events"], [expected])
    assert edata.obsm["events"].dtype == {"any": bool, "count": np.int64}.get(how, np.float64)
    assert (edata.obs["events_first"].iloc[0], edata.obs["events_last"].iloc[0]) == first_and_last


def test_mark_events_first_and_last_positions():
    values = np.full((3, 2, 6), nan)
    values[0, 0, [1, 4]] = 1
    values[0, 1, 2] = 0
    values[2, 1, 5] = 7
    edata = _edata(values)
    ep.tl.mark_events(edata, layer=DEFAULT_TEM_LAYER_NAME, key_added="seen")

    np.testing.assert_array_equal(edata.obs["seen_first"], [1, nan, 5])
    np.testing.assert_array_equal(edata.obs["seen_last"], [4, nan, 5])
    assert edata.obs["seen_first"].dtype == edata.obs["seen_last"].dtype == np.float64


@pytest.mark.parametrize(
    ("ignore_zero", "expected", "first"),
    [(False, [False, True, False, True], 1), (True, [False, False, False, True], 3)],
)
def test_mark_events_value_of_0_marks_unless_ignore_zero(ignore_zero, expected, first):
    # such as a dose recorded as 0 hours long
    edata = _edata([[[nan, 0, nan, 1]]])
    ep.tl.mark_events(edata, ignore_zero=ignore_zero, layer=DEFAULT_TEM_LAYER_NAME)

    np.testing.assert_array_equal(edata.obsm["events"], [expected])
    assert edata.obs["events_first"].iloc[0] == first


@pytest.mark.parametrize(
    ("how", "expected", "first_and_last"), [("sum", [0, 1, nan, 0], (1, 1)), ("max", [2, 1, nan, 0], (0, 1))]
)
def test_mark_events_marker_of_0_marks_nothing(how, expected, first_and_last):
    edata = _edata([[[2, 1, nan, nan], [-2, nan, nan, 0]]])
    ep.tl.mark_events(edata, how=how, layer=DEFAULT_TEM_LAYER_NAME)

    # the marker keeps the 0, but the timepoint is not marked
    np.testing.assert_array_equal(edata.obsm["events"], [expected])
    assert (edata.obs["events_first"].iloc[0], edata.obs["events_last"].iloc[0]) == first_and_last


MASKS = np.array([[[1, 0, 0, 1, 0], [0, 0, 1, 1, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]], dtype=bool)


@pytest.mark.parametrize("kind", ["layer", "array", "frame", "sparse"])
def test_mark_events_boolean_variables_mark_where_true(kind):
    if kind == "layer":
        edata, layer = _edata(MASKS, dtype=bool), DEFAULT_TEM_LAYER_NAME
    else:
        edata, layer = _marks(np.zeros((2, 5))), None
        for i in range(MASKS.shape[1]):
            mask = MASKS[:, i]
            edata.obsm[f"v{i}"] = {
                "array": mask,
                "frame": pd.DataFrame(mask, index=edata.obs_names),
                "sparse": sp.csr_matrix(mask),
            }[kind]
    ep.tl.mark_events(edata, ["v0", "v1"], how="count", layer=layer)

    # False marks nothing: the second observation has no marked timepoint
    np.testing.assert_array_equal(edata.obsm["events"], [[1, 0, 1, 2, 0], [0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(edata.obs["events_first"], [0, nan])
    np.testing.assert_array_equal(edata.obs["events_last"], [3, nan])


def test_mark_events_boolean_and_numeric_variables():
    edata = _edata([[[nan, 0, nan, nan]]])
    edata.obsm["mask"] = np.array([[False, False, True, False]])
    ep.tl.mark_events(edata, ["v0", "mask"], layer=DEFAULT_TEM_LAYER_NAME)

    # the 0 of the numeric variable marks, the False of the boolean one does not
    np.testing.assert_array_equal(edata.obsm["events"], [[False, True, True, False]])


def test_mark_events_of_some_variables():
    edata = _edata(VALUES)
    ep.tl.mark_events(edata, "v1", how="count", layer=DEFAULT_TEM_LAYER_NAME)

    np.testing.assert_array_equal(edata.obsm["events"], [[0, 0, 1, 1, 0, 1]])


@pytest.mark.parametrize("sparse", [False, True])
def test_mark_events_inputs_from_obsm_data_frame_or_sparse(sparse):
    marks = np.array([[0.0, 1.0, 0.0, 2.0], [1.0, 0.0, 0.0, 0.0]])
    edata = _marks(np.zeros((2, 4)))
    edata.obsm["other"] = sp.csr_matrix(marks) if sparse else pd.DataFrame(marks, index=edata.obs_names)
    ep.tl.mark_events(edata, ["other"], how="sum", ignore_zero=True)

    np.testing.assert_array_equal(edata.obsm["events"], [[nan, 1, nan, 2], [1, nan, nan, nan]])


def test_mark_events_inputs_from_obsm():
    edata = _edata(VALUES)
    edata.obsm["other"] = np.array([[1.0, nan, nan, nan, nan, nan]])
    ep.tl.mark_events(edata, ["v1", "other"], how="max", layer=DEFAULT_TEM_LAYER_NAME)
    only_obsm = _marks([[0, 0, 1, 0, 0]], other=[[1.0, nan, nan, nan, nan]])
    ep.tl.mark_events(only_obsm, ["marked", "other"], ignore_zero=True)

    np.testing.assert_array_equal(edata.obsm["events"], [[1, nan, 3, 0, nan, 0]])
    np.testing.assert_array_equal(only_obsm.obsm["events"], [[True, False, True, False, False]])


def test_mark_events_errors():
    edata = _edata(VALUES)
    with pytest.raises(ValueError, match="how must be one of"):
        ep.tl.mark_events(edata, how="mean", layer=DEFAULT_TEM_LAYER_NAME)
    with pytest.raises(ValueError, match="at least one variable"):
        ep.tl.mark_events(edata, [], layer=DEFAULT_TEM_LAYER_NAME)
    with pytest.raises(KeyError, match="Not found in edata.var_names or edata.obsm"):
        ep.tl.mark_events(edata, ["v0", "drug"], layer=DEFAULT_TEM_LAYER_NAME)
    with pytest.raises(ValueError, match="3D"):
        ep.tl.mark_events(ed.EHRData(np.zeros((1, 2))))
    with pytest.warns(UserWarning, match="not unique"):
        duplicated = _edata(VALUES, names=["v0", "v0"])
    with pytest.raises(ValueError, match="more than once"):
        ep.tl.mark_events(duplicated, layer=DEFAULT_TEM_LAYER_NAME)
    edata.obsm["short"] = np.zeros((1, 3))
    with pytest.raises(ValueError, match="a value per observation and timepoint"):
        ep.tl.mark_events(edata, ["short"])
    edata.obsm["v0"] = np.zeros((1, 6))
    with pytest.raises(ValueError, match="both variables and keys of edata.obsm"):
        ep.tl.mark_events(edata, ["v0"], layer=DEFAULT_TEM_LAYER_NAME)


def test_mark_events_sparse_3D_layer_not_implemented():
    sparse = pytest.importorskip("sparse")
    edata = _edata(np.nan_to_num(VALUES))
    edata.layers[DEFAULT_TEM_LAYER_NAME] = sparse.COO.from_numpy(edata.layers[DEFAULT_TEM_LAYER_NAME])
    with pytest.raises(NotImplementedError, match="does not support array type COO"):
        ep.tl.mark_events(edata, layer=DEFAULT_TEM_LAYER_NAME)


def test_mark_events_copy():
    edata = _edata(VALUES)
    result = ep.tl.mark_events(edata, layer=DEFAULT_TEM_LAYER_NAME, key_added="marks", copy=True)

    assert "marks" not in edata.obsm
    assert "marks_first" not in edata.obs
    np.testing.assert_array_equal(result.obsm["marks"], [[False, True, True, True, True, True]])
    assert result.obs["marks_first"].iloc[0] == 1


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
@pytest.mark.parametrize("ignore_zero", [False, True])
@pytest.mark.parametrize("how", ["any", "count", "sum", "max"])
def test_mark_events_array_types(array_type, how, ignore_zero, rng):
    values = rng.choice([nan, 0.0, 1.0, 2.5, -1.0], size=(6, 4, 9))
    expected = _edata(values)
    ep.tl.mark_events(expected, how=how, ignore_zero=ignore_zero, layer=DEFAULT_TEM_LAYER_NAME)
    edata = _edata(values)
    edata.layers[DEFAULT_TEM_LAYER_NAME] = array_type(edata.layers[DEFAULT_TEM_LAYER_NAME])

    with forbid_dask_compute(allowed=1):
        ep.tl.mark_events(edata, how=how, ignore_zero=ignore_zero, layer=DEFAULT_TEM_LAYER_NAME)
    np.testing.assert_array_equal(edata.obsm["events"], expected.obsm["events"])
    assert edata.obsm["events"].dtype == expected.obsm["events"].dtype
    pd.testing.assert_frame_equal(edata.obs, expected.obs)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_mark_events_boolean_array_types(array_type, rng):
    masks = rng.random((6, 3, 9)) < 0.2
    edata = _edata(masks, dtype=bool)
    edata.layers[DEFAULT_TEM_LAYER_NAME] = array_type(edata.layers[DEFAULT_TEM_LAYER_NAME])

    with forbid_dask_compute(allowed=1):
        ep.tl.mark_events(edata, how="count", layer=DEFAULT_TEM_LAYER_NAME)
    np.testing.assert_array_equal(edata.obsm["events"], masks.sum(axis=1))
    any_marked = masks.any(axis=1)
    np.testing.assert_array_equal(
        edata.obs["events_first"], np.where(any_marked.any(axis=1), any_marked.argmax(axis=1), nan)
    )
    assert 0 < any_marked.sum() < any_marked.size


def _sustained(marked, *, lengths=None, end=None, **kwargs) -> np.ndarray:
    """The positions at which the events `marked` of one observation are sustained."""
    edata = _marks(
        marked, obs=None if end is None else _obs(1, end=[end]), **({} if lengths is None else {"lengths": lengths})
    )
    ep.tl.mark_sustained_events(
        edata,
        "marked",
        lengths=None if lengths is None else "lengths",
        end_key=None if end is None else "end",
        **kwargs,
    )
    return np.flatnonzero(edata.obsm["sustained_events"][0])


def test_mark_sustained_events_gaps_up_to_max_gap():
    marked = np.zeros(12)
    marked[[0, 2, 4, 6]] = 1

    # from 2 the window ends 2 timepoints after the last mark at 6, from 4 it ends 4 after it
    np.testing.assert_array_equal(_sustained(marked, duration=6, max_gap=2), [0, 2])
    # every gap is exactly 2: one timepoint less is too short
    np.testing.assert_array_equal(_sustained(marked, duration=6, max_gap=1), [])
    np.testing.assert_array_equal(_sustained(marked, duration=4, max_gap=2), [0, 2, 4])


def test_mark_sustained_events_lengths_cover_the_gaps():
    marked = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    lengths = [5, nan, nan, nan, nan, nan, nan, 0, nan, nan, nan, 10]

    # from 0: 7 - 5 = 2, then the window end 8 - 7 = 1; from 7: 11 - 7 = 4; 11 covers its window
    np.testing.assert_array_equal(_sustained(marked, lengths=lengths, duration=8, max_gap=2), [0, 11])
    np.testing.assert_array_equal(_sustained(marked, duration=8, max_gap=2), [])
    # a missing length counts as 0
    np.testing.assert_array_equal(_sustained(marked, lengths=np.full(12, nan), duration=8, max_gap=2), [])


def test_mark_sustained_events_latest_covered_position_of_all_earlier_marks():
    marked = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    lengths = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    # 2 covers up to 2, but 0 covers up to 10: 11 - 10 = 1 and 12 - 12 = 0
    np.testing.assert_array_equal(_sustained(marked, lengths=lengths, duration=12, max_gap=1), [0])


def test_mark_sustained_events_negative_lengths():
    marked = [1, 1, 0]

    np.testing.assert_array_equal(_sustained(marked, lengths=[-2, 0, nan], duration=2, max_gap=2), [1])
    np.testing.assert_array_equal(_sustained(marked, lengths=[0, 0, nan], duration=2, max_gap=2), [0, 1])


def test_mark_sustained_events_end_of_the_observation():
    marked = np.zeros((4, 8))
    marked[:, [0, 5]] = 1
    edata = _marks(marked, obs=_obs(4, end=[3.7, 4.0, -1.0, nan]))
    ep.tl.mark_sustained_events(edata, "marked", duration=10, max_gap=3, end_key="end")

    # o0: the end at 3.7 is position 3, 3 after the mark at 0, and the mark at 5 after it is left out
    np.testing.assert_array_equal(
        edata.obsm["sustained_events"].astype(int), [[1, 0, 0, 0, 0, 0, 0, 0], [0] * 8, [0] * 8, [0] * 8]
    )
    np.testing.assert_array_equal(edata.obs["sustained_events_first"], [0, nan, nan, nan])


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        (nan, []),
        # beyond the last timepoint at 3: the window ends at the end, not at the last timepoint
        (4, [0, 2]),
        (4.5, [0, 2]),
        (5, []),
        (3, [0, 2]),
        (2.9, [0, 2]),
        # the mark at 2 is left out
        (1.9, [0]),
        (0, [0]),
        # before the first timepoint
        (-0.1, []),
        (-5, []),
    ],
)
def test_mark_sustained_events_end_positions(end, expected):
    # without an end, both windows end 4 or more after the last mark at 2
    np.testing.assert_array_equal(_sustained([1, 0, 1, 0], end=end, duration=6, max_gap=2), expected)


@pytest.mark.parametrize(
    "end",
    [
        pd.array([4, 1, 9], dtype="int64"),
        pd.array([4.0, 1.0, nan], dtype="float32"),
        pd.array([4, 1, None], dtype="Int64"),
        pd.array([4.0, 1.0, None], dtype="Float64"),
    ],
    ids=["int64", "float32", "Int64", "Float64"],
)
def test_mark_sustained_events_end_key_of_numeric_types(end):
    edata = _marks([[1, 0, 1, 0]] * 3, obs=_obs(3, end=end))
    ep.tl.mark_sustained_events(edata, "marked", duration=6, max_gap=2, end_key="end")

    # o1: the end at 1 leaves out the mark at 2; o2: the end at 9 or none is too far after the last mark
    np.testing.assert_array_equal(
        edata.obsm["sustained_events"].astype(int), [[1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
    )


@pytest.mark.parametrize(
    "end",
    [pd.to_timedelta([4, 1], unit="h"), pd.to_datetime(["2020-01-01", "2020-01-02"]), ["4", "1"], [True, False]],
    ids=["time differences", "dates", "text", "booleans"],
)
def test_mark_sustained_events_end_key_must_hold_numbers(end):
    edata = _marks([[1, 0, 1, 0]] * 2, obs=_obs(2, end=end))
    with pytest.raises(TypeError, match="positions of timepoints as numbers"):
        ep.tl.mark_sustained_events(edata, "marked", duration=6, max_gap=2, end_key="end")


@pytest.mark.parametrize("end", [[nan, nan], [None, None], pd.to_timedelta([None, None])], ids=["NaN", "None", "NaT"])
def test_mark_sustained_events_end_key_all_missing(end):
    edata = _marks([[1, 0, 1, 0]] * 2, obs=_obs(2, end=end))
    ep.tl.mark_sustained_events(edata, "marked", duration=2, max_gap=2, end_key="end")

    np.testing.assert_array_equal(edata.obsm["sustained_events"].astype(int), [[1, 0, 1, 0]] * 2)


def test_mark_sustained_events_from_layer_variables(rng):
    values = np.where(rng.random((6, 2, 30)) < 0.3, rng.integers(-3, 8, size=(6, 2, 30)), nan)
    end = np.array([nan, 10.5, 20, 29, 50, -2])
    edata = _edata(values, names=["drug", "hours"], obs=_obs(6, end=end))
    ep.tl.mark_sustained_events(
        edata, "drug", duration=8, max_gap=3, lengths="hours", end_key="end", layer=DEFAULT_TEM_LAYER_NAME
    )

    # a value of 0 marks no timepoint
    marked = ~np.isnan(values[:, 0]) & (values[:, 0] != 0)
    expected = _sustained_reference(marked, np.nan_to_num(values[:, 1]), end, 8, 3)
    np.testing.assert_array_equal(edata.obsm["sustained_events"], expected)
    assert 0 < expected.sum() < marked.sum()


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_mark_sustained_events_array_types(array_type, rng):
    values = np.where(rng.random((8, 2, 40)) < 0.25, rng.integers(0, 6, size=(8, 2, 40)), nan)
    obs = _obs(8, end=rng.choice([nan, -2.5, 5.5, 20, 38, 45], size=8))
    expected = _edata(values, obs=obs)
    ep.tl.mark_sustained_events(
        expected, "v0", lengths="v1", duration=10, max_gap=3, end_key="end", layer=DEFAULT_TEM_LAYER_NAME
    )
    edata = _edata(values, obs=obs)
    edata.layers[DEFAULT_TEM_LAYER_NAME] = array_type(edata.layers[DEFAULT_TEM_LAYER_NAME])
    edata.obsm["lengths"] = values[:, 1].copy()

    with forbid_dask_compute(allowed=1):
        ep.tl.mark_sustained_events(
            edata, "v0", lengths="lengths", duration=10, max_gap=3, end_key="end", layer=DEFAULT_TEM_LAYER_NAME
        )
    np.testing.assert_array_equal(edata.obsm["sustained_events"], expected.obsm["sustained_events"])
    pd.testing.assert_frame_equal(edata.obs, expected.obs)
    assert expected.obsm["sustained_events"].any()


@pytest.mark.parametrize("seed", range(6))
def test_mark_sustained_events_equal_a_reference(seed):
    rng = np.random.default_rng(seed)
    n_obs, n_t = 40, 50
    marked = rng.random((n_obs, n_t)) < 0.25
    lengths = np.where(marked, rng.choice([-3, 0, 1, 2, 5, 12], size=(n_obs, n_t)), nan)
    end = rng.uniform(-5, n_t + 10, size=n_obs)
    end[rng.random(n_obs) < 0.3] = nan
    edata = _marks(marked, obs=_obs(n_obs, end=end), lengths=lengths)
    duration, max_gap = rng.choice([1, 4, 10, 24]), rng.choice([0, 1, 3])
    ep.tl.mark_sustained_events(edata, "marked", duration=duration, max_gap=max_gap, lengths="lengths", end_key="end")

    expected = _sustained_reference(marked, np.nan_to_num(lengths), end, duration, max_gap)
    np.testing.assert_array_equal(edata.obsm["sustained_events"], expected)
    assert 0 < expected.sum() < marked.sum()


def test_mark_sustained_events_errors():
    edata = _marks([[1, 0, 1]])
    with pytest.raises(KeyError, match="Not found"):
        ep.tl.mark_sustained_events(edata, "marked", duration=1, max_gap=1, lengths="hours")
    with pytest.raises(KeyError, match="obs has no column 'death'"):
        ep.tl.mark_sustained_events(edata, "marked", duration=1, max_gap=1, end_key="death")


def test_mark_sustained_events_copy():
    edata = _marks([[1, 0, 1, 0]])
    result = ep.tl.mark_sustained_events(edata, "marked", duration=2, max_gap=2, key_added="abx", copy=True)

    assert "abx" not in edata.obsm
    assert "abx_first" not in edata.obs
    np.testing.assert_array_equal(result.obsm["abx"], [[True, False, True, False]])
    assert (result.obs["abx_first"].iloc[0], result.obs["abx_last"].iloc[0]) == (0, 2)


def _paired(a, b, **kwargs) -> np.ndarray:
    """The positions at which the events `a` and `b` of one observation are paired."""
    edata = _marks(a, b=b)
    ep.tl.mark_paired_events(edata, "marked", "b", **kwargs)
    return np.flatnonzero(edata.obsm["paired_events"][0])


@pytest.mark.parametrize(
    ("a", "b", "a_to_b", "b_to_a", "expected"),
    [
        ([1, 0, 0, 0], [1, 0, 0, 0], 0, 0, [0]),  # the same timepoint
        ([1, 0, 0, 0], [0, 0, 1, 0], 2, 0, [0]),  # b exactly a_to_b after a
        ([1, 0, 0, 0], [0, 0, 1, 0], 1, 10, []),  # b one more than a_to_b after a
        ([0, 0, 1, 0], [1, 0, 0, 0], 10, 2, [0]),  # a exactly b_to_a after b
        ([0, 0, 1, 0], [1, 0, 0, 0], 10, 1, []),  # a one more than b_to_a after b, and b before a does not pair a
        ([1, 0, 1, 1], [0, 1, 0, 0], 1, 1, [0, 1]),
        ([1, 0, 1, 1], [0, 0, 0, 0], 10, 10, []),
        ([nan, 0, 2, nan], [0, 0, 0, 1], 10, 10, [2]),  # missing values and 0 mark nothing
    ],
)
def test_mark_paired_events(a, b, a_to_b, b_to_a, expected):
    np.testing.assert_array_equal(_paired(a, b, a_to_b=a_to_b, b_to_a=b_to_a), expected)


def test_mark_paired_events_first_and_last_positions():
    edata = _marks([[1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]], b=[[0, 0, 1, 0, 0, 1], [0] * 6])
    ep.tl.mark_paired_events(edata, "marked", "b", a_to_b=2, b_to_a=0)

    np.testing.assert_array_equal(edata.obsm["paired_events"].astype(int), [[1, 0, 0, 0, 0, 1], [0] * 6])
    np.testing.assert_array_equal(edata.obs["paired_events_first"], [0, nan])
    np.testing.assert_array_equal(edata.obs["paired_events_last"], [5, nan])


@pytest.mark.parametrize("seed", range(6))
def test_mark_paired_events_equal_a_reference(seed):
    rng = np.random.default_rng(seed)
    a, b = (rng.choice([nan, 0.0, 1.0, 2.5], p=[0.6, 0.3, 0.05, 0.05], size=(40, 60)) for _ in range(2))
    edata = _marks(a, b=b)
    a_to_b, b_to_a = rng.choice([0, 1, 3, 6, 24], size=2)
    ep.tl.mark_paired_events(edata, "marked", "b", a_to_b=a_to_b, b_to_a=b_to_a)

    a_marked, b_marked = (~np.isnan(x) & (x != 0) for x in (a, b))
    expected = _paired_reference(a_marked, b_marked, a_to_b, b_to_a)
    np.testing.assert_array_equal(edata.obsm["paired_events"], expected)
    assert 0 < expected.sum() < (a_marked | b_marked).sum()


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_mark_paired_events_array_types(array_type, rng):
    values = np.where(rng.random((8, 2, 30)) < 0.15, 1.0, nan)
    expected = _edata(values)
    ep.tl.mark_paired_events(expected, "v0", "v1", a_to_b=3, b_to_a=6, layer=DEFAULT_TEM_LAYER_NAME)
    edata = _edata(values)
    edata.layers[DEFAULT_TEM_LAYER_NAME] = array_type(edata.layers[DEFAULT_TEM_LAYER_NAME])

    with forbid_dask_compute(allowed=1):
        ep.tl.mark_paired_events(edata, "v0", "v1", a_to_b=3, b_to_a=6, layer=DEFAULT_TEM_LAYER_NAME)
    np.testing.assert_array_equal(edata.obsm["paired_events"], expected.obsm["paired_events"])
    pd.testing.assert_frame_equal(edata.obs, expected.obs)
    assert 0 < expected.obsm["paired_events"].sum() < (~np.isnan(values)).sum()


def test_mark_paired_events_errors_and_copy():
    edata = _marks([[1, 0, 1]], b=[[0, 1, 0]])
    with pytest.raises(KeyError, match="Not found"):
        ep.tl.mark_paired_events(edata, "marked", "c", a_to_b=1, b_to_a=1)
    result = ep.tl.mark_paired_events(edata, "marked", "b", a_to_b=1, b_to_a=1, copy=True)

    assert "paired_events" not in edata.obsm
    np.testing.assert_array_equal(result.obsm["paired_events"], [[True, True, False]])


@pytest.mark.parametrize("kind", ["array", "boolean array", "frame", "boolean frame", "sparse"])
def test_sustained_and_paired_events_from_obsm(kind):
    edata = _marks(np.zeros((1, 6)))
    for key, marks in (("a", [[1, 0, 0, 1, 0, 0]]), ("b", [[0, 1, 0, 0, 0, 0]])):
        marks = np.array(marks, dtype=bool if kind.startswith("boolean") else np.float64)
        edata.obsm[key] = (
            sp.csr_matrix(marks)
            if kind == "sparse"
            else pd.DataFrame(marks, index=edata.obs_names)
            if kind.endswith("frame")
            else marks
        )
    ep.tl.mark_sustained_events(edata, "a", duration=3, max_gap=3)
    ep.tl.mark_paired_events(edata, "a", "b", a_to_b=1, b_to_a=1)

    np.testing.assert_array_equal(edata.obsm["sustained_events"], [[True, False, False, True, False, False]])
    np.testing.assert_array_equal(edata.obsm["paired_events"], [[True, False, False, False, False, False]])


def test_event_tools_read_the_markers_of_mark_events():
    values = [[[1, nan, 2, nan, nan, nan], [nan, nan, -2, nan, 1, nan], [nan, nan, 1, nan, nan, nan]]]
    edata = _edata(values, names=["drug_a", "drug_b", "sample"])
    ep.tl.mark_events(edata, ["drug_a", "drug_b"], how="sum", layer=DEFAULT_TEM_LAYER_NAME)
    ep.tl.mark_sustained_events(edata, "events", duration=4, max_gap=4, layer=DEFAULT_TEM_LAYER_NAME)
    ep.tl.mark_paired_events(edata, "events", "sample", a_to_b=2, b_to_a=0, layer=DEFAULT_TEM_LAYER_NAME)

    # the doses at 2 sum to 0, a marker that marks no timepoint: no events at 2 continue from 0 or pair the sample
    np.testing.assert_array_equal(edata.obsm["events"], [[1, nan, 0, nan, 1, nan]])
    np.testing.assert_array_equal(edata.obsm["sustained_events"], [[True, False, False, False, True, False]])
    np.testing.assert_array_equal(edata.obsm["paired_events"], [[True, False, False, False, False, False]])


def test_event_tools_without_timepoints():
    edata = _edata(np.empty((2, 2, 0)), obs=_obs(2, end=[nan, 3]))
    ep.tl.mark_events(edata, layer=DEFAULT_TEM_LAYER_NAME)
    ep.tl.mark_sustained_events(
        edata, "v0", lengths="v1", duration=1, max_gap=0, end_key="end", layer=DEFAULT_TEM_LAYER_NAME
    )
    ep.tl.mark_paired_events(edata, "v0", "v1", a_to_b=0, b_to_a=0, layer=DEFAULT_TEM_LAYER_NAME)

    for key in ("events", "sustained_events", "paired_events"):
        assert edata.obsm[key].shape == (2, 0)
        assert edata.obs[f"{key}_first"].isna().all()
        assert edata.obs[f"{key}_last"].isna().all()


def _call_with_count(argument: str, value) -> None:
    """Call the tool that takes the count `argument` with `value`, and 1 for its other count."""
    edata = _marks([[1, 0, 1]], b=[[0, 1, 0]])
    if argument in ("duration", "max_gap"):
        ep.tl.mark_sustained_events(edata, "marked", **{"duration": 1, "max_gap": 1, argument: value})
    else:
        ep.tl.mark_paired_events(edata, "marked", "b", **{"a_to_b": 1, "b_to_a": 1, argument: value})


COUNTS = ["duration", "max_gap", "a_to_b", "b_to_a"]


@pytest.mark.parametrize("argument", COUNTS)
@pytest.mark.parametrize("value", [1.0, True, "24", pd.Timedelta("1h")])
def test_counts_must_be_whole_numbers(argument, value):
    with pytest.raises(TypeError, match=f"{argument} must be a whole number of timepoints"):
        _call_with_count(argument, value)


@pytest.mark.parametrize("argument", COUNTS)
def test_counts_must_not_be_negative(argument):
    with pytest.raises(ValueError, match=f"{argument} must be a number of"):
        _call_with_count(argument, -1)


def test_duration_must_be_more_than_0():
    with pytest.raises(ValueError, match="more than 0 timepoints"):
        _call_with_count("duration", 0)


@pytest.mark.parametrize(
    ("argument", "value"), [("duration", np.int64(2)), ("max_gap", 0), ("a_to_b", 0), ("b_to_a", np.int32(0))]
)
def test_counts_of_0_and_numpy_integers(argument, value):
    _call_with_count(argument, value)


def test_renamed_arguments():
    edata = _marks([[1, 0, 1, 0]], lengths=[[1, nan, 1, nan]], b=[[0, 1, 0, 0]])
    ep.tl.mark_sustained_events(edata, events="marked", lengths="lengths", duration=2, max_gap=1)
    ep.tl.mark_paired_events(edata, a="marked", b="b", a_to_b=1, b_to_a=1)

    np.testing.assert_array_equal(edata.obsm["sustained_events"], [[True, False, True, False]])
    np.testing.assert_array_equal(edata.obsm["paired_events"], [[True, True, False, False]])
    with pytest.raises(TypeError, match="events_key"):
        ep.tl.mark_sustained_events(edata, events_key="marked", duration=2, max_gap=1)
    with pytest.raises(TypeError, match="lengths_key"):
        ep.tl.mark_sustained_events(edata, "marked", lengths_key="lengths", duration=2, max_gap=1)
    with pytest.raises(TypeError, match="a_key"):
        ep.tl.mark_paired_events(edata, a_key="marked", b_key="b", a_to_b=1, b_to_a=1)
    with pytest.raises(TypeError, match="time_key"):
        ep.tl.mark_events(edata, "marked", time_key="time_value")


@pytest.mark.parametrize("dtype", ["boolean", "Float64"])
def test_mark_events_nullable_frame_in_obsm(dtype):
    """A missing value of a nullable data frame in obsm marks no timepoint."""
    edata = ed.EHRData(shape=(1, 0), tem=pd.DataFrame(index=pd.RangeIndex(4).astype(str)))
    edata.obsm["m"] = pd.DataFrame([[True, False, None, True]], index=edata.obs_names, dtype=dtype)
    ep.tl.mark_events(edata, ["m"])
    expected = [True, False, False, True] if dtype == "boolean" else [True, True, False, True]
    np.testing.assert_array_equal(edata.obsm["events"][0], expected)


def test_mark_paired_events_counts_beyond_the_data():
    """A count far beyond the number of timepoints reaches the last timepoint."""
    edata = ed.EHRData(shape=(1, 0), tem=pd.DataFrame(index=pd.RangeIndex(4).astype(str)))
    edata.obsm["a"] = np.array([[1.0, 0, 0, 0]])
    edata.obsm["b"] = np.array([[0.0, 0, 0, 1]])
    ep.tl.mark_paired_events(edata, "a", "b", a_to_b=2**63 - 2, b_to_a=0)
    np.testing.assert_array_equal(edata.obsm["paired_events"][0], [True, False, False, False])
