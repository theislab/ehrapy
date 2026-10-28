from pathlib import Path

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from fast_array_utils.conv import to_dense
from testing.fast_array_utils import Flags

import ehrapy as ep
from ehrapy.core._constants import MISSING_VALUE_COUNT_KEY_2D, MISSING_VALUE_COUNT_KEY_3D
from tests.conftest import forbid_dask_compute

CURRENT_DIR = Path(__file__).parent


@pytest.fixture
def ehr_3d_blobs():
    return ed.dt.ehrdata_blobs(
        n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6, layer=DEFAULT_TEM_LAYER_NAME
    )


def test_filter_features_invalid_args(ehr_3d_blobs):
    # no threshold
    with pytest.raises(ValueError, match="at least one of 'min_obs' and 'max_obs"):
        ep.pp.filter_features(ehr_3d_blobs, time_mode="all", copy=False)
    # invalid time_mode
    with pytest.raises(ValueError, match="must be one of 'all', 'any', 'proportion'"):
        ep.pp.filter_features(ehr_3d_blobs, min_obs=185, time_mode="invalid_mode", copy=False)
    # invalid prop
    with pytest.raises(ValueError, match="prop must be set to a value between 0 and 1"):
        ep.pp.filter_features(ehr_3d_blobs, min_obs=185, time_mode="proportion", prop=3, copy=False)


@pytest.mark.parametrize(
    "fixture, layer, kwargs",
    [
        ("ehr_3d_blobs", DEFAULT_TEM_LAYER_NAME, {"min_obs": 185}),
        ("ehr_3d_blobs", DEFAULT_TEM_LAYER_NAME, {"max_obs": 200}),
        ("mcar_edata", "X", {"min_obs": 90}),
        ("mcar_edata", "X", {"max_obs": 60}),
    ],
)
def test_filter_features_min_max(request, fixture, layer, kwargs):
    """Generic test for min_obs and max_obs filtering on 2D and 3D data"""
    edata = request.getfixturevalue(fixture)

    arr_before = edata.X if layer == "X" else edata.layers[layer]
    n_vars_before = arr_before.shape[1]

    if layer == "X":
        ep.pp.filter_features(edata, time_mode="all", copy=False, **kwargs)
        assert MISSING_VALUE_COUNT_KEY_2D in edata.var.keys()
    else:
        ep.pp.filter_features(edata, layer=layer, time_mode="all", copy=False, **kwargs)
        assert MISSING_VALUE_COUNT_KEY_3D in edata.var.keys()
    arr_after = edata.X if layer == "X" else edata.layers[layer]
    n_vars_after = arr_after.shape[1]

    assert n_vars_after < n_vars_before


def test_filter_features_layers(ehr_3d_blobs):
    edata = ehr_3d_blobs
    with pytest.raises(KeyError):
        ep.pp.filter_features(edata, layer="invalid_layer", min_obs=185, time_mode="all", copy=False)

    layer_before = edata.layers[DEFAULT_TEM_LAYER_NAME].copy()
    n_vars_before = layer_before.shape[1]

    ep.pp.filter_features(edata, layer=DEFAULT_TEM_LAYER_NAME, min_obs=185, time_mode="all", copy=False)

    layer_after = edata.layers[DEFAULT_TEM_LAYER_NAME]
    n_vars_after = layer_after.shape[1]

    assert n_vars_after < n_vars_before


def test_filter_obs_invalid_args(ehr_3d_blobs):
    # no threshold
    with pytest.raises(ValueError, match="at least one of 'min_vars', 'max_vars' and 'query'"):
        ep.pp.filter_observations(ehr_3d_blobs, time_mode="all", copy=False)
    # invalid time_mode
    with pytest.raises(ValueError, match="must be one of 'all', 'any', 'proportion'"):
        ep.pp.filter_observations(ehr_3d_blobs, min_vars=10, time_mode="invalid_mode", copy=False)
    # invalid prop
    with pytest.raises(ValueError, match="prop must be set to a value between 0 and 1"):
        ep.pp.filter_observations(ehr_3d_blobs, min_vars=10, time_mode="proportion", prop=2, copy=False)


@pytest.mark.parametrize(
    "fixture, layer, kwargs",
    [
        ("ehr_3d_blobs", DEFAULT_TEM_LAYER_NAME, {"min_vars": 23}),
        ("ehr_3d_blobs", DEFAULT_TEM_LAYER_NAME, {"max_vars": 18}),
        ("mcar_edata", "X", {"min_vars": 8}),
        ("mcar_edata", "X", {"max_vars": 9}),
    ],
)
def test_filter_obs_min_max(request, fixture, layer, kwargs):
    """Generic test for min_vars and max_vars filtering on 2D and 3D data"""
    edata = request.getfixturevalue(fixture)

    layer_before = edata.X if layer == "X" else edata.layers[layer]
    n_obs_before = layer_before.shape[0]

    if layer == "X":
        ep.pp.filter_observations(edata, time_mode="all", copy=False, **kwargs)
        assert MISSING_VALUE_COUNT_KEY_2D in edata.obs.keys()
    else:
        ep.pp.filter_observations(edata, layer=layer, time_mode="all", copy=False, **kwargs)
        assert MISSING_VALUE_COUNT_KEY_3D in edata.obs.keys()

    layer_after = edata.X if layer == "X" else edata.layers[layer]
    n_obs_after = layer_after.shape[0]

    assert n_obs_after < n_obs_before


def test_filter_obs_layers(ehr_3d_blobs):
    edata = ehr_3d_blobs
    with pytest.raises(KeyError):
        ep.pp.filter_observations(edata, layer="invalid_layer", min_vars=10, time_mode="all", copy=False)

    layer_before = edata.layers[DEFAULT_TEM_LAYER_NAME].copy()
    n_obs_before = layer_before.shape[0]

    ep.pp.filter_observations(edata, layer=DEFAULT_TEM_LAYER_NAME, min_vars=10, time_mode="all", copy=False)

    layer_after = edata.layers[DEFAULT_TEM_LAYER_NAME]
    n_obs_after = layer_after.shape[0]

    assert n_obs_after < n_obs_before


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("time_mode", ["all", "proportion"])
@pytest.mark.parametrize(
    ("filter_func", "kwargs", "annotated"),
    [
        pytest.param(ep.pp.filter_features, {"min_obs": 12}, "var", id="features-min"),
        pytest.param(ep.pp.filter_features, {"max_obs": 15}, "var", id="features-max"),
        pytest.param(ep.pp.filter_observations, {"min_vars": 3, "max_vars": 4}, "obs", id="observations-min-max"),
    ],
)
def test_filter_array_types(array_type, ndim, time_mode, filter_func, kwargs, annotated, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    shape = (20, 6) if ndim == 2 else (20, 6, 3)
    X = np.where(rng.random(shape) < 0.5, 0.0, rng.gamma(2, size=shape))
    X[rng.random(shape) < np.linspace(0, 0.8, 6).reshape(1, 6, *(1,) * (ndim - 2))] = np.nan
    X[:, 0] = np.nan
    X[:, 1] = 2.0
    kwargs = {**kwargs, "time_mode": time_mode, "prop": 0.5 if time_mode == "proportion" else None}
    expected = filter_func(ed.EHRData(X=X), copy=True, **kwargs)
    edata = ed.EHRData(X=array_type(X))

    with forbid_dask_compute(allowed=1):
        result = filter_func(edata, copy=True, **kwargs)

    assert isinstance(result.X, array_type.cls)
    np.testing.assert_allclose(to_dense(result.X, to_cpu_memory=True), expected.X, equal_nan=True)
    pd.testing.assert_frame_equal(getattr(result, annotated), getattr(expected, annotated))


@pytest.fixture
def cohort_edata():
    X = np.array(
        [
            [[7.0, 9.0, 12.0], [120.0, 130.0, np.nan]],
            [[5.0, 6.0, np.nan], [140.0, 150.0, 160.0]],
            [[np.nan, 8.0, 4.0], [110.0, np.nan, 100.0]],
            [[6.0, 5.0, 5.0], [np.nan, np.nan, np.nan]],
            [[9.0, np.nan, 11.0], [125.0, 135.0, 145.0]],
        ]
    )
    obs = pd.DataFrame(
        {"age": [70, 17, 45, 60, 30], "sex": pd.Categorical(["f", "m", "f", "m", "m"])},
        index=[f"patient_{i}" for i in range(5)],
    )
    var = pd.DataFrame(index=["glucose", "blood pressure"])
    tem = pd.DataFrame(index=["day_0", "day_1", "day_2"])
    return ed.EHRData(X=X, obs=obs, var=var, tem=tem)


@pytest.mark.parametrize(
    ("kwargs", "kept"),
    [
        pytest.param({"query": "age >= 18"}, [0, 2, 3, 4], id="obs"),
        pytest.param({"query": "glucose > 6", "tem_names": "day_0"}, [0, 4], id="timepoint"),
        pytest.param({"query": "glucose > 10", "agg": "max"}, [0, 4], id="agg"),
        pytest.param({"query": "glucose > 8", "tem_names": slice(1, 3), "agg": "mean"}, [0, 4], id="agg-window"),
        pytest.param({"query": "`blood pressure`.isna()", "agg": "last"}, [3], id="backticks"),
        pytest.param({"query": ["sex == 'm'", "glucose < 7"], "agg": "first"}, [1, 3], id="sequence"),
        pytest.param({"min_vars": 2, "query": "age < 65", "tem_names": ["day_0"]}, [1, 4], id="min-vars"),
    ],
)
def test_filter_obs_query(cohort_edata, kwargs, kept):
    result = ep.pp.filter_observations(cohort_edata, copy=True, **kwargs)

    assert list(result.obs_names) == [f"patient_{i}" for i in kept]


def test_filter_obs_query_2d(cohort_edata):
    edata = ed.EHRData(X=cohort_edata.X[:, :, 0], obs=cohort_edata.obs, var=cohort_edata.var)

    ep.pp.filter_observations(edata, query="glucose > 5 and age > 40")

    assert list(edata.obs_names) == ["patient_0", "patient_3"]


def test_filter_obs_query_tracker(cohort_edata):
    tracker = ep.tl.CohortTracker(cohort_edata, columns=["age", "sex"])

    ep.pp.filter_observations(
        cohort_edata, min_vars=2, query=["age >= 18", "glucose > 8"], tem_names="day_0", tracker=tracker
    )

    assert list(cohort_edata.obs_names) == ["patient_4"]
    assert tracker._tracked_text == [
        "Cohort 0\n (n=5)",
        "min_vars=2\n (n=3)",
        "age >= 18\n (n=2)",
        "glucose > 8\n (n=1)",
    ]
    assert tracker._tracked_parents == [None, 0, 1, 2]


def test_filter_obs_query_tracker_continues(cohort_edata):
    tracker = ep.tl.CohortTracker(cohort_edata, columns=["age"])
    tracker(cohort_edata, label="Enrolled")

    ep.pp.filter_observations(cohort_edata, query="age >= 18", tracker=tracker)

    assert tracker._tracked_text == ["Enrolled\n (n=5)", "age >= 18\n (n=4)"]


def test_filter_obs_query_invalid(cohort_edata):
    with pytest.raises(ValueError, match="need `agg` or `tem_names`"):
        ep.pp.filter_observations(cohort_edata, query="glucose > 6")
    with pytest.raises(TypeError, match="must evaluate to a boolean"):
        ep.pp.filter_observations(cohort_edata, query="age + 1")
    with pytest.raises(KeyError, match="day_9"):
        ep.pp.filter_observations(cohort_edata, query="glucose > 6", tem_names="day_9")
    cohort_edata.obs["glucose"] = 1.0
    with pytest.raises(ValueError, match="both obs columns and variables"):
        ep.pp.filter_observations(cohort_edata, query="glucose > 6", agg="max")
    edata_2d = ed.EHRData(X=cohort_edata.X[:, :, 0], obs=cohort_edata.obs, var=cohort_edata.var)
    with pytest.raises(ValueError, match="tem_names needs 3D data"):
        ep.pp.filter_observations(edata_2d, query="age > 18", tem_names="day_0")
    assert cohort_edata.n_obs == 5


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
def test_filter_obs_query_array_types(array_type, ndim, cohort_edata):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    X = cohort_edata.X if ndim == 3 else np.nan_to_num(cohort_edata.X[:, :, 1], nan=0.0)
    tem = cohort_edata.tem if ndim == 3 else None
    edata = ed.EHRData(X=array_type(X), obs=cohort_edata.obs, var=cohort_edata.var, tem=tem)
    kwargs = {"min_vars": 1, "query": ["glucose > 5", "`blood pressure` > 0"], "agg": "max"}

    with forbid_dask_compute(allowed=1):
        result = ep.pp.filter_observations(edata, copy=True, **kwargs)

    expected = [0, 1, 2, 4] if ndim == 3 else [0, 1]
    assert list(result.obs_names) == [f"patient_{i}" for i in expected]
    assert isinstance(result.X, array_type.cls)
    np.testing.assert_array_equal(to_dense(result.X, to_cpu_memory=True), X[expected])
