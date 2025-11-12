from pathlib import Path

import ehrdata as ed
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep
from ehrapy.core._constants import MISSING_VALUE_COUNT_KEY_2D, MISSING_VALUE_COUNT_KEY_3D

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
    with pytest.raises(ValueError, match="at least one of 'min_vars' and 'max_vars"):
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
        ep.pp.filter_features(edata, layer="invalid_layer", min_obs=185, time_mode="all", copy=False)

    layer_before = edata.layers[DEFAULT_TEM_LAYER_NAME].copy()
    n_obs_before = layer_before.shape[0]

    ep.pp.filter_observations(edata, layer=DEFAULT_TEM_LAYER_NAME, min_vars=10, time_mode="all", copy=False)

    layer_after = edata.layers[DEFAULT_TEM_LAYER_NAME]
    n_obs_after = layer_after.shape[0]

    assert n_obs_after < n_obs_before
