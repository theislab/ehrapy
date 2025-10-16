from pathlib import Path

import ehrdata as ed
import numpy as np
import pytest

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent


@pytest.fixture
def ehr_3d_blobs(scope="module"):
    return ed.dt.ehrdata_blobs(n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6)


@pytest.fixture
def invalid_data(scope="module"):
    rng = np.random.default_rng()
    return rng.random((10, 5))


def test_filter_features_invalid_args(ehr_3d_blobs, invalid_data):
    edata = ehr_3d_blobs.copy()

    # passing a non AnnData / EHRData object
    with pytest.raises(TypeError):
        ep.pp.filter_features(invalid_data, min_obs=10, time_mode="all", copy=False)
    # no threshold
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata, time_mode="all", copy=False)
    # invalid time_mode
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata, min_obs=185, time_mode="invalid_mode", copy=False)
    # invalid prop
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata, min_obs=185, time_mode="proportion", prop=3, copy=False)


@pytest.mark.parametrize(
    "fixture, shape, kwargs",
    [
        ("ehr_3d_blobs", "R", {"min_obs": 185}),
        ("ehr_3d_blobs", "R", {"max_obs": 200}),
        ("mcar_edata", "X", {"min_obs": 90}),
        ("mcar_edata", "X", {"max_obs": 60}),
    ],
)
def test_filter_features_min_max(request, fixture, shape, kwargs):
    # generic test for min_obs and max_obs filtering on 2d and 3d data
    edata = request.getfixturevalue(fixture).copy()

    layer = getattr(edata, shape)
    n_vars_before = layer.shape[1]

    ep.pp.filter_features(edata, time_mode="all", copy=False, **kwargs)

    layer_after = getattr(edata, shape)
    n_vars_after = layer_after.shape[1]

    assert n_vars_after < n_vars_before


def test_filter_features_layers(ehr_3d_blobs):
    edata = ehr_3d_blobs.copy()
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata, layers="invalid_layer", min_obs=185, time_mode="all", copy=False)

    layer_before = edata.layers["R_layer"].copy()
    n_vars_before = layer_before.shape[1]

    ep.pp.filter_features(edata, layers="R_layer", min_obs=185, time_mode="all", copy=False)

    layer_after = edata.layers["R_layer"]
    n_vars_after = layer_after.shape[1]

    assert n_vars_after < n_vars_before


def test_filter_obs_invalid_args(ehr_3d_blobs, invalid_data):
    edata = ehr_3d_blobs.copy()

    # passing a non AnnData / EHRData object
    with pytest.raises(TypeError):
        ep.pp.filter_observations(invalid_data, min_vars=10, time_mode="all", copy=False)
    # no threshold
    with pytest.raises(ValueError):
        ep.pp.filter_observations(edata, time_mode="all", copy=False)
    # invalid time_mode
    with pytest.raises(ValueError):
        ep.pp.filter_observations(edata, min_vars=10, time_mode="invalid_mode", copy=False)
    # invalid prop
    with pytest.raises(ValueError):
        ep.pp.filter_observations(edata, min_vars=10, time_mode="proportion", prop=2, copy=False)


@pytest.mark.parametrize(
    "fixture, shape, kwargs",
    [
        ("ehr_3d_blobs", "R", {"min_vars": 10}),
        ("ehr_3d_blobs", "R", {"max_vars": 12}),
        ("mcar_edata", "X", {"min_vars": 10}),
        ("mcar_edata", "X", {"max_vars": 6}),
    ],
)
def test_filter_obs_min_max(request, fixture, shape, kwargs):
    # generic test for min_obs and max_obs filtering on 2d and 3d data
    edata = request.getfixturevalue(fixture).copy()

    layer = getattr(edata, shape)
    n_obs_before = layer.shape[0]

    ep.pp.filter_observations(edata, time_mode="all", copy=False, **kwargs)

    layer_after = getattr(edata, shape)
    n_obs_after = layer_after.shape[0]

    assert n_obs_after < n_obs_before


def test_filter_obs_layers(ehr_3d_blobs):
    edata = ehr_3d_blobs.copy()
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata, layers="invalid_layer", min_obs=185, time_mode="all", copy=False)

    layer_before = edata.layers["R_layer"].copy()
    n_obs_before = layer_before.shape[0]

    ep.pp.filter_observations(edata, layers="R_layer", min_vars=10, time_mode="all", copy=False)

    layer_after = edata.layers["R_layer"]
    n_obs_after = layer_after.shape[0]

    assert n_obs_after < n_obs_before
