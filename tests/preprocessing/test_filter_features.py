from pathlib import Path

import ehrdata as ed
import numpy as np
import pytest

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent


def test_filter_features_invalid_args_min_max_obs():
    edata = ed.dt.ehrdata_blobs(n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6)

    # no threshold
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata)
    # invalid time_mode
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata, min_obs=185, time_mode="invalid_mode", copy=False)
    # invalid prop
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata, min_obs=185, time_mode="proportion", prop=2, copy=False)

    # min_obs filtering
    n_vars_before = edata.R.shape[1]
    ep.pp.filter_features(edata, min_obs=185, time_mode="all", copy=False)
    n_vars_after = edata.R.shape[1]
    assert n_vars_after < n_vars_before

    # max_obs filtering
    n_vars_before = edata.R.shape[1]
    ep.pp.filter_features(edata, max_obs=200, time_mode="all", copy=False)
    n_vars_after = edata.R.shape[1]
    assert n_vars_after < n_vars_before


def test_filter_obs_invalid_args_min_max_vars():
    edata = ed.dt.ehrdata_blobs(n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6)

    # no threshold
    with pytest.raises(ValueError):
        ep.pp.filter_observations(edata)
    # invalid time_mode
    with pytest.raises(ValueError):
        ep.pp.filter_observations(edata, min_vars=10, time_mode="invalid_mode", copy=False)
    # invalid prop
    with pytest.raises(ValueError):
        ep.pp.filter_observations(edata, min_vars=10, time_mode="proportion", prop=2, copy=False)

    # min_vars filtering
    n_obs_before = edata.R.shape[0]
    ep.pp.filter_observations(edata, min_vars=10, time_mode="all", copy=False)
    n_obs_after = edata.R.shape[0]
    assert n_obs_after < n_obs_before

    # max_vars filtering
    n_obs_before = edata.R.shape[0]
    ep.pp.filter_observations(edata, max_vars=12, time_mode="all", copy=False)
    n_obs_after = edata.R.shape[0]
    assert n_obs_after < n_obs_before
