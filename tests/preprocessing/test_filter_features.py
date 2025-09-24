from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent


def test_filter_features_basic(mcar_edata_3d):
    edata = mcar_edata_3d.copy()
    edata.R[:, -1, :] = np.nan

    # no threshold
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata, inplace=False)
    # multiple thresholds
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata, min_obs=5, max_counts=5, inplace=True)
    # invalid time_mode
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata, min_obs=5, time_mode="invalid_mode", inplace=True, copy=False)
    # invalid prop
    with pytest.raises(ValueError):
        ep.pp.filter_features(edata, min_obs=5, time_mode="proportion", prop=2, inplace=True, copy=False)

    n_vars_before = edata.R.shape[1]
    ep.pp.filter_features(edata, min_obs=5, time_mode="any", inplace=True, copy=False)
    n_vars_after = edata.R.shape[1]
    assert n_vars_after < n_vars_before


def test_filter_obs_basic(mcar_edata_3d):
    edata = mcar_edata_3d.copy()

    # no threshold
    with pytest.raises(ValueError):
        ep.pp.filter_observations(edata, inplace=False)
    # multiple thresholds
    with pytest.raises(ValueError):
        ep.pp.filter_observations(edata, min_vars=5, max_counts=5, inplace=True)
    # invalid time_mode
    with pytest.raises(ValueError):
        ep.pp.filter_observations(edata, min_vars=5, time_mode="invalid_mode", inplace=True, copy=False)
    # invalid prop
    with pytest.raises(ValueError):
        ep.pp.filter_observations(edata, min_vars=5, time_mode="proportion", prop=2, inplace=True, copy=False)

    n_obs_before = edata.R.shape[0]
    ep.pp.filter_observations(edata, min_counts=10, time_mode="any", inplace=True)
    n_obs_after = edata.R.shape[0]
    assert n_obs_after < n_obs_before
