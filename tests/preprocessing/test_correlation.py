import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_compute_variable_correlations_pearson(edata_blobs_timeseries_small, method):
    edata = edata_blobs_timeseries_small
    corr_df, pval_df, sig_df = ep.pp.variable_correlations(edata=edata, layer=DEFAULT_TEM_LAYER_NAME, method=method)

    assert corr_df.shape == (11, 11)
    assert pval_df.shape == (11, 11)
    assert sig_df.shape == (11, 11)

    assert np.array_equal(corr_df.values, corr_df.values.T)
    assert np.array_equal(np.diag(corr_df.values), np.ones(11))

    # Bounds
    assert (corr_df.values >= -1).all()
    assert (corr_df.values <= 1).all()

    assert (pval_df.values >= 0).all()
    assert (pval_df.values <= 1).all()

    assert sig_df.values.dtype == bool


@pytest.mark.parametrize("agg", ["mean", "last", "first"])
def test_compute_variable_correlations_aggregation(edata_blobs_timeseries_small, agg):
    edata = edata_blobs_timeseries_small

    corr_df, _, _ = ep.pp.variable_correlations(edata=edata, layer=DEFAULT_TEM_LAYER_NAME, agg=agg)

    assert corr_df.shape == (11, 11)
    assert not np.isnan(corr_df.values).all()


def test_compute_variable_correlations_errors(edata_blobs_timeseries_small, edata_mini_3D_missing_values):
    edata = edata_blobs_timeseries_small
    cat_edata = edata_mini_3D_missing_values
    with pytest.raises(KeyError, match="Layer .* not found"):
        ep.pp.variable_correlations(edata=edata, layer="unsupported")
    with pytest.raises(KeyError, match="Variables not found"):
        ep.pp.variable_correlations(edata=edata, layer=DEFAULT_TEM_LAYER_NAME, var_names=["var_0", "nonexistent_var"])
    with pytest.raises(ValueError, match="Non-numeric variables"):
        ep.pp.variable_correlations(edata=cat_edata, layer=DEFAULT_TEM_LAYER_NAME, var_names=["5"])
    with pytest.raises(ValueError, match="Unsupported correlation method"):
        ep.pp.variable_correlations(edata=edata, layer=DEFAULT_TEM_LAYER_NAME, method="unsupported")
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        ep.pp.variable_correlations(edata=edata, layer=DEFAULT_TEM_LAYER_NAME, agg="median")


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("agg", ["mean", "first", "last"])
def test_variable_correlations_array_types(array_type, ndim, agg, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    shape = (30, 5) if ndim == 2 else (30, 5, 3)
    X = np.where(rng.random(shape) < 0.5, 0, rng.normal(size=shape))
    X[:, 1] += X[:, 0]
    X[rng.random(shape) < 0.1] = np.nan
    X[:, 3] = 2.0
    X[:, 4] = np.nan

    def make_edata(X):
        return ed.EHRData(shape=shape[:2], layers={DEFAULT_TEM_LAYER_NAME: X})

    expected = ep.pp.variable_correlations(make_edata(X), layer=DEFAULT_TEM_LAYER_NAME, agg=agg)
    edata = make_edata(array_type(X))

    if array_type.flags & Flags.Sparse:
        with pytest.raises(NotImplementedError):
            ep.pp.variable_correlations(edata, layer=DEFAULT_TEM_LAYER_NAME, agg=agg)
        return

    with forbid_dask_compute(allowed=1):
        result = ep.pp.variable_correlations(edata, layer=DEFAULT_TEM_LAYER_NAME, agg=agg)

    for result_df, expected_df in zip(result, expected, strict=True):
        pd.testing.assert_frame_equal(result_df, expected_df)
