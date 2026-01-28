import numpy as np
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_compute_variable_correlations_pearson(edata_blobs_timeseries_small, method):
    edata = edata_blobs_timeseries_small
    corr_df, pval_df, sig_df = ep.tl.compute_variable_correlations(
        edata=edata, layer=DEFAULT_TEM_LAYER_NAME, method=method
    )

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

    corr_df, _, _ = ep.tl.compute_variable_correlations(edata=edata, layer=DEFAULT_TEM_LAYER_NAME, agg=agg)

    assert corr_df.shape == (11, 11)
    assert not np.isnan(corr_df.values).all()


def test_compute_variable_correlations_errors(edata_blobs_timeseries_small):
    edata = edata_blobs_timeseries_small
    with pytest.raises(KeyError, match="Layer .* not found"):
        ep.tl.compute_variable_correlations(edata=edata, layer="unsupported")
    with pytest.raises(KeyError, match="Variables not found"):
        ep.tl.compute_variable_correlations(
            edata=edata, layer=DEFAULT_TEM_LAYER_NAME, var_names=["var_0", "nonexistent_var"]
        )
    with pytest.raises(ValueError, match="Unsupported correlation method"):
        ep.tl.compute_variable_correlations(edata=edata, layer=DEFAULT_TEM_LAYER_NAME, method="unsupported")
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        ep.tl.compute_variable_correlations(edata=edata, layer=DEFAULT_TEM_LAYER_NAME, agg="median")
