import ehrdata as ed
import numpy as np
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep
from ehrapy.core._constants import TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY


def test_neighbors_simple(edata_blob_small):
    ep.pp.neighbors(edata_blob_small, n_neighbors=5)


@pytest.mark.parametrize("metric", ["dtw", "soft_dtw", "gak"])
def test_neighbors_with_timeseries_metrics(edata_and_distances_dtw, metric):
    """Test neighbors computation with timeseries metrics."""
    edata, _ = edata_and_distances_dtw

    ep.pp.neighbors(edata, n_neighbors=3, metric=metric, use_rep=DEFAULT_TEM_LAYER_NAME)

    assert "neighbors" in edata.uns
    assert "distances" in edata.obsp
    assert "connectivities" in edata.obsp
    assert edata.obsp["distances"].shape == (5, 5)
    assert edata.obsp["connectivities"].shape == (5, 5)
    assert TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY not in edata.obsm


def test_neighbors_with_timeseries_metric_dtw_tight_test(edata_and_distances_dtw):
    edata, distances = edata_and_distances_dtw
    ep.pp.neighbors(edata, n_neighbors=5, metric="dtw")

    assert np.allclose(edata.obsp["distances"].toarray(), distances)


# TODO: neighbors does not have layer support. Once X can be 3D (https://github.com/scverse/anndata/pull/1707), this function could however encounter a 3D object in X; then test this
# def test_neighbors_3D_edata(edata_blob_small):
#     ep.pp.neighbors(edata_blob_small, n_neighbors=5)
#     with pytest.raises(ValueError, match=r"only supports 2D data"):
#         ep.pp.neighbors(edata_blob_small, n_neighbors=5)
