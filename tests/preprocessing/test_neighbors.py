import ehrdata as ed
import numpy as np
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep
from ehrapy.core._constants import TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY


def test_neighbors_simple(edata_blob_small):
    ep.pp.neighbors(edata_blob_small, n_neighbors=5)


def test_neighbors_jaccard(edata_blob_small):
    ep.pp.neighbors(edata_blob_small, n_neighbors=5, method="jaccard")
    assert edata_blob_small.uns["neighbors"]["params"]["method"] == "jaccard"


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
    ep.pp.neighbors(edata, n_neighbors=5, metric="dtw", use_rep=DEFAULT_TEM_LAYER_NAME)

    assert np.allclose(edata.obsp["distances"].toarray(), distances)


@pytest.mark.parametrize("metric", ["dtw", "soft_dtw", "gak"])
def test_neighbors_with_timeseries_sparse_patient(rng, metric):
    layer = rng.standard_normal((12, 2, 10))
    layer[0, :, 2:] = np.nan
    edata = ed.EHRData(shape=(12, 2), layers={DEFAULT_TEM_LAYER_NAME: layer})

    ep.pp.neighbors(edata, n_neighbors=4, metric=metric, use_rep=DEFAULT_TEM_LAYER_NAME)

    distances, connectivities = edata.obsp["distances"], edata.obsp["connectivities"]
    assert np.isfinite(distances.data).all()
    assert np.isfinite(connectivities.data).all()
    assert distances[0].nnz == distances[:, 0].nnz == 0
    assert connectivities[0].nnz == connectivities[:, 0].nnz == 0
    assert (distances[1:].getnnz(axis=1) == 3).all()


@pytest.mark.parametrize("metric", ["dtw", "soft_dtw", "gak"])
def test_neighbors_with_timeseries_illegal_arguments(edata_and_distances_dtw, metric):
    edata, _ = edata_and_distances_dtw
    with pytest.raises(ValueError, match=rf"If metric is {metric}, use_rep must be a 3D array"):
        ep.pp.neighbors(edata, n_neighbors=5, metric=metric)

    wrong_use_rep_key = "nonexisting_layer_or_obsm_key"
    with pytest.raises(ValueError, match=rf"use_rep {wrong_use_rep_key} not found in edata.layers or edata.obsm"):
        ep.pp.neighbors(edata, n_neighbors=5, metric=metric, use_rep=wrong_use_rep_key)


@pytest.mark.parametrize("use_rep", [None, "X"])
def test_neighbors_3D_X(edata_and_distances_dtw, use_rep):
    edata, distances = edata_and_distances_dtw
    edata.X = edata.layers[DEFAULT_TEM_LAYER_NAME]

    ep.pp.neighbors(edata, n_neighbors=5, metric="dtw", use_rep=use_rep)
    assert np.allclose(edata.obsp["distances"].toarray(), distances)

    with pytest.raises(ValueError, match=r"neighbors\(\) only supports 2D data"):
        ep.pp.neighbors(edata, n_neighbors=5, use_rep=use_rep)
