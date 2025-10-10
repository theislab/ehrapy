import ehrdata as ed
import pytest

import ehrapy as ep


def test_neighbors_simple(edata_blob_small):
    ep.pp.neighbors(edata_blob_small, n_neighbors=5)


@pytest.mark.parametrize("metric", ["dtw", "soft_dtw", "gak"])
def test_neighbors_with_timeseries_metrics(edata_blobs_timeseries_small, metric):
    """Test neighbors computation with timeseries metrics."""
    edata = edata_blobs_timeseries_small

    ep.pp.neighbors(edata, n_neighbors=5, metric=metric)

    assert "neighbors" in edata.uns
    assert "distances" in edata.obsp
    assert "connectivities" in edata.obsp
    assert edata.obsp["distances"].shape == (20, 20)
    assert edata.obsp["connectivities"].shape == (20, 20)


# TODO: neighbors does not have layer support. Once X can be 3D (https://github.com/scverse/anndata/pull/1707), this function could however encounter a 3D object in X; then test this
# def test_neighbors_3D_edata(edata_blob_small):
#     ep.pp.neighbors(edata_blob_small, n_neighbors=5)
#     with pytest.raises(ValueError, match=r"only supports 2D data"):
#         ep.pp.neighbors(edata_blob_small, n_neighbors=5)
