import ehrdata as ed
import numpy as np
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import DASK_WITH_SPARSE_CHUNKS, forbid_dask_compute


def test_tsne(edata_blob_small):
    ep.tl.tsne(edata_blob_small, use_rep="X")


def test_tsne_key_added(edata_blob_small):
    ep.tl.tsne(edata_blob_small, use_rep="X", n_components=3, key_added="X_tsne_3d")
    assert edata_blob_small.obsm["X_tsne_3d"].shape == (edata_blob_small.n_obs, 3)
    assert "X_tsne" not in edata_blob_small.obsm


def test_umap(edata_blob_small):
    ep.tl.umap(edata_blob_small)


def test_umap_key_added(edata_blob_small):
    ep.tl.umap(edata_blob_small, key_added="X_umap_custom")
    assert "X_umap_custom" in edata_blob_small.obsm
    assert "X_umap" not in edata_blob_small.obsm


def test_umap_with_timeseries_metric_dtw(edata_and_distances_dtw):
    edata, _ = edata_and_distances_dtw
    ep.pp.neighbors(edata, n_neighbors=4, metric="dtw", use_rep=DEFAULT_TEM_LAYER_NAME)
    ep.tl.umap(edata)


def test_draw_graph(edata_blob_small):
    ep.tl.draw_graph(edata_blob_small)


def test_draw_graph_with_timeseries_metric_dtw(edata_and_distances_dtw):
    edata, _ = edata_and_distances_dtw
    ep.pp.neighbors(edata, n_neighbors=4, metric="dtw", use_rep=DEFAULT_TEM_LAYER_NAME)
    ep.tl.draw_graph(edata)


def test_diffmap(edata_blob_small):
    ep.tl.diffmap(edata_blob_small)


def test_diffmap_with_timeseries_metric_dtw(edata_and_distances_dtw):
    edata, _ = edata_and_distances_dtw
    ep.pp.neighbors(edata, n_neighbors=4, metric="dtw", use_rep=DEFAULT_TEM_LAYER_NAME)
    ep.tl.diffmap(edata)


def test_embedding_density(edata_blob_small):
    ep.pp.pca(edata_blob_small)
    edata_copy = ep.tl.embedding_density(edata_blob_small, basis="pca", copy=True)
    assert "pca_density" in edata_copy.obs
    assert "pca_density" not in edata_blob_small.obs

    assert ep.tl.embedding_density(edata_blob_small, basis="pca") is None
    assert "pca_density" in edata_blob_small.obs


def test_leiden(edata_blob_small):
    ep.tl.leiden(edata_blob_small)


def test_leiden_with_timeseries_metric_dtw(edata_and_distances_dtw):
    edata, _ = edata_and_distances_dtw
    ep.pp.neighbors(edata, n_neighbors=4, metric="dtw", use_rep=DEFAULT_TEM_LAYER_NAME)
    ep.tl.leiden(edata)


def test_dendrogram(edata_blob_small):
    edata_copy = ep.tl.dendrogram(edata_blob_small, groupby="cluster", copy=True)
    assert "dendrogram_cluster" in edata_copy.uns
    assert "dendrogram_cluster" not in edata_blob_small.uns

    assert ep.tl.dendrogram(edata_blob_small, groupby="cluster") is None
    assert "dendrogram_cluster" in edata_blob_small.uns


def test_dpt(edata_blob_small):
    ep.tl.dpt(edata_blob_small)


def test_paga(edata_blob_small):
    # ep.pp.neighbors(edata_blob_small)
    ep.tl.leiden(edata_blob_small, resolution=2)
    ep.tl.paga(edata_blob_small)


def test_paga_with_timeseries_metric_dtw(edata_and_distances_dtw):
    edata, _ = edata_and_distances_dtw
    ep.pp.neighbors(edata, n_neighbors=4, metric="dtw", use_rep=DEFAULT_TEM_LAYER_NAME)
    ep.tl.leiden(edata, resolution=2)
    ep.tl.paga(edata)


def test_ingest(edata_blob_small):
    edata_ref = edata_blob_small.copy()
    ep.pp.pca(edata_ref)

    edata_copy = ep.tl.ingest(edata_blob_small, edata_ref=edata_ref, embedding_method="pca", copy=True)
    assert "X_pca" in edata_copy.obsm
    assert "X_pca" not in edata_blob_small.obsm

    assert ep.tl.ingest(edata_blob_small, edata_ref=edata_ref, embedding_method="pca") is None
    assert "X_pca" in edata_blob_small.obsm


def test_x_based_tools_3D_raise(edata_blob_small):
    edata_3d = ed.EHRData(X=edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME], obs=edata_blob_small.obs)
    edata_3d.obsm["X_pca"] = edata_blob_small.X[:, :3]
    edata_ref = edata_blob_small.copy()
    ep.pp.pca(edata_ref)

    for tool in (
        lambda: ep.tl.tsne(edata_3d),
        lambda: ep.tl.dendrogram(edata_3d, groupby="cluster"),
        lambda: ep.tl.ingest(edata_3d, edata_ref, embedding_method="pca"),
    ):
        with pytest.raises(ValueError, match="only supports 2D data"):
            tool()

    ep.tl.tsne(edata_3d, use_rep="X_pca", perplexity=5)
    ep.tl.dendrogram(edata_3d, groupby="cluster", use_rep="X_pca")


@pytest.mark.array_type(skip={*DASK_WITH_SPARSE_CHUNKS, Flags.Disk, Flags.Gpu})
@pytest.mark.parametrize("var_names", [None, ["feature_1", "feature_2", "feature_3"]])
def test_dendrogram_array_types(array_type, var_names, edata_blob_small):
    expected = ep.tl.dendrogram(edata_blob_small, groupby="cluster", var_names=var_names, copy=True)
    edata_blob_small.X = array_type(edata_blob_small.X)

    with forbid_dask_compute(allowed=1):
        ep.tl.dendrogram(edata_blob_small, groupby="cluster", var_names=var_names)

    np.testing.assert_allclose(
        edata_blob_small.uns["dendrogram_cluster"]["correlation_matrix"],
        expected.uns["dendrogram_cluster"]["correlation_matrix"],
    )


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_ingest_array_types(array_type, edata_blob_small):
    edata_ref = edata_blob_small.copy()
    ep.pp.pca(edata_ref)
    edata_blob_small.X = array_type(edata_blob_small.X)

    if array_type.cls is not np.ndarray:
        with pytest.raises(NotImplementedError, match="only supports numpy arrays"):
            ep.tl.ingest(edata_blob_small, edata_ref, embedding_method="pca")
        return

    ep.tl.ingest(edata_blob_small, edata_ref, embedding_method="pca")
    assert "X_pca" in edata_blob_small.obsm
