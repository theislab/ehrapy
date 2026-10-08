from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep


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
