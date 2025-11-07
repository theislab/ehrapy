from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep


def test_tsne(edata_blob_small):
    ep.tl.tsne(edata_blob_small, use_rep="X")


def test_umap(edata_blob_small):
    ep.tl.umap(edata_blob_small)


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
    ep.tl.embedding_density(edata_blob_small, basis="pca")


def test_leiden(edata_blob_small):
    ep.tl.leiden(edata_blob_small)


def test_leiden_with_timeseries_metric_dtw(edata_and_distances_dtw):
    edata, _ = edata_and_distances_dtw
    ep.pp.neighbors(edata, n_neighbors=4, metric="dtw", use_rep=DEFAULT_TEM_LAYER_NAME)
    ep.tl.leiden(edata)


def test_dendrogram(edata_blob_small):
    ep.tl.dendrogram(edata_blob_small, groupby="cluster")


def test_dpt(edata_blob_small):
    ep.tl.dpt(edata_blob_small)


def test_paga(edata_blob_small):
    # ep.pp.neighbors(adata)
    ep.tl.leiden(edata_blob_small, resolution=2)
    ep.tl.paga(edata_blob_small)


def test_paga_with_timeseries_metric_dtw(edata_and_distances_dtw):
    edata, _ = edata_and_distances_dtw
    ep.pp.neighbors(edata, n_neighbors=4, metric="dtw", use_rep=DEFAULT_TEM_LAYER_NAME)
    ep.tl.leiden(edata, resolution=2)
    ep.tl.paga(edata)


def test_ingest(edata_blob_small):
    edata_blob_small_copy = edata_blob_small.copy()
    ep.pp.pca(edata_blob_small_copy)
    ep.pp.pca(edata_blob_small)
    ep.tl.ingest(edata_blob_small, edata_ref=edata_blob_small_copy, embedding_method="pca")
