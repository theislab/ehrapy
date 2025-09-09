import ehrapy as ep


def test_tsne(edata_blob_small):
    ep.tl.tsne(edata_blob_small, use_rep="X")


def test_umap(edata_blob_small):
    ep.tl.umap(edata_blob_small)


def test_draw_graph(edata_blob_small):
    ep.tl.draw_graph(edata_blob_small)


def test_diffmap(edata_blob_small):
    ep.tl.diffmap(edata_blob_small)


def test_embedding_density(edata_blob_small):
    ep.pp.pca(edata_blob_small)
    ep.tl.embedding_density(edata_blob_small, basis="pca")


def test_leiden(edata_blob_small):
    ep.tl.leiden(edata_blob_small)


def test_dendrogram(edata_blob_small):
    ep.tl.dendrogram(edata_blob_small, groupby="cluster")


def test_dpt(edata_blob_small):
    ep.tl.dpt(edata_blob_small)


def test_paga(edata_blob_small):
    ep.tl.leiden(edata_blob_small)
    ep.tl.paga(edata_blob_small)


def test_ingest(edata_blob_small):
    edata_blob_small_copy = edata_blob_small.copy()
    ep.pp.pca(edata_blob_small_copy)
    ep.pp.pca(edata_blob_small)
    ep.tl.ingest(edata_blob_small, edata_ref=edata_blob_small_copy, embedding_method="pca")
