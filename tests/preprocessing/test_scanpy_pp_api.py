import pytest
from anndata._core.anndata import Layers

import ehrapy as ep


def test_pca(edata_blob_small):
    ep.pp.pca(edata_blob_small)


def test_pca_3D_edata(edata_blob_small):
    ep.pp.pca(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.pca(edata_blob_small, layer="R_layer")


def test_regress_out(edata_blob_small):
    ep.pp.regress_out(edata_blob_small, keys=["cluster"])


def test_regress_out_3D_edata(edata_blob_small):
    ep.pp.regress_out(edata_blob_small, keys=["cluster"], layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.regress_out(edata_blob_small, layer="R_layer")


def test_subsample(edata_blob_small):
    ep.pp.subsample(edata_blob_small, fraction=0.5)


def test_combat(edata_blob_small):
    ep.pp.combat(edata_blob_small, key="cluster")


def test_combat_3D_edata(edata_blob_small):
    ep.pp.combat(edata_blob_small, key="cluster", layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.combat(edata_blob_small, key="cluster", layer="R_layer")


def test_neighbors(edata_blob_small):
    ep.pp.neighbors(edata_blob_small, n_neighbors=5)


# TODO: neighbors does not have layer support. Once X can be 3D (https://github.com/scverse/anndata/pull/1707), this function could however encounter a 3D object in X; then test this
# def test_neighbors_3D_edata(edata_blob_small):
#     ep.pp.neighbors(edata_blob_small, n_neighbors=5)
#     with pytest.raises(ValueError, match=r"only supports 2D data"):
#         ep.pp.neighbors(edata_blob_small, n_neighbors=5)
