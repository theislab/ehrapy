import pytest

import ehrapy as ep


def test_pca(edata_blob_small):
    ep.pp.pca(edata_blob_small)


def test_pca_3D_edata(edata_blob_small):
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.pca(edata_blob_small, layer="R_layer")


def test_regress_out(edata_blob_small):
    ep.pp.regress_out(edata_blob_small, keys=["cluster"])


def test_regress_out_3D_edata(edata_blob_small):
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.regress_out(edata_blob_small, layer="R_layer")


def test_subsample(edata_blob_small):
    ep.pp.subsample(edata_blob_small, fraction=0.5)


def test_subsample_3D_edata(edata_blob_small):
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.subsample(edata_blob_small, layer="R_layer")


def test_combat(edata_blob_small):
    ep.pp.combat(edata_blob_small, key="cluster")


def test_combat_3D_edata(edata_blob_small):
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.combat(edata_blob_small, layer="R_layer")


def test_neighbors(edata_blob_small):
    ep.pp.neighbors(edata_blob_small, n_neighbors=5)


def test_neighbors_3D_edata(edata_blob_small):
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.neighbors(edata_blob_small, layer="R_layer")
