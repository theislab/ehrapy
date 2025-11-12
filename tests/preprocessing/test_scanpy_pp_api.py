import numpy as np
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep


def test_pca(edata_blob_small):
    ep.pp.pca(edata_blob_small)


def test_pca_3D_edata(edata_blob_small):
    ep.pp.pca(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.pca(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


def test_regress_out(edata_blob_small):
    ep.pp.regress_out(edata_blob_small, keys=["cluster"])


def test_regress_out_3D_edata(edata_blob_small):
    ep.pp.regress_out(edata_blob_small, keys=["cluster"], layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.regress_out(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


def test_subsample(edata_blob_small):
    ep.pp.subsample(edata_blob_small, fraction=0.5)


def test_combat(edata_blob_small):
    ep.pp.combat(edata_blob_small, key="cluster")


def test_combat_3D_edata(edata_blob_small):
    ep.pp.combat(edata_blob_small, key="cluster", layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.combat(edata_blob_small, key="cluster", layer=DEFAULT_TEM_LAYER_NAME)


def test_neighbors(edata_blob_small):
    ep.pp.neighbors(edata_blob_small, n_neighbors=5)

    # since use_rep="..." is possible, check edge case where X is None and layers invalid
    edata_blob_small.obsm["X_pca"] = np.random.default_rng(42).random((edata_blob_small.n_obs, 5))
    edata_blob_small.X = None
    ep.pp.neighbors(edata_blob_small, use_rep="X_pca", n_neighbors=5)
