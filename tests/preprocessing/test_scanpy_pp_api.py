import anndata as ad
import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from fast_array_utils.conv import to_dense
from sklearn.decomposition import PCA
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute


@pytest.fixture
def batched_data(rng) -> tuple[np.ndarray, pd.DataFrame]:
    X = rng.normal(5, 2, size=(40, 6))
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(["a", "b", "c", "d"] * 10),
            "condition": rng.choice(["x", "y"], size=40),
            "covariate": rng.normal(size=40),
            "covariate_2": rng.normal(size=40),
        },
        index=[str(i) for i in range(40)],
    )
    obs["covariate_copy"] = obs["covariate"]
    return X, obs


def test_pca(edata_blob_small):
    ep.pp.pca(edata_blob_small)


def test_pca_mask_var_defaults_to_highly_variable(edata_blob_small):
    edata_blob_small.var["highly_variable"] = [True] * 5 + [False] * 5

    ep.pp.pca(edata_blob_small, n_comps=2)
    used_features = np.abs(edata_blob_small.varm["PCs"]).sum(axis=1) > 0
    np.testing.assert_array_equal(used_features, edata_blob_small.var["highly_variable"])

    ep.pp.pca(edata_blob_small, n_comps=2, mask_var=None)
    assert (np.abs(edata_blob_small.varm["PCs"]).sum(axis=1) > 0).all()


def test_pca_key_added(edata_blob_small):
    ep.pp.pca(edata_blob_small, n_comps=2, key_added="pca_custom")
    assert edata_blob_small.obsm["pca_custom"].shape == (edata_blob_small.n_obs, 2)
    assert "pca_custom" in edata_blob_small.varm
    assert "X_pca" not in edata_blob_small.obsm


def test_pca_3D_unfolds_timepoints(edata_blob_small):
    X = edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME]
    n_obs, n_vars, n_timepoints = X.shape
    expected = PCA(n_components=5, svd_solver="full").fit(X.reshape(n_obs, -1))

    ep.pp.pca(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME, n_comps=5, svd_solver="full", dtype="float64")

    loadings = edata_blob_small.varm["PCs"]
    assert loadings.shape == (n_vars, n_timepoints, 5)
    signs = np.sign(np.sum(loadings.reshape(-1, 5) * expected.components_.T, axis=0))
    np.testing.assert_allclose(
        loadings * signs, np.moveaxis(expected.components_.reshape(5, n_vars, n_timepoints), 0, -1)
    )
    np.testing.assert_allclose(
        edata_blob_small.obsm["X_pca"] * signs, expected.transform(X.reshape(n_obs, -1)), atol=1e-10
    )
    np.testing.assert_allclose(edata_blob_small.uns["pca"]["variance_ratio"], expected.explained_variance_ratio_)


def test_pca_3D_mask_var(edata_blob_small):
    edata_blob_small.var["highly_variable"] = [True] * 5 + [False] * 5

    ep.pp.pca(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME, n_comps=2, key_added="pca_3D")

    used = np.abs(edata_blob_small.varm["pca_3D"]).sum(axis=2) > 0
    np.testing.assert_array_equal(used, np.repeat(edata_blob_small.var[["highly_variable"]].to_numpy(), 10, axis=1))
    assert edata_blob_small.obsm["pca_3D"].shape == (edata_blob_small.n_obs, 2)
    assert edata_blob_small.uns["pca_3D"]["params"]["mask_var"] == "highly_variable"
    assert edata_blob_small.uns["pca_3D"]["params"]["use_highly_variable"] is True

    ep.pp.pca(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME, n_comps=2, mask_var=np.arange(10) < 3)
    used = np.abs(edata_blob_small.varm["PCs"]).sum(axis=2) > 0
    np.testing.assert_array_equal(used, np.repeat((np.arange(10) < 3)[:, None], 10, axis=1))


def test_pca_3D_single_timepoint_and_arrays(edata_blob_small):
    edata_blob_small.layers["single"] = edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME][:, :, :1]

    ep.pp.pca(edata_blob_small, layer="single", n_comps=2)

    assert edata_blob_small.varm["PCs"].shape == (edata_blob_small.n_vars, 2)
    with pytest.raises(ValueError, match="only supports 2D data"):
        ep.pp.pca(edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME])


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_pca_3D_array_types(array_type, edata_blob_small):
    X = edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME]
    expected = ed.EHRData(X=X)
    ep.pp.pca(expected, n_comps=3)
    edata = ed.EHRData(X=array_type(X))

    with forbid_dask_compute(allowed=1):
        ep.pp.pca(edata, n_comps=3)

    assert isinstance(edata.obsm["X_pca"], array_type.cls)
    np.testing.assert_allclose(
        np.abs(to_dense(edata.obsm["X_pca"], to_cpu_memory=True)), np.abs(expected.obsm["X_pca"]), atol=1e-4
    )
    np.testing.assert_allclose(np.abs(edata.varm["PCs"]), np.abs(expected.varm["PCs"]), atol=1e-4)


def test_pca_3D_neighbors_umap_leiden(edata_blob_small):
    edata = edata_blob_small
    edata.X = edata.layers[DEFAULT_TEM_LAYER_NAME]

    ep.pp.pca(edata, n_comps=5)
    ep.pp.neighbors(edata)
    ep.tl.umap(edata)
    ep.tl.leiden(edata)

    assert edata.uns["neighbors"]["params"]["use_rep"] == "X_pca"
    assert edata.obsm["X_umap"].shape == (edata.n_obs, 2)
    assert edata.obs["leiden"].nunique() > 1


def test_regress_out(edata_blob_small):
    ep.pp.regress_out(edata_blob_small, keys=["cluster"])


def test_regress_out_3D_edata(edata_blob_small):
    ep.pp.regress_out(edata_blob_small, keys=["cluster"], layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.regress_out(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize(
    "keys",
    [
        pytest.param("covariate", id="numeric"),
        pytest.param(["covariate", "covariate_2"], id="numerics"),
        pytest.param(["covariate", "covariate_copy"], id="singular"),
        pytest.param("batch", id="categorical"),
        pytest.param("condition", id="string"),
    ],
)
def test_regress_out_matches_scanpy(batched_data, keys):
    X, obs = batched_data
    X = X.astype(np.float32)
    X[:, 2] = 3.0
    adata = ad.AnnData(X=X.copy(), obs=obs.copy())
    sc.pp.regress_out(adata, keys)
    edata = ed.EHRData(X=X.copy(), obs=obs.copy())

    ep.pp.regress_out(edata, keys)

    np.testing.assert_allclose(edata.X, adata.X, atol=1e-5)


def test_regress_out_multiple_categorical_keys_raises(batched_data):
    X, obs = batched_data
    with pytest.raises(ValueError, match="single categorical key"):
        ep.pp.regress_out(ed.EHRData(X=X, obs=obs), ["batch", "covariate"])


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("keys", ["covariate", "batch"])
def test_regress_out_array_types(array_type, batched_data, keys):
    X, obs = batched_data
    expected = ep.pp.regress_out(ed.EHRData(X=X, obs=obs), keys, copy=True).X
    edata = ed.EHRData(X=array_type(X), obs=obs)

    if array_type.flags & Flags.Sparse:
        with pytest.raises(NotImplementedError, match="sparse"):
            ep.pp.regress_out(edata, keys)
        return

    with forbid_dask_compute():
        result = ep.pp.regress_out(edata, keys, copy=True).X

    assert type(result) is type(edata.X)
    if array_type.flags & Flags.Dask:
        assert type(result._meta) is type(edata.X._meta)
        assert result.chunks == edata.X.chunks
    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, rtol=1e-10)


def test_combat(edata_blob_small):
    ep.pp.combat(edata_blob_small, batch_key="cluster")


@pytest.mark.parametrize("covariates", [None, ["covariate"], ["covariate", "covariate_2"]])
def test_combat_matches_scanpy(batched_data, covariates):
    X, obs = batched_data
    adata = ad.AnnData(X=X.copy(), obs=obs.copy())
    sc.pp.combat(adata, "batch", covariates=covariates)
    edata = ed.EHRData(X=X.copy(), obs=obs.copy())

    ep.pp.combat(edata, batch_key="batch", covariates=covariates)

    np.testing.assert_allclose(edata.X, adata.X, rtol=1e-10)


def test_combat_categorical_covariate(batched_data):
    X, obs = batched_data
    dummies = pd.get_dummies(obs["condition"], drop_first=True, dtype=np.float64)
    adata = ad.AnnData(X=X.copy(), obs=pd.concat([obs, dummies], axis=1))
    sc.pp.combat(adata, "batch", covariates=list(dummies.columns))
    edata = ed.EHRData(X=X.copy(), obs=obs.copy())

    ep.pp.combat(edata, batch_key="batch", covariates=["condition"])

    np.testing.assert_allclose(edata.X, adata.X, rtol=1e-10)


def test_combat_small_batch_raises(batched_data):
    X, obs = batched_data
    obs["batch"] = ["a"] * 39 + ["b"]
    with pytest.raises(ValueError, match="fewer than 2 observations"):
        ep.pp.combat(ed.EHRData(X=X, obs=obs), batch_key="batch")


@pytest.mark.parametrize("covariates", [["batch"], ["covariate", "covariate"]])
def test_combat_invalid_covariates_raise(batched_data, covariates):
    X, obs = batched_data
    with pytest.raises(ValueError, match="Covariates must be unique"):
        ep.pp.combat(ed.EHRData(X=X, obs=obs), batch_key="batch", covariates=covariates)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_combat_array_types(array_type, batched_data):
    X, obs = batched_data
    expected = ep.pp.combat(ed.EHRData(X=X, obs=obs), batch_key="batch", covariates=["covariate"], copy=True).X
    edata = ed.EHRData(X=array_type(X), obs=obs)

    if array_type.flags & Flags.Sparse:
        with pytest.raises(NotImplementedError, match="sparse"):
            ep.pp.combat(edata, batch_key="batch")
        return

    with forbid_dask_compute():
        result = ep.pp.combat(edata, batch_key="batch", covariates=["covariate"], copy=True).X

    assert type(result) is type(edata.X)
    if array_type.flags & Flags.Dask:
        assert type(result._meta) is type(edata.X._meta)
        assert result.chunks == edata.X.chunks
    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, rtol=1e-10)


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_combat_copy(edata_blob_small, layer):
    X_before = edata_blob_small.X.copy()
    layer_before = edata_blob_small.layers["layer_2"].copy()

    corrected = ep.pp.combat(edata_blob_small, batch_key="cluster", layer=layer, copy=True)

    np.testing.assert_array_equal(edata_blob_small.X, X_before)
    np.testing.assert_array_equal(edata_blob_small.layers["layer_2"], layer_before)
    corrected_mtx = corrected.X if layer is None else corrected.layers[layer]
    assert not np.allclose(corrected_mtx, X_before)

    assert ep.pp.combat(edata_blob_small, batch_key="cluster", layer=layer) is None
    mtx = edata_blob_small.X if layer is None else edata_blob_small.layers[layer]
    np.testing.assert_allclose(mtx, corrected_mtx)
    if layer is not None:
        np.testing.assert_array_equal(edata_blob_small.X, X_before)


def test_combat_3D_edata(edata_blob_small):
    ep.pp.combat(edata_blob_small, batch_key="cluster", layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.combat(edata_blob_small, batch_key="cluster", layer=DEFAULT_TEM_LAYER_NAME)


def test_neighbors(edata_blob_small):
    ep.pp.neighbors(edata_blob_small, n_neighbors=5)

    # since use_rep="..." is possible, check edge case where X is None and layers invalid
    edata_blob_small.obsm["X_pca"] = np.random.default_rng(42).random((edata_blob_small.n_obs, 5))
    edata_blob_small.X = None
    ep.pp.neighbors(edata_blob_small, use_rep="X_pca", n_neighbors=5)
