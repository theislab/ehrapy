from __future__ import annotations

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute


@pytest.fixture
def edata_3d() -> ed.EHRData:
    """Small EHRData with a non-negative 3D layer suitable for NCP."""
    rng = np.random.default_rng(0)
    n_obs, n_vars, n_time = 30, 10, 8
    # Use absolute values so all entries are non-negative
    layer = np.abs(rng.standard_normal((n_obs, n_vars, n_time)))
    var_names = [f"var_{i}" for i in range(n_vars)]
    return ed.EHRData(
        shape=(n_obs, n_vars),
        layers={DEFAULT_TEM_LAYER_NAME: layer},
        var=pd.DataFrame(index=var_names),
    )


def test_ncp_stores_factors(edata_3d: ed.EHRData) -> None:
    rank = 2
    ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=rank, n_iter_max=20)

    assert "X_ncp" in edata_3d.obsm
    assert "ncp_loadings" in edata_3d.varm
    assert "ncp" in edata_3d.uns

    A = edata_3d.obsm["X_ncp"]
    B = edata_3d.varm["ncp_loadings"]
    C = edata_3d.uns["ncp"]["temporal_factors"]

    assert A.shape == (30, rank)
    assert B.shape == (10, rank)
    assert C.shape == (8, rank)


def test_ncp_factors_non_negative(edata_3d: ed.EHRData) -> None:
    ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=2, n_iter_max=20)

    assert np.all(edata_3d.obsm["X_ncp"] >= 0)
    assert np.all(edata_3d.varm["ncp_loadings"] >= 0)
    assert np.all(edata_3d.uns["ncp"]["temporal_factors"] >= 0)


def test_ncp_uns_params(edata_3d: ed.EHRData) -> None:
    ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=3, n_iter_max=15)

    params = edata_3d.uns["ncp"]["params"]
    assert params["rank"] == 3
    assert params["layer"] == DEFAULT_TEM_LAYER_NAME
    assert params["n_iter_max"] == 15


def test_ncp_sigmoid_transform(edata_3d: ed.EHRData) -> None:
    """Sigmoid transform: factors must still be valid and layer is unchanged."""
    layer_before = edata_3d.layers[DEFAULT_TEM_LAYER_NAME].copy()
    ep.tl.ncp(
        edata_3d,
        layer=DEFAULT_TEM_LAYER_NAME,
        rank=2,
        n_iter_max=10,
        sigmoid_transform=True,
    )
    # original layer must not be mutated
    np.testing.assert_array_equal(edata_3d.layers[DEFAULT_TEM_LAYER_NAME], layer_before)
    # factors must be present and non-negative
    assert edata_3d.obsm["X_ncp"].shape[1] == 2
    assert np.all(edata_3d.obsm["X_ncp"] >= 0)


def test_ncp_key_added(edata_3d: ed.EHRData) -> None:
    ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=2, key_added="mykey", n_iter_max=10)

    assert "X_mykey" in edata_3d.obsm
    assert "mykey_loadings" in edata_3d.varm
    assert "mykey" in edata_3d.uns
    # default key must not have been written
    assert "X_ncp" not in edata_3d.obsm


def test_ncp_copy(edata_3d: ed.EHRData) -> None:
    original_obsm_keys = set(edata_3d.obsm.keys())
    result = ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=2, n_iter_max=10, copy=True)

    assert result is not None
    assert "X_ncp" in result.obsm
    # original must be untouched
    assert set(edata_3d.obsm.keys()) == original_obsm_keys


def test_ncp_inplace_returns_none(edata_3d: ed.EHRData) -> None:
    result = ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=2, n_iter_max=10)
    assert result is None


def test_ncp_missing_layer_raises(edata_3d: ed.EHRData) -> None:
    with pytest.raises(KeyError, match="not found"):
        ep.tl.ncp(edata_3d, layer="nonexistent_layer", rank=2)


def test_ncp_2d_layer_raises() -> None:
    rng = np.random.default_rng(0)
    edata = ed.EHRData(np.abs(rng.standard_normal((20, 5))), layers={"flat": np.abs(rng.standard_normal((20, 5)))})
    with pytest.raises(ValueError, match="3D"):
        ep.tl.ncp(edata, layer="flat", rank=2)


def test_ncp_nan_raises(edata_3d: ed.EHRData) -> None:
    edata_3d.layers[DEFAULT_TEM_LAYER_NAME][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Impute"):
        ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=2)


def test_ncp_negative_raises(edata_3d: ed.EHRData) -> None:
    edata_3d.layers[DEFAULT_TEM_LAYER_NAME][:, 0, :] *= -1
    with pytest.raises(ValueError, match="non-negative"):
        ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=2)


def test_ncp_negative_with_sigmoid_transform(edata_3d: ed.EHRData) -> None:
    edata_3d.layers[DEFAULT_TEM_LAYER_NAME][:, 0, :] *= -1
    ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=2, n_iter_max=10, sigmoid_transform=True)

    assert np.all(edata_3d.varm["ncp_loadings"] >= 0)


def test_ncp_reproducibility(edata_3d: ed.EHRData) -> None:
    ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=2, n_iter_max=30, random_state=42)
    A1 = edata_3d.obsm["X_ncp"].copy()

    # overwrite with same seed
    ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=2, n_iter_max=30, random_state=42)
    A2 = edata_3d.obsm["X_ncp"]

    np.testing.assert_allclose(A1, A2)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_ncp_array_types(array_type, edata_3d: ed.EHRData) -> None:
    tensor = edata_3d.layers[DEFAULT_TEM_LAYER_NAME]
    if array_type.flags & Flags.Sparse:
        edata_3d.layers["flat"] = array_type(tensor[:, :, 0])
        with pytest.raises(ValueError, match="3D"):
            ep.tl.ncp(edata_3d, layer="flat")
        return

    expected = ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=2, n_iter_max=20, copy=True)
    edata_3d.layers[DEFAULT_TEM_LAYER_NAME] = array_type(tensor)

    with forbid_dask_compute(allowed=1):
        result = ep.tl.ncp(edata_3d, layer=DEFAULT_TEM_LAYER_NAME, rank=2, n_iter_max=20, copy=True)

    assert type(result.layers[DEFAULT_TEM_LAYER_NAME]) is type(edata_3d.layers[DEFAULT_TEM_LAYER_NAME])
    assert isinstance(result.obsm["X_ncp"], np.ndarray)
    np.testing.assert_allclose(result.obsm["X_ncp"], expected.obsm["X_ncp"])
    np.testing.assert_allclose(result.varm["ncp_loadings"], expected.varm["ncp_loadings"])
    np.testing.assert_allclose(result.uns["ncp"]["temporal_factors"], expected.uns["ncp"]["temporal_factors"])
