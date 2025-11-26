import ehrdata as ed
import numpy as np
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep


@pytest.fixture
def mixed_data_array(rng: np.random.Generator) -> np.ndarray:
    n_obs = 50

    arr = np.empty((n_obs, 5, 1), dtype=object)
    arr[:, 0, 0] = rng.standard_normal(n_obs).astype(float)
    arr[:, 1, 0] = rng.choice(["A", "B", "C"], n_obs)
    arr[:, 2, 0] = rng.standard_normal(n_obs).astype(float)
    arr[:, 3, 0] = rng.choice(["X", "Y"], n_obs)
    arr[:, 4, 0] = rng.standard_normal(n_obs).astype(float)

    return arr


@pytest.fixture
def pure_quant_array(rng: np.random.Generator) -> np.ndarray:
    data = rng.standard_normal((30, 5))
    return data[:, :, np.newaxis]


@pytest.fixture
def pure_qual_array(rng: np.random.Generator) -> np.ndarray:
    n_obs = 40
    data = np.column_stack(
        [rng.choice(["A", "B", "C"], n_obs), rng.choice(["X", "Y", "Z"], n_obs), rng.choice(["P", "Q"], n_obs)]
    )
    return data[:, :, np.newaxis]


def test_famd_numpy_mixed_data(mixed_data_array: np.ndarray) -> None:
    factor_scores, loadings, metadata = ep.tl.famd(mixed_data_array, n_components=2)

    assert factor_scores.shape == (50, 2)
    assert loadings.shape == (len(metadata["feature_names"]), 2)
    assert metadata["n_components"] == 2
    assert "variance" in metadata
    assert "variance_ratio" in metadata
    assert "quant_mask" in metadata
    assert len(metadata["feature_names"]) > 0
    assert metadata["variance_ratio"].sum() <= 1.0
    assert metadata["variance_ratio"].sum() > 0.0


def test_famd_numpy_pure_quantitative(pure_quant_array: np.ndarray) -> None:
    factor_scores, loadings, metadata = ep.tl.famd(pure_quant_array, n_components=3)

    assert factor_scores.shape == (30, 3)
    assert metadata["n_components"] == 3


def test_famd_numpy_pure_qualitative(pure_qual_array: np.ndarray) -> None:
    factor_scores, loadings, metadata = ep.tl.famd(pure_qual_array, n_components=2)

    assert factor_scores.shape == (40, 2)
    assert metadata["n_components"] == 2


def test_famd_ehrdata_integration(edata_blobs_timeseries_small: ed.EHRData) -> None:
    edata = edata_blobs_timeseries_small
    edata.layers[DEFAULT_TEM_LAYER_NAME] = edata.X[:, :, np.newaxis]

    ep.tl.famd(edata, n_components=2, layer=DEFAULT_TEM_LAYER_NAME)

    assert "X_famd" in edata.obsm
    assert "famd" in edata.uns
    assert "famd_loadings" in edata.varm
    assert edata.obsm["X_famd"].shape == (20, 2)
    assert edata.uns["famd"]["params"]["n_components"] == 2
    assert "variance" in edata.uns["famd"]
    assert "variance_ratio" in edata.uns["famd"]


def test_famd_n_components_exceeds_dimensions(rng: np.random.Generator) -> None:
    arr = rng.standard_normal((10, 5, 1))

    factor_scores, loadings, metadata = ep.tl.famd(arr, n_components=20)

    assert metadata["n_components"] == min(20, 5)
    assert factor_scores.shape[1] == metadata["n_components"]


def test_famd_with_nans(mixed_data_array: np.ndarray) -> None:
    data = mixed_data_array.copy()
    data[0, 0, 0] = np.nan
    data[5, 2, 0] = np.nan

    factor_scores, loadings, metadata = ep.tl.famd(data, n_components=2)

    assert not np.any(np.isnan(factor_scores))
    assert not np.any(np.isnan(loadings))


def test_famd_single_category(rng: np.random.Generator) -> None:
    n_obs = 30
    arr = np.empty((n_obs, 2, 1), dtype=object)
    arr[:, 0, 0] = rng.standard_normal(n_obs).astype(float)
    arr[:, 1, 0] = np.full(n_obs, "A")

    factor_scores, loadings, metadata = ep.tl.famd(arr, n_components=1)

    assert factor_scores.shape[0] == n_obs


def test_famd_variance_explained_monotonic(pure_quant_array: np.ndarray) -> None:
    _, _, metadata = ep.tl.famd(pure_quant_array, n_components=4)

    variance = metadata["variance"]
    assert np.all(variance[:-1] >= variance[1:])


def test_famd_feature_names(mixed_data_array: np.ndarray) -> None:
    _, _, metadata = ep.tl.famd(mixed_data_array, n_components=2)

    assert "feature_names" in metadata
    assert len(metadata["feature_names"]) > 0


def test_famd_loadings_shape(mixed_data_array: np.ndarray) -> None:
    _, loadings, metadata = ep.tl.famd(mixed_data_array, n_components=2)

    assert loadings.shape[0] == len(metadata["feature_names"])
    assert loadings.shape[1] == metadata["n_components"]


def test_famd_reproducibility(pure_quant_array: np.ndarray) -> None:
    scores1, loadings1, _ = ep.tl.famd(pure_quant_array, n_components=2)
    scores2, loadings2, _ = ep.tl.famd(pure_quant_array, n_components=2)

    assert np.allclose(np.abs(scores1), np.abs(scores2))
    assert np.allclose(np.abs(loadings1), np.abs(loadings2))


def test_famd_zero_variance_column(rng: np.random.Generator) -> None:
    arr = np.empty((20, 3, 1), dtype=object)
    arr[:, 0, 0] = rng.standard_normal(20).astype(float)
    arr[:, 1, 0] = np.zeros(20).astype(float)
    arr[:, 2, 0] = rng.standard_normal(20).astype(float)

    factor_scores, loadings, metadata = ep.tl.famd(arr, n_components=2)

    assert not np.any(np.isnan(factor_scores))
    assert not np.any(np.isinf(factor_scores))
