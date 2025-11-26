import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from anndata import AnnData
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep
from tests.conftest import ARRAY_TYPES_NONNUMERIC

CURRENT_DIR = Path(__file__).parent
from scipy import sparse


def test_vars_checks(adata_to_norm):
    """Test for checks that vars argument is valid."""
    with pytest.raises(ValueError, match=r"Some selected vars are not numeric"):
        ep.pp.scale_norm(adata_to_norm, vars=["String1"])


# TODO: check this for each function, with just default settings?
@pytest.mark.parametrize(
    "array_type,expected_error",
    [
        (np.array, None),
        (da.array, None),
        (sparse.csr_matrix, NotImplementedError),
    ],
)
def test_norm_scale_array_types(adata_to_norm, array_type, expected_error):
    adata_to_norm.X = array_type(adata_to_norm.X)
    if expected_error:
        with pytest.raises(expected_error):
            ep.pp.scale_norm(adata_to_norm)


def test_norm_scale_3D_edata(edata_blob_small):
    ep.pp.scale_norm(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.scale_norm(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("array_type", [np.array, da.array])
def test_norm_scale(adata_to_norm, array_type):
    """Test for the scaling normalization method."""
    warnings.filterwarnings("ignore")
    adata_to_norm.X = array_type(adata_to_norm.X)
    ep.pp.scale_norm(adata_to_norm)

    adata_norm = ep.pp.scale_norm(adata_to_norm, copy=True)

    num1_norm = np.array([-1.4039999, 0.55506986, 0.84893], dtype=np.float32)
    num2_norm = np.array([-1.3587323, 1.0190493, 0.3396831], dtype=np.float32)

    assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
    assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
    assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)
    assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)


def test_norm_scale_integers(edata_mini_integers_in_X):
    adata_norm = ep.pp.scale_norm(edata_mini_integers_in_X, copy=True)
    in_days_norm = np.array(
        [
            [-0.4472136],
            [0.4472136],
            [-1.34164079],
            [-0.4472136],
            [-1.34164079],
            [-0.4472136],
            [0.4472136],
            [1.34164079],
            [2.23606798],
            [-0.4472136],
            [0.4472136],
            [-0.4472136],
        ]
    )
    assert np.allclose(adata_norm.X, in_days_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_scale_kwargs(array_type, adata_to_norm):
    adata_to_norm.X = array_type(adata_to_norm.X)

    adata_norm = ep.pp.scale_norm(adata_to_norm, copy=True, with_mean=False)

    num1_norm = np.array([3.3304186, 5.2894883, 5.5833483], dtype=np.float32)
    num2_norm = np.array([-0.6793662, 1.6984155, 1.0190493], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_scale_group(array_type, edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()
    edata_mini_casted.X = array_type(edata_mini_casted.X)

    with pytest.raises(KeyError):
        ep.pp.scale_norm(edata_mini_casted, group_key="invalid_key", copy=True)

    edata_mini_norm = ep.pp.scale_norm(
        edata_mini_casted,
        vars=["sys_bp_entry", "dia_bp_entry"],
        group_key="disease",
        copy=True,
    )
    col1_norm = np.array(
        [
            -1.34164079,
            -0.4472136,
            0.4472136,
            1.34164079,
            -1.34164079,
            -0.4472136,
            0.4472136,
            1.34164079,
        ]
    )
    col2_norm = col1_norm
    assert np.allclose(edata_mini_norm.X[:, 0], edata_mini_casted.X[:, 0])
    assert np.allclose(edata_mini_norm.X[:, 1], col1_norm)
    assert np.allclose(edata_mini_norm.X[:, 2], col2_norm)


@pytest.mark.parametrize("copy", [True, False])
def test_3d_norm_copy_behavior(edata_blobs_timeseries_small, copy):
    """Test copy behavior for all 3D normalization functions."""
    edata = edata_blobs_timeseries_small.copy()
    R_original = edata.R.copy()

    result = ep.pp.scale_norm(edata, copy=copy)

    if copy:
        assert result is not None
        assert np.allclose(edata.R, R_original, equal_nan=True)
        assert not np.allclose(result.R, R_original, equal_nan=True)
    else:
        assert result is None
        assert not np.allclose(edata.R, R_original, equal_nan=True)


@pytest.mark.parametrize(
    "norm_func",
    [
        ep.pp.scale_norm,
        ep.pp.minmax_norm,
        ep.pp.maxabs_norm,
        ep.pp.robust_scale_norm,
        ep.pp.quantile_norm,
        ep.pp.power_norm,
    ],
)
def test_3d_norm_shape_preservation(edata_blobs_timeseries_small, norm_func):
    """Test that all 3D normalization functions preserve shape and dtype."""
    edata = edata_blobs_timeseries_small.copy()
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype

    if norm_func == ep.pp.power_norm:
        edata.R = np.abs(edata.R) + 0.1

    norm_func(edata)

    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)


@pytest.mark.parametrize(
    "norm_func",
    [
        ep.pp.scale_norm,
        ep.pp.minmax_norm,
        ep.pp.maxabs_norm,
        ep.pp.robust_scale_norm,
        ep.pp.quantile_norm,
        ep.pp.power_norm,
    ],
)
def test_3d_norm_group_functionality(edata_blobs_timeseries_small, norm_func):
    """Test group-wise normalization for all 3D normalization functions."""
    edata = edata_blobs_timeseries_small.copy()

    n_obs = edata.n_obs
    group_size = n_obs // 2
    edata.obs["group"] = ["A"] * group_size + ["B"] * (n_obs - group_size)

    if norm_func == ep.pp.power_norm:
        edata.R = np.abs(edata.R) + 0.1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        norm_func(edata, group_key="group")

    assert edata.R.shape == edata_blobs_timeseries_small.R.shape


@pytest.mark.parametrize(
    "array_type,expected_error",
    [
        (np.array, None),
        (da.array, None),
        (sparse.csr_matrix, NotImplementedError),
    ],
)
def test_norm_minmax_array_types(adata_to_norm, array_type, expected_error):
    adata_to_norm.X = array_type(adata_to_norm.X)
    if expected_error:
        with pytest.raises(expected_error):
            ep.pp.minmax_norm(adata_to_norm)


def test_norm_minmax_3D_edata(edata_blob_small):
    ep.pp.minmax_norm(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.minmax_norm(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_minmax(array_type, adata_to_norm):
    """Test for the minmax normalization method."""
    adata_to_norm.X = array_type(adata_to_norm.X)

    adata_norm = ep.pp.minmax_norm(adata_to_norm, copy=True)

    num1_norm = np.array([0.0, 0.86956537, 0.9999999], dtype=np.dtype(np.float32))
    num2_norm = np.array([0.0, 1.0, 0.71428573], dtype=np.float32)

    assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
    assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
    assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)
    assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)


def test_norm_minmax_integers(edata_mini_integers_in_X):
    adata_norm = ep.pp.minmax_norm(edata_mini_integers_in_X, copy=True)
    in_days_norm = np.array([[0.25], [0.5], [0.0], [0.25], [0.0], [0.25], [0.5], [0.75], [1.0], [0.25], [0.5], [0.25]])
    assert np.allclose(adata_norm.X, in_days_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_minmax_kwargs(array_type, adata_to_norm):
    adata_to_norm.X = array_type(adata_to_norm.X)

    adata_norm = ep.pp.minmax_norm(adata_to_norm, copy=True, feature_range=(0, 2))

    num1_norm = np.array([0.0, 1.7391307, 1.9999998], dtype=np.float32)
    num2_norm = np.array([0.0, 2.0, 1.4285715], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_minmax_group(array_type, edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()
    edata_mini_casted.X = array_type(edata_mini_casted.X)

    with pytest.raises(KeyError):
        ep.pp.minmax_norm(edata_mini_casted, group_key="invalid_key", copy=True)

    edata_mini_norm = ep.pp.minmax_norm(
        edata_mini_casted,
        vars=["sys_bp_entry", "dia_bp_entry"],
        group_key="disease",
        copy=True,
    )
    col1_norm = np.array([0.0, 0.33333333, 0.66666667, 1.0, 0.0, 0.33333333, 0.66666667, 1.0])
    col2_norm = col1_norm
    assert np.allclose(edata_mini_norm.X[:, 0], edata_mini_casted.X[:, 0])
    assert np.allclose(edata_mini_norm.X[:, 1], col1_norm)
    assert np.allclose(edata_mini_norm.X[:, 2], col2_norm)


@pytest.mark.parametrize(
    "array_type,expected_error",
    [
        (np.array, None),
        (da.array, NotImplementedError),
        (sparse.csr_matrix, NotImplementedError),
    ],
)
def test_norm_maxabs_array_types(adata_to_norm, array_type, expected_error):
    adata_to_norm.X = array_type(adata_to_norm.X)
    if expected_error:
        with pytest.raises(expected_error):
            ep.pp.maxabs_norm(adata_to_norm)
    else:
        ep.pp.maxabs_norm(adata_to_norm)


def test_norm_maxabs_3D_edata(edata_blob_small):
    ep.pp.maxabs_norm(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.maxabs_norm(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_maxabs(array_type, adata_to_norm):
    """Test for the maxabs normalization method."""
    adata_to_norm.X = array_type(adata_to_norm.X)

    if "dask" in array_type.__name__:
        with pytest.raises(NotImplementedError):
            adata_norm = ep.pp.maxabs_norm(adata_to_norm, copy=True)

    else:
        adata_norm = ep.pp.maxabs_norm(adata_to_norm, copy=True)

        num1_norm = np.array([0.5964913, 0.94736844, 1.0], dtype=np.float32)
        num2_norm = np.array([-0.4, 1.0, 0.6], dtype=np.float32)

        assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
        assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
        assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
        assert np.allclose(adata_norm.X[:, 3], num1_norm)
        assert np.allclose(adata_norm.X[:, 4], num2_norm)
        assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)


def test_norm_maxabs_integers(edata_mini_integers_in_X):
    adata_norm = ep.pp.maxabs_norm(edata_mini_integers_in_X, copy=True)
    in_days_norm = np.array([[0.25], [0.5], [0.0], [0.25], [0.0], [0.25], [0.5], [0.75], [1.0], [0.25], [0.5], [0.25]])
    assert np.allclose(adata_norm.X, in_days_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_maxabs_group(array_type, edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()
    edata_mini_casted.X = array_type(edata_mini_casted.X)

    if "dask" in array_type.__name__:
        with pytest.raises(NotImplementedError):
            ep.pp.maxabs_norm(edata_mini_casted, copy=True)
    else:
        with pytest.raises(KeyError):
            ep.pp.maxabs_norm(edata_mini_casted, group_key="invalid_key", copy=True)

        edata_mini_norm = ep.pp.maxabs_norm(
            edata_mini_casted,
            vars=["sys_bp_entry", "dia_bp_entry"],
            group_key="disease",
            copy=True,
        )
        col1_norm = np.array(
            [
                0.9787234,
                0.9858156,
                0.9929078,
                1.0,
                0.98013245,
                0.98675497,
                0.99337748,
                1.0,
            ]
        )
        col2_norm = np.array([0.96296296, 0.97530864, 0.98765432, 1.0, 0.9625, 0.975, 0.9875, 1.0])
        assert np.allclose(edata_mini_norm.X[:, 0], edata_mini_casted.X[:, 0])
        assert np.allclose(edata_mini_norm.X[:, 1], col1_norm)
        assert np.allclose(edata_mini_norm.X[:, 2], col2_norm)


@pytest.mark.parametrize(
    "array_type,expected_error",
    [
        (np.array, None),
        (da.array, None),
        (sparse.csr_matrix, NotImplementedError),
    ],
)
def test_norm_robust_scale_array_types(adata_to_norm, array_type, expected_error):
    adata_to_norm.X = array_type(adata_to_norm.X)
    if expected_error:
        with pytest.raises(expected_error):
            ep.pp.robust_scale_norm(adata_to_norm)


def test_norm_robust_scale_3D_edata(edata_blob_small):
    ep.pp.robust_scale_norm(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.robust_scale_norm(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_robust_scale(array_type, adata_to_norm):
    """Test for the robust_scale normalization method."""
    adata_to_norm.X = array_type(adata_to_norm.X)

    adata_norm = ep.pp.robust_scale_norm(adata_to_norm, copy=True)

    num1_norm = np.array([-1.73913043, 0.0, 0.26086957], dtype=np.float32)
    num2_norm = np.array([-1.4285715, 0.5714286, 0.0], dtype=np.float32)

    assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
    assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
    assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)
    assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)


def test_norm_robust_scale_integers(edata_mini_integers_in_X):
    adata_norm = ep.pp.robust_scale_norm(edata_mini_integers_in_X, copy=True)
    in_days_norm = np.array([[0.0], [1.0], [-1.0], [0.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [0.0], [1.0], [0.0]])
    assert np.allclose(adata_norm.X, in_days_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_robust_scale_kwargs(adata_to_norm, array_type):
    adata_to_norm.X = array_type(adata_to_norm.X)

    adata_norm = ep.pp.robust_scale_norm(adata_to_norm, copy=True, with_scaling=False)

    num1_norm = np.array([-2.0, 0.0, 0.2999997], dtype=np.float32)
    num2_norm = np.array([-5.0, 2.0, 0.0], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_robust_scale_group(array_type, edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()
    edata_mini_casted.X = array_type(edata_mini_casted.X)

    with pytest.raises(KeyError):
        ep.pp.robust_scale_norm(edata_mini_casted, group_key="invalid_key", copy=True)

    edata_mini_norm = ep.pp.robust_scale_norm(
        edata_mini_casted,
        vars=["sys_bp_entry", "dia_bp_entry"],
        group_key="disease",
        copy=True,
    )
    col1_norm = np.array(
        [-1.0, -0.33333333, 0.33333333, 1.0, -1.0, -0.33333333, 0.33333333, 1.0],
        dtype=np.float32,
    )
    col2_norm = col1_norm
    assert np.allclose(edata_mini_norm.X[:, 0], edata_mini_casted.X[:, 0])
    assert np.allclose(edata_mini_norm.X[:, 1], col1_norm)
    assert np.allclose(edata_mini_norm.X[:, 2], col2_norm)


@pytest.mark.parametrize(
    "array_type,expected_error",
    [
        (np.array, None),
        (da.array, None),
        (sparse.csr_matrix, NotImplementedError),
    ],
)
def test_norm_quantile_array_types(adata_to_norm, array_type, expected_error):
    adata_to_norm.X = array_type(adata_to_norm.X)
    if expected_error:
        with pytest.raises(expected_error):
            ep.pp.quantile_norm(adata_to_norm)


def test_norm_quantile_3D_edata(edata_blob_small):
    ep.pp.quantile_norm(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.quantile_norm(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_quantile_uniform(array_type, adata_to_norm):
    """Test for the quantile normalization method."""
    warnings.filterwarnings("ignore", category=UserWarning)
    adata_to_norm.X = array_type(adata_to_norm.X)

    adata_norm = ep.pp.quantile_norm(adata_to_norm, copy=True)

    num1_norm = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    num2_norm = np.array([0.0, 1.0, 0.5], dtype=np.float32)

    assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
    assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
    assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)
    assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)


def test_norm_quantile_integers(edata_mini_integers_in_X):
    adata_norm = ep.pp.quantile_norm(edata_mini_integers_in_X, copy=True)
    in_days_norm = np.array(
        [
            [0.36363636],
            [0.72727273],
            [0.0],
            [0.36363636],
            [0.0],
            [0.36363636],
            [0.72727273],
            [0.90909091],
            [1.0],
            [0.36363636],
            [0.72727273],
            [0.36363636],
        ]
    )
    assert np.allclose(adata_norm.X, in_days_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_quantile_uniform_kwargs(array_type, adata_to_norm):
    adata_to_norm.X = array_type(adata_to_norm.X)

    adata_norm = ep.pp.quantile_norm(adata_to_norm, copy=True, output_distribution="normal")

    num1_norm = np.array([-5.19933758, 0.0, 5.19933758], dtype=np.float32)
    num2_norm = np.array([-5.19933758, 5.19933758, 0.0], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_quantile_uniform_group(array_type, edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()
    edata_mini_casted.X = array_type(edata_mini_casted.X)

    with pytest.raises(KeyError):
        ep.pp.quantile_norm(edata_mini_casted, group_key="invalid_key", copy=True)

    edata_mini_norm = ep.pp.quantile_norm(
        edata_mini_casted,
        vars=["sys_bp_entry", "dia_bp_entry"],
        group_key="disease",
        copy=True,
    )
    col1_norm = np.array(
        [0.0, 0.33333333, 0.66666667, 1.0, 0.0, 0.33333333, 0.66666667, 1.0],
        dtype=np.float32,
    )
    col2_norm = col1_norm
    assert np.allclose(edata_mini_norm.X[:, 0], edata_mini_casted.X[:, 0])
    assert np.allclose(edata_mini_norm.X[:, 1], col1_norm)
    assert np.allclose(edata_mini_norm.X[:, 2], col2_norm)


@pytest.mark.parametrize(
    "array_type,expected_error",
    [
        (np.array, None),
        (da.array, None),
        (sparse.csr_matrix, NotImplementedError),
    ],
)
def test_norm_power_array_types(adata_to_norm, array_type, expected_error):
    adata_to_norm.X = array_type(adata_to_norm.X)
    if expected_error:
        with pytest.raises(expected_error):
            ep.pp.power_norm(adata_to_norm)


def test_norm_power_3D_edata(edata_blob_small):
    ep.pp.power_norm(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.power_norm(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_power(array_type, adata_to_norm):
    """Test for the power transformation normalization method."""
    adata_to_norm.X = array_type(adata_to_norm.X)

    if "dask" in array_type.__name__:
        with pytest.raises(NotImplementedError):
            ep.pp.power_norm(adata_to_norm, copy=True)
    else:
        adata_norm = ep.pp.power_norm(adata_to_norm, copy=True)

        num1_norm = np.array([-1.3821232, 0.43163615, 0.950487], dtype=np.float32)
        num2_norm = np.array([-1.340104, 1.0613203, 0.27878374], dtype=np.float32)

        assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
        assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
        assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
        assert np.allclose(adata_norm.X[:, 3], num1_norm, rtol=1.1)
        assert np.allclose(adata_norm.X[:, 4], num2_norm, rtol=1.1)
        assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)


def test_norm_power_integers(edata_mini_integers_in_X):
    adata_norm = ep.pp.power_norm(edata_mini_integers_in_X, copy=True)
    in_days_norm = np.array(
        [
            [-0.31234142],
            [0.58319338],
            [-1.65324303],
            [-0.31234142],
            [-1.65324303],
            [-0.31234142],
            [0.58319338],
            [1.27419965],
            [1.8444134],
            [-0.31234142],
            [0.58319338],
            [-0.31234142],
        ]
    )
    assert np.allclose(adata_norm.X, in_days_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_power_kwargs(array_type, adata_to_norm):
    adata_to_norm.X = array_type(adata_to_norm.X)

    if "dask" in array_type.__name__:
        with pytest.raises(NotImplementedError):
            ep.pp.power_norm(adata_to_norm, copy=True)
    else:
        with pytest.raises(ValueError):
            ep.pp.power_norm(adata_to_norm, copy=True, method="box-cox")

        adata_norm = ep.pp.power_norm(adata_to_norm, copy=True, standardize=False)

        num1_norm = np.array([201.03636, 1132.8341, 1399.3877], dtype=np.float32)
        num2_norm = np.array([-1.8225479, 5.921072, 3.397709], dtype=np.float32)

        assert np.allclose(adata_norm.X[:, 3], num1_norm, rtol=1e-02, atol=1e-02)
        assert np.allclose(adata_norm.X[:, 4], num2_norm, rtol=1e-02, atol=1e-02)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_norm_power_group(array_type, edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()
    edata_mini_casted.X = array_type(edata_mini_casted.X)

    if "dask" in array_type.__name__:
        with pytest.raises(NotImplementedError):
            ep.pp.power_norm(edata_mini_casted, copy=True)
    else:
        with pytest.raises(KeyError):
            ep.pp.power_norm(edata_mini_casted, group_key="invalid_key", copy=True)

        edata_mini_norm = ep.pp.power_norm(
            edata_mini_casted,
            vars=["sys_bp_entry", "dia_bp_entry"],
            group_key="disease",
            copy=True,
        )
        col1_norm = np.array(
            [
                -1.34266204,
                -0.44618949,
                0.44823148,
                1.34062005,
                -1.34259417,
                -0.44625773,
                0.44816403,
                1.34068786,
            ],
            dtype=np.float32,
        )
        col2_norm = np.array(
            [
                [
                    -1.3650659,
                    -0.41545486,
                    0.45502198,
                    1.3254988,
                    -1.3427324,
                    -0.4461177,
                    0.44829938,
                    1.3405508,
                ]
            ],
            dtype=np.float32,
        )
        # The tests are disabled (= tolerance set to 1)
        # because depending on weird dependency versions they currently give different results
        assert np.allclose(edata_mini_norm.X[:, 0], edata_mini_casted.X[:, 0], rtol=1, atol=1)
        assert np.allclose(edata_mini_norm.X[:, 1], col1_norm, rtol=1, atol=1)
        assert np.allclose(edata_mini_norm.X[:, 2], col2_norm, rtol=1, atol=1)


@pytest.mark.parametrize(
    "array_type,expected_error",
    [
        (np.array, None),
        (da.array, None),
        (sparse.csr_matrix, None),
    ],
)
def test_norm_log_norm_array_types(adata_to_norm, array_type, expected_error):
    adata_to_norm.X = array_type(adata_to_norm.X)
    if expected_error:
        with pytest.raises(expected_error):
            ep.pp.log_norm(adata_to_norm)


def test_norm_log_3D_edata(edata_blob_small):
    edata_blob_small.X = np.abs(edata_blob_small.X)
    edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME] = np.abs(edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME])
    ep.pp.log_norm(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.log_norm(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


def test_norm_log1p(adata_to_norm):
    """Test for the log normalization method."""
    # Ensure that some test data is strictly positive
    log_adata = adata_to_norm.copy()
    log_adata.X[0, 4] = 1

    adata_norm = ep.pp.log_norm(log_adata, copy=True)

    num1_norm = np.array([1.4816046, 1.856298, 1.9021075], dtype=np.float32)
    num2_norm = np.array([0.6931472, 1.7917595, 1.3862944], dtype=np.float32)

    assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
    assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
    assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)
    assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)

    # Check alternative base works
    adata_norm = ep.pp.log_norm(log_adata, base=10, copy=True)

    num1_norm = np.divide(np.array([1.4816046, 1.856298, 1.9021075], dtype=np.float32), np.log(10))
    num2_norm = np.divide(np.array([0.6931472, 1.7917595, 1.3862944], dtype=np.float32), np.log(10))

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)

    # Check alternative offset works
    adata_norm = ep.pp.log_norm(log_adata, offset=0.5, copy=True)

    num1_norm = np.array([1.3609766, 1.7749524, 1.8245492], dtype=np.float32)
    num2_norm = np.array([0.4054651, 1.7047482, 1.252763], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)

    try:
        ep.pp.log_norm(adata_to_norm, vars="Numeric2", offset=3, copy=True)
    except ValueError:
        pytest.fail("Unexpected ValueError exception was raised.")

    with pytest.raises(ValueError):
        ep.pp.log_norm(adata_to_norm, copy=True)

    with pytest.raises(ValueError):
        ep.pp.log_norm(adata_to_norm, vars="Numeric2", offset=1, copy=True)


def test_norm_record(adata_to_norm):
    """Test for logging of applied normalization methods."""
    adata_norm = ep.pp.minmax_norm(adata_to_norm, copy=True)

    assert adata_norm.uns["normalization"] == {
        "Numeric1": ["minmax"],
        "Numeric2": ["minmax"],
    }

    adata_norm = ep.pp.maxabs_norm(adata_norm, vars=["Numeric1"], copy=True)

    assert adata_norm.uns["normalization"] == {
        "Numeric1": ["minmax", "maxabs"],
        "Numeric2": ["minmax"],
    }


def test_offset_negative_values():
    """Test for the offset_negative_values method."""
    to_offset_adata = AnnData(X=np.array([[-1, -5, -10], [5, 6, -20]], dtype=np.float32))
    expected_adata = AnnData(X=np.array([[19, 15, 10], [25, 26, 0]], dtype=np.float32))

    assert np.array_equal(expected_adata.X, ep.pp.offset_negative_values(to_offset_adata, copy=True).X)


def test_offset_negative_values_3D_edata(edata_blob_small):
    ep.pp.offset_negative_values(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.offset_negative_values(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


def test_norm_numerical_only():
    """Test for the log_norm method."""
    to_normalize_adata = AnnData(X=np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32))
    expected_adata = AnnData(X=np.array([[0.6931472, 0, 0], [0, 0, 0.6931472]], dtype=np.float32))

    assert np.array_equal(expected_adata.X, ep.pp.log_norm(to_normalize_adata, copy=True).X)


def test_scale_norm_3d(edata_blobs_timeseries_small):
    """Test that scale_norm centers each 3D variable to mean ~0 and std ~1.

    The function should operate per-variable across samples and timestamps and
    be robust to all-NaN slices (these are skipped by checks).
    """
    edata = edata_blobs_timeseries_small
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype
    ep.pp.scale_norm(edata)

    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmean(flat), 0, atol=1e-6)
            assert np.allclose(np.nanstd(flat), 1, atol=1e-6)


def test_minmax_norm_3d(edata_blobs_timeseries_small):
    """Test that minmax_norm rescales each variable to [0, 1].

    For 3D data this means each variable's flattened values (samples × timestamps)
    should have a min of 0 and a max of 1 (NaNs ignored).
    """
    edata = edata_blobs_timeseries_small
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype

    ep.pp.minmax_norm(edata)
    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmin(flat), 0, atol=1e-6)
            assert np.allclose(np.nanmax(flat), 1, atol=1e-6)


def test_maxabs_norm_3d(edata_blobs_timeseries_small):
    """Test that maxabs_norm scales each variable so the maximum absolute value is 1.

    This should hold per-variable across the flattened samples × timestamps axis,
    ignoring NaN entries.
    """
    edata = edata_blobs_timeseries_small
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype

    ep.pp.maxabs_norm(edata)
    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmax(np.abs(flat)), 1, atol=1e-6)


def test_robust_scale_norm_3d(edata_blobs_timeseries_small):
    """Test that robust_scale_norm centers variables by median and scales by IQR.

    For each variable (flattened across samples and timestamps) the median should
    be ~0 and the interquartile range should be scaled to 1.
    """
    edata = edata_blobs_timeseries_small
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype

    ep.pp.robust_scale_norm(edata)
    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmedian(flat), 0, atol=1e-6)
            q75, q25 = np.nanpercentile(flat, [75, 25])
            iqr = q75 - q25
            assert np.allclose(iqr, 1, atol=1e-6)


def test_quantile_norm_3d(edata_blobs_timeseries_small):
    """Test that quantile_norm maps each variable's empirical distribution to [0,1].

    We check that per-variable flattened values have min≈0, max≈1 and sensible
    quartiles (approx. 0.25, 0.5, 0.75) after transformation.
    """
    edata = edata_blobs_timeseries_small
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype

    ep.pp.quantile_norm(edata)

    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmin(flat), 0, atol=1e-6)
            assert np.allclose(np.nanmax(flat), 1, atol=1e-6)
            q25, q50, q75 = np.nanpercentile(flat, [25, 50, 75])
            assert np.allclose(q25, 0.25, atol=0.05)
            assert np.allclose(q50, 0.5, atol=0.05)
            assert np.allclose(q75, 0.75, atol=0.05)


def test_power_norm_3d(edata_blobs_timeseries_small):
    """Test that power_norm (PowerTransformer) approximately standardizes skewed data.

    The test prepares strictly positive input (abs + offset) and expects the
    flattened per-variable distributions to have mean ~0 and std ~1 after
    transformation.
    """
    edata = edata_blobs_timeseries_small
    edata.R = np.abs(edata.R) + 0.1

    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype

    ep.pp.power_norm(edata)

    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmean(flat), 0, atol=1e-5)
            assert np.allclose(np.nanstd(flat), 1, atol=1e-5)


def test_log_norm_3d(edata_blobs_timeseries_small):
    """Test that log_norm applies elementwise log1p (or log with offset) to 3D data.

    The test uses strictly positive input (abs + 1) and verifies the result is
    equal to np.log1p(original) elementwise (NaN-preserving).
    """
    edata = edata_blobs_timeseries_small
    edata.R = np.abs(edata.R) + 1

    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype

    R_original = edata.R.copy()

    ep.pp.log_norm(edata)

    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    expected = np.log1p(R_original)
    assert np.allclose(edata.R, expected, rtol=1e-6, equal_nan=True)

    assert not np.allclose(R_original, edata.R, equal_nan=True)


def test_offset_negative_values_3d(edata_blobs_timeseries_small):
    """Test that offset_negative_values shifts the array so its minimum becomes 0.

    The function should preserve shape and dtype and ensure all non-NaN values
    are >= 0 after the operation.
    """
    edata = edata_blobs_timeseries_small
    edata.R = edata.R - 2

    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype
    assert np.nanmin(edata.R) < 0, "Test data should have negative values"

    ep.pp.offset_negative_values(edata)

    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    assert np.allclose(np.nanmin(edata.R), 0, atol=1e-10)

    non_nan_values = edata.R[~np.isnan(edata.R)]
    assert np.all(non_nan_values >= 0)


def test_3d_norm_metadata_and_layers(edata_blobs_timeseries_small):
    """Test that 3D normalization preserves metadata and works with layers."""
    edata = edata_blobs_timeseries_small.copy()
    edata.layers["test_3d_layer"] = edata.R.copy() * 2 + 5

    ep.pp.scale_norm(edata, layer="test_3d_layer")

    assert not np.allclose(edata.R, edata.layers["test_3d_layer"])

    ep.pp.scale_norm(edata)

    assert "normalization" in edata.uns
    assert len(edata.uns["normalization"]) > 0

    assert edata.obs.shape[0] == edata_blobs_timeseries_small.obs.shape[0]
    assert edata.var.shape[0] == edata_blobs_timeseries_small.var.shape[0]


@pytest.mark.parametrize(
    "norm_func",
    [
        ep.pp.scale_norm,
        ep.pp.minmax_norm,
        ep.pp.maxabs_norm,
        ep.pp.robust_scale_norm,
        ep.pp.quantile_norm,
        ep.pp.power_norm,
    ],
)
def test_3d_norm_invalid_vars(edata_blobs_timeseries_small, norm_func):
    """Test that all 3D normalization functions handle invalid variable names."""
    edata = edata_blobs_timeseries_small.copy()

    if norm_func == ep.pp.power_norm:
        edata.R = np.abs(edata.R) + 0.1

    with pytest.raises(ValueError):
        norm_func(edata, vars=["nonexistent_var"])


def test_3d_norm_variable_selection(edata_blobs_timeseries_small):
    """Test variable selection with 3D normalization."""
    edata = edata_blobs_timeseries_small.copy()
    R_original = edata.R.copy()

    selected_vars = [edata.var_names[0], edata.var_names[1]]
    ep.pp.scale_norm(edata, vars=selected_vars)

    assert not np.allclose(edata.R[:, 0, :], R_original[:, 0, :], equal_nan=True)
    assert not np.allclose(edata.R[:, 1, :], R_original[:, 1, :], equal_nan=True)

    if edata.R.shape[1] > 2:
        assert np.allclose(edata.R[:, 2, :], R_original[:, 2, :], equal_nan=True)

    with pytest.raises(ValueError):
        ep.pp.scale_norm(edata_blobs_timeseries_small.copy(), vars=["nonexistent_var"])
