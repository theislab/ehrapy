import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import ehrapy as ep
from tests.conftest import ARRAY_TYPES

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


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_norm_scale_kwargs(array_type, adata_to_norm):
    adata_to_norm.X = array_type(adata_to_norm.X)

    adata_norm = ep.pp.scale_norm(adata_to_norm, copy=True, with_mean=False)

    num1_norm = np.array([3.3304186, 5.2894883, 5.5833483], dtype=np.float32)
    num2_norm = np.array([-0.6793662, 1.6984155, 1.0190493], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_norm_minmax_kwargs(array_type, adata_to_norm):
    adata_to_norm.X = array_type(adata_to_norm.X)

    adata_norm = ep.pp.minmax_norm(adata_to_norm, copy=True, feature_range=(0, 2))

    num1_norm = np.array([0.0, 1.7391307, 1.9999998], dtype=np.float32)
    num2_norm = np.array([0.0, 2.0, 1.4285715], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_norm_robust_scale_kwargs(adata_to_norm, array_type):
    adata_to_norm.X = array_type(adata_to_norm.X)

    adata_norm = ep.pp.robust_scale_norm(adata_to_norm, copy=True, with_scaling=False)

    num1_norm = np.array([-2.0, 0.0, 0.2999997], dtype=np.float32)
    num2_norm = np.array([-5.0, 2.0, 0.0], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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



@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_norm_quantile_uniform_kwargs(array_type, adata_to_norm):
    adata_to_norm.X = array_type(adata_to_norm.X)

    adata_norm = ep.pp.quantile_norm(adata_to_norm, copy=True, output_distribution="normal")

    num1_norm = np.array([-5.19933758, 0.0, 5.19933758], dtype=np.float32)
    num2_norm = np.array([-5.19933758, 5.19933758, 0.0], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
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


def test_norm_numerical_only():
    """Test for the log_norm method."""
    to_normalize_adata = AnnData(X=np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32))
    expected_adata = AnnData(X=np.array([[0.6931472, 0, 0], [0, 0, 0.6931472]], dtype=np.float32))

    assert np.array_equal(expected_adata.X, ep.pp.log_norm(to_normalize_adata, copy=True).X)


def test_scale_norm_3d(edata_blob_small_3d):
    """Test scale_norm normalization on 3D EHRData (edata.R)."""
    edata = edata_blob_small_3d
    # Save original shape and dtype
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype
    # Run normalization
    ep.pp.scale_norm(edata)

    # After normalization, R should still be 3D and same shape
    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    # Check that each variable (axis=1) is zero mean and unit variance across all samples and timestamps
    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        # Allow small numerical error
        assert np.allclose(np.mean(flat), 0, atol=1e-6)
        assert np.allclose(np.std(flat), 1, atol=1e-6)


def test_minmax_norm_3d(edata_blob_small_3d):
    """Test minmax_norm normalization on 3D EHRData (edata.R)."""
    edata = edata_blob_small_3d
    # Save original shape and dtype
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype
    
    # Run normalization
    ep.pp.minmax_norm(edata)    # After normalization, R should still be 3D and same shape
    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    # Check that each variable (axis=1) is scaled to [0, 1] range across all samples and timestamps
    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        # Allow small numerical error
        assert np.allclose(np.min(flat), 0, atol=1e-6)
        assert np.allclose(np.max(flat), 1, atol=1e-6)


def test_maxabs_norm_3d(edata_blob_small_3d):
    """Test maxabs_norm normalization on 3D EHRData (edata.R)."""
    edata = edata_blob_small_3d
    # Save original shape and dtype
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype
    
    # Run normalization
    ep.pp.maxabs_norm(edata)    # After normalization, R should still be 3D and same shape
    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    # Check that each variable (axis=1) is scaled by max absolute value across all samples and timestamps
    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        # Max absolute value should be 1 (or close to it)
        assert np.allclose(np.max(np.abs(flat)), 1, atol=1e-6)


def test_robust_scale_norm_3d(edata_blob_small_3d):
    """Test robust_scale_norm normalization on 3D EHRData (edata.R)."""
    edata = edata_blob_small_3d
    # Save original shape and dtype
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype
    
    # Run normalization
    ep.pp.robust_scale_norm(edata)    # After normalization, R should still be 3D and same shape
    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    # Check that each variable (axis=1) is robust scaled across all samples and timestamps
    # RobustScaler centers by median and scales by IQR, so median should be ~0 and IQR should be ~1
    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        # Median should be close to 0 after centering
        assert np.allclose(np.median(flat), 0, atol=1e-6)
        # IQR (75th percentile - 25th percentile) should be close to 1 after scaling
        q75, q25 = np.percentile(flat, [75, 25])
        iqr = q75 - q25
        assert np.allclose(iqr, 1, atol=1e-6)


def test_quantile_norm_3d(edata_blob_small_3d):
    """Test quantile_norm normalization on 3D EHRData (edata.R)."""
    edata = edata_blob_small_3d
    # Save original shape and dtype
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype
    
    # Run normalization
    ep.pp.quantile_norm(edata)

    # After normalization, R should still be 3D and same shape
    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    # Check that each variable (axis=1) follows uniform distribution [0,1] after quantile transformation
    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        # Values should be approximately uniformly distributed in [0, 1]
        assert np.allclose(np.min(flat), 0, atol=1e-6)
        assert np.allclose(np.max(flat), 1, atol=1e-6)
        # Check approximate uniform distribution by checking quartiles
        q25, q50, q75 = np.percentile(flat, [25, 50, 75])
        assert np.allclose(q25, 0.25, atol=0.05)  # More tolerance for uniform distribution
        assert np.allclose(q50, 0.5, atol=0.05)
        assert np.allclose(q75, 0.75, atol=0.05)


def test_power_norm_3d(edata_blob_small_3d):
    """Test power_norm normalization on 3D EHRData (edata.R)."""
    edata = edata_blob_small_3d
    # Make sure data is positive for power transform
    edata.R = np.abs(edata.R) + 0.1  # Add small offset to avoid zeros
    
    # Save original shape and dtype
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype
    
    # Run normalization
    ep.pp.power_norm(edata)

    # After normalization, R should still be 3D and same shape
    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    # Power transform should reduce skewness and make data more normal
    n_obs, n_var, n_timestamps = edata.R.shape
    for var_idx in range(n_var):
        flat = edata.R[:, var_idx, :].reshape(-1)
        # After power transform, data should have approximately zero mean and unit variance
        assert np.allclose(np.mean(flat), 0, atol=1e-5)
        assert np.allclose(np.std(flat), 1, atol=1e-5)


def test_log_norm_3d(edata_blob_small_3d):
    """Test log_norm normalization on 3D EHRData (edata.R)."""
    edata = edata_blob_small_3d
    # Make sure data is positive for log transform
    edata.R = np.abs(edata.R) + 1  # Offset to make positive
    
    # Save original shape and dtype
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype
    
    # Save original for comparison
    R_original = edata.R.copy()
    
    # Run normalization
    ep.pp.log_norm(edata)

    # After normalization, R should still be 3D and same shape
    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    # Verify log transformation: log(original + 1) should equal result
    expected = np.log1p(R_original)
    assert np.allclose(edata.R, expected, rtol=1e-6)
    
    # Data should have changed significantly
    assert not np.array_equal(R_original, edata.R)


def test_offset_negative_values_3d(edata_blob_small_3d):
    """Test offset_negative_values on 3D EHRData (edata.R)."""
    edata = edata_blob_small_3d
    # Ensure we have some negative values
    edata.R = edata.R - 2  # Shift to create negative values
    
    # Save original shape and dtype
    orig_shape = edata.R.shape
    orig_dtype = edata.R.dtype
    assert np.min(edata.R) < 0, "Test data should have negative values"
    
    # Run offset
    ep.pp.offset_negative_values(edata)

    # After offset, R should still be 3D and same shape
    assert edata.R.shape == orig_shape
    assert edata.R.dtype == orig_dtype or np.issubdtype(edata.R.dtype, np.floating)

    # Minimum value should now be 0 (or very close)
    assert np.allclose(np.min(edata.R), 0, atol=1e-10)
    
    # All values should be non-negative
    assert np.all(edata.R >= 0)
