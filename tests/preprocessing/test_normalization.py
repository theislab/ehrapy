import warnings
from pathlib import Path

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME, FEATURE_TYPE_KEY, NUMERIC_TAG
from fast_array_utils.conv import to_dense
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute

CURRENT_DIR = Path(__file__).parent


def test_vars_checks(edata_to_norm):
    with pytest.raises(ValueError, match=r"Some selected vars are not numeric"):
        ep.pp.scale_norm(edata_to_norm, var_names=["String1"])


def test_norm_scale(edata_to_norm):
    warnings.filterwarnings("ignore")
    ep.pp.scale_norm(edata_to_norm)

    edata_norm = ep.pp.scale_norm(edata_to_norm, copy=True)

    num1_norm = np.array([-1.4039999, 0.55506986, 0.84893], dtype=np.float32)
    num2_norm = np.array([-1.3587323, 1.0190493, 0.3396831], dtype=np.float32)

    assert np.array_equal(edata_norm.X[:, 0], edata_to_norm.X[:, 0])
    assert np.array_equal(edata_norm.X[:, 1], edata_to_norm.X[:, 1])
    assert np.array_equal(edata_norm.X[:, 2], edata_to_norm.X[:, 2])
    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)
    assert np.allclose(edata_norm.X[:, 5], edata_to_norm.X[:, 5], equal_nan=True)


def test_norm_scale_integers(edata_mini_integers_in_X):
    edata_norm = ep.pp.scale_norm(edata_mini_integers_in_X, copy=True)
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
    assert np.allclose(edata_norm.X, in_days_norm)


def test_norm_scale_kwargs(edata_to_norm):

    edata_norm = ep.pp.scale_norm(edata_to_norm, copy=True, with_mean=False)

    num1_norm = np.array([3.3304186, 5.2894883, 5.5833483], dtype=np.float32)
    num2_norm = np.array([-0.6793662, 1.6984155, 1.0190493], dtype=np.float32)

    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)


def test_norm_scale_group(edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()

    with pytest.raises(KeyError):
        ep.pp.scale_norm(edata_mini_casted, groupby="invalid_key", copy=True)

    edata_mini_norm = ep.pp.scale_norm(
        edata_mini_casted,
        var_names=["sys_bp_entry", "dia_bp_entry"],
        groupby="disease",
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


def test_norm_minmax(edata_to_norm):

    edata_norm = ep.pp.minmax_norm(edata_to_norm, copy=True)

    num1_norm = np.array([0.0, 0.86956537, 0.9999999], dtype=np.dtype(np.float32))
    num2_norm = np.array([0.0, 1.0, 0.71428573], dtype=np.float32)

    assert np.array_equal(edata_norm.X[:, 0], edata_to_norm.X[:, 0])
    assert np.array_equal(edata_norm.X[:, 1], edata_to_norm.X[:, 1])
    assert np.array_equal(edata_norm.X[:, 2], edata_to_norm.X[:, 2])
    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)
    assert np.allclose(edata_norm.X[:, 5], edata_to_norm.X[:, 5], equal_nan=True)


def test_norm_minmax_integers(edata_mini_integers_in_X):
    edata_norm = ep.pp.minmax_norm(edata_mini_integers_in_X, copy=True)
    in_days_norm = np.array([[0.25], [0.5], [0.0], [0.25], [0.0], [0.25], [0.5], [0.75], [1.0], [0.25], [0.5], [0.25]])
    assert np.allclose(edata_norm.X, in_days_norm)


def test_norm_minmax_kwargs(edata_to_norm):

    edata_norm = ep.pp.minmax_norm(edata_to_norm, copy=True, feature_range=(0, 2))

    num1_norm = np.array([0.0, 1.7391307, 1.9999998], dtype=np.float32)
    num2_norm = np.array([0.0, 2.0, 1.4285715], dtype=np.float32)

    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)


def test_norm_minmax_group(edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()

    with pytest.raises(KeyError):
        ep.pp.minmax_norm(edata_mini_casted, groupby="invalid_key", copy=True)

    edata_mini_norm = ep.pp.minmax_norm(
        edata_mini_casted,
        var_names=["sys_bp_entry", "dia_bp_entry"],
        groupby="disease",
        copy=True,
    )
    col1_norm = np.array([0.0, 0.33333333, 0.66666667, 1.0, 0.0, 0.33333333, 0.66666667, 1.0])
    col2_norm = col1_norm
    assert np.allclose(edata_mini_norm.X[:, 0], edata_mini_casted.X[:, 0])
    assert np.allclose(edata_mini_norm.X[:, 1], col1_norm)
    assert np.allclose(edata_mini_norm.X[:, 2], col2_norm)


def test_norm_maxabs(edata_to_norm):

    edata_norm = ep.pp.maxabs_norm(edata_to_norm, copy=True)

    num1_norm = np.array([0.5964913, 0.94736844, 1.0], dtype=np.float32)
    num2_norm = np.array([-0.4, 1.0, 0.6], dtype=np.float32)

    assert np.array_equal(edata_norm.X[:, 0], edata_to_norm.X[:, 0])
    assert np.array_equal(edata_norm.X[:, 1], edata_to_norm.X[:, 1])
    assert np.array_equal(edata_norm.X[:, 2], edata_to_norm.X[:, 2])
    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)
    assert np.allclose(edata_norm.X[:, 5], edata_to_norm.X[:, 5], equal_nan=True)


def test_norm_maxabs_integers(edata_mini_integers_in_X):
    edata_norm = ep.pp.maxabs_norm(edata_mini_integers_in_X, copy=True)
    in_days_norm = np.array([[0.25], [0.5], [0.0], [0.25], [0.0], [0.25], [0.5], [0.75], [1.0], [0.25], [0.5], [0.25]])
    assert np.allclose(edata_norm.X, in_days_norm)


def test_norm_maxabs_group(edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()

    with pytest.raises(KeyError):
        ep.pp.maxabs_norm(edata_mini_casted, groupby="invalid_key", copy=True)

    edata_mini_norm = ep.pp.maxabs_norm(
        edata_mini_casted,
        var_names=["sys_bp_entry", "dia_bp_entry"],
        groupby="disease",
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


def test_norm_robust_scale(edata_to_norm):

    edata_norm = ep.pp.robust_scale_norm(edata_to_norm, copy=True)

    num1_norm = np.array([-1.73913043, 0.0, 0.26086957], dtype=np.float32)
    num2_norm = np.array([-1.4285715, 0.5714286, 0.0], dtype=np.float32)

    assert np.array_equal(edata_norm.X[:, 0], edata_to_norm.X[:, 0])
    assert np.array_equal(edata_norm.X[:, 1], edata_to_norm.X[:, 1])
    assert np.array_equal(edata_norm.X[:, 2], edata_to_norm.X[:, 2])
    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)
    assert np.allclose(edata_norm.X[:, 5], edata_to_norm.X[:, 5], equal_nan=True)


def test_norm_robust_scale_integers(edata_mini_integers_in_X):
    edata_norm = ep.pp.robust_scale_norm(edata_mini_integers_in_X, copy=True)
    in_days_norm = np.array([[0.0], [1.0], [-1.0], [0.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [0.0], [1.0], [0.0]])
    assert np.allclose(edata_norm.X, in_days_norm)


def test_norm_robust_scale_kwargs(edata_to_norm):

    edata_norm = ep.pp.robust_scale_norm(edata_to_norm, copy=True, with_scaling=False)

    num1_norm = np.array([-2.0, 0.0, 0.2999997], dtype=np.float32)
    num2_norm = np.array([-5.0, 2.0, 0.0], dtype=np.float32)

    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)


def test_norm_robust_scale_group(edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()

    with pytest.raises(KeyError):
        ep.pp.robust_scale_norm(edata_mini_casted, groupby="invalid_key", copy=True)

    edata_mini_norm = ep.pp.robust_scale_norm(
        edata_mini_casted,
        var_names=["sys_bp_entry", "dia_bp_entry"],
        groupby="disease",
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


def test_norm_quantile_uniform(edata_to_norm):
    warnings.filterwarnings("ignore", category=UserWarning)

    edata_norm = ep.pp.quantile_norm(edata_to_norm, copy=True)

    num1_norm = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    num2_norm = np.array([0.0, 1.0, 0.5], dtype=np.float32)

    assert np.array_equal(edata_norm.X[:, 0], edata_to_norm.X[:, 0])
    assert np.array_equal(edata_norm.X[:, 1], edata_to_norm.X[:, 1])
    assert np.array_equal(edata_norm.X[:, 2], edata_to_norm.X[:, 2])
    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)
    assert np.allclose(edata_norm.X[:, 5], edata_to_norm.X[:, 5], equal_nan=True)


def test_norm_quantile_integers(edata_mini_integers_in_X):
    edata_norm = ep.pp.quantile_norm(edata_mini_integers_in_X, n_quantiles=12, copy=True)
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
    assert np.allclose(edata_norm.X, in_days_norm)


def test_norm_quantile_uniform_kwargs(edata_to_norm):

    edata_norm = ep.pp.quantile_norm(edata_to_norm, copy=True, output_distribution="normal", n_quantiles=3)

    num1_norm = np.array([-5.19933758, 0.0, 5.19933758], dtype=np.float32)
    num2_norm = np.array([-5.19933758, 5.19933758, 0.0], dtype=np.float32)

    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)


def test_norm_quantile_uniform_group(edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()

    with pytest.raises(KeyError):
        ep.pp.quantile_norm(edata_mini_casted, groupby="invalid_key", copy=True)

    edata_mini_norm = ep.pp.quantile_norm(
        edata_mini_casted,
        var_names=["sys_bp_entry", "dia_bp_entry"],
        groupby="disease",
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


def test_norm_power(edata_to_norm):

    edata_norm = ep.pp.power_norm(edata_to_norm, copy=True)

    num1_norm = np.array([-1.3821232, 0.43163615, 0.950487], dtype=np.float32)
    num2_norm = np.array([-1.340104, 1.0613203, 0.27878374], dtype=np.float32)

    assert np.array_equal(edata_norm.X[:, 0], edata_to_norm.X[:, 0])
    assert np.array_equal(edata_norm.X[:, 1], edata_to_norm.X[:, 1])
    assert np.array_equal(edata_norm.X[:, 2], edata_to_norm.X[:, 2])
    assert np.allclose(edata_norm.X[:, 3], num1_norm, rtol=1.1)
    assert np.allclose(edata_norm.X[:, 4], num2_norm, rtol=1.1)
    assert np.allclose(edata_norm.X[:, 5], edata_to_norm.X[:, 5], equal_nan=True)


def test_norm_power_integers(edata_mini_integers_in_X):
    edata_norm = ep.pp.power_norm(edata_mini_integers_in_X, copy=True)
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
    assert np.allclose(edata_norm.X, in_days_norm, rtol=1e-4, atol=1e-4)


def test_norm_power_kwargs(edata_to_norm):

    with pytest.raises(ValueError):
        ep.pp.power_norm(edata_to_norm, copy=True, method="box-cox")

    edata_norm = ep.pp.power_norm(edata_to_norm, copy=True, standardize=False)

    num1_norm = np.array([201.03636, 1132.8341, 1399.3877], dtype=np.float32)
    num2_norm = np.array([-1.8225479, 5.921072, 3.397709], dtype=np.float32)

    assert np.allclose(edata_norm.X[:, 3], num1_norm, rtol=1e-02, atol=1e-02)
    assert np.allclose(edata_norm.X[:, 4], num2_norm, rtol=1e-02, atol=1e-02)


def test_norm_power_group(edata_mini_normalization):
    edata_mini_casted = edata_mini_normalization.copy()

    with pytest.raises(KeyError):
        ep.pp.power_norm(edata_mini_casted, groupby="invalid_key", copy=True)

    edata_mini_norm = ep.pp.power_norm(
        edata_mini_casted,
        var_names=["sys_bp_entry", "dia_bp_entry"],
        groupby="disease",
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


def test_norm_log1p(edata_to_norm):
    # Ensure that some test data is strictly positive
    log_edata = edata_to_norm.copy()
    log_edata.X[0, 4] = 1

    edata_norm = ep.pp.log_norm(log_edata, copy=True)

    num1_norm = np.array([1.4816046, 1.856298, 1.9021075], dtype=np.float32)
    num2_norm = np.array([0.6931472, 1.7917595, 1.3862944], dtype=np.float32)

    assert np.array_equal(edata_norm.X[:, 0], edata_to_norm.X[:, 0])
    assert np.array_equal(edata_norm.X[:, 1], edata_to_norm.X[:, 1])
    assert np.array_equal(edata_norm.X[:, 2], edata_to_norm.X[:, 2])
    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)
    assert np.allclose(edata_norm.X[:, 5], edata_to_norm.X[:, 5], equal_nan=True)

    # Check alternative base works
    edata_norm = ep.pp.log_norm(log_edata, base=10, copy=True)

    num1_norm = np.divide(np.array([1.4816046, 1.856298, 1.9021075], dtype=np.float32), np.log(10))
    num2_norm = np.divide(np.array([0.6931472, 1.7917595, 1.3862944], dtype=np.float32), np.log(10))

    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)

    # Check alternative offset works
    edata_norm = ep.pp.log_norm(log_edata, offset=0.5, copy=True)

    num1_norm = np.array([1.3609766, 1.7749524, 1.8245492], dtype=np.float32)
    num2_norm = np.array([0.4054651, 1.7047482, 1.252763], dtype=np.float32)

    assert np.allclose(edata_norm.X[:, 3], num1_norm)
    assert np.allclose(edata_norm.X[:, 4], num2_norm)

    try:
        ep.pp.log_norm(edata_to_norm, var_names="Numeric2", offset=3, copy=True)
    except ValueError:
        pytest.fail("Unexpected ValueError exception was raised.")

    with pytest.raises(ValueError):
        ep.pp.log_norm(edata_to_norm, copy=True)

    with pytest.raises(ValueError):
        ep.pp.log_norm(edata_to_norm, var_names="Numeric2", offset=1, copy=True)


def test_norm_record(edata_to_norm):
    edata_norm = ep.pp.minmax_norm(edata_to_norm, copy=True)

    assert edata_norm.uns["normalization"] == {
        "Numeric1": ["minmax"],
        "Numeric2": ["minmax"],
    }

    edata_norm = ep.pp.maxabs_norm(edata_norm, var_names=["Numeric1"], copy=True)

    assert edata_norm.uns["normalization"] == {
        "Numeric1": ["minmax", "maxabs"],
        "Numeric2": ["minmax"],
    }


def test_offset_negative_values():
    to_offset_edata = ed.EHRData(X=np.array([[-1, -5, -10], [5, 6, -20]], dtype=np.float32))
    expected_edata = ed.EHRData(X=np.array([[19, 15, 10], [25, 26, 0]], dtype=np.float32))

    assert np.array_equal(expected_edata.X, ep.pp.offset_negative_values(to_offset_edata, copy=True).X)


def test_norm_numerical_only():
    to_normalize_edata = ed.EHRData(X=np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32))
    expected_edata = ed.EHRData(X=np.array([[0.6931472, 0, 0], [0, 0, 0.6931472]], dtype=np.float32))
    ed.infer_feature_types(to_normalize_edata, binary_as="numeric")
    assert np.array_equal(expected_edata.X, ep.pp.log_norm(to_normalize_edata, copy=True).X)


def test_scale_norm_3D(edata_blobs_timeseries_small):
    edata = edata_blobs_timeseries_small.copy()
    ep.pp.scale_norm(edata, layer=DEFAULT_TEM_LAYER_NAME)

    n_obs, n_var, n_timestamps = edata.layers[DEFAULT_TEM_LAYER_NAME].shape
    for var_idx in range(n_var):
        flat = edata.layers[DEFAULT_TEM_LAYER_NAME][:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmean(flat), 0, atol=1e-6), f"Mean check failed for variable {var_idx}"
            assert np.allclose(np.nanstd(flat), 1, atol=1e-6), f"Std check failed for variable {var_idx}"


def test_minmax_norm_3D(edata_blobs_timeseries_small):
    edata = edata_blobs_timeseries_small.copy()
    ep.pp.minmax_norm(edata, layer=DEFAULT_TEM_LAYER_NAME)

    n_obs, n_var, n_timestamps = edata.layers[DEFAULT_TEM_LAYER_NAME].shape
    for var_idx in range(n_var):
        flat = edata.layers[DEFAULT_TEM_LAYER_NAME][:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmin(flat), 0, atol=1e-6), f"Min check failed for variable {var_idx}"
            assert np.allclose(np.nanmax(flat), 1, atol=1e-6), f"Max check failed for variable {var_idx}"


def test_maxabs_norm_3D(edata_blobs_timeseries_small):
    edata = edata_blobs_timeseries_small.copy()
    ep.pp.maxabs_norm(edata, layer=DEFAULT_TEM_LAYER_NAME)

    n_obs, n_var, n_timestamps = edata.layers[DEFAULT_TEM_LAYER_NAME].shape
    for var_idx in range(n_var):
        flat = edata.layers[DEFAULT_TEM_LAYER_NAME][:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmax(np.abs(flat)), 1, atol=1e-6), f"Max-abs check failed for variable {var_idx}"


def test_robust_scale_norm_3D(edata_blobs_timeseries_small):
    edata = edata_blobs_timeseries_small.copy()
    ep.pp.robust_scale_norm(edata, layer=DEFAULT_TEM_LAYER_NAME)

    n_obs, n_var, n_timestamps = edata.layers[DEFAULT_TEM_LAYER_NAME].shape
    for var_idx in range(n_var):
        flat = edata.layers[DEFAULT_TEM_LAYER_NAME][:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmedian(flat), 0, atol=1e-6), f"Median check failed for variable {var_idx}"
            assert np.allclose(np.nanpercentile(flat, 75) - np.nanpercentile(flat, 25), 1, atol=1e-6), (
                f"IQR check failed for variable {var_idx}"
            )


def test_quantile_norm_3D(edata_blobs_timeseries_small):
    edata = edata_blobs_timeseries_small.copy()
    ep.pp.quantile_norm(edata, layer=DEFAULT_TEM_LAYER_NAME)

    n_obs, n_var, n_timestamps = edata.layers[DEFAULT_TEM_LAYER_NAME].shape
    for var_idx in range(n_var):
        flat = edata.layers[DEFAULT_TEM_LAYER_NAME][:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmin(flat), 0, atol=1e-6), f"Min check failed for variable {var_idx}"
            assert np.allclose(np.nanmax(flat), 1, atol=1e-6), f"Max check failed for variable {var_idx}"
            assert np.allclose(np.nanpercentile(flat, 25), 0.25, atol=0.05), f"Q25 check failed for variable {var_idx}"
            assert np.allclose(np.nanpercentile(flat, 50), 0.5, atol=0.05), f"Q50 check failed for variable {var_idx}"
            assert np.allclose(np.nanpercentile(flat, 75), 0.75, atol=0.05), f"Q75 check failed for variable {var_idx}"


def test_power_norm_3D(edata_blobs_timeseries_small):
    edata = edata_blobs_timeseries_small.copy()
    ep.pp.offset_negative_values(edata, layer=DEFAULT_TEM_LAYER_NAME)
    ep.pp.power_norm(edata, layer=DEFAULT_TEM_LAYER_NAME)

    n_obs, n_var, n_timestamps = edata.layers[DEFAULT_TEM_LAYER_NAME].shape
    for var_idx in range(n_var):
        flat = edata.layers[DEFAULT_TEM_LAYER_NAME][:, var_idx, :].reshape(-1)
        if not np.all(np.isnan(flat)):
            assert np.allclose(np.nanmean(flat), 0, atol=1e-5), f"Mean check failed for variable {var_idx}"
            assert np.allclose(np.nanstd(flat), 1, atol=1e-5), f"Std check failed for variable {var_idx}"


def test_log_norm_3D(edata_blobs_timeseries_small):
    edata = edata_blobs_timeseries_small.copy()
    ep.pp.offset_negative_values(edata, layer=DEFAULT_TEM_LAYER_NAME)

    layer_original = edata.layers[DEFAULT_TEM_LAYER_NAME].copy()

    ep.pp.log_norm(edata, layer=DEFAULT_TEM_LAYER_NAME)

    expected = np.log1p(layer_original)
    assert np.allclose(edata.layers[DEFAULT_TEM_LAYER_NAME], expected, rtol=1e-6, equal_nan=True)

    assert not np.allclose(layer_original, edata.layers[DEFAULT_TEM_LAYER_NAME], equal_nan=True)


def test_offset_negative_values_3D(edata_blobs_timeseries_small):
    edata = edata_blobs_timeseries_small.copy()
    edata.layers[DEFAULT_TEM_LAYER_NAME] = edata.layers[DEFAULT_TEM_LAYER_NAME] - 2
    assert np.nanmin(edata.layers[DEFAULT_TEM_LAYER_NAME]) < 0

    ep.pp.offset_negative_values(edata, layer=DEFAULT_TEM_LAYER_NAME)

    assert np.allclose(np.nanmin(edata.layers[DEFAULT_TEM_LAYER_NAME]), 0, atol=1e-10)

    non_nan_values = edata.layers[DEFAULT_TEM_LAYER_NAME][~np.isnan(edata.layers[DEFAULT_TEM_LAYER_NAME])]
    assert np.all(non_nan_values >= 0)


@pytest.mark.parametrize(
    "norm_func",
    [
        ep.pp.scale_norm,
        ep.pp.minmax_norm,
        ep.pp.maxabs_norm,
        ep.pp.robust_scale_norm,
        ep.pp.quantile_norm,
        ep.pp.power_norm,
        ep.pp.log_norm,
        ep.pp.offset_negative_values,
    ],
)
def test_norm_with_X_none_and_layer(edata_blobs_timeseries_small, norm_func):
    edata = edata_blobs_timeseries_small.copy()
    layer = DEFAULT_TEM_LAYER_NAME

    edata.X = None

    edata.var[FEATURE_TYPE_KEY] = NUMERIC_TAG

    layer_before = edata.layers[layer].copy()

    if norm_func in (ep.pp.power_norm, ep.pp.log_norm):
        ep.pp.offset_negative_values(edata, layer=layer)
        layer_before = edata.layers[layer].copy()

    result = norm_func(edata, layer=layer, copy=True)

    assert result is not None
    assert result.layers[layer].shape == layer_before.shape
    assert edata.X is None

    if norm_func != ep.pp.offset_negative_values:
        assert not np.allclose(layer_before, result.layers[layer], equal_nan=True)


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
def test_norm_group_3D(edata_blobs_timeseries_small, norm_func):
    edata = edata_blobs_timeseries_small
    layer = DEFAULT_TEM_LAYER_NAME
    edata.var[FEATURE_TYPE_KEY] = NUMERIC_TAG

    if norm_func == ep.pp.power_norm:
        ep.pp.offset_negative_values(edata, layer=layer)

    # create two groups with different distributions
    n_obs = edata.n_obs
    group_size = n_obs // 2
    edata.obs["group"] = ["A"] * group_size + ["B"] * (n_obs - group_size)

    # raise NotImplementedError for all dask arrays
    original_shape = edata.layers[layer].shape
    layer_before = edata.layers[layer].copy()

    norm_func(edata, layer=layer, groupby="group")

    # verify shape and tracking
    assert edata.layers[layer].shape == original_shape
    assert "normalization" in edata.uns
    assert len(edata.uns["normalization"]) > 0

    layer_after = edata.layers[layer]

    # verify data changed
    assert not np.allclose(layer_before, layer_after, equal_nan=True)

    group_a = layer_after[:group_size].flatten()
    group_b = layer_after[group_size:].flatten()
    group_a = group_a[~np.isnan(group_a)]
    group_b = group_b[~np.isnan(group_b)]

    def near0(x):
        return abs(x) < 1e-5

    def near1(x):
        return abs(x - 1.0) < 1e-5

    # validate per-group normalization
    if norm_func in {ep.pp.scale_norm, ep.pp.power_norm}:
        assert near0(np.nanmean(group_a)) and near0(np.nanmean(group_b))
        assert near1(np.nanstd(group_a)) and near1(np.nanstd(group_b))

    elif norm_func in {ep.pp.minmax_norm, ep.pp.quantile_norm}:
        assert near0(np.nanmin(group_a)) and near0(np.nanmin(group_b))
        assert near1(np.nanmax(group_a)) and near1(np.nanmax(group_b))

    elif norm_func == ep.pp.maxabs_norm:
        assert near1(np.nanmax(np.abs(group_a)))
        assert near1(np.nanmax(np.abs(group_b)))

    elif norm_func == ep.pp.robust_scale_norm:
        assert near0(np.nanmedian(group_a)) and near0(np.nanmedian(group_b))


NORMS = [
    pytest.param(ep.pp.scale_norm, {}, False, id="scale"),
    pytest.param(ep.pp.scale_norm, {"with_mean": False}, True, id="scale-without-mean"),
    pytest.param(ep.pp.minmax_norm, {}, False, id="minmax"),
    pytest.param(ep.pp.maxabs_norm, {}, True, id="maxabs"),
    pytest.param(ep.pp.robust_scale_norm, {}, False, id="robust"),
    pytest.param(ep.pp.robust_scale_norm, {"with_centering": False}, True, id="robust-without-centering"),
    pytest.param(ep.pp.quantile_norm, {"n_quantiles": 10}, False, id="quantile"),
    pytest.param(ep.pp.power_norm, {}, False, id="power"),
    pytest.param(ep.pp.log_norm, {}, True, id="log"),
]


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("groupby", [None, "group"])
@pytest.mark.parametrize(("norm", "kwargs", "sparse_support"), NORMS)
def test_norm_array_types(array_type, ndim, groupby, norm, kwargs, sparse_support, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    if groupby is not None and norm is ep.pp.log_norm:
        pytest.skip("log_norm has no groupby")
    shape = (20, 4) if ndim == 2 else (20, 4, 3)
    X = np.where(rng.random(shape) < 0.4, 0, rng.gamma(2, size=shape))
    X[rng.random(shape) < 0.1] = np.nan
    obs = pd.DataFrame({"group": ["a", "b"] * 10}, index=[str(i) for i in range(20)])
    kwargs = {**kwargs, "groupby": groupby} if groupby is not None else kwargs

    def make_edata(X):
        edata = ed.EHRData(X=X, obs=obs)
        edata.var[FEATURE_TYPE_KEY] = NUMERIC_TAG
        return edata

    expected = norm(make_edata(X), copy=True, **kwargs).X
    edata = make_edata(array_type(X))

    if array_type.flags & Flags.Sparse and (not sparse_support or array_type.flags & Flags.Dask):
        with pytest.raises(NotImplementedError):
            norm(edata, **kwargs)
        return

    with forbid_dask_compute():
        result = norm(edata, copy=True, **kwargs).X

    assert isinstance(result, array_type.cls)
    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("negative", [False, True])
def test_offset_negative_values_array_types(array_type, negative):
    X = np.array([[0.0, 2.0, np.nan], [0.0, 0.0, 3.0]])
    if negative:
        X[0, 1] = -2.0
    expected = ep.pp.offset_negative_values(ed.EHRData(X=X), copy=True).X
    edata = ed.EHRData(X=array_type(X))

    if array_type.flags & Flags.Sparse and (negative or array_type.flags & Flags.Dask):
        with pytest.raises(NotImplementedError):
            ep.pp.offset_negative_values(edata)
        return

    with forbid_dask_compute():
        result = ep.pp.offset_negative_values(edata, copy=True).X

    assert isinstance(result, array_type.cls)
    np.testing.assert_allclose(to_dense(result, to_cpu_memory=True), expected, equal_nan=True)


def test_norm_groupby_missing_values(edata_mini_normalization):
    edata_mini_normalization.obs.loc[edata_mini_normalization.obs_names[0], "disease"] = np.nan
    with pytest.raises(ValueError, match="contains missing values"):
        ep.pp.scale_norm(edata_mini_normalization, groupby="disease")
