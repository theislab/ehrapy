import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import ehrapy as ep
from ehrapy.anndata._constants import CATEGORICAL_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from ehrapy.io._read import read_csv

CURRENT_DIR = Path(__file__).parent
_TEST_DATA_PATH = f"{CURRENT_DIR.parent}/test_data"


@pytest.fixture
def adata_mini():
    return read_csv(f"{_TEST_DATA_PATH}/dataset1.csv", columns_obs_only=["glucose", "weight", "disease", "station"])[:8]


@pytest.fixture
def adata_to_norm():
    obs_data = {"ID": ["Patient1", "Patient2", "Patient3"], "Age": [31, 94, 62]}

    X_data = np.array(
        [
            [1, 3.4, -2.0, 1.0, "A string", "A different string"],
            [2, 5.4, 5.0, 2.0, "Silly string", "A different string"],
            [2, 5.7, 3.0, np.nan, "A string", "What string?"],
        ],
        dtype=np.dtype(object),
    )
    # the "ignore" tag is used to make the column being ignored; the original test selecting a few
    # columns induces a specific ordering which is kept for now
    var_data = {
        "Feature": ["Integer1", "Numeric1", "Numeric2", "Numeric3", "String1", "String2"],
        "Type": ["Integer", "Numeric", "Numeric", "Numeric", "String", "String"],
        FEATURE_TYPE_KEY: [CATEGORICAL_TAG, NUMERIC_TAG, NUMERIC_TAG, "ignore", CATEGORICAL_TAG, CATEGORICAL_TAG],
    }
    adata = AnnData(
        X=X_data,
        obs=pd.DataFrame(data=obs_data),
        var=pd.DataFrame(data=var_data, index=var_data["Feature"]),
        uns=OrderedDict(),
    )

    adata = ep.pp.encode(adata, autodetect=True, encodings="label")

    return adata


def test_vars_checks(adata_to_norm):
    """Test for checks that vars argument is valid."""
    with pytest.raises(ValueError, match=r"Some selected vars are not numeric"):
        ep.pp.scale_norm(adata_to_norm, vars=["String1"])


def test_norm_scale(adata_to_norm, adata_mini):
    """Test for the scaling normalization method."""
    warnings.filterwarnings("ignore")

    adata_norm = ep.pp.scale_norm(adata_to_norm, copy=True)

    num1_norm = np.array([-1.4039999, 0.55506986, 0.84893], dtype=np.float32)
    num2_norm = np.array([-1.3587323, 1.0190493, 0.3396831], dtype=np.float32)

    assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
    assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
    assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)
    assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)

    # Test passing kwargs works
    adata_norm = ep.pp.scale_norm(adata_to_norm, copy=True, with_mean=False)

    num1_norm = np.array([3.3304186, 5.2894883, 5.5833483], dtype=np.float32)
    num2_norm = np.array([-0.6793662, 1.6984155, 1.0190493], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)

    # Test the group key works
    with pytest.raises(KeyError):
        ep.pp.scale_norm(adata_mini, group_key="invalid_key", copy=True)

    adata_mini_norm = ep.pp.scale_norm(
        adata_mini, vars=["sys_bp_entry", "dia_bp_entry"], group_key="disease", copy=True
    )
    col1_norm = np.array(
        [-1.34164079, -0.4472136, 0.4472136, 1.34164079, -1.34164079, -0.4472136, 0.4472136, 1.34164079]
    )
    col2_norm = col1_norm
    assert np.allclose(adata_mini_norm.X[:, 0], adata_mini.X[:, 0])
    assert np.allclose(adata_mini_norm.X[:, 1], col1_norm)
    assert np.allclose(adata_mini_norm.X[:, 2], col2_norm)


def test_norm_minmax(adata_to_norm, adata_mini):
    """Test for the minmax normalization method."""
    adata_norm = ep.pp.minmax_norm(adata_to_norm, copy=True)

    num1_norm = np.array([0.0, 0.86956537, 0.9999999], dtype=np.dtype(np.float32))
    num2_norm = np.array([0.0, 1.0, 0.71428573], dtype=np.float32)

    assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
    assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
    assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)
    assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)

    # Test passing kwargs works
    adata_norm = ep.pp.minmax_norm(adata_to_norm, copy=True, feature_range=(0, 2))

    num1_norm = np.array([0.0, 1.7391307, 1.9999998], dtype=np.float32)
    num2_norm = np.array([0.0, 2.0, 1.4285715], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)

    # Test the group key works
    with pytest.raises(KeyError):
        ep.pp.minmax_norm(adata_mini, group_key="invalid_key", copy=True)

    adata_mini_norm = ep.pp.minmax_norm(
        adata_mini, vars=["sys_bp_entry", "dia_bp_entry"], group_key="disease", copy=True
    )
    col1_norm = np.array([0.0, 0.33333333, 0.66666667, 1.0, 0.0, 0.33333333, 0.66666667, 1.0])
    col2_norm = col1_norm
    assert np.allclose(adata_mini_norm.X[:, 0], adata_mini.X[:, 0])
    assert np.allclose(adata_mini_norm.X[:, 1], col1_norm)
    assert np.allclose(adata_mini_norm.X[:, 2], col2_norm)


def test_norm_maxabs(adata_to_norm, adata_mini):
    """Test for the maxabs normalization method."""
    adata_norm = ep.pp.maxabs_norm(adata_to_norm, copy=True)

    num1_norm = np.array([0.5964913, 0.94736844, 1.0], dtype=np.float32)
    num2_norm = np.array([-0.4, 1.0, 0.6], dtype=np.float32)

    assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
    assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
    assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)
    assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)

    # Test the group key works
    with pytest.raises(KeyError):
        ep.pp.maxabs_norm(adata_mini, group_key="invalid_key", copy=True)

    adata_mini_norm = ep.pp.maxabs_norm(
        adata_mini, vars=["sys_bp_entry", "dia_bp_entry"], group_key="disease", copy=True
    )
    col1_norm = np.array([0.9787234, 0.9858156, 0.9929078, 1.0, 0.98013245, 0.98675497, 0.99337748, 1.0])
    col2_norm = np.array([0.96296296, 0.97530864, 0.98765432, 1.0, 0.9625, 0.975, 0.9875, 1.0])
    assert np.allclose(adata_mini_norm.X[:, 0], adata_mini.X[:, 0])
    assert np.allclose(adata_mini_norm.X[:, 1], col1_norm)
    assert np.allclose(adata_mini_norm.X[:, 2], col2_norm)


def test_norm_robust_scale(adata_to_norm, adata_mini):
    """Test for the robust_scale normalization method."""
    adata_norm = ep.pp.robust_scale_norm(adata_to_norm, copy=True)

    num1_norm = np.array([-1.73913043, 0.0, 0.26086957], dtype=np.float32)
    num2_norm = np.array([-1.4285715, 0.5714286, 0.0], dtype=np.float32)

    assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
    assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
    assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)
    assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)

    # Test passing kwargs works
    adata_norm = ep.pp.robust_scale_norm(adata_to_norm, copy=True, with_scaling=False)

    num1_norm = np.array([-2.0, 0.0, 0.2999997], dtype=np.float32)
    num2_norm = np.array([-5.0, 2.0, 0.0], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)

    # Test the group key works
    with pytest.raises(KeyError):
        ep.pp.robust_scale_norm(adata_mini, group_key="invalid_key", copy=True)

    adata_mini_norm = ep.pp.robust_scale_norm(
        adata_mini, vars=["sys_bp_entry", "dia_bp_entry"], group_key="disease", copy=True
    )
    col1_norm = np.array([-1.0, -0.33333333, 0.33333333, 1.0, -1.0, -0.33333333, 0.33333333, 1.0], dtype=np.float32)
    col2_norm = col1_norm
    assert np.allclose(adata_mini_norm.X[:, 0], adata_mini.X[:, 0])
    assert np.allclose(adata_mini_norm.X[:, 1], col1_norm)
    assert np.allclose(adata_mini_norm.X[:, 2], col2_norm)


def test_norm_quantile_uniform(adata_to_norm, adata_mini):
    """Test for the quantile normalization method."""
    warnings.filterwarnings("ignore", category=UserWarning)
    adata_norm = ep.pp.quantile_norm(adata_to_norm, copy=True)

    num1_norm = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    num2_norm = np.array([0.0, 1.0, 0.5], dtype=np.float32)

    assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
    assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
    assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)
    assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)

    # Test passing kwargs works
    adata_norm = ep.pp.quantile_norm(adata_to_norm, copy=True, output_distribution="normal")

    num1_norm = np.array([-5.19933758, 0.0, 5.19933758], dtype=np.float32)
    num2_norm = np.array([-5.19933758, 5.19933758, 0.0], dtype=np.float32)

    assert np.allclose(adata_norm.X[:, 3], num1_norm)
    assert np.allclose(adata_norm.X[:, 4], num2_norm)

    # Test the group key works
    with pytest.raises(KeyError):
        ep.pp.quantile_norm(adata_mini, group_key="invalid_key", copy=True)

    adata_mini_norm = ep.pp.quantile_norm(
        adata_mini, vars=["sys_bp_entry", "dia_bp_entry"], group_key="disease", copy=True
    )
    col1_norm = np.array([0.0, 0.33333333, 0.66666667, 1.0, 0.0, 0.33333333, 0.66666667, 1.0], dtype=np.float32)
    col2_norm = col1_norm
    assert np.allclose(adata_mini_norm.X[:, 0], adata_mini.X[:, 0])
    assert np.allclose(adata_mini_norm.X[:, 1], col1_norm)
    assert np.allclose(adata_mini_norm.X[:, 2], col2_norm)


def test_norm_power(adata_to_norm, adata_mini):
    """Test for the power transformation normalization method."""
    adata_norm = ep.pp.power_norm(adata_to_norm, copy=True)

    num1_norm = np.array([-1.3821232, 0.43163615, 0.950487], dtype=np.float32)
    num2_norm = np.array([-1.340104, 1.0613203, 0.27878374], dtype=np.float32)

    assert np.array_equal(adata_norm.X[:, 0], adata_to_norm.X[:, 0])
    assert np.array_equal(adata_norm.X[:, 1], adata_to_norm.X[:, 1])
    assert np.array_equal(adata_norm.X[:, 2], adata_to_norm.X[:, 2])
    assert np.allclose(adata_norm.X[:, 3], num1_norm, rtol=1.1)
    assert np.allclose(adata_norm.X[:, 4], num2_norm, rtol=1.1)
    assert np.allclose(adata_norm.X[:, 5], adata_to_norm.X[:, 5], equal_nan=True)

    with pytest.raises(ValueError):
        ep.pp.power_norm(adata_to_norm, copy=True, method="box-cox")

    with pytest.raises(KeyError):
        ep.pp.power_norm(adata_mini, group_key="invalid_key", copy=True)

    adata_mini_norm = ep.pp.power_norm(
        adata_mini, vars=["sys_bp_entry", "dia_bp_entry"], group_key="disease", copy=True
    )
    col1_norm = np.array(
        [-1.34266204, -0.44618949, 0.44823148, 1.34062005, -1.34259417, -0.44625773, 0.44816403, 1.34068786],
        dtype=np.float32,
    )
    col2_norm = np.array(
        [-1.34342372, -0.44542197, 0.44898626, 1.33985944, -1.34344617, -0.4453993, 0.44900845, 1.33983703],
        dtype=np.float32,
    )
    assert np.allclose(adata_mini_norm.X[:, 0], adata_mini.X[:, 0])
    assert np.allclose(adata_mini_norm.X[:, 1], col1_norm, rtol=1e-02, atol=1e-02)
    assert np.allclose(adata_mini_norm.X[:, 2], col2_norm, rtol=1e-02, atol=1e-02)


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

    assert adata_norm.uns["normalization"] == {"Numeric1": ["minmax"], "Numeric2": ["minmax"]}

    adata_norm = ep.pp.maxabs_norm(adata_norm, vars=["Numeric1"], copy=True)

    assert adata_norm.uns["normalization"] == {"Numeric1": ["minmax", "maxabs"], "Numeric2": ["minmax"]}


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
