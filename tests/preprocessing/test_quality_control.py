from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import ehrapy as ep
from ehrapy.io._read import read_csv
from ehrapy.preprocessing._encoding import encode
from ehrapy.preprocessing._quality_control import _compute_obs_metrics, _compute_var_metrics, mcar_test
from tests.conftest import ARRAY_TYPES, TEST_DATA_PATH, as_dense_dask_array

CURRENT_DIR = Path(__file__).parent
_TEST_PATH_ENCODE = f"{TEST_DATA_PATH}/encode"


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_qc_metrics_vanilla(array_type, missing_values_adata):
    adata = missing_values_adata
    adata.X = array_type(adata.X)
    modification_copy = adata.copy()
    obs_metrics, var_metrics = ep.pp.qc_metrics(adata)

    assert np.array_equal(obs_metrics["missing_values_abs"].values, np.array([1, 2]))
    assert np.allclose(obs_metrics["missing_values_pct"].values, np.array([33.3333, 66.6667]))

    assert np.array_equal(var_metrics["missing_values_abs"].values, np.array([1, 2, 0]))
    assert np.allclose(var_metrics["missing_values_pct"].values, np.array([50.0, 100.0, 0.0]))
    assert np.allclose(var_metrics["mean"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
    assert np.allclose(var_metrics["median"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
    assert np.allclose(var_metrics["min"].values, np.array([0.21, np.nan, 7.234]), equal_nan=True)
    assert np.allclose(var_metrics["max"].values, np.array([0.21, np.nan, 41.419998]), equal_nan=True)
    assert (~var_metrics["iqr_outliers"]).all()

    # check that none of the columns were modified
    for key in modification_copy.obs.keys():
        assert np.array_equal(modification_copy.obs[key], adata.obs[key])
    for key in modification_copy.var.keys():
        assert np.array_equal(modification_copy.var[key], adata.var[key])


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_obs_qc_metrics(array_type, missing_values_adata):
    missing_values_adata.X = array_type(missing_values_adata.X)
    mtx = missing_values_adata.X
    obs_metrics = _compute_obs_metrics(mtx, missing_values_adata)

    assert np.array_equal(obs_metrics["missing_values_abs"].values, np.array([1, 2]))
    assert np.allclose(obs_metrics["missing_values_pct"].values, np.array([33.3333, 66.6667]))


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_var_qc_metrics(array_type, missing_values_adata):
    missing_values_adata.X = array_type(missing_values_adata.X)
    mtx = missing_values_adata.X
    var_metrics = _compute_var_metrics(mtx, missing_values_adata)

    assert np.array_equal(var_metrics["missing_values_abs"].values, np.array([1, 2, 0]))
    assert np.allclose(var_metrics["missing_values_pct"].values, np.array([50.0, 100.0, 0.0]))
    assert np.allclose(var_metrics["mean"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
    assert np.allclose(var_metrics["median"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
    assert np.allclose(var_metrics["min"].values, np.array([0.21, np.nan, 7.234]), equal_nan=True)
    assert np.allclose(var_metrics["max"].values, np.array([0.21, np.nan, 41.419998]), equal_nan=True)
    assert (~var_metrics["iqr_outliers"]).all()


@pytest.mark.parametrize(
    "array_type, expected_error",
    [
        (np.array, None),
        (as_dense_dask_array, None),
        # can't test sparse matrices because they don't support string values
    ],
)
def test_obs_qc_metrics_array_types(array_type, expected_error):
    adata = read_csv(dataset_path=f"{_TEST_PATH_ENCODE}/dataset1.csv")
    adata.X = array_type(adata.X)
    mtx = adata.X
    if expected_error:
        with pytest.raises(expected_error):
            _compute_obs_metrics(mtx, adata)


def test_obs_nan_qc_metrics():
    adata = read_csv(dataset_path=f"{_TEST_PATH_ENCODE}/dataset1.csv")
    adata.X[0][4] = np.nan
    adata2 = encode(adata, encodings={"one-hot": ["clinic_day"]})
    mtx = adata2.X
    obs_metrics = _compute_obs_metrics(mtx, adata2)
    assert obs_metrics.iloc[0].iloc[0] == 1


@pytest.mark.parametrize(
    "array_type, expected_error",
    [
        (np.array, None),
        (as_dense_dask_array, None),
        # can't test sparse matrices because they don't support string values
    ],
)
def test_var_qc_metrics_array_types(array_type, expected_error):
    adata = read_csv(dataset_path=f"{_TEST_PATH_ENCODE}/dataset1.csv")
    adata.X = array_type(adata.X)
    mtx = adata.X
    if expected_error:
        with pytest.raises(expected_error):
            _compute_var_metrics(mtx, adata)


def test_var_encoding_mode_does_not_modify_original_matrix():
    adata = read_csv(dataset_path=f"{_TEST_PATH_ENCODE}/dataset1.csv")
    adata2 = encode(adata, encodings={"one-hot": ["clinic_day"]})
    mtx_copy = adata2.X.copy()
    _compute_var_metrics(adata2.X, adata2)
    assert np.array_equal(mtx_copy, adata2.X)


def test_var_nan_qc_metrics():
    adata = read_csv(dataset_path=f"{_TEST_PATH_ENCODE}/dataset1.csv")
    adata.X[0][4] = np.nan
    adata2 = encode(adata, encodings={"one-hot": ["clinic_day"]})
    mtx = adata2.X
    var_metrics = _compute_var_metrics(mtx, adata2)
    assert var_metrics.iloc[0].iloc[0] == 1
    assert var_metrics.iloc[1].iloc[0] == 1
    assert var_metrics.iloc[2].iloc[0] == 1
    assert var_metrics.iloc[3].iloc[0] == 1
    assert var_metrics.iloc[4].iloc[0] == 1


def test_calculate_qc_metrics(missing_values_adata):
    obs_metrics, var_metrics = ep.pp.qc_metrics(missing_values_adata)

    assert obs_metrics is not None
    assert var_metrics is not None

    assert missing_values_adata.obs.missing_values_abs is not None
    assert missing_values_adata.var.missing_values_abs is not None


def test_encode_3D_edata(edata_blob_small):
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.qc_metrics(edata_blob_small, layer="R_layer")


def test_qc_lab_measurements_simple(lab_measurements_simple_adata):
    expected_obs_data = pd.Series(
        data={
            "Acetaminophen normal": [True, True],
            "Acetoacetic acid normal": [True, False],
            "Beryllium, toxic normal": [False, True],
        }
    )

    ep.pp.qc_lab_measurements(
        lab_measurements_simple_adata,
        measurements=list(lab_measurements_simple_adata.var_names),
        unit="SI",
    )

    assert (
        list(lab_measurements_simple_adata.obs["Acetaminophen normal"]) == (expected_obs_data["Acetaminophen normal"])
    )
    assert (
        list(lab_measurements_simple_adata.obs["Acetoacetic acid normal"])
        == (expected_obs_data["Acetoacetic acid normal"])
    )
    assert (
        list(lab_measurements_simple_adata.obs["Beryllium, toxic normal"])
        == (expected_obs_data["Beryllium, toxic normal"])
    )


def test_qc_lab_measurements_simple_layer(lab_measurements_layer_adata):
    expected_obs_data = pd.Series(
        data={
            "Acetaminophen normal": [True, True],
            "Acetoacetic acid normal": [True, False],
            "Beryllium, toxic normal": [False, True],
        }
    )

    ep.pp.qc_lab_measurements(
        lab_measurements_layer_adata,
        measurements=list(lab_measurements_layer_adata.var_names),
        unit="SI",
        layer="layer_copy",
    )

    assert list(lab_measurements_layer_adata.obs["Acetaminophen normal"]) == (expected_obs_data["Acetaminophen normal"])
    assert (
        list(lab_measurements_layer_adata.obs["Acetoacetic acid normal"])
        == (expected_obs_data["Acetoacetic acid normal"])
    )
    assert (
        list(lab_measurements_layer_adata.obs["Beryllium, toxic normal"])
        == (expected_obs_data["Beryllium, toxic normal"])
    )


def test_qc_lab_measurements_3D_edata(edata_blob_small):
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.qc_lab_measurements(edata_blob_small, layer="R_layer")


def test_qc_lab_measurements_age():
    # TODO
    pass


def test_qc_lab_measurements_sex():
    # TODO
    pass


def test_qc_lab_measurements_ethnicity():
    # TODO
    pass


def test_qc_lab_measurements_multiple_measurements():
    data = pd.DataFrame(np.array([[100, 98], [162, 107]]), columns=["oxygen saturation", "glucose"], index=[0, 1])

    with pytest.raises(ValueError):
        adata = ep.ad.df_to_anndata(data)
        ep.pp.qc_lab_measurements(adata, measurements=["oxygen saturation", "glucose"], unit="SI")


@pytest.mark.parametrize(
    "method,expected_output_type",
    [
        ("little", float),
        ("ttest", pd.DataFrame),
    ],
)
def test_mcar_test_method_output_types(mar_adata, method, expected_output_type):
    """Tests if mcar_test returns the correct output type for different methods."""
    output = mcar_test(mar_adata, method=method)
    assert isinstance(output, expected_output_type), (
        f"Output type for method '{method}' should be {expected_output_type}, got {type(output)} instead."
    )


def test_mcar_test_3D_edata(edata_blob_small):
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        mcar_test(edata_blob_small, layer="R_layer")


def test_mar_data_identification(mar_adata):
    """Test that mcar_test correctly identifies data as not MCAR (i.e., MAR or NMAR)."""
    p_value = mcar_test(mar_adata, method="little")
    assert p_value <= 0.05, "The test should significantly reject the MCAR hypothesis for MAR data."


def test_mcar_identification(mcar_adata):
    """Test that mcar_test correctly identifies data as MCAR."""
    p_value = mcar_test(mcar_adata, method="little")
    assert p_value > 0.05, "The test should not significantly accept the MCAR hypothesis for MCAR data."
