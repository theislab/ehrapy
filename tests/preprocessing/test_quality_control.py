from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import ehrapy as ep
from ehrapy.io._read import read_csv
from ehrapy.preprocessing._encoding import encode
from ehrapy.preprocessing._quality_control import _obs_qc_metrics, _var_qc_metrics, mcar_test

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_preprocessing"
_TEST_PATH_ENCODE = f"{CURRENT_DIR}/test_data_encode"


@pytest.fixture
def obs_data():
    return {
        "disease": ["cancer", "tumor"],
        "country": ["Germany", "switzerland"],
        "sex": ["male", "female"],
    }


@pytest.fixture
def var_data():
    return {
        "alive": ["yes", "no", "maybe"],
        "hospital": ["hospital 1", "hospital 2", "hospital 1"],
        "crazy": ["yes", "yes", "yes"],
    }


@pytest.fixture
def missing_values_adata(obs_data, var_data):
    return AnnData(
        X=np.array([[0.21, np.nan, 41.42], [np.nan, np.nan, 7.234]], dtype=np.float32),
        obs=pd.DataFrame(data=obs_data),
        var=pd.DataFrame(data=var_data, index=["Acetaminophen", "hospital", "crazy"]),
    )


@pytest.fixture
def lab_measurements_simple_adata(obs_data, var_data):
    X = np.array([[73, 0.02, 1.00], [148, 0.25, 3.55]], dtype=np.float32)
    return AnnData(
        X=X,
        obs=pd.DataFrame(data=obs_data),
        var=pd.DataFrame(data=var_data, index=["Acetaminophen", "Acetoacetic acid", "Beryllium, toxic"]),
    )


@pytest.fixture
def lab_measurements_layer_adata(obs_data, var_data):
    X = np.array([[73, 0.02, 1.00], [148, 0.25, 3.55]], dtype=np.float32)
    return AnnData(
        X=X,
        obs=pd.DataFrame(data=obs_data),
        var=pd.DataFrame(data=var_data, index=["Acetaminophen", "Acetoacetic acid", "Beryllium, toxic"]),
        layers={"layer_copy": X},
    )


def test_obs_qc_metrics(missing_values_adata):
    obs_metrics = _obs_qc_metrics(missing_values_adata)

    assert np.array_equal(obs_metrics["missing_values_abs"].values, np.array([1, 2]))
    assert np.allclose(obs_metrics["missing_values_pct"].values, np.array([33.3333, 66.6667]))


def test_var_qc_metrics(missing_values_adata):
    var_metrics = _var_qc_metrics(missing_values_adata)

    assert np.array_equal(var_metrics["missing_values_abs"].values, np.array([1, 2, 0]))
    assert np.allclose(var_metrics["missing_values_pct"].values, np.array([50.0, 100.0, 0.0]))
    assert np.allclose(var_metrics["mean"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
    assert np.allclose(var_metrics["median"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
    assert np.allclose(var_metrics["min"].values, np.array([0.21, np.nan, 7.234]), equal_nan=True)
    assert np.allclose(var_metrics["max"].values, np.array([0.21, np.nan, 41.419998]), equal_nan=True)


def test_obs_nan_qc_metrics():
    adata = read_csv(dataset_path=f"{_TEST_PATH_ENCODE}/dataset1.csv")
    adata.X[0][4] = np.nan
    adata2 = encode(adata, encodings={"one-hot": ["clinic_day"]})
    obs_metrics = _obs_qc_metrics(adata2)
    assert obs_metrics.iloc[0].iloc[0] == 1


def test_var_nan_qc_metrics():
    adata = read_csv(dataset_path=f"{_TEST_PATH_ENCODE}/dataset1.csv")
    adata.X[0][4] = np.nan
    adata2 = encode(adata, encodings={"one-hot": ["clinic_day"]})
    var_metrics = _var_qc_metrics(adata2)
    assert var_metrics.iloc[0].iloc[0] == 1
    assert var_metrics.iloc[1].iloc[0] == 1
    assert var_metrics.iloc[2].iloc[0] == 1
    assert var_metrics.iloc[3].iloc[0] == 1
    assert var_metrics.iloc[4].iloc[0] == 1


def test_calculate_qc_metrics(missing_values_adata):
    obs_metrics, var_metrics = ep.pp.qc_metrics(missing_values_adata, inplace=True)

    assert obs_metrics is not None
    assert var_metrics is not None

    assert missing_values_adata.obs.missing_values_abs is not None
    assert missing_values_adata.var.missing_values_abs is not None


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
    assert isinstance(
        output, expected_output_type
    ), f"Output type for method '{method}' should be {expected_output_type}, got {type(output)} instead."


def test_mar_data_identification(mar_adata):
    """Test that mcar_test correctly identifies data as not MCAR (i.e., MAR or NMAR)."""
    p_value = mcar_test(mar_adata, method="little")
    assert p_value <= 0.05, "The test should significantly reject the MCAR hypothesis for MAR data."


def test_mcar_identification(mcar_adata):
    """Test that mcar_test correctly identifies data as MCAR."""
    p_value = mcar_test(mcar_adata, method="little")
    assert p_value > 0.05, "The test should not significantly accept the MCAR hypothesis for MCAR data."
