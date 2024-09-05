from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import ehrapy as ep
from ehrapy.io._read import read_csv
from ehrapy.preprocessing._encoding import encode
from ehrapy.preprocessing._quality_control import _obs_qc_metrics, _var_qc_metrics, mcar_test, ks_test
from tests.conftest import TEST_DATA_PATH

CURRENT_DIR = Path(__file__).parent
_TEST_PATH_ENCODE = f"{TEST_DATA_PATH}/encode"


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
    assert (~var_metrics["iqr_outliers"]).all()


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
    obs_metrics, var_metrics = ep.pp.qc_metrics(missing_values_adata)

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
    assert p_value > 0.05, "The test should significantly accept the MCAR hypothesis for MCAR data."


def test_ks_same_datasets():
    """Test that ks_test correctly considers as statistically equivalent dataset if one is the other minus some removed values."""
    rng = np.random.default_rng()
    n_obs = rng.integers(1000, 10000)
    n_vars = rng.integers(3, 10)
    proba_missing = rng.uniform(0.01, 0.2)

    # Create a randomly generated AnnData
    full_adata = AnnData(pd.DataFrame(
        np.random.uniform(0.0, 100.0, size=(n_obs, n_vars)),
        columns=[f'Var{i + 1}' for i in range(n_vars)]))
    hollowed_adata = full_adata.copy() # Save it so we can compare against it later

    # Randomly remove value in some cells in one of the two datasets
    for i in range(hollowed_adata.shape[0]):
        for j in range(hollowed_adata.shape[1]):
            if rng.uniform() < proba_missing:
                hollowed_adata.X[i, j] = np.nan

    assert not ks_test(full_adata, hollowed_adata), "The KS test shouldn't conclude the datasets are statistically different"


def test_ks_different_datasets():
    """Test that ks_test correctly considers as statistically equivalent dataset if one is NOT the other minus some removed values."""
    rng = np.random.default_rng()
    n_obs = rng.integers(1000, 10000)
    n_vars = rng.integers(3, 10)
    proba_missing = rng.uniform(0.01, 0.2)

    # Create two different randomly generated AnnDatas
    full_adata = AnnData(pd.DataFrame(
        np.random.uniform(0.0, 100.0, size=(n_obs, n_vars)),
        columns=[f'Var{i + 1}' for i in range(n_vars)]))
    hollowed_adata = AnnData(pd.DataFrame(
        np.random.uniform(0.0, 100.0, size=(n_obs, n_vars)),
        columns=[f'Var{i + 1}' for i in range(n_vars)]))

    # Randomly remove value in some cells in one of the two datasets
    for i in range(hollowed_adata.shape[0]):
        for j in range(hollowed_adata.shape[1]):
            if rng.uniform() < proba_missing:
                hollowed_adata.X[i, j] = np.nan

    assert ks_test(full_adata, hollowed_adata), "The KS test shouldn't conclude the datasets are statistically equivalent"


def test_ks_differently_shaped_datasets():
    """Test that ks_test rejects two differently-shaped datasets."""
    rng = np.random.default_rng()
    n_obs = rng.integers(1000, 10000)
    n_vars = rng.integers(3, 10)

    # Create an AnnData with a given shape...
    adata1 = AnnData(pd.DataFrame(
        np.random.uniform(0.0, 100.0, size=(n_obs, n_vars)),
        columns=[f'Var{i + 1}' for i in range(n_vars)]))

    # ... And create another AnnData with a different shape
    n_vars += 1
    adata2 = AnnData(pd.DataFrame(
        np.random.uniform(0.0, 100.0, size=(n_obs, n_vars)),
        columns=[f'Var{i + 1}' for i in range(n_vars)]))

    # It should fail!
    try:
        ks_test(adata1, adata2)
        thrown = False
    except ValueError:
        thrown = True
    assert thrown, "The KS test should have rejected the datasets"
