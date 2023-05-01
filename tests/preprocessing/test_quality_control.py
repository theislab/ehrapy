from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
import pytest

import ehrapy as ep
from ehrapy.io._read import read_csv
from ehrapy.preprocessing._encode import encode
from ehrapy.preprocessing._quality_control import _obs_qc_metrics, _var_qc_metrics

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_preprocessing"
_TEST_PATH_ENCODE = f"{CURRENT_DIR}/test_data_encode"


class TestQualityControl:
    def setup_method(self):
        obs_data = {
            "disease": ["cancer", "tumor"],
            "country": ["Germany", "switzerland"],
            "sex": ["male", "female"],
        }
        var_data = {
            "alive": ["yes", "no", "maybe"],
            "hospital": ["hospital 1", "hospital 2", "hospital 1"],
            "crazy": ["yes", "yes", "yes"],
        }
        self.test_missing_values_adata = AnnData(
            X=np.array([[0.21, np.nan, 41.42], [np.nan, np.nan, 7.234]], dtype=np.float32),
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_data, index=["Acetaminophen", "hospital", "crazy"]),
        )

        X = np.array([[73, 0.02, 1.00], [148, 0.25, 3.55]], dtype=np.float32)
        self.test_lab_measurements_simple_adata = AnnData(
            X=X,
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_data, index=["Acetaminophen", "Acetoacetic acid", "Beryllium, toxic"]),
        )
        self.test_lab_measurements_layer_adata = AnnData(
            X=X,
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_data, index=["Acetaminophen", "Acetoacetic acid", "Beryllium, toxic"]),
            layers={"layer_copy": X},
        )

    def test_obs_qc_metrics(self):
        obs_metrics = _obs_qc_metrics(self.test_missing_values_adata)

        assert np.array_equal(obs_metrics["missing_values_abs"].values, np.array([1, 2]))
        assert np.allclose(obs_metrics["missing_values_pct"].values, np.array([33.3333, 66.6667]))

    def test_var_qc_metrics(self):
        var_metrics = _var_qc_metrics(self.test_missing_values_adata)

        assert np.array_equal(var_metrics["missing_values_abs"].values, np.array([1, 2, 0]))
        assert np.allclose(var_metrics["missing_values_pct"].values, np.array([50.0, 100.0, 0.0]))
        assert np.allclose(var_metrics["mean"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
        assert np.allclose(var_metrics["median"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
        assert np.allclose(var_metrics["min"].values, np.array([0.21, np.nan, 7.234]), equal_nan=True)
        assert np.allclose(var_metrics["max"].values, np.array([0.21, np.nan, 41.419998]), equal_nan=True)

    def test_obs_nan_qc_metrics(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH_ENCODE}/dataset1.csv")
        adata.X[0][4] = np.nan
        adata2 = encode(adata, encodings={"one_hot_encoding": ["clinic_day"]})
        obs_metrics = _obs_qc_metrics(adata2)
        assert obs_metrics.iloc[0][0] == 1

    def test_var_nan_qc_metrics(self):
        adata = read_csv(dataset_path=f"{_TEST_PATH_ENCODE}/dataset1.csv")
        adata.X[0][4] = np.nan
        adata2 = encode(adata, encodings={"one_hot_encoding": ["clinic_day"]})
        var_metrics = _var_qc_metrics(adata2)
        assert var_metrics.iloc[0][0] == 1
        assert var_metrics.iloc[1][0] == 1
        assert var_metrics.iloc[2][0] == 1
        assert var_metrics.iloc[3][0] == 1
        assert var_metrics.iloc[4][0] == 1

    def test_calculate_qc_metrics(self):
        obs_metrics, var_metrics = ep.pp.qc_metrics(self.test_missing_values_adata, inplace=True)

        assert obs_metrics is not None
        assert var_metrics is not None

        assert self.test_missing_values_adata.obs.missing_values_abs is not None
        assert self.test_missing_values_adata.var.missing_values_abs is not None

    def test_qc_lab_measurements_simple(self):
        expected_obs_data = pd.Series(
            data={
                "Acetaminophen normal": [True, True],
                "Acetoacetic acid normal": [True, False],
                "Beryllium, toxic normal": [False, True],
            }
        )

        ep.pp.qc_lab_measurements(
            self.test_lab_measurements_simple_adata,
            measurements=list(self.test_lab_measurements_simple_adata.var_names),
            unit="SI",
        )

        assert list(self.test_lab_measurements_simple_adata.obs["Acetaminophen normal"]) == (
            expected_obs_data["Acetaminophen normal"]
        )
        assert list(self.test_lab_measurements_simple_adata.obs["Acetoacetic acid normal"]) == (
            expected_obs_data["Acetoacetic acid normal"]
        )
        assert list(self.test_lab_measurements_simple_adata.obs["Beryllium, toxic normal"]) == (
            expected_obs_data["Beryllium, toxic normal"]
        )

    def test_qc_lab_measurements_simple_layer(self):
        expected_obs_data = pd.Series(
            data={
                "Acetaminophen normal": [True, True],
                "Acetoacetic acid normal": [True, False],
                "Beryllium, toxic normal": [False, True],
            }
        )

        ep.pp.qc_lab_measurements(
            self.test_lab_measurements_layer_adata,
            measurements=list(self.test_lab_measurements_layer_adata.var_names),
            unit="SI",
            layer="layer_copy",
        )

        assert list(self.test_lab_measurements_layer_adata.obs["Acetaminophen normal"]) == (
            expected_obs_data["Acetaminophen normal"]
        )
        assert list(self.test_lab_measurements_layer_adata.obs["Acetoacetic acid normal"]) == (
            expected_obs_data["Acetoacetic acid normal"]
        )
        assert list(self.test_lab_measurements_layer_adata.obs["Beryllium, toxic normal"]) == (
            expected_obs_data["Beryllium, toxic normal"]
        )

    def test_qc_lab_measurements_age(self):
        # TODO
        pass

    def test_qc_lab_measurements_sex(self):
        # TODO
        pass

    def test_qc_lab_measurements_ethnicity(self):
        # TODO
        pass

    def test_qc_lab_measurements_multiple_measurements(self):
        data = pd.DataFrame(np.array([[100, 98], [162, 107]]), columns=["oxygen saturation", "glucose"], index=[0, 1])

        with pytest.raises(ValueError):
            adata = ep.ad.df_to_anndata(data)
            ep.pp.qc_lab_measurements(adata, measurements=["oxygen saturation", "glucose"], unit="SI")
