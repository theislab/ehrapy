from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

import ehrapy as ep
from ehrapy.preprocessing._quality_control import _obs_qc_metrics, _var_qc_metrics

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_preprocessing"


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

        self.test_lab_measurements_simple_adata = AnnData(
            X=np.array([[73, 0.02, 0.04], [148, 0.25, 0.34]], dtype=np.float32),
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_data, index=["Acetaminophen", "Acetoacetic acid", "Acetone"]),
        )

    def test_obs_qc_metrics(self):
        obs_metrics = _obs_qc_metrics(self.test_missing_values_adata)

        assert np.array_equal(obs_metrics["missing_values_abs"].values, np.array([1, 2]))
        assert np.allclose(obs_metrics["missing_values_pct"].values, np.array([33.3333, 66.6667]))

    def test_var_qc_metrics(self):
        var_metrics = _var_qc_metrics(self.test_missing_values_adata)

        assert np.array_equal(var_metrics["missing_values_abs"].values, np.array([1, 2, 0]))
        assert np.allclose(var_metrics["missing_values_pct"].values, np.array([50.0, 100.0, 0.0]))
        assert np.allclose(var_metrics["mean"].values, np.array([np.nan, np.nan, 24.327]), equal_nan=True)
        assert np.allclose(var_metrics["median"].values, np.array([np.nan, np.nan, 24.327]), equal_nan=True)
        assert np.allclose(var_metrics["min"].values, np.array([np.nan, np.nan, 7.234]), equal_nan=True)
        assert np.allclose(var_metrics["max"].values, np.array([np.nan, np.nan, 41.419998]), equal_nan=True)

    def test_calculate_qc_metrics(self):
        obs_metrics, var_metrics = ep.pp.qc_metrics(self.test_missing_values_adata, inplace=True)

        assert obs_metrics is not None
        assert var_metrics is not None

        assert self.test_missing_values_adata.obs.missing_values_abs is not None
        assert self.test_missing_values_adata.var.missing_values_abs is not None

    def test_qc_lab_measurements_simple(self):
        expected_var_data = {  # noqa F841
            "Acetaminophen_reference_range": [True, True],
            "Acetoacetic acid": [True, False],
            "Acetone": [True, False],
        }

        print(
            ep.pp.qc_lab_measurements(
                self.test_lab_measurements_simple_adata,
                measurements=list(self.test_lab_measurements_simple_adata.var_names),
            )
        )
