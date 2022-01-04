from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from ehrapy.api.preprocessing._quality_control import _obs_qc_metrics

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
        self.test_adata = AnnData(
            X=np.array([[0.21, np.nan, 41.42], [np.nan, np.nan, 7.234]], dtype=np.float32),
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_data, index=["alive", "hospital", "crazy"]),
        )

    def test_obs_qc_metrics(self):
        obs_metrics = _obs_qc_metrics(self.test_adata)

        assert np.array_equal(obs_metrics["missing_values_abs"].values, np.array([1, 2]))
        assert np.allclose(obs_metrics["missing_values_pct"].values, np.array([0.333333, 0.666667]))

    def test_var_qc_metrics(self):
        pass
