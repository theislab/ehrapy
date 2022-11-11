from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_preprocessing"


class TestOutliers:
    def setup_method(self):
        obs_data = {
            "disease": ["cancer", "tumor"],
            "country": ["Germany", "switzerland"],
            "age": [17, 36],
        }
        var_data = {
            "alive": ["yes", "no", "maybe"],
            "hospital": ["hospital 1", "hospital 2", "hospital 1"],
            "crazy": ["yes", "yes", "yes"],
        }
        self.test_outliers = AnnData(
            X=np.array([[0.21, 14.52, 41.42], [5.42, 96.2, 7.234]], dtype=np.float32),
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_data, index=["Acetaminophen", "co2", "po2"]),
        )

    def test_winsorize(self):
        #ep.pp.winsorize(self.test_outliers, vars=["co2"], limits=[13, 65], copy=False)
        pass

    def test_quantile(self):
        # adata_filtered = ep.pp.filter_quantiles(self.test_outliers, vars=["feature 1, feature 2"], quantile_top=1, copy=True)
        pass
