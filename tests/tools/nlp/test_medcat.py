from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_nlp"


class TestMedCAT:
    def setup_method(self):
        obs_data = {
            "Krankheit": ["Krebs", "Tumor"],
            "Land": ["Deutschland", "Schweiz"],
            "Geschlecht": ["männlich", "weiblich"],
        }
        var_data = {
            "Krankheit": ["Krebs", "Tumor", "Krebs"],
            "Land": ["Deutschland", "Schweiz", "Österreich"],
            "Geschlecht": ["männlich", "weiblich", "männlich"],
        }
        self.test_adata = AnnData(
            X=np.array([["Deutschland", "Zöliakie", "Tumor"], ["Frankreich", "Allergie", "Krebs"]], np.dtype(object)),
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_data, index=["Land", "Prädisposition", "Krankheit"]),
            dtype=np.dtype(object),
        )

        # patient_notes = pd.read_csv(f"{_TEST_PATH}/pt_notes.csv")
