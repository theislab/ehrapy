from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_nlp"


class TestMedCAT:
    def test_add_medcat_annotation_to_obs(self):
        # created manually a small dataset with annotations to use here
        adata = ep.io.read_csv(f"{_TEST_PATH}/dataset1.csv")
        adata.uns["medcat_annotations"] = pd.read_csv(f"{_TEST_PATH}/medcat_annotations1.csv")

        ep.tl.add_medcat_annotation_to_obs(adata, name="Diabetes")
        assert "Diabetes" in adata.obs.columns
