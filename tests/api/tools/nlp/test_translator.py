import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from ehrapy.api.tools.nlp._translators import Translator

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_nlp"

deepl_token: str = os.environ.get("DEEPL_TOKEN")


@pytest.mark.parametrize("flavour", ["deepl", "googletranslate"])
class TestTranslator:
    def setup_translator(self, flavour):
        if deepl_token is None and flavour == "deepl":
            print("Skipping DeepL translation tests. Require DEEPL_TOKEN as environment variable")
            assert True

        target = "en-us" if flavour == "deepl" else "en"
        token = deepl_token if flavour == "deepl" else None
        self.translator = Translator(flavour, target=target, token=token)

    def setup_method(self, method):
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

    def test_text_translation(self, flavour):
        self.setup_translator(flavour)
        assert self.translator.flavour == flavour
        result = self.translator.translate_text("Ich mag Züge.")
        assert result == "I like trains."

    def test_translate_obs_column(self, flavour):
        self.setup_translator(flavour)
        self.translator.translate_obs_column(
            self.test_adata,
            columns="Krankheit",
            translate_column_name=True,
            inplace=True,
        )
        assert self.test_adata.obs.columns.str.lower().isin(["disease", "illness"]).any()
        assert self.test_adata.obs.melt()["value"].str.lower().isin(["cancer"]).any()

    def test_translate_var_column(self, flavour):
        self.setup_translator(flavour)
        self.translator.translate_var_column(
            self.test_adata,
            columns="Krankheit",
            translate_column_name=True,
            inplace=True,
        )
        assert self.test_adata.var.columns.str.lower().isin(["disease", "illness"]).any()
        assert self.test_adata.var.melt()["value"].str.lower().isin(["cancer"]).any()

    def test_translate_X_column(self, flavour):
        self.setup_translator(flavour)
        self.translator.translate_X_column(
            self.test_adata,
            columns="Krankheit",
            translate_column_name=True,
        )
        assert pd.Series(self.test_adata.X[0]).str.lower().isin(["tumor"]).any()
        assert pd.Series(self.test_adata.X[1]).str.lower().isin(["cancer"]).any()

        assert self.test_adata.var_names.str.lower().isin(["disease", "illness"]).any()
