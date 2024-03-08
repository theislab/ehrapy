import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from deep_translator.exceptions import TooManyRequests

from ehrapy.tools.nlp._translators import Translator

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_nlp"

deepl_token: str = os.environ.get("DEEPL_TOKEN")
microsoft_token: str = os.environ.get("MICROSOFT_TOKEN")
yandex_token: str = os.environ.get("YANDEX_TOKEN")


@pytest.mark.parametrize(
    "flavour",
    ["deepl", "googletranslate", "microsoft", "yandex"],  # "mymemory" currently unavailable
)  # "libre" temporarily removed
class TestTranslator:
    def setup_translator(self, flavour):
        target = "en-us" if flavour == "deepl" else "en"
        if flavour == "deepl":
            token = microsoft_token
        elif flavour == "microsoft":
            token = deepl_token
        elif flavour == "yandex":
            token = yandex_token
        else:
            token = None
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
        )

    def test_text_translation(self, flavour):
        flavour_token = f"{flavour}_token"
        if flavour_token in list(globals()) and globals()[flavour_token] is None:
            pytest.skip(
                f"Skipping {flavour} translation tests. Require {flavour.upper()}_TOKEN as environment variable"
            )

        self.setup_translator(flavour)
        assert self.translator.flavour == flavour
        try:
            result = self.translator.translate_text("Ich mag Züge.")
        except TooManyRequests:
            pytest.skip("Request limit reached")
        if flavour == "libre":
            print(result)
        assert pd.Series([result]).isin(["I like trains.", "I like moves."]).any()

    def test_translate_obs_column(self, flavour):
        flavour_token = f"{flavour}_token"
        if flavour_token in list(globals()) and globals()[flavour_token] is None:
            pytest.skip(
                f"Skipping {flavour} translation tests. Require {flavour.upper()}_TOKEN as environment variable"
            )

        self.setup_translator(flavour)
        try:
            self.translator.translate_obs_column(
                self.test_adata,
                columns="Krankheit",
                translate_column_name=True,
                inplace=True,
            )
        except TooManyRequests:
            pytest.skip("Request limit reached")
        assert self.test_adata.obs.columns.str.lower().isin(["disease", "illness", "health"]).any()
        assert self.test_adata.obs.melt()["value"].str.lower().isin(["cancer"]).any()

    def test_translate_var_column(self, flavour):
        flavour_token = f"{flavour}_token"
        if flavour_token in list(globals()) and globals()[flavour_token] is None:
            pytest.skip(
                f"Skipping {flavour} translation tests. Require {flavour.upper()}_TOKEN as environment variable"
            )

        try:
            self.setup_translator(flavour)
            self.translator.translate_var_column(
                self.test_adata,
                columns="Krankheit",
                translate_column_name=True,
                inplace=True,
            )
        except TooManyRequests:
            pytest.skip("Request limit reached")
        assert self.test_adata.var.columns.str.lower().isin(["disease", "illness", "health"]).any()
        assert self.test_adata.var.melt()["value"].str.lower().isin(["cancer"]).any()

    def test_translate_X_column(self, flavour):
        flavour_token = f"{flavour}_token"
        if flavour_token in list(globals()) and globals()[flavour_token] is None:
            pytest.skip(
                f"Skipping {flavour} translation tests. Require {flavour.upper()}_TOKEN as environment variable"
            )

        self.setup_translator(flavour)
        try:
            self.translator.translate_X_column(
                self.test_adata,
                columns="Krankheit",
                translate_column_name=True,
            )
        except TooManyRequests:
            pytest.skip("Request limit reached")
        assert pd.Series(self.test_adata.X[0]).str.lower().isin(["tumor"]).any()
        assert pd.Series(self.test_adata.X[1]).str.lower().isin(["cancer"]).any()

        assert self.test_adata.var_names.str.lower().isin(["disease", "illness", "prädisposition"]).any()
