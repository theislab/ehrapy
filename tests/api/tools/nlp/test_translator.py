import os
from pathlib import Path

import pytest

from ehrapy.api.tools.nlp import DeepL

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_nlp"

deepl_token: str = os.environ.get("DEEPL_TOKEN")

if deepl_token is None:
    pytest.skip(
        "Skipping DeepL translation tests. Require DEEPL_TOKEN as environment variable", allow_module_level=True
    )


class TestDeepL:
    def test_authentication(self):
        translator = DeepL(deepl_token)
        assert translator is not None
        translator.authenticate(deepl_token)

    def test_text_translation(self):
        translator = DeepL(deepl_token)
        result = translator.translate_text("Ich mag Züge.", target_language="EN-US")
        assert result.text == "I like trains."
