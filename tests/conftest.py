from pathlib import Path

import pytest


@pytest.fixture
def root_dir():
    return Path(__file__).resolve().parent
