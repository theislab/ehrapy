import os

import pytest


@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))
