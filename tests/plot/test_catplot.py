from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import ehrapy as ep
from ehrapy.io._read import read_csv

CURRENT_DIR = Path(__file__).parent
_TEST_DATA_PATH = f"{CURRENT_DIR.parent}/test_data"
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


@pytest.fixture
def adata_mini():
    return read_csv(f"{_TEST_DATA_PATH}/dataset1.csv", columns_obs_only=["glucose", "weight", "disease", "station"])


def test_catplot_vanilla(adata_mini, check_same_image):
    fig = ep.pl.catplot(adata_mini, jitter=False)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/catplot_vanilla",
        tol=2e-1,
    )
