from pathlib import Path

import pytest

import ehrapy as ep
from ehrapy.io._read import read_csv

CURRENT_DIR = Path(__file__).parent
_TEST_DATA_PATH = f"{CURRENT_DIR}/test_data_encode"


@pytest.fixture
def adata_mini():
    return read_csv(f"{_TEST_DATA_PATH}/dataset1.csv", columns_obs_only=["clinic_day"])


def test_sampling_basic(adata_mini):
    # no key
    with pytest.raises(TypeError):
        ep.pp.sample(adata_mini)

    # invalid key
    with pytest.raises(ValueError):
        ep.pp.sample(adata_mini, key="non_existing_column")

    # invalid method
    with pytest.raises(ValueError):
        ep.pp.sample(adata_mini, key="clinic_day", method="non_existing_method")

    # undersampling
    adata_sampled = ep.pp.sample(adata_mini, key="clinic_day", method="RandomUnderSampler")
    assert adata_sampled.n_obs == 4
    assert adata_sampled.obs.clinic_day.value_counts().min() == adata_sampled.obs.clinic_day.value_counts().max()

    # oversampling
    adata_sampled = ep.pp.sample(adata_mini, key="clinic_day", method="RandomOverSampler")
    assert adata_sampled.n_obs == 8
    assert adata_sampled.obs.clinic_day.value_counts().min() == adata_sampled.obs.clinic_day.value_counts().max()
