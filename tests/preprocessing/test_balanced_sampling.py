from pathlib import Path

import pytest

import ehrapy as ep
from ehrapy.io._read import read_csv
from tests.conftest import TEST_DATA_PATH

CURRENT_DIR = Path(__file__).parent


@pytest.fixture
def edata_mini():
    return read_csv(f"{TEST_DATA_PATH}/encode/dataset1.csv", columns_obs_only=["clinic_day"])


def test_balanced_sampling_basic(edata_mini):
    # no key
    with pytest.raises(TypeError):
        ep.pp.balanced_sample(edata_mini)

    # invalid key
    with pytest.raises(ValueError):
        ep.pp.balanced_sample(edata_mini, key="non_existing_column")

    # invalid method
    with pytest.raises(ValueError):
        ep.pp.balanced_sample(edata_mini, key="clinic_day", method="non_existing_method")

    # undersampling
    adata_sampled = ep.pp.balanced_sample(edata_mini, key="clinic_day", method="RandomUnderSampler", copy=True)
    assert adata_sampled.n_obs == 4
    assert adata_sampled.obs.clinic_day.value_counts().min() == adata_sampled.obs.clinic_day.value_counts().max()

    # oversampling
    adata_sampled = ep.pp.balanced_sample(edata_mini, key="clinic_day", method="RandomOverSampler", copy=True)
    assert adata_sampled.n_obs == 8
    assert adata_sampled.obs.clinic_day.value_counts().min() == adata_sampled.obs.clinic_day.value_counts().max()

    # undersampling, no copy
    edata_mini_for_undersampling = edata_mini.copy()
    output = ep.pp.balanced_sample(
        edata_mini_for_undersampling, key="clinic_day", method="RandomUnderSampler", copy=False
    )
    assert output is None
    assert edata_mini_for_undersampling.n_obs == 4
    assert (
        edata_mini_for_undersampling.obs.clinic_day.value_counts().min()
        == edata_mini_for_undersampling.obs.clinic_day.value_counts().max()
    )

    # oversampling, no copy
    edata_mini_for_oversampling = edata_mini.copy()
    output = ep.pp.balanced_sample(
        edata_mini_for_oversampling, key="clinic_day", method="RandomOverSampler", copy=False
    )
    assert output is None
    assert edata_mini_for_oversampling.n_obs == 8
    assert (
        edata_mini_for_oversampling.obs.clinic_day.value_counts().min()
        == edata_mini_for_oversampling.obs.clinic_day.value_counts().max()
    )
