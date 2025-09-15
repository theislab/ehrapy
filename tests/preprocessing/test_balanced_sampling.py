from pathlib import Path

import pytest

import ehrapy as ep
from ehrapy.io._read import read_csv
from tests.conftest import TEST_DATA_PATH

CURRENT_DIR = Path(__file__).parent


@pytest.fixture
def adata_mini():
    return read_csv(f"{TEST_DATA_PATH}/encode/dataset1.csv", columns_obs_only=["clinic_day"])


def test_balanced_sampling_basic(adata_mini):
    # no key
    with pytest.raises(TypeError):
        ep.pp.sample(adata_mini, balanced=True)

    # invalid key
    with pytest.raises(ValueError):
        ep.pp.sample(adata_mini, balanced=True, balanced_key="non_existing_column")

    # invalid method
    with pytest.raises(ValueError):
        ep.pp.sample(adata_mini, balanced=True, balanced_key="clinic_day", balanced_method="non_existing_method")

    # undersampling
    adata_sampled = ep.pp.sample(
        adata_mini, balanced=True, balanced_key="clinic_day", balanced_method="under", copy=True
    )
    assert adata_sampled.n_obs == 4
    assert adata_sampled.obs.clinic_day.value_counts().min() == adata_sampled.obs.clinic_day.value_counts().max()

    # oversampling
    adata_sampled = ep.pp.sample(
        adata_mini, balanced=True, balanced_key="clinic_day", balanced_method="over", copy=True
    )
    assert adata_sampled.n_obs == 8
    assert adata_sampled.obs.clinic_day.value_counts().min() == adata_sampled.obs.clinic_day.value_counts().max()

    # undersampling, no copy
    adata_mini_for_undersampling = adata_mini.copy()
    output = ep.pp.sample(
        adata_mini_for_undersampling, balanced=True, balanced_key="clinic_day", balanced_method="under", copy=False
    )
    assert output is None
    assert adata_mini_for_undersampling.n_obs == 4
    assert (
        adata_mini_for_undersampling.obs.clinic_day.value_counts().min()
        == adata_mini_for_undersampling.obs.clinic_day.value_counts().max()
    )

    # oversampling, no copy
    adata_mini_for_oversampling = adata_mini.copy()
    output = ep.pp.sample(
        adata_mini_for_oversampling, balanced=True, balanced_key="clinic_day", balanced_method="over", copy=False
    )
    assert output is None
    assert adata_mini_for_oversampling.n_obs == 8
    assert (
        adata_mini_for_oversampling.obs.clinic_day.value_counts().min()
        == adata_mini_for_oversampling.obs.clinic_day.value_counts().max()
    )
