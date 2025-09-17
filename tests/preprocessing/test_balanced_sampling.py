from pathlib import Path

import pytest

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent


def test_balanced_sampling_basic(encode_ds_1_edata):
    encode_ds_1_edata.obs["clinic_day"] = list(encode_ds_1_edata[:, ["clinic_day"]].X.flatten())

    # no key
    with pytest.raises(TypeError):
        ep.pp.balanced_sample(encode_ds_1_edata)

    # invalid key
    with pytest.raises(ValueError):
        ep.pp.balanced_sample(encode_ds_1_edata, key="non_existing_column")

    # invalid method
    with pytest.raises(ValueError):
        ep.pp.balanced_sample(encode_ds_1_edata, key="clinic_day", method="non_existing_method")

    # undersampling
    adata_sampled = ep.pp.balanced_sample(encode_ds_1_edata, key="clinic_day", method="RandomUnderSampler", copy=True)
    assert adata_sampled.n_obs == 4
    assert adata_sampled.obs.clinic_day.value_counts().min() == adata_sampled.obs.clinic_day.value_counts().max()

    # oversampling
    adata_sampled = ep.pp.balanced_sample(encode_ds_1_edata, key="clinic_day", method="RandomOverSampler", copy=True)
    assert adata_sampled.n_obs == 8
    assert adata_sampled.obs.clinic_day.value_counts().min() == adata_sampled.obs.clinic_day.value_counts().max()

    # undersampling, no copy
    edata_mini_for_undersampling = encode_ds_1_edata.copy()
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
    edata_mini_for_oversampling = encode_ds_1_edata.copy()
    output = ep.pp.balanced_sample(
        edata_mini_for_oversampling, key="clinic_day", method="RandomOverSampler", copy=False
    )
    assert output is None
    assert edata_mini_for_oversampling.n_obs == 8
    assert (
        edata_mini_for_oversampling.obs.clinic_day.value_counts().min()
        == edata_mini_for_oversampling.obs.clinic_day.value_counts().max()
    )
