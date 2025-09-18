from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent


@pytest.mark.parametrize("sparse_input", [False, True])
def test_balanced_sampling_basic(encode_ds_1_edata, sparse_input):
    encode_ds_1_edata.obs["clinic_day"] = list(encode_ds_1_edata[:, ["clinic_day"]].X.flatten())
    encode_ds_1_edata = encode_ds_1_edata[:, :-1].copy()
    if sparse_input:
        edata_sparse = encode_ds_1_edata.copy()
        edata_sparse.X = sp.csr_matrix(np.asarray(edata_sparse.X, dtype=np.float64))
        encode_ds_1_edata = edata_sparse

    # no key
    with pytest.raises(TypeError):
        ep.pp.sample(encode_ds_1_edata, balanced=True)

    # invalid key
    with pytest.raises(ValueError):
        ep.pp.sample(encode_ds_1_edata, balanced=True, balanced_key="non_existing_column")

    # invalid method
    with pytest.raises(ValueError):
        ep.pp.sample(encode_ds_1_edata, balanced=True, balanced_key="clinic_day", balanced_method="non_existing_method")

    # undersampling
    edata_sampled = ep.pp.sample(
        encode_ds_1_edata, balanced=True, balanced_key="clinic_day", balanced_method="RandomUnderSampler", copy=True
    )
    assert edata_sampled.n_obs == 4
    assert edata_sampled.obs.clinic_day.value_counts().min() == edata_sampled.obs.clinic_day.value_counts().max()

    # oversampling
    edata_sampled = ep.pp.sample(
        encode_ds_1_edata, balanced=True, balanced_key="clinic_day", balanced_method="RandomOverSampler", copy=True
    )
    assert edata_sampled.n_obs == 8
    assert edata_sampled.obs.clinic_day.value_counts().min() == edata_sampled.obs.clinic_day.value_counts().max()

    # undersampling, no copy
    encode_ds_1_edata_for_undersampling = encode_ds_1_edata.copy()
    output = ep.pp.sample(
        encode_ds_1_edata_for_undersampling,
        balanced=True,
        balanced_key="clinic_day",
        balanced_method="RandomUnderSampler",
        copy=False,
    )
    assert output is None
    assert encode_ds_1_edata_for_undersampling.n_obs == 4

    # oversampling, no copy
    encode_ds_1_edata_for_oversampling = encode_ds_1_edata.copy()
    output = ep.pp.sample(
        encode_ds_1_edata_for_oversampling,
        balanced=True,
        balanced_key="clinic_day",
        balanced_method="RandomOverSampler",
        copy=False,
    )
    assert output is None
