import numpy as np

from ehrapy._utils_data import _are_ndarrays_equal, _is_val_missing


def test_are_ndarrays_equal(impute_num_adata):
    impute_num_adata_copy = impute_num_adata.copy()
    assert _are_ndarrays_equal(impute_num_adata.X, impute_num_adata_copy.X)
    impute_num_adata_copy.X[0, 0] = 42.0
    assert not _are_ndarrays_equal(impute_num_adata.X, impute_num_adata_copy.X)


def test_is_val_missing(impute_num_adata):
    assert np.array_equal(
        _is_val_missing(impute_num_adata.X),
        np.array([[False, False, True], [False, False, False], [True, False, False], [False, False, True]]),
    )
