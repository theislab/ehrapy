from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def root_dir():
    return Path(__file__).resolve().parent


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def mar_adata(rng) -> AnnData:
    """Generate MAR data using dependent columns."""
    data = rng.random((100, 10))
    # Assume missingness in the last column depends on the values of the first column
    missing_indicator = data[:, 0] < np.percentile(data[:, 0], 0.1 * 100)
    data[missing_indicator, -1] = np.nan  # Only last column has missing values dependent on the first column

    return AnnData(data)


@pytest.fixture
def mcar_adata(rng) -> AnnData:
    """Generate MCAR data by randomly sampling."""
    data = rng.random((100, 10))
    missing_indices = np.random.choice(a=[False, True], size=data.shape, p=[1 - 0.1, 0.1])
    data[missing_indices] = np.nan

    return AnnData(data)
