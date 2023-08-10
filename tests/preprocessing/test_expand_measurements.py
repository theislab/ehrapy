import numpy as np
import pytest
from anndata import AnnData
from pandas import DataFrame

from ehrapy.preprocessing import expand_measurements

np.random.seed(42)


@pytest.fixture
def adata_to_expand():
    row_ids = ["pat1", "pat1", "pat1", "pat2", "pat2", "pat3"]
    measurement1 = np.random.choice([0, 1], size=6)
    measurement2 = np.random.uniform(0, 20, size=6)
    measurement3 = np.random.uniform(0, 20, size=6)
    data_dict = {"measurement1": measurement1, "measurement2": measurement2, "measurement3": measurement3}
    data_df = DataFrame(data_dict, index=row_ids)
    adata = AnnData(X=data_df)

    return adata


def test_all_measures(adata_to_expand):
    transformed_adata = expand_measurements(
        adata_to_expand,
    )

    assert transformed_adata.shape == (3, 9)  # (3 patients, 3 measurements * 3 statistics)


def test_var_names_subset(adata_to_expand):
    transformed_adata = expand_measurements(
        adata_to_expand,
        var_names=["measurement1", "measurement2"],
    )

    assert transformed_adata.shape == (3, 6)  # (3 patients, 2 measurements * 3 statistics)


def test_measurements_subset(adata_to_expand):
    transformed_adata = expand_measurements(adata_to_expand, statistics=["min"])

    assert transformed_adata.shape == (3, 3)  # (3 patients, 3 measurements * 1 statistics)
