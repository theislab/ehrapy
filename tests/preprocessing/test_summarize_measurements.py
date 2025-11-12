import numpy as np
import pytest
from anndata import AnnData
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from pandas import DataFrame

from ehrapy.preprocessing import summarize_measurements


@pytest.fixture
def adata_to_expand(rng):
    row_ids = ["pat1", "pat1", "pat1", "pat2", "pat2", "pat3"]
    measurement1 = rng.choice([0, 1], size=6)
    measurement2 = rng.uniform(0, 20, size=6)
    measurement3 = rng.uniform(0, 20, size=6)
    data_dict = {"measurement1": measurement1, "measurement2": measurement2, "measurement3": measurement3}
    data_df = DataFrame(data_dict, index=row_ids)
    adata = AnnData(X=data_df)

    return adata


def test_all_statistics(adata_to_expand):
    transformed_adata = summarize_measurements(
        adata_to_expand,
    )

    assert transformed_adata.shape == (3, 9)  # (3 patients, 3 measurements * 3 statistics)
    assert np.allclose(
        transformed_adata[:, "measurement2_min"].X.reshape(-1), np.array([1.883547, 15.222794, 2.5622725])
    )
    assert np.allclose(
        transformed_adata[:, "measurement2_max"].X.reshape(-1), np.array([19.512447, 15.721286, 2.5622725])
    )
    assert np.allclose(
        transformed_adata[:, "measurement2_mean"].X.reshape(-1), np.array([11.781118, 15.47204, 2.5622725])
    )


def test_var_names_subset(adata_to_expand):
    transformed_adata = summarize_measurements(
        adata_to_expand,
        var_names=["measurement1", "measurement2"],
    )

    assert transformed_adata.shape == (3, 6)  # (3 patients, 2 measurements * 3 statistics)


def test_statistics_subset(adata_to_expand):
    transformed_adata = summarize_measurements(adata_to_expand, statistics=["min"])

    assert transformed_adata.shape == (3, 3)  # (3 patients, 3 measurements * 1 statistics)


def test_summarize_measurements_3D_edata(edata_blob_small):
    summarize_measurements(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        summarize_measurements(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)
