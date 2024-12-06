from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from anndata import AnnData
from matplotlib.testing.compare import compare_images

import ehrapy as ep
from ehrapy.io import read_csv

if TYPE_CHECKING:
    import os

    from matplotlib.figure import Figure

TEST_DATA_PATH = Path(__file__).parent / "data"


@pytest.fixture
def root_dir():
    return Path(__file__).resolve().parent


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)

@pytest.fixture
def mimic_2():
    adata = ep.dt.mimic_2()
    return adata

@pytest.fixture
def mimic_2_encoded():
    adata = ep.dt.mimic_2(encoded=True)
    return adata


@pytest.fixture
def mimic_2_10():
    mimic_2_10 = ep.dt.mimic_2()[:10]

    return mimic_2_10


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


@pytest.fixture
def adata_mini():
    return read_csv(f"{TEST_DATA_PATH}/dataset1.csv", columns_obs_only=["glucose", "weight", "disease", "station"])


@pytest.fixture
def adata_move_obs_num() -> AnnData:
    return read_csv(TEST_DATA_PATH / "io/dataset_move_obs_num.csv")


@pytest.fixture
def adata_move_obs_mix() -> AnnData:
    return read_csv(TEST_DATA_PATH / "io/dataset_move_obs_mix.csv")


@pytest.fixture
def impute_num_adata() -> AnnData:
    adata = read_csv(dataset_path=f"{TEST_DATA_PATH}/imputation/test_impute_num.csv")
    return adata


@pytest.fixture
def impute_adata() -> AnnData:
    adata = read_csv(dataset_path=f"{TEST_DATA_PATH}/imputation/test_impute.csv")
    return adata


@pytest.fixture
def impute_iris_adata() -> AnnData:
    adata = read_csv(dataset_path=f"{TEST_DATA_PATH}/imputation/test_impute_iris.csv")
    return adata


@pytest.fixture
def impute_titanic_adata():
    adata = read_csv(dataset_path=f"{TEST_DATA_PATH}/imputation/test_impute_titanic.csv")
    return adata


@pytest.fixture
def encode_ds_1_adata() -> AnnData:
    adata = read_csv(dataset_path=f"{TEST_DATA_PATH}/encode/dataset1.csv")
    return adata


@pytest.fixture
def encode_ds_2_adata() -> AnnData:
    adata = read_csv(dataset_path=f"{TEST_DATA_PATH}/encode/dataset2.csv")
    return adata


# simplified from https://github.com/scverse/scanpy/blob/main/scanpy/tests/conftest.py
@pytest.fixture
def check_same_image(tmp_path):
    def check_same_image(
        fig: Figure,
        base_path: Path | os.PathLike,
        *,
        tol: float,
    ) -> None:
        expected = Path(base_path).parent / (Path(base_path).name + "_expected.png")
        if not Path(expected).is_file():
            raise OSError(f"No expected output found at {expected}.")
        actual = tmp_path / "actual.png"

        fig.savefig(actual, dpi=80)

        result = compare_images(expected, actual, tol=tol, in_decorator=True)

        if result is None:
            return None

        raise AssertionError(result)

    return check_same_image


def asarray(a):
    import numpy as np

    return np.asarray(a)


def as_dense_dask_array(a):
    import dask.array as da

    return da.asarray(a)


ARRAY_TYPES = (asarray, as_dense_dask_array)
