
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from ehrdata.core.constants import CATEGORICAL_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from ehrdata.io import read_csv
from matplotlib.testing.compare import compare_images

import ehrapy as ep

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
def obs_data():
    return {
        "disease": ["cancer", "tumor"],
        "country": ["Germany", "switzerland"],
        "sex": ["male", "female"],
    }


@pytest.fixture
def var_data():
    return {
        "alive": ["yes", "no", "maybe"],
        "hospital": ["hospital 1", "hospital 2", "hospital 1"],
        "crazy": ["yes", "yes", "yes"],
    }


@pytest.fixture
def edata_feature_type_specifications():
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 2, 0],
            "feature2": ["a", "b", "c", "d"],
            "feature3": [1.0, 2.0, 3.0, 2.0],
            "feature4": [0.0, 0.3, 0.5, 4.6],
            "feature5": ["a", "b", np.nan, "d"],
            "feature6": [1.4, 0.2, np.nan, np.nan],
            "feature7": pd.to_datetime(["2021-01-01", "2024-04-16", "2021-01-03", "2067-07-02"]),
        }
    )
    edata = ed.io.from_pandas(df)

    return edata


@pytest.fixture
def missing_values_edata(obs_data, var_data):
    return ed.EHRData(
        X=np.array([[0.21, np.nan, 41.42], [np.nan, np.nan, 7.234]], dtype=np.float32),
        obs=pd.DataFrame(data=obs_data),
        var=pd.DataFrame(data=var_data, index=["Acetaminophen", "hospital", "crazy"]),
    )


@pytest.fixture
def lab_measurements_simple_edata(obs_data, var_data):
    X = np.array([[73, 0.02, 1.00], [148, 0.25, 3.55]], dtype=np.float32)
    return ed.EHRData(
        X=X,
        obs=pd.DataFrame(data=obs_data),
        var=pd.DataFrame(data=var_data, index=["Acetaminophen", "Acetoacetic acid", "Beryllium, toxic"]),
    )


@pytest.fixture
def lab_measurements_layer_edata(obs_data, var_data):
    X = np.array([[73, 0.02, 1.00], [148, 0.25, 3.55]], dtype=np.float32)
    return ed.EHRData(
        X=X,
        obs=pd.DataFrame(data=obs_data),
        var=pd.DataFrame(data=var_data, index=["Acetaminophen", "Acetoacetic acid", "Beryllium, toxic"]),
        layers={"layer_copy": X},
    )


@pytest.fixture
def mimic_2():
    edata = ed.dt.mimic_2()
    ed.infer_feature_types(edata)
    edata.layers["layer_2"] = edata.X.copy()
    return edata


@pytest.fixture
def mimic_2_encoded():
    edata = ed.dt.mimic_2()
    ed.infer_feature_types(edata)
    edata = ep.pp.encode(edata, autodetect=True)

    return edata


@pytest.fixture
def mimic_2_10():
    mimic_2_10 = ed.dt.mimic_2()[:10].copy()
    ed.infer_feature_types(mimic_2_10)
    return mimic_2_10


@pytest.fixture
def mar_edata(rng) -> ed.EHRData:
    """Generate MAR data using dependent columns."""
    data = rng.random((100, 10))
    # Assume missingness in the last column depends on the values of the first column
    missing_indicator = data[:, 0] < np.percentile(data[:, 0], 0.1 * 100)
    data[missing_indicator, -1] = np.nan  # Only last column has missing values dependent on the first column

    return ed.EHRData(data)


@pytest.fixture
def mcar_edata(rng) -> ed.EHRData:
    """Generate MCAR data by randomly sampling."""
    data = rng.random((100, 10))
    missing_indices = rng.choice(a=[False, True], size=data.shape, p=[1 - 0.1, 0.1])
    data[missing_indices] = np.nan

    return ed.EHRData(data)


@pytest.fixture
def edata_mini():
    return read_csv(f"{TEST_DATA_PATH}/dataset1.csv", columns_obs_only=["glucose", "weight", "disease", "station"])


@pytest.fixture
def edata_mini_sample():
    return read_csv(f"{TEST_DATA_PATH}/dataset1.csv", columns_obs_only=["clinic_day"])


@pytest.fixture
def edata_mini_normalization():
    return read_csv(
        f"{TEST_DATA_PATH}/dataset1.csv",
        columns_obs_only=["glucose", "weight", "disease", "station"],
    )[:8]


@pytest.fixture
def edata_mini_integers_in_X():
    adata = read_csv(
        f"{TEST_DATA_PATH}/dataset1.csv",
        columns_obs_only=["idx", "sys_bp_entry", "dia_bp_entry", "glucose", "weight", "disease", "station"],
    )
    # cast data in X to integers; pd.read generates floats generously, but want to test integer normalization
    adata.X = adata.X.astype(np.int32)
    ep.ad.infer_feature_types(adata)
    ep.ad.replace_feature_types(adata, ["in_days"], "numeric")
    return adata


@pytest.fixture
def diabetes_130_fairlearn_sample():
    edata = ed.dt.diabetes_130_fairlearn(
        columns_obs_only=[
            "race",
            "gender",
            "age",
            "readmitted",
            "readmit_binary",
            "discharge_disposition_id",
        ]
    )[:200]
    ed.infer_feature_types(edata)
    return edata


@pytest.fixture
def mimic_2_sample_serv_unit_day_icu():
    edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
    ed.infer_feature_types(edata)
    return edata


@pytest.fixture
def mimic_2_sa():
    edata = ed.dt.mimic_2()
    ed.infer_feature_types(edata)
    edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
    edata = edata[:, ["mort_day_censored", "censor_flg"]].copy()
    duration_col, event_col = "mort_day_censored", "censor_flg"

    edata.layers["layer_2"] = edata.X.copy()

    return edata, duration_col, event_col


@pytest.fixture
def edata_move_obs_num() -> ed.EHRData:
    return read_csv(TEST_DATA_PATH / "io/dataset_move_obs_num.csv")


@pytest.fixture
def edata_move_obs_mix() -> ed.EHRData:
    return read_csv(TEST_DATA_PATH / "io/dataset_move_obs_mix.csv")


@pytest.fixture
def impute_num_edata() -> ed.EHRData:
    edata = read_csv(f"{TEST_DATA_PATH}/imputation/test_impute_num.csv")
    return edata


@pytest.fixture
def impute_edata() -> ed.EHRData:
    edata = read_csv(f"{TEST_DATA_PATH}/imputation/test_impute.csv")
    return edata


@pytest.fixture
def impute_iris_edata() -> ed.EHRData:
    edata = read_csv(f"{TEST_DATA_PATH}/imputation/test_impute_iris.csv")
    return edata


@pytest.fixture
def impute_titanic_edata():
    edata = read_csv(f"{TEST_DATA_PATH}/imputation/test_impute_titanic.csv")
    return edata


@pytest.fixture
def encode_ds_1_edata() -> ed.EHRData:
    edata = read_csv(f"{TEST_DATA_PATH}/encode/dataset1.csv")
    edata.layers["layer_2"] = edata.X.copy()
    return edata


@pytest.fixture
def encode_ds_2_edata() -> ed.EHRData:
    edata = read_csv(f"{TEST_DATA_PATH}/encode/dataset2.csv")
    edata.layers["layer_2"] = edata.X.copy()
    return edata


@pytest.fixture
def edata_small_bias() -> ed.EHRData:
    rng = np.random.default_rng(seed=42)
    corr = rng.integers(0, 100, 100)
    df = pd.DataFrame(
        {
            "corr1": corr,
            "corr2": corr * 2,
            "corr3": corr * -1,
            "contin1": rng.integers(0, 20, 50).tolist() + rng.integers(20, 40, 50).tolist(),
            "cat1": [0] * 50 + [1] * 50,
            "cat2": [10] * 10 + [11] * 40 + [10] * 30 + [11] * 20,
        }
    )
    edata = ed.io.from_pandas(df)
    edata.var[FEATURE_TYPE_KEY] = [NUMERIC_TAG] * 4 + [CATEGORICAL_TAG] * 2
    return edata


@pytest.fixture
def edata_blob_small() -> ed.EHRData:
    edata = ed.dt.ehrdata_blobs(n_variables=10, n_centers=2, n_observations=50, base_timepoints=10)
    edata.layers["layer_2"] = edata.X.copy()
    edata.obs["cluster"] = edata.obs["cluster"].astype("category")
    ep.pp.neighbors(edata)
    return edata


@pytest.fixture
def edata_blob_small_3d() -> ed.EHRData:
    """EHRData with 3D R array, all numeric features."""
    n_obs = 50
    n_var = 10
    n_timestamps = 10
    rng = np.random.default_rng(seed=42)
    # Generate random numeric data for R
    R = rng.normal(loc=0, scale=1, size=(n_obs, n_var, n_timestamps)).astype(np.float32)
    # X is the mean over time for each obs/var
    X = R.mean(axis=2)
    # obs and var DataFrames
    obs = pd.DataFrame({"obs_id": [f"obs_{i}" for i in range(n_obs)]})
    var = pd.DataFrame({"feature_type": [NUMERIC_TAG] * n_var}, index=[f"var_{i}" for i in range(n_var)])
    # Create EHRData
    edata = ed.EHRData(X=X, obs=obs, var=var, R=R)
    return edata

@pytest.fixture
def adata_to_norm():
    obs_data = {"ID": ["Patient1", "Patient2", "Patient3"], "Age": [31, 94, 62]}

    X_data = np.array(
        [
            [1, 3.4, -2.0, 1.0, "A string", "A different string"],
            [2, 5.4, 5.0, 2.0, "Silly string", "A different string"],
            [2, 5.7, 3.0, np.nan, "A string", "What string?"],
        ],
        dtype=np.dtype(object),
    )
    # the "ignore" tag is used to make the column being ignored; the original test selecting a few
    # columns induces a specific ordering which is kept for now
    var_data = {
        "Feature": [
            "Integer1",
            "Numeric1",
            "Numeric2",
            "Numeric3",
            "String1",
            "String2",
        ],
        "Type": ["Integer", "Numeric", "Numeric", "Numeric", "String", "String"],
        FEATURE_TYPE_KEY: [
            CATEGORICAL_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
            "ignore",
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
        ],
    }
    adata = AnnData(
        X=X_data,
        obs=pd.DataFrame(data=obs_data),
        var=pd.DataFrame(data=var_data, index=var_data["Feature"]),
        uns=OrderedDict(),
    )

    adata = ep.pp.encode(adata, autodetect=True, encodings="label")

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


def as_dense_dask_array(a, chunk_size=1000):
    import dask.array as da

    return da.from_array(a, chunks=chunk_size)


ARRAY_TYPES = (asarray, as_dense_dask_array)
