from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import ehrdata as ed
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from ehrdata.core.constants import CATEGORICAL_TAG, DEFAULT_TEM_LAYER_NAME, FEATURE_TYPE_KEY, NUMERIC_TAG
from matplotlib.testing.compare import compare_images

import ehrapy as ep
from ehrapy._types import (
    ARRAY_TYPES_NONNUMERIC,
    ARRAY_TYPES_NUMERIC,
    ARRAY_TYPES_NUMERIC_3D_ABLE,
    as_dense_dask_array,
    asarray,
)

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
    ed.infer_feature_types(edata, output=None)
    edata.layers["layer_2"] = edata.X.copy()
    return edata


@pytest.fixture
def mimic_2_encoded():
    edata = ed.dt.mimic_2()
    ed.infer_feature_types(edata, output=None)
    edata = ep.pp.encode(edata, autodetect=True)

    return edata


@pytest.fixture
def mimic_2_10():
    mimic_2_10 = ed.dt.mimic_2()[:10].copy()
    ed.infer_feature_types(mimic_2_10, output=None)
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

    data_3d = rng.random((100, 10, 3))
    missing_indices = rng.choice(a=[False, True], size=data_3d.shape, p=[1 - 0.1, 0.1])
    data_3d[missing_indices] = np.nan

    return ed.EHRData(data, layers={DEFAULT_TEM_LAYER_NAME: data_3d})


@pytest.fixture
def edata_mini():
    return ed.io.read_csv(
        f"{TEST_DATA_PATH}/dataset1.csv", columns_obs_only=["glucose", "weight", "disease", "station"]
    )


@pytest.fixture
def edata_mini_3D_missing_values():
    tiny_mixed_array = np.array(
        [
            [[138, 139], [78, np.nan], [77, 76], [1, 2], ["A", "B"], ["Yes", np.nan]],
            [[140, 141], [80, 81], [60, 90], [0, 1], ["A", "A"], ["Yes", "Yes"]],
            [[148, 149], [77, 78], [110, np.nan], [0, 1], [np.nan, "B"], ["Yes", "Yes"]],
            [[150, 151], [79, 80], [56, np.nan], [2, 3], ["B", "B"], ["Yes", "No"]],
        ],
        dtype=object,
    )
    n_obs, n_vars, _ = tiny_mixed_array.shape
    return ed.EHRData(shape=(n_obs, n_vars), layers={DEFAULT_TEM_LAYER_NAME: tiny_mixed_array})


@pytest.fixture
def edata_mini_sample():
    return ed.io.read_csv(f"{TEST_DATA_PATH}/dataset1.csv", columns_obs_only=["clinic_day"])


@pytest.fixture
def edata_mini_normalization():
    return ed.io.read_csv(
        f"{TEST_DATA_PATH}/dataset1.csv",
        columns_obs_only=["glucose", "weight", "disease", "station"],
    )[:8]


@pytest.fixture
def edata_mini_integers_in_X():
    adata = ed.io.read_csv(
        f"{TEST_DATA_PATH}/dataset1.csv",
        columns_obs_only=["idx", "sys_bp_entry", "dia_bp_entry", "glucose", "weight", "disease", "station"],
    )
    # cast data in X to integers; pd.read generates floats generously, but want to test integer normalization
    adata.X = adata.X.astype(np.int32)
    ep.ad.infer_feature_types(adata)
    ep.ad.replace_feature_types(adata, ["in_days"], "numeric")
    return adata


@pytest.fixture
def edata_and_distances_dtw():
    """See tests/_scripts/dtw_test_reference.ipynb."""
    data = np.random.default_rng(42).integers(0, 5, (5, 2, 4))
    edata = ed.EHRData(X=None, layers={DEFAULT_TEM_LAYER_NAME: data})

    distances = np.array(
        [
            [0.0, 2.98118805, 2.44948974, 3.30277564, 2.34277886],
            [2.98118805, 0.0, 3.43649167, 3.05492646, 3.28629768],
            [2.44948974, 3.43649167, 0.0, 3.16227766, 3.31318964],
            [3.30277564, 3.05492646, 3.16227766, 0.0, 4.35228539],
            [2.34277886, 3.28629768, 3.31318964, 4.35228539, 0.0],
        ]
    )

    return edata, distances


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
def diabetes_130_fairlearn_sample_100():
    edata = ed.dt.diabetes_130_fairlearn(
        columns_obs_only=[
            "race",
            "gender",
        ]
    )[:100]

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
    return ed.io.read_csv(TEST_DATA_PATH / "io/dataset_move_obs_num.csv")


@pytest.fixture
def edata_move_obs_mix() -> ed.EHRData:
    return ed.io.read_csv(TEST_DATA_PATH / "io/dataset_move_obs_mix.csv")


@pytest.fixture
def impute_num_edata() -> ed.EHRData:
    edata = ed.io.read_csv(f"{TEST_DATA_PATH}/imputation/test_impute_num.csv")
    return edata


@pytest.fixture
def impute_edata() -> ed.EHRData:
    edata = ed.io.read_csv(f"{TEST_DATA_PATH}/imputation/test_impute.csv")
    return edata


@pytest.fixture
def impute_iris_edata() -> ed.EHRData:
    edata = ed.io.read_csv(f"{TEST_DATA_PATH}/imputation/test_impute_iris.csv")
    return edata


@pytest.fixture
def impute_titanic_edata():
    edata = ed.io.read_csv(f"{TEST_DATA_PATH}/imputation/test_impute_titanic.csv")
    return edata


@pytest.fixture
def encode_ds_1_edata() -> ed.EHRData:
    edata = ed.io.read_csv(f"{TEST_DATA_PATH}/encode/dataset1.csv")
    edata.layers["layer_2"] = edata.X.copy()
    return edata


@pytest.fixture
def encode_ds_2_edata() -> ed.EHRData:
    edata = ed.io.read_csv(f"{TEST_DATA_PATH}/encode/dataset2.csv")
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
    ep.pp.neighbors(edata)
    return edata


@pytest.fixture
def edata_blobs_timeseries_small() -> ed.EHRData:
    edata = ed.dt.ehrdata_blobs(
        n_observations=20,
        base_timepoints=15,
        cluster_std=0.5,
        n_centers=3,
        seasonality=True,
        time_shifts=True,
        variable_length=False,
        layer=DEFAULT_TEM_LAYER_NAME,
    )
    edata.layers["layer_2"] = edata.X.copy()

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
def check_same_image(tmp_path: Path):
    def check_same_image(
        fig: Figure | hv.core.overlay.Overlay | hv.element.chart.Scatter | hv.Element,
        base_path: Path | os.PathLike,
        *,
        tol: float,
    ) -> None:
        expected = Path(base_path).parent / (Path(base_path).name + "_expected.png")
        if not Path(expected).is_file():
            raise OSError(f"No expected output found at {expected}.")
        actual = tmp_path / "actual.png"

        if hasattr(fig, "savefig"):
            fig.savefig(actual, dpi=80)
        else:
            hv.save(fig, actual, backend="matplotlib", size=80)

        result = compare_images(expected, actual, tol=tol, in_decorator=True)
        if result is None:
            return None
        raise AssertionError(result)

    return check_same_image


@pytest.fixture
def clean_up_plots():
    plt.close("all")
    yield
    plt.clf()
    plt.cla()
    plt.close("all")


@pytest.fixture
def hv_backend():
    baseline = "matplotlib"
    hv.extension(baseline)

    def _set(name: str):
        hv.extension(name)
        return name

    try:
        yield _set
    finally:
        hv.extension(baseline)
