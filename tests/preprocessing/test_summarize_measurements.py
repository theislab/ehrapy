import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from fast_array_utils.conv import to_dense
from pandas import DataFrame
from testing.fast_array_utils import Flags

from ehrapy.preprocessing import summarize_measurements
from tests.conftest import forbid_dask_compute

STATISTICS = ["min", "max", "mean", "median", "first", "last", "count", "std"]


@pytest.fixture
def edata_to_expand(rng):
    row_ids = ["pat1", "pat1", "pat1", "pat2", "pat2", "pat3"]
    measurement1 = rng.choice([0, 1], size=6)
    measurement2 = rng.uniform(0, 20, size=6)
    measurement3 = rng.uniform(0, 20, size=6)
    data_dict = {"measurement1": measurement1, "measurement2": measurement2, "measurement3": measurement3}
    data_df = DataFrame(data_dict, index=row_ids)
    edata = ed.EHRData(X=data_df)

    return edata


def test_all_statistics(edata_to_expand):
    transformed_edata = summarize_measurements(
        edata_to_expand,
    )

    assert transformed_edata.shape == (3, 9, 1)  # (3 patients, 3 measurements * 3 statistics)
    assert np.allclose(
        transformed_edata[:, "measurement2_min"].X.reshape(-1), np.array([1.883547, 15.222794, 2.5622725])
    )
    assert np.allclose(
        transformed_edata[:, "measurement2_max"].X.reshape(-1), np.array([19.512447, 15.721286, 2.5622725])
    )
    assert np.allclose(
        transformed_edata[:, "measurement2_mean"].X.reshape(-1), np.array([11.781118, 15.47204, 2.5622725])
    )


def test_var_names_subset(edata_to_expand):
    transformed_edata = summarize_measurements(
        edata_to_expand,
        var_names=["measurement1", "measurement2"],
    )

    assert transformed_edata.shape == (3, 6, 1)  # (3 patients, 2 measurements * 3 statistics)


def test_statistics_subset(edata_to_expand):
    transformed_edata = summarize_measurements(edata_to_expand, statistics=["min"])

    assert transformed_edata.shape == (3, 3, 1)  # (3 patients, 3 measurements * 1 statistics)


def _non_missing_at(x: np.ndarray, position: int) -> float:
    x = x[~np.isnan(x)]
    return x[position] if len(x) else np.nan


def _slope(x: np.ndarray) -> float:
    observed = ~np.isnan(x)
    return np.polyfit(np.flatnonzero(observed), x[observed], 1)[0] if observed.sum() > 1 else np.nan


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_summarize_measurements_3D(rng):
    X = rng.normal(size=(5, 3, 4))
    X[rng.random(X.shape) < 0.3] = np.nan
    X[0, 0] = np.nan
    obs = pd.DataFrame({"group": list("aabbc")}, index=[f"pat{i}" for i in range(5)])
    edata = ed.EHRData(shape=(5, 3), obs=obs, layers={DEFAULT_TEM_LAYER_NAME: X})

    summary = summarize_measurements(edata, layer=DEFAULT_TEM_LAYER_NAME, statistics=[*STATISTICS, "slope"])

    expected = {
        "min": np.nanmin(X, axis=2),
        "max": np.nanmax(X, axis=2),
        "mean": np.nanmean(X, axis=2),
        "median": np.nanmedian(X, axis=2),
        "first": np.apply_along_axis(_non_missing_at, 2, X, 0),
        "last": np.apply_along_axis(_non_missing_at, 2, X, -1),
        "count": (~np.isnan(X)).sum(axis=2),
        "std": np.nanstd(X, axis=2, ddof=1),
        "slope": np.apply_along_axis(_slope, 2, X),
    }
    assert summary.shape == (5, 3 * len(expected), 1)
    assert summary.var_names.tolist() == [f"{var}_{stat}" for var in edata.var_names for stat in expected]
    pd.testing.assert_frame_equal(summary.obs, edata.obs)
    for stat, values in expected.items():
        np.testing.assert_allclose(summary[:, [f"{var}_{stat}" for var in edata.var_names]].X, values)


def test_summarize_measurements_unknown_var(edata_blob_small):
    with pytest.raises(KeyError, match="Variables not found"):
        summarize_measurements(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME, var_names=["unknown"])


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
def test_summarize_measurements_array_types(array_type, ndim, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    shape = (12, 3) if ndim == 2 else (6, 3, 4)
    X = np.where(rng.random(shape) < 0.5, 0, rng.normal(size=shape))
    X[rng.random(shape) < 0.2] = np.nan
    X[:, 1] = 2.0
    X[:, 2] = np.nan
    obs = pd.DataFrame(index=list("bacbabccabda") if ndim == 2 else list("abcdef"))

    def make_edata(X):
        return ed.EHRData(shape=shape[:2], obs=obs, layers={DEFAULT_TEM_LAYER_NAME: X})

    statistics = STATISTICS if ndim == 2 else [*STATISTICS, "slope"]
    expected = summarize_measurements(make_edata(X), layer=DEFAULT_TEM_LAYER_NAME, statistics=statistics)
    edata = make_edata(array_type(X))

    with forbid_dask_compute():
        result = summarize_measurements(edata, layer=DEFAULT_TEM_LAYER_NAME, statistics=statistics)

    assert type(result.X) is type(edata.layers[DEFAULT_TEM_LAYER_NAME])
    if array_type.flags & Flags.Dask:
        assert type(result.X._meta) is type(edata.layers[DEFAULT_TEM_LAYER_NAME]._meta)
    pd.testing.assert_index_equal(result.obs_names, expected.obs_names)
    pd.testing.assert_index_equal(result.var_names, expected.var_names)
    np.testing.assert_allclose(to_dense(result.X, to_cpu_memory=True), expected.X, equal_nan=True)


def test_summarize_measurements_2D_count_std(edata_to_expand):
    summary = summarize_measurements(edata_to_expand, var_names=["measurement2"], statistics=["count", "std"])

    values = edata_to_expand[:, "measurement2"].X.ravel()
    np.testing.assert_allclose(summary.X[:, 0].ravel(), [3, 2, 1])
    np.testing.assert_allclose(
        summary.X[:, 1].ravel(), [np.std(values[:3], ddof=1), np.std(values[3:5], ddof=1), np.nan]
    )


def test_summarize_measurements_slope_needs_3D(edata_to_expand):
    with pytest.raises(ValueError, match="needs 3D data"):
        summarize_measurements(edata_to_expand, statistics=["slope"])
