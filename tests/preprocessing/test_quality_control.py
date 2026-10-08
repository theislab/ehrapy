import json

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import CATEGORICAL_TAG, DEFAULT_TEM_LAYER_NAME, FEATURE_TYPE_KEY, NUMERIC_TAG
from ehrdata.io import read_csv
from testing.fast_array_utils import Flags

import ehrapy as ep
from ehrapy.preprocessing._encoding import encode
from ehrapy.preprocessing._quality_control import mcar_test
from tests.conftest import TEST_DATA_PATH, forbid_dask_compute

_TEST_PATH_ENCODE = f"{TEST_DATA_PATH}/encode"

_BLOB_KWARGS = {"n_centers": 1, "cluster_std": 1.0, "base_timepoints": 1}

_SCENARIOS_LITTLE = {
    "mcar_small": {"n_obs": 100, "n_vars": 10, "missing_rate": 0.10, "seed": 42},
    "mar_small": {"n_obs": 100, "n_vars": 10, "missing_pct": 0.10, "seed": 42},
    "mcar_medium_high_missing": {"n_obs": 900, "n_vars": 50, "missing_rate": 0.50, "seed": 7},
}
_SCENARIO_TTEST = {"n_obs": 200, "n_vars": 8, "missing_pct": 0.20, "seed": 99}


def _make_mcar_edata(*, n_obs, n_vars, missing_rate, seed):
    return ed.dt.ehrdata_blobs(
        n_observations=n_obs, n_variables=n_vars, missing_values=missing_rate, random_state=seed, **_BLOB_KWARGS
    )


def _make_mar_edata(*, n_obs, n_vars, missing_pct, seed):
    edata = ed.dt.ehrdata_blobs(
        n_observations=n_obs, n_variables=n_vars, missing_values=0.0, random_state=seed, **_BLOB_KWARGS
    )
    X = np.asarray(edata.X, dtype=float)
    X = X[:, :, 0] if X.ndim == 3 else X.copy()
    X[X[:, 0] < np.percentile(X[:, 0], missing_pct * 100), -1] = np.nan
    edata.X = X
    return edata


def _build_little_scenario(name):
    cfg = _SCENARIOS_LITTLE[name]
    return _make_mcar_edata(**cfg) if "missing_rate" in cfg else _make_mar_edata(**cfg)


def test_qc_metrics_vanilla(missing_values_edata):
    edata = missing_values_edata
    modification_copy = edata.copy()

    ep.pp.qc_metrics(edata)
    obs_metrics, var_metrics = edata.obs, edata.var

    assert np.array_equal(obs_metrics["missing_values_abs"].values, np.array([1, 2]))
    assert np.allclose(obs_metrics["missing_values_pct"].values, np.array([33.3333, 66.6667]))
    assert np.allclose(obs_metrics["entropy_of_missingness"].values, np.array([0.9183, 0.9183]))

    assert np.array_equal(var_metrics["missing_values_abs"].values, np.array([1, 2, 0]))
    assert np.allclose(var_metrics["missing_values_pct"].values, np.array([50.0, 100.0, 0.0]))
    assert np.allclose(var_metrics["entropy_of_missingness"].values, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(var_metrics["mean"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
    assert np.allclose(var_metrics["median"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
    assert np.allclose(var_metrics["min"].values, np.array([0.21, np.nan, 7.234]), equal_nan=True)
    assert np.allclose(var_metrics["max"].values, np.array([0.21, np.nan, 41.419998]), equal_nan=True)
    assert (~var_metrics["iqr_outliers"]).all()

    # check that none of the columns were modified
    for key in modification_copy.obs.keys():
        assert np.array_equal(modification_copy.obs[key], edata.obs[key])
    for key in modification_copy.var.keys():
        assert np.array_equal(modification_copy.var[key], edata.var[key])


def test_qc_metrics_vanilla_advanced(missing_values_edata):
    edata = missing_values_edata

    edata.var["feature_type"] = ["numeric", "numeric", "categorical"]
    modification_copy = edata.copy()
    ep.pp.qc_metrics(edata)
    obs_metrics, var_metrics = edata.obs, edata.var

    assert np.array_equal(obs_metrics["missing_values_abs"].values, np.array([1, 2]))
    assert np.allclose(obs_metrics["missing_values_pct"].values, np.array([33.3333, 66.6667]))
    assert np.array_equal(obs_metrics["unique_values_abs"].values, np.array([1, 1]))
    assert np.allclose(obs_metrics["unique_values_ratio"].values, np.array([100.0, 100.0]))
    assert np.allclose(obs_metrics["entropy_of_missingness"].values, np.array([0.9183, 0.9183]))

    assert np.array_equal(var_metrics["missing_values_abs"].values, np.array([1, 2, 0]))
    assert np.allclose(var_metrics["missing_values_pct"].values, np.array([50.0, 100.0, 0.0]))
    assert np.allclose(var_metrics["unique_values_abs"].values, np.array([np.nan, np.nan, 2.0]), equal_nan=True)
    assert np.allclose(var_metrics["unique_values_ratio"].values, np.array([np.nan, np.nan, 100.0]), equal_nan=True)
    assert np.allclose(var_metrics["entropy_of_missingness"].values, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(var_metrics["mean"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
    assert np.allclose(var_metrics["median"].values, np.array([0.21, np.nan, 24.327]), equal_nan=True)
    assert np.allclose(var_metrics["min"].values, np.array([0.21, np.nan, 7.234]), equal_nan=True)
    assert np.allclose(var_metrics["max"].values, np.array([0.21, np.nan, 41.419998]), equal_nan=True)
    assert np.allclose(var_metrics["coefficient_of_variation"].values, np.array([0.0, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(var_metrics["is_constant"].values, np.array([1, 0, np.nan]), equal_nan=True)
    assert np.allclose(var_metrics["constant_variable_ratio"].values, np.array([50.0, 50.0, 50.0]), equal_nan=True)
    assert np.allclose(var_metrics["range_ratio"].values, np.array([0.0, np.nan, np.nan]), equal_nan=True)
    assert (~var_metrics["iqr_outliers"]).all()

    # check that none of the columns were modified
    for key in modification_copy.obs.keys():
        assert np.array_equal(modification_copy.obs[key], edata.obs[key])
    for key in modification_copy.var.keys():
        assert np.array_equal(modification_copy.var[key], edata.var[key])


def test_qc_metrics_3d_vanilla(edata_mini_3D_missing_values):
    edata = edata_mini_3D_missing_values[:, :4].copy()
    modification_copy = edata.copy()

    ep.pp.qc_metrics(edata, layer=DEFAULT_TEM_LAYER_NAME)
    obs_metrics, var_metrics = edata.obs, edata.var

    assert np.array_equal(obs_metrics["missing_values_abs"].values, np.array([1, 0, 1, 1]))
    assert np.allclose(obs_metrics["missing_values_pct"].values, np.array([12.5, 0.0, 12.5, 12.5]))
    assert np.allclose(obs_metrics["entropy_of_missingness"].values, np.array([0.54356, 0, 0.54356, 0.54356]))

    assert np.array_equal(
        var_metrics["missing_values_abs"].values,
        np.array(
            [
                0,
                1,
                2,
                0,
            ]
        ),
    )
    assert np.allclose(var_metrics["missing_values_pct"].values, np.array([0.0, 12.5, 25.0, 0.0]))
    assert np.allclose(var_metrics["entropy_of_missingness"].values, np.array([0, 0.54356, 0.811278, 0]))
    assert np.allclose(var_metrics["mean"].values, np.array([144.5, 79.0, 78.16667, 1.25]))
    assert np.allclose(var_metrics["median"].values, np.array([144.5, 79.0, 76.5, 1.0]))
    assert np.allclose(
        var_metrics["standard_deviation"].values,
        np.array([5.12347538, 1.30930734, 18.16972451, 0.96824584]),
        equal_nan=True,
    )
    assert np.allclose(var_metrics["min"].values, np.array([138.0, 77.0, 56.0, 0.0]))
    assert np.allclose(var_metrics["max"].values, np.array([151.0, 81.0, 110.0, 3.0]))
    assert (~var_metrics["iqr_outliers"]).all()

    # check that none of the columns were modified
    for key in modification_copy.obs.keys():
        assert np.array_equal(modification_copy.obs[key], edata.obs[key])
    for key in modification_copy.var.keys():
        assert np.array_equal(modification_copy.var[key], edata.var[key])


def test_qc_metrics_3d_vanilla_advanced(edata_mini_3D_missing_values):
    edata = edata_mini_3D_missing_values[:, :4].copy()
    edata.var["feature_type"] = ["numeric", "numeric", "numeric", "categorical"]
    modification_copy = edata.copy()

    ep.pp.qc_metrics(edata, layer=DEFAULT_TEM_LAYER_NAME)
    obs_metrics, var_metrics = edata.obs, edata.var

    assert np.array_equal(obs_metrics["missing_values_abs"].values, np.array([1, 0, 1, 1]))
    assert np.allclose(obs_metrics["missing_values_pct"].values, np.array([12.5, 0.0, 12.5, 12.5]))
    assert np.allclose(obs_metrics["entropy_of_missingness"].values, np.array([0.54356, 0, 0.54356, 0.54356]))
    assert np.array_equal(obs_metrics["unique_values_abs"].values, np.array([2, 2, 2, 2]))
    assert np.allclose(obs_metrics["unique_values_ratio"].values, np.array([100.0, 100.0, 100.0, 100.0]))

    assert np.array_equal(
        var_metrics["missing_values_abs"].values,
        np.array(
            [
                0,
                1,
                2,
                0,
            ]
        ),
    )
    assert np.allclose(var_metrics["missing_values_pct"].values, np.array([0.0, 12.5, 25.0, 0.0]))
    assert np.allclose(var_metrics["entropy_of_missingness"].values, np.array([0, 0.54356, 0.811278, 0]))
    assert np.allclose(var_metrics["mean"].values, np.array([144.5, 79.0, 78.16667, 1.25]))
    assert np.allclose(var_metrics["median"].values, np.array([144.5, 79.0, 76.5, 1.0]))
    assert np.allclose(
        var_metrics["standard_deviation"].values,
        np.array([5.12347538, 1.30930734, 18.16972451, 0.96824584]),
        equal_nan=True,
    )
    assert np.allclose(var_metrics["min"].values, np.array([138.0, 77.0, 56.0, 0.0]))
    assert np.allclose(var_metrics["max"].values, np.array([151.0, 81.0, 110.0, 3.0]))
    assert np.allclose(var_metrics["unique_values_abs"].values, np.array([np.nan, np.nan, np.nan, 4.0]), equal_nan=True)
    assert np.allclose(
        var_metrics["unique_values_ratio"].values, np.array([np.nan, np.nan, np.nan, 50.0]), equal_nan=True
    )
    assert np.allclose(
        var_metrics["coefficient_of_variation"].values,
        np.array([0.03545658, 0.01657351, 0.2324485, np.nan]),
        equal_nan=True,
    )
    assert np.array_equal(var_metrics["is_constant"].values, np.array([0, 0, 0, np.nan]), equal_nan=True)
    assert np.allclose(var_metrics["constant_variable_ratio"].values, np.array([0, 0, 0, 0]))
    assert np.allclose(var_metrics["range_ratio"].values, np.array([8.9965, 5.0633, 69.0832, np.nan]), equal_nan=True)
    assert (~var_metrics["iqr_outliers"]).all()

    # check that none of the columns were modified
    for key in modification_copy.obs.keys():
        assert np.array_equal(modification_copy.obs[key], edata.obs[key])
    for key in modification_copy.var.keys():
        assert np.array_equal(modification_copy.var[key], edata.var[key])


def test_qc_metrics_heterogeneous_columns():
    mtx = np.array([[11, "a"], [True, 22]], dtype=object)

    edata = ed.EHRData(shape=(2, 2), layers={"tem_data": mtx})
    with pytest.raises(ValueError, match="Mixed or unsupported"):
        ep.pp.qc_metrics(edata, layer="tem_data")


def test_qc_metrics_encoded_uses_original_values():
    edata = read_csv(f"{_TEST_PATH_ENCODE}/dataset1.csv")
    edata.X[0][4] = np.nan
    edata = encode(edata, encodings={"one-hot": ["clinic_day"]})
    X_before = edata.X.copy()

    ep.pp.qc_metrics(edata)

    np.testing.assert_array_equal(edata.X, X_before)
    assert edata.obs["missing_values_abs"].iloc[0] == 1
    encoded = edata.var_names.str.startswith("ehrapycat_clinic_day")
    assert (edata.var.loc[encoded, "missing_values_abs"] == 1).all()
    assert (edata.var.loc[encoded, "unique_values_abs"] == 4).all()
    assert edata.var.loc[encoded, ["mean", "median", "min", "max"]].isna().all().all()
    assert not edata.var.loc[encoded, "iqr_outliers"].any()


def _array_type_data(rng: np.random.Generator, ndim: int) -> np.ndarray:
    """Values with many zeros, missing values, an outlier, and all-NaN, constant zero, constant and categorical variables."""
    shape = (30, 6) if ndim == 2 else (30, 6, 3)
    X = np.where(rng.random(shape) < 0.5, 0.0, rng.gamma(2, size=shape))
    X[rng.random(shape) < 0.2] = np.nan
    X[:, 0] = np.nan
    X[:, 1] = 0.0
    X[:, 2] = 3.0
    X[:, 3] = np.where(np.isnan(X[:, 3]), np.nan, rng.integers(0, 3, size=X[:, 3].shape))
    X[0, 4] = 50.0
    return X


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("extended", [False, True])
def test_qc_metrics_array_types(array_type, ndim, extended, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    X = _array_type_data(rng, ndim)

    def make_edata(X):
        var = pd.DataFrame({"qc": [False, False, True, False, False, False]}, index=[f"var{i}" for i in range(6)])
        if extended:
            var[FEATURE_TYPE_KEY] = [NUMERIC_TAG] * 3 + [CATEGORICAL_TAG] + [NUMERIC_TAG] * 2
        return ed.EHRData(X=X, var=var)

    expected = ep.pp.qc_metrics(make_edata(X), qc_vars=["qc"], copy=True)
    edata = make_edata(array_type(X))

    if array_type.flags & Flags.Sparse and array_type.flags & Flags.Dask:
        with pytest.raises(NotImplementedError):
            ep.pp.qc_metrics(edata, qc_vars=["qc"])
        return

    with forbid_dask_compute(allowed=1):
        result = ep.pp.qc_metrics(edata, qc_vars=["qc"], copy=True)

    assert isinstance(result.X, array_type.cls)
    pd.testing.assert_frame_equal(result.obs, expected.obs)
    pd.testing.assert_frame_equal(result.var, expected.var)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_qc_metrics_object_array_types(array_type, edata_mini_3D_missing_values):
    edata = edata_mini_3D_missing_values
    expected = ep.pp.qc_metrics(edata, layer=DEFAULT_TEM_LAYER_NAME, copy=True)
    edata.layers[DEFAULT_TEM_LAYER_NAME] = array_type(edata.layers[DEFAULT_TEM_LAYER_NAME])

    with forbid_dask_compute(allowed=1):
        result = ep.pp.qc_metrics(edata, layer=DEFAULT_TEM_LAYER_NAME, copy=True)

    assert isinstance(result.layers[DEFAULT_TEM_LAYER_NAME], array_type.cls)
    pd.testing.assert_frame_equal(result.obs, expected.obs)
    pd.testing.assert_frame_equal(result.var, expected.var)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_qc_metrics_encoded_array_types(array_type):
    edata = read_csv(f"{_TEST_PATH_ENCODE}/dataset1.csv")
    edata.X[0][4] = np.nan
    edata = encode(edata, encodings={"one-hot": ["clinic_day"]})
    expected = ep.pp.qc_metrics(edata, copy=True)
    edata.X = array_type(edata.X)

    if array_type.flags & Flags.Sparse:
        with pytest.raises(NotImplementedError):
            ep.pp.qc_metrics(edata)
        return

    with forbid_dask_compute(allowed=1):
        result = ep.pp.qc_metrics(edata, copy=True)

    assert isinstance(result.X, array_type.cls)
    pd.testing.assert_frame_equal(result.obs, expected.obs)
    pd.testing.assert_frame_equal(result.var, expected.var)


@pytest.mark.parametrize("copy", [False, True])
def test_calculate_qc_metrics(missing_values_edata, copy):
    result = ep.pp.qc_metrics(missing_values_edata, copy=copy)

    if copy:
        assert "missing_values_abs" not in missing_values_edata.obs
        assert "missing_values_abs" not in missing_values_edata.var
        missing_values_edata = result
    else:
        assert result is None
    assert "missing_values_abs" in missing_values_edata.obs
    assert "missing_values_abs" in missing_values_edata.var


def _make_lab_edata(n_obs: int = 20, seed: int = 0) -> ed.EHRData:
    """Create a small synthetic EHRData for lab measurement QC tests."""
    rng = np.random.default_rng(seed)
    data = np.column_stack(
        [
            rng.normal(5.0, 0.5, n_obs),  # potassium — tight cluster
            rng.normal(140.0, 5.0, n_obs),  # sodium
        ]
    ).astype(float)
    # Inject one obvious outlier in potassium
    data[0, 0] = 99.0
    edata = ed.EHRData(X=data, var=pd.DataFrame(index=["potassium", "sodium"]))
    return edata


def test_qc_lab_measurements_flags_and_scores():
    """Basic check: flags and scores appear in obs and have the right shape."""
    edata = _make_lab_edata()
    ep.pp.qc_lab_measurements(edata, var_names=["potassium", "sodium"])

    assert "potassium_outlier" in edata.obs.columns
    assert "potassium_score" in edata.obs.columns
    assert "sodium_outlier" in edata.obs.columns
    assert "sodium_score" in edata.obs.columns
    assert len(edata.obs["potassium_outlier"]) == edata.n_obs
    assert edata.obs["potassium_outlier"].dtype == bool


def test_qc_lab_measurements_outlier_detected():
    """The injected extreme value should be flagged as an outlier."""
    edata = _make_lab_edata()
    ep.pp.qc_lab_measurements(edata, var_names=["potassium"], method="quantile")
    # Index 0 has value 99, far outside the normal distribution
    assert edata.obs["potassium_outlier"].iloc[0]
    # Most other values should be within the reference interval
    assert edata.obs["potassium_outlier"].iloc[1:].sum() < edata.n_obs - 1


def test_qc_lab_measurements_score_direction():
    """High values should produce positive scores (z-score / IQR distance)."""
    edata = _make_lab_edata()
    ep.pp.qc_lab_measurements(edata, var_names=["potassium"], score_type="zscore")
    scores = edata.obs["potassium_score"].values
    # The extreme value at index 0 (99.0) must have the highest score in the column
    assert scores[0] == scores.max()
    assert scores[0] > 0


def test_qc_lab_measurements_add_flag_false():
    """With add_flag=False the flag column must not be created."""
    edata = _make_lab_edata()
    ep.pp.qc_lab_measurements(edata, var_names=["potassium"], add_flag=False)
    assert "potassium_outlier" not in edata.obs.columns
    assert "potassium_score" in edata.obs.columns


def test_qc_lab_measurements_add_score_false():
    """With add_score=False the score column must not be created."""
    edata = _make_lab_edata()
    ep.pp.qc_lab_measurements(edata, var_names=["potassium"], add_score=False)
    assert "potassium_score" not in edata.obs.columns
    assert "potassium_outlier" in edata.obs.columns


def test_qc_lab_measurements_methods():
    """All four methods should run and produce flags of the correct dtype."""
    edata_base = _make_lab_edata(n_obs=50)
    for method in ("quantile", "iqr", "zscore", "modified_zscore"):
        edata = edata_base.copy()
        ep.pp.qc_lab_measurements(edata, var_names=["potassium"], method=method)
        assert edata.obs["potassium_outlier"].dtype == bool, f"method={method}"


def test_qc_lab_measurements_score_types():
    """All three score types should run and produce finite floats for non-NaN inputs."""
    edata_base = _make_lab_edata(n_obs=50)
    for score_type in ("zscore", "iqr_distance", "percentile"):
        edata = edata_base.copy()
        ep.pp.qc_lab_measurements(edata, var_names=["potassium"], score_type=score_type)
        scores = edata.obs["potassium_score"]
        assert np.isfinite(scores).all(), f"score_type={score_type}"


def test_qc_lab_measurements_groupby():
    """Scores are computed relative to each group's own distribution.

    Two non-overlapping groups (M≈5, F≈50) without any injected extreme values.
    Without groupby, M values appear as extreme negatives relative to the combined
    mean ≈ 27.5, so their mean |z-score| is large.  With groupby each group is
    scored against itself, so within-group scores should be centred near zero.
    """
    rng = np.random.default_rng(42)
    n = 100
    values = np.concatenate([rng.normal(5.0, 0.5, n), rng.normal(50.0, 0.5, n)])
    sex = ["M"] * n + ["F"] * n
    edata = ed.EHRData(
        X=values[:, None].astype(float),
        var=pd.DataFrame(index=["potassium"]),
        obs=pd.DataFrame({"sex": sex}),
    )

    # Without groupby: M values look wildly below the combined mean
    edata_global = edata.copy()
    ep.pp.qc_lab_measurements(edata_global, var_names=["potassium"], score_type="zscore")
    m_abs_score_global = np.abs(edata_global.obs["potassium_score"].iloc[:n].mean())

    # With groupby: M values are scored against M peers, scores centre near 0
    edata_grouped = edata.copy()
    ep.pp.qc_lab_measurements(edata_grouped, var_names=["potassium"], groupby="sex", score_type="zscore")
    m_abs_score_grouped = np.abs(edata_grouped.obs["potassium_score"].iloc[:n].mean())

    # Stratification must bring group-relative scores much closer to zero
    assert m_abs_score_grouped < m_abs_score_global / 5


def test_qc_lab_measurements_groupby_invalid_col():
    edata = _make_lab_edata()
    with pytest.raises(ValueError, match="groupby columns not found"):
        ep.pp.qc_lab_measurements(edata, var_names=["potassium"], groupby="nonexistent")


def test_qc_lab_measurements_invalid_var():
    edata = _make_lab_edata()
    with pytest.raises(ValueError, match="Variables not found"):
        ep.pp.qc_lab_measurements(edata, var_names=["nonexistent_var"])


def test_qc_lab_measurements_nan_handling():
    """NaN values should not be flagged as outliers; their score should be NaN."""
    edata = _make_lab_edata(n_obs=20)
    edata.X[5, 0] = np.nan
    ep.pp.qc_lab_measurements(edata, var_names=["potassium"])
    assert not edata.obs["potassium_outlier"].iloc[5]  # NaN → not flagged
    assert np.isnan(edata.obs["potassium_score"].iloc[5])


def test_qc_lab_measurements_copy():
    """copy=True must not modify the original object."""
    edata = _make_lab_edata()
    original_obs_cols = set(edata.obs.columns)
    result = ep.pp.qc_lab_measurements(edata, var_names=["potassium"], copy=True)
    assert set(edata.obs.columns) == original_obs_cols  # original untouched
    assert "potassium_outlier" in result.obs.columns


def test_qc_lab_measurements_layer():
    """Function should operate on the specified layer rather than X."""
    edata = _make_lab_edata()
    edata.layers["measurements"] = edata.X.copy()
    edata.X[:] = 0  # zero out X so any result must come from the layer
    ep.pp.qc_lab_measurements(edata, var_names=["potassium"], layer="measurements")
    assert "potassium_outlier" in edata.obs.columns


def test_qc_lab_measurements_3D_flags_any_timepoint_and_averages_scores(rng):
    X = rng.normal(5.0, 0.5, size=(20, 2, 3))
    X[0, 0, 1] = 99.0
    X[3, 1, 2] = np.nan
    edata = ed.EHRData(X=X)
    flat = ed.EHRData(X=np.moveaxis(X, 1, 2).reshape(-1, 2))

    ep.pp.qc_lab_measurements(edata)
    ep.pp.qc_lab_measurements(flat)

    assert edata.obs["0_outlier"].iloc[0]
    for var in edata.var_names:
        flags = flat.obs[f"{var}_outlier"].to_numpy().reshape(20, 3)
        scores = flat.obs[f"{var}_score"].to_numpy().reshape(20, 3)
        np.testing.assert_array_equal(edata.obs[f"{var}_outlier"], flags.any(axis=1))
        np.testing.assert_allclose(edata.obs[f"{var}_score"], np.nanmean(scores, axis=1))


def test_qc_lab_measurements_groupby_missing_values():
    edata = _make_lab_edata()
    edata.obs["sex"] = ["M", None] * 10
    with pytest.raises(ValueError, match="contains missing values"):
        ep.pp.qc_lab_measurements(edata, groupby="sex")


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("groupby", [None, "group"])
@pytest.mark.parametrize(
    ("method", "score_type"),
    [("iqr", "zscore"), ("quantile", "percentile"), ("zscore", "iqr_distance"), ("modified_zscore", "zscore")],
)
def test_qc_lab_measurements_array_types(array_type, ndim, groupby, method, score_type, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    X = _array_type_data(rng, ndim)
    obs = pd.DataFrame({"group": ["a", "b"] * 15}, index=[str(i) for i in range(30)])
    kwargs = {"method": method, "score_type": score_type, "groupby": groupby}
    expected = ep.pp.qc_lab_measurements(ed.EHRData(X=X, obs=obs), copy=True, **kwargs).obs

    with forbid_dask_compute(allowed=1):
        result = ep.pp.qc_lab_measurements(ed.EHRData(X=array_type(X), obs=obs), copy=True, **kwargs)

    assert isinstance(result.X, array_type.cls)
    pd.testing.assert_frame_equal(result.obs, expected)


def test_qc_lab_measurements_defaults_to_all_vars():
    """When var_names=None, all variables should be evaluated."""
    edata = _make_lab_edata()
    ep.pp.qc_lab_measurements(edata)
    assert "potassium_outlier" in edata.obs.columns
    assert "sodium_outlier" in edata.obs.columns


@pytest.mark.parametrize(
    "method,expected_output_type",
    [
        ("little", float),
        ("ttest", pd.DataFrame),
    ],
)
def test_mcar_test_method_output_types(mar_edata, method, expected_output_type):
    output = mcar_test(mar_edata, method=method)
    assert isinstance(output, expected_output_type)


def test_mar_data_identification(mar_edata):
    p_value = mcar_test(mar_edata, method="little")
    assert p_value <= 0.05


def test_mcar_identification(mcar_edata):
    p_value = mcar_test(mcar_edata, method="little")
    assert p_value > 0.05


def test_mcar_test_multi_timepoint_3d_raises(mcar_edata):
    with pytest.raises(ValueError, match="only supports 2D data"):
        mcar_test(mcar_edata, layer=DEFAULT_TEM_LAYER_NAME)


def test_mcar_test_ttest_detects_mar(mar_edata):
    result = mcar_test(mar_edata, method="ttest")
    assert result.shape == (mar_edata.n_vars, mar_edata.n_vars)
    p_col0_given_miss9 = result.iloc[-1, 0]
    assert not np.isnan(p_col0_given_miss9)
    assert p_col0_given_miss9 < 0.05


@pytest.fixture(params=list(_SCENARIOS_LITTLE))
def little_scenario(request):
    return request.param, _build_little_scenario(request.param)


def test_mcar_test_little_matches_pyampute_reference(little_scenario):
    name, edata = little_scenario
    with (TEST_DATA_PATH / "preprocessing/mcar_refs/little_expected.json").open(encoding="utf-8") as f:
        expected = json.load(f)[name]

    observed = mcar_test(edata, method="little")
    assert np.isclose(observed, expected, rtol=1e-2, atol=2e-3), (
        f"Mismatch for {name}: observed={observed}, expected={expected}"
    )


def test_mcar_test_ttest_matches_pyampute_reference():
    edata = _make_mar_edata(**_SCENARIO_TTEST)
    expected = pd.read_csv(TEST_DATA_PATH / "preprocessing/mcar_refs/ttest_mar_expected.csv", index_col=0)
    observed = mcar_test(edata, method="ttest")
    expected = expected.reindex(index=observed.index, columns=observed.columns)

    obs_vals = observed.to_numpy(dtype=float)
    exp_vals = expected.to_numpy(dtype=float)
    finite = np.isfinite(obs_vals) & np.isfinite(exp_vals)
    nan_match = np.isnan(obs_vals) & np.isnan(exp_vals)

    assert np.all(finite | nan_match), "NaN pattern mismatch vs pyampute reference"
    assert np.allclose(obs_vals[finite], exp_vals[finite], rtol=1e-6, atol=1e-10)


def test_mcar_test_single_timepoint_3d(mar_edata):
    edata = ed.EHRData(X=mar_edata.X[:, :, None])
    assert mcar_test(edata) == mcar_test(mar_edata)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("method", ["little", "ttest"])
def test_mcar_test_array_types(array_type, mar_edata, method):
    expected = mcar_test(mar_edata, method=method)
    edata = ed.EHRData(X=array_type(mar_edata.X))

    if array_type.cls is not np.ndarray:
        with pytest.raises(NotImplementedError):
            mcar_test(edata, method=method)
        return

    result = mcar_test(edata, method=method)

    if method == "little":
        assert result == expected
    else:
        pd.testing.assert_frame_equal(result, expected)
