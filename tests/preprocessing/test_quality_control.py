from pathlib import Path

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from ehrdata.io import read_csv

import ehrapy as ep
from ehrapy.preprocessing._encoding import encode
from ehrapy.preprocessing._quality_control import _compute_obs_metrics, _compute_var_metrics, mcar_test
from tests.conftest import ARRAY_TYPES_NONNUMERIC, TEST_DATA_PATH, as_dense_dask_array

CURRENT_DIR = Path(__file__).parent
_TEST_PATH_ENCODE = f"{TEST_DATA_PATH}/encode"


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_qc_metrics_vanilla(array_type, missing_values_edata):
    edata = missing_values_edata
    edata.X = array_type(edata.X)
    modification_copy = edata.copy()

    obs_metrics, var_metrics = ep.pp.qc_metrics(edata)

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


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NONNUMERIC)
def test_qc_metrics_vanilla_advanced(array_type, missing_values_edata):
    edata = missing_values_edata

    edata.var["feature_type"] = ["numeric", "numeric", "categorical"]
    edata.X = array_type(missing_values_edata.X)
    modification_copy = edata.copy()
    obs_metrics, var_metrics = ep.pp.qc_metrics(edata)

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

    obs_metrics, var_metrics = ep.pp.qc_metrics(edata, layer=DEFAULT_TEM_LAYER_NAME)

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

    obs_metrics, var_metrics = ep.pp.qc_metrics(edata, layer=DEFAULT_TEM_LAYER_NAME)

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


@pytest.mark.parametrize(
    "array_type, expected_error",
    [
        (np.array, None),
        (as_dense_dask_array, None),
        # can't test sparse matrices because they don't support string values
    ],
)
def test_obs_qc_metrics_array_types(array_type, expected_error):
    edata = read_csv(f"{_TEST_PATH_ENCODE}/dataset1.csv")
    edata.X = array_type(edata.X)
    mtx = edata.X
    if expected_error:
        with pytest.raises(expected_error):
            _compute_obs_metrics(mtx, edata)


def test_obs_nan_qc_metrics():
    edata = read_csv(f"{_TEST_PATH_ENCODE}/dataset1.csv")
    edata.X[0][4] = np.nan
    edata2 = encode(edata, encodings={"one-hot": ["clinic_day"]})
    mtx = edata2.X
    obs_metrics = _compute_obs_metrics(mtx, edata2)
    assert obs_metrics.iloc[0].iloc[0] == 1


@pytest.mark.parametrize(
    "array_type, expected_error",
    [
        (np.array, None),
        (as_dense_dask_array, None),
        # can't test sparse matrices because they don't support string values
    ],
)
def test_var_qc_metrics_array_types(array_type, expected_error):
    edata = read_csv(f"{_TEST_PATH_ENCODE}/dataset1.csv")
    edata.X = array_type(edata.X)
    mtx = edata.X
    if expected_error:
        with pytest.raises(expected_error):
            _compute_var_metrics(mtx, edata)


def test_var_encoding_mode_does_not_modify_original_matrix():
    edata = read_csv(f"{_TEST_PATH_ENCODE}/dataset1.csv")
    edata2 = encode(edata, encodings={"one-hot": ["clinic_day"]})
    mtx_copy = edata2.X.copy()
    _compute_var_metrics(edata2.X, edata2)
    assert np.array_equal(mtx_copy, edata2.X)


def test_var_nan_qc_metrics():
    edata = read_csv(f"{_TEST_PATH_ENCODE}/dataset1.csv")
    edata.X[0][4] = np.nan
    edata2 = encode(edata, encodings={"one-hot": ["clinic_day"]})
    mtx = edata2.X
    var_metrics = _compute_var_metrics(mtx, edata2)
    assert var_metrics.iloc[0].iloc[0] == 1
    assert var_metrics.iloc[1].iloc[0] == 1
    assert var_metrics.iloc[2].iloc[0] == 1
    assert var_metrics.iloc[3].iloc[0] == 1
    assert var_metrics.iloc[4].iloc[0] == 1


def test_calculate_qc_metrics(missing_values_edata):
    obs_metrics, var_metrics = ep.pp.qc_metrics(missing_values_edata)

    assert obs_metrics is not None
    assert var_metrics is not None

    assert missing_values_edata.obs.missing_values_abs is not None
    assert missing_values_edata.var.missing_values_abs is not None


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
    ep.pp.qc_lab_measurements(edata, vars=["potassium", "sodium"])

    assert "potassium_outlier" in edata.obs.columns
    assert "potassium_score" in edata.obs.columns
    assert "sodium_outlier" in edata.obs.columns
    assert "sodium_score" in edata.obs.columns
    assert len(edata.obs["potassium_outlier"]) == edata.n_obs
    assert edata.obs["potassium_outlier"].dtype == bool


def test_qc_lab_measurements_outlier_detected():
    """The injected extreme value should be flagged as an outlier."""
    edata = _make_lab_edata()
    ep.pp.qc_lab_measurements(edata, vars=["potassium"], method="quantile")
    # Index 0 has value 99, far outside the normal distribution
    assert edata.obs["potassium_outlier"].iloc[0]
    # Most other values should be within the reference interval
    assert edata.obs["potassium_outlier"].iloc[1:].sum() < edata.n_obs - 1


def test_qc_lab_measurements_score_direction():
    """High values should produce positive scores (z-score / IQR distance)."""
    edata = _make_lab_edata()
    ep.pp.qc_lab_measurements(edata, vars=["potassium"], score_type="zscore")
    scores = edata.obs["potassium_score"].values
    # The extreme value at index 0 (99.0) must have the highest score in the column
    assert scores[0] == scores.max()
    assert scores[0] > 0


def test_qc_lab_measurements_add_flag_false():
    """With add_flag=False the flag column must not be created."""
    edata = _make_lab_edata()
    ep.pp.qc_lab_measurements(edata, vars=["potassium"], add_flag=False)
    assert "potassium_outlier" not in edata.obs.columns
    assert "potassium_score" in edata.obs.columns


def test_qc_lab_measurements_add_score_false():
    """With add_score=False the score column must not be created."""
    edata = _make_lab_edata()
    ep.pp.qc_lab_measurements(edata, vars=["potassium"], add_score=False)
    assert "potassium_score" not in edata.obs.columns
    assert "potassium_outlier" in edata.obs.columns


def test_qc_lab_measurements_methods():
    """All four methods should run and produce flags of the correct dtype."""
    edata_base = _make_lab_edata(n_obs=50)
    for method in ("quantile", "iqr", "zscore", "modified_zscore"):
        edata = edata_base.copy()
        ep.pp.qc_lab_measurements(edata, vars=["potassium"], method=method)
        assert edata.obs["potassium_outlier"].dtype == bool, f"method={method}"


def test_qc_lab_measurements_score_types():
    """All three score types should run and produce finite floats for non-NaN inputs."""
    edata_base = _make_lab_edata(n_obs=50)
    for score_type in ("zscore", "iqr_distance", "percentile"):
        edata = edata_base.copy()
        ep.pp.qc_lab_measurements(edata, vars=["potassium"], score_type=score_type)
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
    ep.pp.qc_lab_measurements(edata_global, vars=["potassium"], score_type="zscore")
    m_abs_score_global = edata_global.obs["potassium_score"].iloc[:n].abs().mean()

    # With groupby: M values are scored against M peers, scores centre near 0
    edata_grouped = edata.copy()
    ep.pp.qc_lab_measurements(edata_grouped, vars=["potassium"], groupby="sex", score_type="zscore")
    m_abs_score_grouped = edata_grouped.obs["potassium_score"].iloc[:n].abs().mean()

    # Stratification must bring group-relative scores much closer to zero
    assert m_abs_score_grouped < m_abs_score_global / 5


def test_qc_lab_measurements_groupby_invalid_col():
    edata = _make_lab_edata()
    with pytest.raises(ValueError, match="groupby columns not found"):
        ep.pp.qc_lab_measurements(edata, vars=["potassium"], groupby="nonexistent")


def test_qc_lab_measurements_invalid_var():
    edata = _make_lab_edata()
    with pytest.raises(ValueError, match="Variables not found"):
        ep.pp.qc_lab_measurements(edata, vars=["nonexistent_var"])


def test_qc_lab_measurements_nan_handling():
    """NaN values should not be flagged as outliers; their score should be NaN."""
    edata = _make_lab_edata(n_obs=20)
    edata.X[5, 0] = np.nan
    ep.pp.qc_lab_measurements(edata, vars=["potassium"])
    assert not edata.obs["potassium_outlier"].iloc[5]  # NaN → not flagged
    assert np.isnan(edata.obs["potassium_score"].iloc[5])


def test_qc_lab_measurements_copy():
    """copy=True must not modify the original object."""
    edata = _make_lab_edata()
    original_obs_cols = set(edata.obs.columns)
    result = ep.pp.qc_lab_measurements(edata, vars=["potassium"], copy=True)
    assert set(edata.obs.columns) == original_obs_cols  # original untouched
    assert "potassium_outlier" in result.obs.columns


def test_qc_lab_measurements_layer():
    """Function should operate on the specified layer rather than X."""
    edata = _make_lab_edata()
    edata.layers["measurements"] = edata.X.copy()
    edata.X[:] = 0  # zero out X so any result must come from the layer
    ep.pp.qc_lab_measurements(edata, vars=["potassium"], layer="measurements")
    assert "potassium_outlier" in edata.obs.columns


def test_qc_lab_measurements_3D_edata(edata_blob_small):
    ep.pp.qc_lab_measurements(edata_blob_small, vars=list(edata_blob_small.var_names), layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.qc_lab_measurements(edata_blob_small, vars=list(edata_blob_small.var_names), layer=DEFAULT_TEM_LAYER_NAME)


def test_qc_lab_measurements_defaults_to_all_vars():
    """When vars=None, all variables should be evaluated."""
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
    """Tests if mcar_test returns the correct output type for different methods."""
    output = mcar_test(mar_edata, method=method)
    assert isinstance(output, expected_output_type), (
        f"Output type for method '{method}' should be {expected_output_type}, got {type(output)} instead."
    )


def test_mcar_test_3D_edata(edata_blob_small):
    mcar_test(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        mcar_test(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


def test_mar_data_identification(mar_edata):
    """Test that mcar_test correctly identifies data as not MCAR (i.e., MAR or NMAR)."""
    p_value = mcar_test(mar_edata, method="little")
    assert p_value <= 0.05, "The test should significantly reject the MCAR hypothesis for MAR data."


def test_mcar_identification(mcar_edata):
    """Test that mcar_test correctly identifies data as MCAR."""
    p_value = mcar_test(mcar_edata, method="little")
    assert p_value > 0.05, "The test should not significantly accept the MCAR hypothesis for MCAR data."
