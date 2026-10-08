import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
import statsmodels
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from lifelines import (
    CoxPHFitter,
    KaplanMeierFitter,
    LogLogisticAFTFitter,
    NelsonAalenFitter,
    WeibullAFTFitter,
    WeibullFitter,
)
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_ols(mimic_2, layer):
    edata = mimic_2
    # If use layer argument, set X to None to avoid it being used
    if layer is not None:
        edata.X = None

    formula = "tco2_first ~ pco2_first"
    var_names = ["tco2_first", "pco2_first"]
    ols = ep.tl.ols(edata, var_names=var_names, formula=formula, missing="drop", layer=layer)
    s = ols.fit().params.iloc[1]
    i = ols.fit().params.iloc[0]
    assert isinstance(ols, statsmodels.regression.linear_model.OLS)
    assert 0.18857179158259973 == pytest.approx(s)
    assert 16.210859352601442 == pytest.approx(i)


def test_ols_3D(edata_blob_small):
    formula = "feature_1 ~ feature_2"
    var_names = ["feature_1", "feature_2"]
    ep.tl.ols(edata_blob_small, var_names=var_names, formula=formula, missing="drop", layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.tl.ols(edata_blob_small, var_names=var_names, formula=formula, missing="drop", layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_glm(mimic_2, layer):
    edata = mimic_2
    # If use layer argument, set X to None to avoid it being used
    if layer is not None:
        edata.X = None

    formula = "day_28_flg ~ age"
    var_names = ["day_28_flg", "age"]
    family = "Binomial"
    glm = ep.tl.glm(
        edata, var_names=var_names, formula=formula, family=family, missing="drop", as_continuous=["age"], layer=layer
    )
    Intercept = glm.fit().params.iloc[0]
    age = glm.fit().params.iloc[1]
    assert isinstance(glm, statsmodels.genmod.generalized_linear_model.GLM)
    assert -5.778006344870297 == pytest.approx(Intercept)
    assert 0.06523274132877163 == pytest.approx(age)


def test_glm_3D(edata_blob_small):
    formula = "feature_1 ~ feature_2"
    var_names = ["feature_1", "feature_2"]
    ep.tl.glm(edata_blob_small, var_names=var_names, formula=formula, missing="drop", layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.tl.glm(edata_blob_small, var_names=var_names, formula=formula, missing="drop", layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize(
    "weightings",
    [
        "wilcoxon",
        "tarone-ware",
        "peto",
        # "fleming-harrington"
    ],
)
def test_calculate_logrank_pvalue(weightings):
    durations_A = np.array([1, 2, 3], dtype=float)
    event_observed_A = np.array([1, 1, 0], dtype=int)
    durations_B = np.array([1, 2, 3, 4], dtype=float)
    event_observed_B = np.array([1, 0, 0, 1], dtype=int)

    kmf1 = KaplanMeierFitter()
    kmf1.fit(durations_A, event_observed_A)
    kmf2 = KaplanMeierFitter()
    kmf2.fit(durations_B, event_observed_B)

    results_pairwise = ep.tl.test_kmf_logrank(kmf1, kmf2, weightings=weightings)
    p_value_pairwise = results_pairwise.p_value
    assert 0 < p_value_pairwise < 1


def test_anova_glm(mimic_2):
    edata = mimic_2
    formula = "day_28_flg ~ age"
    var_names = ["day_28_flg", "age"]
    family = "Binomial"
    age_glm = ep.tl.glm(
        edata, var_names=var_names, formula=formula, family=family, missing="drop", as_continuous=["age"]
    )
    age_glm_result = age_glm.fit()
    formula = "day_28_flg ~ age + service_unit"
    var_names = ["day_28_flg", "age", "service_unit"]
    ageunit_glm = ep.tl.glm(
        edata, var_names=var_names, formula=formula, family=family, missing="drop", as_continuous=["age"]
    )
    ageunit_glm_result = ageunit_glm.fit()
    dataframe = ep.tl.anova_glm(
        age_glm_result,
        ageunit_glm_result,
        formula_1="day_28_flg ~ age",
        formula_2="day_28_flg ~ age + service_unit",
    )

    assert len(dataframe) == 2
    assert dataframe.shape == (2, 6)
    assert dataframe.iloc[1, 4] == 2
    assert pytest.approx(dataframe.iloc[1, 5], 0.1) == 0.103185


@pytest.mark.parametrize(
    "sa_function,sa_class",
    [
        (ep.tl.kaplan_meier, KaplanMeierFitter),
        (ep.tl.cox_ph, CoxPHFitter),
        (ep.tl.nelson_aalen, NelsonAalenFitter),
        (ep.tl.weibull, WeibullFitter),
        (ep.tl.weibull_aft, WeibullAFTFitter),
        (ep.tl.log_logistic_aft, LogLogisticAFTFitter),
    ],
)
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_survival_models(sa_function, sa_class, mimic_2_sa, layer):
    edata, duration_col, event_col = mimic_2_sa
    # If use layer argument, set X to None to avoid it being used
    if layer is not None:
        edata.X = None

    sa = sa_function(edata, duration_col=duration_col, event_col=event_col, key_added="test", layer=layer)

    assert isinstance(sa, sa_class)
    assert len(sa.durations) == 1776
    assert sum(sa.event_observed) == 497

    model_summary = edata.uns.get("test")
    assert model_summary is not None

    expected_attr = "event_table" if isinstance(sa, KaplanMeierFitter | NelsonAalenFitter) else "summary"
    assert model_summary.equals(getattr(sa, expected_attr))


@pytest.mark.parametrize(
    "sa_function,sa_class",
    [
        (ep.tl.kaplan_meier, KaplanMeierFitter),
        (ep.tl.cox_ph, CoxPHFitter),
        (ep.tl.nelson_aalen, NelsonAalenFitter),
        (ep.tl.weibull, WeibullFitter),
        (ep.tl.weibull_aft, WeibullAFTFitter),
        (ep.tl.log_logistic_aft, LogLogisticAFTFitter),
    ],
)
def test_survival_models_3D(sa_function, sa_class, edata_blob_small):
    duration_col = "feature_1"
    event_col = "feature_0"
    # Assigning through a subset view does not propagate on anndata >=0.13, so set X on the parent.
    X = np.asarray(edata_blob_small.X).copy()
    X[:, edata_blob_small.var_names.get_loc(duration_col)] = np.arange(len(edata_blob_small), dtype=np.int32)
    X[:, edata_blob_small.var_names.get_loc(event_col)] = 1
    edata_blob_small.X = X

    edata_blob_small.layers["layer_2"] = edata_blob_small.X.copy()

    sa_function(edata_blob_small, duration_col=duration_col, event_col=event_col, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        sa_function(edata_blob_small, duration_col=duration_col, event_col=event_col, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("method", ["average", "conditional"])
@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_cox_ph_adjusted_curves_basic(mimic_2_adjusted_sa, method, layer):
    """Results are stored in edata.uns with correct structure."""
    edata = mimic_2_adjusted_sa
    duration_col, event_col = "mort_day_censored", "censor_flg"

    if layer is not None:
        edata.X = None

    cph = ep.tl.cox_ph(
        edata,
        duration_col=duration_col,
        event_col=event_col,
        formula="sapsi_first + afib_flg",
        layer=layer,
    )
    ep.tl.cox_ph_adjusted_curves(
        edata,
        cph=cph,
        strata="aline_flg",
        duration_col=duration_col,
        event_col=event_col,
        method=method,
        n_bootstrap=10,
        key_added="test_adjusted",
        layer=layer,
    )

    assert "test_adjusted" in edata.uns
    result = edata.uns["test_adjusted"]

    # _meta is stored correctly
    assert "_meta" in result
    assert result["_meta"]["strata"] == "aline_flg"
    assert result["_meta"]["method"] == method
    assert result["_meta"]["duration_col"] == duration_col
    assert result["_meta"]["event_col"] == event_col

    # one entry per group plus _meta
    groups = [k for k in result if k != "_meta"]
    assert len(groups) == 2  # aline_flg is binary

    # each group entry has the expected keys and shapes
    for group in groups:
        entry = result[group]
        assert "times" in entry
        assert "survival" in entry

        if method == "average":
            assert entry["ci_lower"] is not None
            assert entry["ci_upper"] is not None
        else:
            assert entry["ci_lower"] is None
            assert entry["ci_upper"] is None

        assert len(entry["times"]) == len(entry["survival"]) == 100
        # survival probabilities must be in [0, 1]
        assert np.all(entry["survival"] >= 0)
        assert np.all(entry["survival"] <= 1)
        # survival must be non-increasing
        assert np.all(np.diff(entry["survival"]) <= 1e-8)


def test_cox_ph_adjusted_curves_copy(mimic_2_adjusted_sa):
    edata = mimic_2_adjusted_sa
    duration_col, event_col = "mort_day_censored", "censor_flg"
    cph = ep.tl.cox_ph(edata, duration_col=duration_col, event_col=event_col, formula="sapsi_first + afib_flg")
    kwargs = {"cph": cph, "strata": "aline_flg", "duration_col": duration_col, "event_col": event_col}

    edata_copy = ep.tl.cox_ph_adjusted_curves(edata, method="conditional", copy=True, **kwargs)
    assert "cox_ph_adjusted_curves" in edata_copy.uns
    assert "cox_ph_adjusted_curves" not in edata.uns

    assert ep.tl.cox_ph_adjusted_curves(edata, method="conditional", **kwargs) is None
    assert "cox_ph_adjusted_curves" in edata.uns


@pytest.fixture
def survival_obs_edata(rng):
    n = 200
    X = rng.normal(size=(n, 3))
    X[:5, 2] = np.nan
    obs = pd.DataFrame(
        {
            "duration": rng.exponential(10, n) + 0.1,
            "event": rng.integers(0, 2, n).astype(bool),
            "sex": rng.choice(["female", "male"], n),
            "entry": rng.uniform(0, 0.05, n),
            "weight": rng.uniform(0.5, 1.5, n),
        },
        index=[str(i) for i in range(n)],
    )
    return ed.EHRData(X=X, obs=obs, var=pd.DataFrame(index=["age", "bmi", "unused"]))


@pytest.mark.parametrize(
    "sa_function", [ep.tl.kaplan_meier, ep.tl.nelson_aalen, ep.tl.weibull, ep.tl.weibull_aft, ep.tl.log_logistic_aft]
)
def test_survival_models_obs_columns(survival_obs_edata, sa_function):
    kwargs = {} if sa_function in (ep.tl.kaplan_meier, ep.tl.nelson_aalen, ep.tl.weibull) else {"covariates": ["age"]}
    model = sa_function(survival_obs_edata, "duration", event_col="event", **kwargs)
    assert len(model.durations) == survival_obs_edata.n_obs


def test_cox_ph_only_drops_rows_missing_model_columns(survival_obs_edata):
    cph = ep.tl.cox_ph(survival_obs_edata, "duration", event_col="event", formula="age + bmi + C(sex)")
    assert len(cph.durations) == survival_obs_edata.n_obs

    cph = ep.tl.cox_ph(survival_obs_edata, "duration", event_col="event", covariates=["age", "unused"])
    assert len(cph.durations) == survival_obs_edata.n_obs - 5
    assert set(cph.params_.index) == {"age", "unused"}


def test_cox_ph_obs_only_on_3D_data(survival_obs_edata):
    edata = ed.EHRData(
        X=np.stack([survival_obs_edata.X] * 4, axis=2), obs=survival_obs_edata.obs, var=survival_obs_edata.var
    )
    cph = ep.tl.cox_ph(edata, "duration", event_col="event", formula="C(sex)")
    assert set(cph.params_.index) == {"C(sex)[T.male]"}


def test_kaplan_meier_entry_and_weights(survival_obs_edata):
    kmf = ep.tl.kaplan_meier(survival_obs_edata, "duration", event_col="event", entry_col="entry", weights_col="weight")
    assert np.allclose(kmf.entry, survival_obs_edata.obs["entry"])
    assert np.allclose(kmf.weights, survival_obs_edata.obs["weight"])


def test_kaplan_meier_without_event_col(survival_obs_edata):
    kmf = ep.tl.kaplan_meier(survival_obs_edata, "duration")
    assert kmf.event_observed.all()


def _survival_edata(X) -> ed.EHRData:
    return ed.EHRData(X=X, var=pd.DataFrame(index=["duration", "event", "a", "b"]))


def _survival_data(rng: np.random.Generator) -> np.ndarray:
    return np.column_stack(
        [rng.integers(1, 50, 60), rng.integers(0, 2, 60), rng.standard_normal(60), np.where(rng.random(60) < 0.5, 0, 1)]
    ).astype(float)


def _raises_or_runs(array_type, func, /, *args, **kwargs):
    if array_type.flags & Flags.Sparse and array_type.flags & Flags.Dask:
        with pytest.raises(NotImplementedError):
            func(*args, **kwargs)
        return None
    with forbid_dask_compute(allowed=1):
        return func(*args, **kwargs)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize(
    "sa_function",
    [ep.tl.kaplan_meier, ep.tl.cox_ph, ep.tl.nelson_aalen, ep.tl.weibull, ep.tl.weibull_aft, ep.tl.log_logistic_aft],
)
def test_survival_models_array_types(array_type, sa_function, rng):
    X = _survival_data(rng)
    expected = _survival_edata(X)
    sa_function(expected, duration_col="duration", event_col="event", key_added="test")
    edata = _survival_edata(array_type(X))

    if _raises_or_runs(array_type, sa_function, edata, duration_col="duration", event_col="event", key_added="test"):
        pd.testing.assert_frame_equal(edata.uns["test"], expected.uns["test"])


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("model", [ep.tl.ols, ep.tl.glm])
def test_ols_glm_array_types(array_type, model, rng):
    X = _survival_data(rng)
    kwargs = {"var_names": ["a", "duration"], "formula": "a ~ duration"}
    expected = model(_survival_edata(X), **kwargs).fit().params

    if result := _raises_or_runs(array_type, model, _survival_edata(array_type(X)), **kwargs):
        pd.testing.assert_series_equal(result.fit().params, expected)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_cox_ph_adjusted_curves_array_types(array_type, rng):
    X = _survival_data(rng)
    expected = _survival_edata(X)
    cph = ep.tl.cox_ph(expected, duration_col="duration", event_col="event", formula="a + b")
    kwargs = {"cph": cph, "strata": "b", "duration_col": "duration", "event_col": "event", "method": "conditional"}
    ep.tl.cox_ph_adjusted_curves(expected, **kwargs)
    edata = _survival_edata(array_type(X))

    if result := _raises_or_runs(array_type, ep.tl.cox_ph_adjusted_curves, edata, copy=True, **kwargs):
        assert type(result.X) is type(edata.X)
        for group in ("0.0", "1.0"):
            np.testing.assert_allclose(
                result.uns["cox_ph_adjusted_curves"][group]["survival"],
                expected.uns["cox_ph_adjusted_curves"][group]["survival"],
            )
