import numpy as np
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

import ehrapy as ep


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_ols(mimic_2, layer):
    adata = mimic_2
    # If use layer argument, set X to None to avoid it being used
    if layer is not None:
        adata.X = None

    formula = "tco2_first ~ pco2_first"
    var_names = ["tco2_first", "pco2_first"]
    ols = ep.tl.ols(adata, var_names, formula=formula, missing="drop", layer=layer)
    s = ols.fit().params.iloc[1]
    i = ols.fit().params.iloc[0]
    assert isinstance(ols, statsmodels.regression.linear_model.OLS)
    assert 0.18857179158259973 == pytest.approx(s)
    assert 16.210859352601442 == pytest.approx(i)


def test_ols_3D(edata_blob_small):
    formula = "feature_1 ~ feature_2"
    var_names = ["feature_1", "feature_2"]
    ep.tl.ols(edata_blob_small, var_names, formula=formula, missing="drop", layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.tl.ols(edata_blob_small, var_names, formula=formula, missing="drop", layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_glm(mimic_2, layer):
    adata = mimic_2
    # If use layer argument, set X to None to avoid it being used
    if layer is not None:
        adata.X = None

    formula = "day_28_flg ~ age"
    var_names = ["day_28_flg", "age"]
    family = "Binomial"
    glm = ep.tl.glm(
        adata, var_names, formula=formula, family=family, missing="drop", as_continuous=["age"], layer=layer
    )
    Intercept = glm.fit().params.iloc[0]
    age = glm.fit().params.iloc[1]
    assert isinstance(glm, statsmodels.genmod.generalized_linear_model.GLM)
    assert 5.778006344870297 == pytest.approx(Intercept)
    assert -0.06523274132877163 == pytest.approx(age)


def test_glm_3D(edata_blob_small):
    formula = "feature_1 ~ feature_2"
    var_names = ["feature_1", "feature_2"]
    ep.tl.glm(edata_blob_small, var_names, formula=formula, missing="drop", layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.tl.glm(edata_blob_small, var_names, formula=formula, missing="drop", layer=DEFAULT_TEM_LAYER_NAME)


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
    adata = mimic_2
    formula = "day_28_flg ~ age"
    var_names = ["day_28_flg", "age"]
    family = "Binomial"
    age_glm = ep.tl.glm(adata, var_names, formula=formula, family=family, missing="drop", as_continuous=["age"])
    age_glm_result = age_glm.fit()
    formula = "day_28_flg ~ age + service_unit"
    var_names = ["day_28_flg", "age", "service_unit"]
    ageunit_glm = ep.tl.glm(adata, var_names, formula=formula, family=family, missing="drop", as_continuous=["age"])
    ageunit_glm_result = ageunit_glm.fit()
    dataframe = ep.tl.anova_glm(
        age_glm_result, ageunit_glm_result, "day_28_flg ~ age", "day_28_flg ~ age + service_unit"
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
    adata, duration_col, event_col = mimic_2_sa
    # If use layer argument, set X to None to avoid it being used
    if layer is not None:
        adata.X = None

    sa = sa_function(adata, duration_col=duration_col, event_col=event_col, uns_key="test", layer=layer)

    assert isinstance(sa, sa_class)
    assert len(sa.durations) == 1776
    assert sum(sa.event_observed) == 497

    model_summary = adata.uns.get("test")
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
    edata_blob_small[:, [duration_col]].X = np.arange(len(edata_blob_small), dtype=np.int32)
    edata_blob_small[:, [event_col]].X = 1

    edata_blob_small.layers["layer_2"] = edata_blob_small.X.copy()

    sa_function(edata_blob_small, duration_col=duration_col, event_col=event_col, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        sa_function(edata_blob_small, duration_col=duration_col, event_col=event_col, layer=DEFAULT_TEM_LAYER_NAME)


def test_kmf(mimic_2_sa):
    with pytest.warns(DeprecationWarning):
        adata, _, _ = mimic_2_sa
        kmf = ep.tl.kmf(adata[:, ["mort_day_censored"]].X, adata[:, ["censor_flg"]].X)

        assert isinstance(kmf, KaplanMeierFitter)
        assert len(kmf.durations) == 1776
        assert sum(kmf.event_observed) == 497
