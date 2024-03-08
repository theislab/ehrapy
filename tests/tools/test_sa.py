import numpy as np
import pytest
import statsmodels
from lifelines import CoxPHFitter, KaplanMeierFitter

import ehrapy as ep


class TestSA:
    def test_ols(self):
        adata = ep.dt.mimic_2(encoded=False)
        formula = "tco2_first ~ pco2_first"
        var_names = ["tco2_first", "pco2_first"]
        ols = ep.tl.ols(adata, var_names, formula, missing="drop")
        s = ols.fit().params.iloc[1]
        i = ols.fit().params.iloc[0]
        assert isinstance(ols, statsmodels.regression.linear_model.OLS)
        assert 0.18857179158259973 == pytest.approx(s)
        assert 16.210859352601442 == pytest.approx(i)

    def test_glm(self):
        adata = ep.dt.mimic_2(encoded=False)
        formula = "day_28_flg ~ age"
        var_names = ["day_28_flg", "age"]
        family = "Binomial"
        glm = ep.tl.glm(adata, var_names, formula, family, missing="drop", as_continuous=["age"])
        Intercept = glm.fit().params.iloc[0]
        age = glm.fit().params.iloc[1]
        assert isinstance(glm, statsmodels.genmod.generalized_linear_model.GLM)
        assert 5.778006344870297 == pytest.approx(Intercept)
        assert -0.06523274132877163 == pytest.approx(age)

    def test_kmf(self):
        adata = ep.dt.mimic_2(encoded=False)
        adata[:, ["censor_flg"]].X = np.where(adata[:, ["censor_flg"]].X == 0, 1, 0)
        kmf = ep.tl.kmf(adata[:, ["mort_day_censored"]].X, adata[:, ["censor_flg"]].X)

        assert isinstance(kmf, KaplanMeierFitter)
        assert len(kmf.durations) == 1776
        assert sum(kmf.event_observed) == 497

    @pytest.mark.parametrize("weightings", ["wilcoxon", "tarone-ware", "peto", "fleming-harrington"])
    def test_calculate_logrank_pvalue(self, weightings):
        durations_A = [1, 2, 3]
        event_observed_A = [1, 1, 0]
        durations_B = [1, 2, 3, 4]
        event_observed_B = [1, 0, 0, 1]

        kmf1 = KaplanMeierFitter()
        kmf1.fit(durations_A, event_observed_A)

        kmf2 = KaplanMeierFitter()
        kmf2.fit(durations_B, event_observed_B)

        results_pairwise = ep.tl.test_kmf_logrank(kmf1, kmf2)
        p_value_pairwise = results_pairwise.p_value
        assert 0 < p_value_pairwise < 1

    def test_anova_glm(self):
        adata = ep.dt.mimic_2(encoded=False)
        formula = "day_28_flg ~ age"
        var_names = ["day_28_flg", "age"]
        family = "Binomial"
        age_glm = ep.tl.glm(adata, var_names, formula, family, missing="drop", as_continuous=["age"])
        age_glm_result = age_glm.fit()
        formula = "day_28_flg ~ age + service_unit"
        var_names = ["day_28_flg", "age", "service_unit"]
        ageunit_glm = ep.tl.glm(adata, var_names, formula, family="Binomial", missing="drop", as_continuous=["age"])
        ageunit_glm_result = ageunit_glm.fit()
        dataframe = ep.tl.anova_glm(
            age_glm_result, ageunit_glm_result, "day_28_flg ~ age", "day_28_flg ~ age + service_unit"
        )

        assert len(dataframe) == 2
        assert dataframe.shape == (2, 6)
        assert dataframe.iloc[1, 4] == 2
        assert pytest.approx(dataframe.iloc[1, 5], 0.1) == 0.103185

    def test_cox_ph(self):
        adata = ep.dt.mimic_2(encoded=False)
        adata[:, ["censor_flg"]].X = np.where(adata[:, ["censor_flg"]].X == 0, 1, 0)
        cph = ep.tl.cox_ph(adata, "mort_day_censored", "censor_flg")

        assert isinstance(cph, CoxPHFitter)
        assert len(cph.durations) == 1776
        assert sum(cph.event_observed) == 497
