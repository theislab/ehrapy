import lifelines
import numpy as np
import pytest
import statsmodels

import ehrapy as ep


class TestSA:
    def test_ols(self):
        import ehrapy as ep

        adata = ep.dt.mimic_2(encoded=False)
        formula = "tco2_first ~ pco2_first"
        var_names = ["tco2_first", "pco2_first"]
        ols = ep.tl.ols(adata, var_names, formula, missing="drop")
        s = ols.fit().params[1]
        i = ols.fit().params[0]
        assert isinstance(ols, statsmodels.regression.linear_model.OLS)
        assert 0.18857179158259973 == pytest.approx(s)
        assert 16.210859352601442 == pytest.approx(i)

    def test_glm(self):
        adata = ep.dt.mimic_2(encoded=False)
        formula = "day_28_flg ~ age"
        var_names = ["day_28_flg", "age"]
        family = "Binomial"
        glm = ep.tl.glm(adata, var_names, formula, family, missing="drop", ascontinus=["age"])
        Intercept = glm.fit().params[0]
        age = glm.fit().params[1]
        assert isinstance(glm, statsmodels.genmod.generalized_linear_model.GLM)
        assert 5.778006344870297 == pytest.approx(Intercept)
        assert -0.06523274132877163 == pytest.approx(age)

    def test_kmf(self):
        import ehrapy as ep

        adata = ep.dt.mimic_2(encoded=False)
        adata[:, ["censor_flg"]].X = np.where(adata[:, ["censor_flg"]].X == 0, 1, 0)
        kmf = ep.tl.kmf(adata[:, ["mort_day_censored"]].X, adata[:, ["censor_flg"]].X)

        assert isinstance(kmf, lifelines.fitters.kaplan_meier_fitter.KaplanMeierFitter)
        assert len(kmf.durations) == 1776
        assert sum(kmf.event_observed) == 497
