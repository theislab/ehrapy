"""Tests for the in-house causal inference module."""

from __future__ import annotations

import ehrdata as ed
import holoviews as hv
import numpy as np
import pandas as pd
import pytest

import ehrapy as ep


def _synth_dataset(n: int = 2000, true_ate: float = 3.0, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    age = rng.normal(60, 10, n)
    sex = rng.integers(0, 2, n).astype(float)
    bmi = rng.normal(27, 4, n)
    lp = -2.0 + 0.05 * (age - 60) + 0.3 * sex
    T = rng.binomial(1, 1 / (1 + np.exp(-lp))).astype(float)
    Y = 10 + 0.1 * (age - 60) + 1.5 * sex - 0.2 * (bmi - 27) + true_ate * T + rng.normal(0, 2, n)
    X = np.column_stack([age, sex, bmi, T, Y]).astype(float)
    var = pd.DataFrame(index=["age", "sex", "bmi", "tx", "y"])
    return ed.EHRData(X=X, var=var)


def _synth_hte_dataset(n: int = 3000, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    age = rng.normal(60, 10, n)
    sex = rng.integers(0, 2, n).astype(float)
    bmi = rng.normal(27, 4, n)
    lp = -2.0 + 0.05 * (age - 60) + 0.3 * sex
    T = rng.binomial(1, 1 / (1 + np.exp(-lp))).astype(float)
    true_cate = 5.0 - 0.05 * (age - 60)
    Y = 10 + 0.1 * (age - 60) + 1.5 * sex - 0.2 * (bmi - 27) + true_cate * T + rng.normal(0, 2, n)
    X = np.column_stack([age, sex, bmi, T, Y]).astype(float)
    return ed.EHRData(X=X, var=pd.DataFrame(index=["age", "sex", "bmi", "tx", "y"])), age


class TestATE:
    def setup_method(self):
        self.edata = _synth_dataset()
        self.covariates = ["age", "sex", "bmi"]

    def test_iptw_recovers_ate(self):
        est = ep.tl.iptw(self.edata, "tx", "y", covariates=self.covariates, n_bootstrap=50, random_state=0)
        assert isinstance(est, ep.tl.CausalEstimate)
        assert est.value == pytest.approx(3.0, abs=0.4)
        assert est.ci_lower < 3.0 < est.ci_upper
        assert est.params["weights"].shape == (2000,)

    def test_iptw_unstabilized(self):
        est = ep.tl.iptw(self.edata, "tx", "y", covariates=self.covariates, stabilized=False, n_bootstrap=0)
        assert est.method == "iptw"

    def test_g_computation_recovers_ate(self):
        est = ep.tl.g_computation(self.edata, "tx", "y", covariates=self.covariates, n_bootstrap=50, random_state=0)
        assert est.value == pytest.approx(3.0, abs=0.4)
        assert est.ci_lower < 3.0 < est.ci_upper

    def test_aipw_recovers_ate(self):
        est = ep.tl.aipw(self.edata, "tx", "y", covariates=self.covariates)
        assert est.value == pytest.approx(3.0, abs=0.4)
        assert est.se is not None
        assert est.ci_lower < 3.0 < est.ci_upper

    def test_aipw_doubly_robust(self):
        # With a misspecified outcome model (constant predictor), correct propensity should still recover ATE.
        from sklearn.dummy import DummyRegressor

        est = ep.tl.aipw(
            self.edata,
            "tx",
            "y",
            covariates=self.covariates,
            outcome_model=DummyRegressor(strategy="mean"),
        )
        assert est.value == pytest.approx(3.0, abs=0.5)

    def test_propensity_score_matching_recovers_ate(self):
        est = ep.tl.propensity_score_matching(
            self.edata, "tx", "y", covariates=self.covariates, n_bootstrap=30, random_state=0
        )
        assert est.value == pytest.approx(3.0, abs=0.6)
        assert est.params["matches"]["n_matched_pairs"] > 0

    def test_propensity_score_matching_ate_target(self):
        est = ep.tl.propensity_score_matching(
            self.edata, "tx", "y", covariates=self.covariates, target="ate", n_bootstrap=0
        )
        assert est.method.endswith("_ate")


class TestHTE:
    def setup_method(self):
        self.edata, self.age = _synth_hte_dataset()
        self.covariates = ["age", "sex", "bmi"]

    def test_t_learner_recovers_heterogeneity(self):
        est = ep.tl.t_learner(self.edata, "tx", "y", covariates=self.covariates)
        cate = est.params["cate"]
        assert cate.shape == (3000,)
        # CATE should be strongly negatively correlated with age
        assert np.corrcoef(cate, -self.age)[0, 1] > 0.5
        assert est.value == pytest.approx(5.0, abs=0.5)

    def test_x_learner_recovers_heterogeneity(self):
        est = ep.tl.x_learner(self.edata, "tx", "y", covariates=self.covariates)
        cate = est.params["cate"]
        assert np.corrcoef(cate, -self.age)[0, 1] > 0.5
        assert est.value == pytest.approx(5.0, abs=0.5)

    def test_s_learner_returns_cate(self):
        est = ep.tl.s_learner(self.edata, "tx", "y", covariates=self.covariates)
        assert est.params["cate"].shape == (3000,)

    def test_key_added_writes_obs(self):
        ep.tl.t_learner(self.edata, "tx", "y", covariates=self.covariates, key_added="cate_tlearner")
        assert "cate_tlearner" in self.edata.obs.columns
        assert not self.edata.obs["cate_tlearner"].isna().any()

    def test_x_learner_rejects_classifier_for_cate(self):
        from sklearn.linear_model import LogisticRegression

        with pytest.raises(TypeError, match="must be a regressor"):
            ep.tl.x_learner(self.edata, "tx", "y", covariates=self.covariates, cate_model=LogisticRegression())


class TestDiagnostics:
    def setup_method(self):
        self.edata = _synth_dataset()
        self.covariates = ["age", "sex", "bmi"]

    def test_covariate_balance_improves_after_weighting(self):
        bal = ep.tl.covariate_balance(self.edata, "tx", covariates=self.covariates)
        assert {"smd_unweighted", "smd_weighted", "var_ratio_unweighted", "var_ratio_weighted"} <= set(bal.columns)
        assert (bal["smd_weighted"].abs() < bal["smd_unweighted"].abs()).all()

    def test_covariate_balance_accepts_external_weights(self):
        est = ep.tl.iptw(self.edata, "tx", "y", covariates=self.covariates, n_bootstrap=0)
        w_full = np.full(self.edata.n_obs, np.nan)
        pos = self.edata.obs.index.get_indexer(est.params["index"])
        w_full[pos] = est.params["weights"]
        bal = ep.tl.covariate_balance(self.edata, "tx", covariates=self.covariates, weights=w_full)
        assert (bal["smd_weighted"].abs() < 0.15).all()

    def test_positivity_check_summary_shape(self):
        info = ep.tl.positivity_check(self.edata, "tx", covariates=self.covariates)
        assert 0.0 <= info["support_fraction"] <= 1.0
        assert set(info["summary_treated"]) == {"min", "max", "mean", "median", "p05", "p95"}
        assert info["propensity_scores"].shape[0] == self.edata.n_obs


class TestPlots:
    def setup_method(self):
        self.edata = _synth_dataset()
        self.covariates = ["age", "sex", "bmi"]

    def test_love_plot(self):
        bal = ep.tl.covariate_balance(self.edata, "tx", covariates=self.covariates)
        plot = ep.pl.love_plot(bal)
        assert isinstance(plot, hv.Overlay)

    def test_propensity_overlap(self):
        info = ep.tl.positivity_check(self.edata, "tx", covariates=self.covariates)
        plot = ep.pl.propensity_overlap(info)
        assert isinstance(plot, hv.Overlay)

    def test_causal_effect_forest(self):
        est_iptw = ep.tl.iptw(self.edata, "tx", "y", covariates=self.covariates, n_bootstrap=30, random_state=0)
        est_aipw = ep.tl.aipw(self.edata, "tx", "y", covariates=self.covariates)
        plot = ep.pl.causal_effect(est_iptw, other={"aipw": est_aipw})
        assert isinstance(plot, hv.Overlay)


class TestBackends:
    def test_rejects_sparse_x_explicitly(self):
        import scipy.sparse as sp

        edata = _synth_dataset()
        edata.X = sp.csr_array(edata.X)
        with pytest.raises(NotImplementedError, match="numpy-backed"):
            ep.tl.iptw(edata, "tx", "y", covariates=["age", "sex", "bmi"], n_bootstrap=0)

    def test_rejects_dask_x_explicitly(self):
        da = pytest.importorskip("dask.array")

        edata = _synth_dataset()
        edata.X = da.from_array(edata.X, chunks=500)
        with pytest.raises(NotImplementedError, match="numpy-backed"):
            ep.tl.iptw(edata, "tx", "y", covariates=["age", "sex", "bmi"], n_bootstrap=0)


class TestGuards:
    def test_rejects_3d_layer(self):
        from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

        edata = _synth_dataset()
        edata.layers[DEFAULT_TEM_LAYER_NAME] = np.stack([edata.X, edata.X], axis=2)
        with pytest.raises(ValueError, match=r"only supports 2D data"):
            ep.tl.iptw(edata, "tx", "y", covariates=["age", "sex", "bmi"], layer=DEFAULT_TEM_LAYER_NAME, n_bootstrap=0)

    def test_rejects_non_binary_treatment(self):
        edata = _synth_dataset()
        edata.X[:, 3] = np.random.default_rng(0).integers(0, 3, edata.n_obs).astype(float)
        with pytest.raises(ValueError, match="must be binary"):
            ep.tl.iptw(edata, "tx", "y", covariates=["age", "sex", "bmi"], n_bootstrap=0)

    def test_rejects_treatment_in_covariates(self):
        edata = _synth_dataset()
        with pytest.raises(ValueError, match="must not appear"):
            ep.tl.iptw(edata, "tx", "y", covariates=["age", "tx"], n_bootstrap=0)

    def test_rejects_unknown_column(self):
        edata = _synth_dataset()
        with pytest.raises(KeyError, match="not found"):
            ep.tl.iptw(edata, "tx", "y", covariates=["age", "not_a_column"], n_bootstrap=0)
