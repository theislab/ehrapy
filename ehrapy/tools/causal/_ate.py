"""Average treatment effect (ATE) estimators for binary treatments.

All estimators in this module accept an :class:`~ehrdata.EHRData` object and return a :class:`~ehrapy.tools.causal.CausalEstimate`.
They share a common interface: a ``treatment`` and ``outcome`` column name plus a list of ``covariates`` (the adjustment set) that may come from either ``edata.var_names`` or ``edata.obs.columns``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_compat
import numpy as np

from ehrapy._compat import function_2D_only
from ehrapy.tools.causal._design import assert_binary_treatment, build_design
from ehrapy.tools.causal._estimate import CausalEstimate
from ehrapy.tools.causal._models import fit_propensity, predict_mean, resolve_outcome_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData
    from sklearn.base import BaseEstimator


_DEFAULT_CLIP = (0.01, 0.99)


def _bootstrap_ate(
    estimator_fn,
    *,
    n: int,
    n_bootstrap: int,
    random_state: int | None,
) -> tuple[float, float, float]:
    """Bootstrap an estimator function to obtain (SE, ci_lower, ci_upper).

    ``estimator_fn`` is a callable that takes a boolean index array of length ``n`` and returns a scalar ATE.
    """
    rng = np.random.default_rng(random_state)
    values = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values[b] = estimator_fn(idx)
    se = float(np.std(values, ddof=1))
    ci_lower, ci_upper = (float(x) for x in np.quantile(values, [0.025, 0.975]))
    return se, ci_lower, ci_upper


@function_2D_only()
def iptw(
    edata: EHRData,
    treatment: str,
    outcome: str,
    *,
    covariates: Sequence[str],
    propensity_model: str | BaseEstimator = "logistic",
    stabilized: bool = True,
    clip: tuple[float, float] | None = _DEFAULT_CLIP,
    n_bootstrap: int = 200,
    random_state: int | None = None,
    layer: str | None = None,
) -> CausalEstimate:
    """Estimate the average treatment effect by inverse probability of treatment weighting (IPTW).

    Fits a propensity model ``e(X) = P(T=1 | X)`` and forms weights ``w_i = T_i / e_i + (1-T_i) / (1-e_i)``.
    With ``stabilized=True`` the weights are multiplied by the marginal treatment probabilities, which typically reduces variance with negligible bias.
    The ATE is the difference of weighted means of ``Y`` between treated and untreated groups.

    Args:
        edata: Central data object.
        treatment: Column name of the binary (0/1) treatment variable.
        outcome: Column name of the outcome variable.
        covariates: Adjustment set used to fit the propensity model.
            Each entry must refer to a name in ``edata.var_names`` or ``edata.obs.columns``.
        propensity_model: Specification of the propensity model.
            Accepts one of the strings ``'logistic'``, ``'gradient_boosting'``, ``'random_forest'``, or any sklearn-compatible classifier (it must implement ``predict_proba``).
        stabilized: Whether to use stabilized weights instead of the basic inverse-probability weights.
        clip: ``(lo, hi)`` propensity-score clipping range applied before forming weights.
            Use ``None`` to disable clipping.
        n_bootstrap: Number of bootstrap resamples used for the SE and 95% percentile confidence interval.
            Set to ``0`` to skip uncertainty estimation.
        random_state: Seed for the bootstrap resampler.
        layer: Layer of ``edata`` to draw the var-side variables from.
            If ``None``, ``edata.X`` is used.

    Returns:
        A :class:`~ehrapy.tools.causal.CausalEstimate` whose ``params`` dict contains the fitted ``propensity_scores`` and the IPTW ``weights``.

    Examples:
        >>> import ehrapy as ep
        >>> est = ep.tl.iptw(edata, "treatment", "outcome", covariates=["age", "sex", "bmi"])
        >>> print(est.summary())
    """
    design = build_design(edata, treatment=treatment, outcome=outcome, covariates=covariates, layer=layer)
    assert_binary_treatment(design.T, treatment)

    ps, _ = fit_propensity(propensity_model, design.X, design.T, clip=clip)
    weights = _iptw_weights(design.T, ps, stabilized=stabilized)
    ate = _weighted_diff_in_means(design.Y, design.T, weights)

    se: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    if n_bootstrap > 0:

        def _refit(idx: np.ndarray) -> float:
            X_b, T_b, Y_b = design.X[idx], design.T[idx], design.Y[idx]
            if len(np.unique(T_b)) < 2:
                return np.nan
            ps_b, _ = fit_propensity(propensity_model, X_b, T_b, clip=clip)
            w_b = _iptw_weights(T_b, ps_b, stabilized=stabilized)
            return _weighted_diff_in_means(Y_b, T_b, w_b)

        se, ci_lower, ci_upper = _bootstrap_ate(
            _refit, n=len(design.T), n_bootstrap=n_bootstrap, random_state=random_state
        )

    return CausalEstimate(
        method="iptw" + ("_stabilized" if stabilized else ""),
        treatment=treatment,
        outcome=outcome,
        value=float(ate),
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=int(len(design.T)),
        params={
            "propensity_scores": ps,
            "weights": weights,
            "index": design.index,
            "feature_names": design.feature_names,
        },
    )


@function_2D_only()
def g_computation(
    edata: EHRData,
    treatment: str,
    outcome: str,
    *,
    covariates: Sequence[str],
    outcome_model: str | BaseEstimator = "auto",
    n_bootstrap: int = 200,
    random_state: int | None = None,
    layer: str | None = None,
) -> CausalEstimate:
    """Estimate the ATE by parametric g-computation (a.k.a. the g-formula or standardisation).

    Fits an outcome model ``μ(T, X)`` on the observed data, then predicts counterfactual outcomes by setting ``T`` to 1 and 0 for every row.
    The ATE is ``mean(μ(1, X)) − mean(μ(0, X))``.

    Args:
        edata: Central data object.
        treatment: Column name of the binary (0/1) treatment variable.
        outcome: Column name of the outcome variable.
        covariates: Adjustment set used to fit the outcome model.
            Each entry must refer to a name in ``edata.var_names`` or ``edata.obs.columns``.
        outcome_model: Specification of the outcome model.
            Accepts one of the strings ``'auto'``, ``'linear'``, ``'logistic'``, ``'gradient_boosting'``, ``'random_forest'``, or any sklearn-compatible regressor/classifier.
            ``'auto'`` picks logistic regression when the outcome is binary 0/1 and linear regression otherwise.
        n_bootstrap: Number of bootstrap resamples used for the SE and 95% percentile confidence interval.
            Set to ``0`` to skip uncertainty estimation.
        random_state: Seed for the bootstrap resampler.
        layer: Layer of ``edata`` to draw the var-side variables from.
            If ``None``, ``edata.X`` is used.

    Returns:
        A :class:`~ehrapy.tools.causal.CausalEstimate` whose ``params`` dict contains the counterfactual predictions ``mu1`` and ``mu0``.
    """
    design = build_design(edata, treatment=treatment, outcome=outcome, covariates=covariates, layer=layer)
    assert_binary_treatment(design.T, treatment)

    mu1, mu0 = _g_predict(design.X, design.T, design.Y, outcome_model)
    ate = float(np.mean(mu1) - np.mean(mu0))

    se: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    if n_bootstrap > 0:

        def _refit(idx: np.ndarray) -> float:
            X_b, T_b, Y_b = design.X[idx], design.T[idx], design.Y[idx]
            if len(np.unique(T_b)) < 2:
                return np.nan
            mu1_b, mu0_b = _g_predict(X_b, T_b, Y_b, outcome_model)
            return float(np.mean(mu1_b) - np.mean(mu0_b))

        se, ci_lower, ci_upper = _bootstrap_ate(
            _refit, n=len(design.T), n_bootstrap=n_bootstrap, random_state=random_state
        )

    return CausalEstimate(
        method="g_computation",
        treatment=treatment,
        outcome=outcome,
        value=ate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=int(len(design.T)),
        params={"mu1": mu1, "mu0": mu0, "index": design.index, "feature_names": design.feature_names},
    )


@function_2D_only()
def aipw(
    edata: EHRData,
    treatment: str,
    outcome: str,
    *,
    covariates: Sequence[str],
    propensity_model: str | BaseEstimator = "logistic",
    outcome_model: str | BaseEstimator = "auto",
    clip: tuple[float, float] | None = _DEFAULT_CLIP,
    n_bootstrap: int = 0,
    random_state: int | None = None,
    layer: str | None = None,
) -> CausalEstimate:
    """Estimate the ATE by the augmented inverse-probability-weighted (AIPW) doubly robust estimator.

    AIPW is consistent if *either* the propensity model *or* the outcome model is correctly specified.
    The point estimate is the mean of the influence function::

        ψ_i = μ_1(X_i) − μ_0(X_i)
              + (T_i / e_i) (Y_i − μ_1(X_i))
              − ((1 − T_i) / (1 − e_i)) (Y_i − μ_0(X_i))

    By default the standard error is computed analytically from the empirical variance of ψ; setting ``n_bootstrap > 0`` switches to a bootstrap SE/CI instead.

    Args:
        edata: Central data object.
        treatment: Column name of the binary (0/1) treatment variable.
        outcome: Column name of the outcome variable.
        covariates: Adjustment set used for both nuisance models.
            Each entry must refer to a name in ``edata.var_names`` or ``edata.obs.columns``.
        propensity_model: Propensity model specification (see :func:`iptw` for the accepted values).
        outcome_model: Outcome model specification (see :func:`g_computation` for the accepted values).
        clip: ``(lo, hi)`` propensity-score clipping range applied before forming the influence function.
            Use ``None`` to disable clipping.
        n_bootstrap: If positive, use a bootstrap SE/CI instead of the analytic influence-function SE.
            Set to ``0`` (the default) to use the influence-function SE.
        random_state: Seed for the bootstrap resampler.
        layer: Layer of ``edata`` to draw the var-side variables from.
            If ``None``, ``edata.X`` is used.

    Returns:
        A :class:`~ehrapy.tools.causal.CausalEstimate` whose ``params`` dict contains ``propensity_scores``, ``mu1``, ``mu0``, and the per-observation ``influence`` values.
    """
    design = build_design(edata, treatment=treatment, outcome=outcome, covariates=covariates, layer=layer)
    assert_binary_treatment(design.T, treatment)

    ps, _ = fit_propensity(propensity_model, design.X, design.T, clip=clip)
    mu1, mu0 = _g_predict(design.X, design.T, design.Y, outcome_model)
    psi = _aipw_influence(design.T, design.Y, ps, mu1, mu0)
    ate = float(np.mean(psi))

    se: float | None
    ci_lower: float | None
    ci_upper: float | None
    if n_bootstrap > 0:

        def _refit(idx: np.ndarray) -> float:
            X_b, T_b, Y_b = design.X[idx], design.T[idx], design.Y[idx]
            if len(np.unique(T_b)) < 2:
                return np.nan
            ps_b, _ = fit_propensity(propensity_model, X_b, T_b, clip=clip)
            mu1_b, mu0_b = _g_predict(X_b, T_b, Y_b, outcome_model)
            psi_b = _aipw_influence(T_b, Y_b, ps_b, mu1_b, mu0_b)
            return float(np.mean(psi_b))

        se, ci_lower, ci_upper = _bootstrap_ate(
            _refit, n=len(design.T), n_bootstrap=n_bootstrap, random_state=random_state
        )
    else:
        se = float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
        ci_lower = float(ate - 1.96 * se)
        ci_upper = float(ate + 1.96 * se)

    return CausalEstimate(
        method="aipw",
        treatment=treatment,
        outcome=outcome,
        value=ate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=int(len(design.T)),
        params={
            "propensity_scores": ps,
            "mu1": mu1,
            "mu0": mu0,
            "influence": psi,
            "index": design.index,
            "feature_names": design.feature_names,
        },
    )


@function_2D_only()
def propensity_score_matching(
    edata: EHRData,
    treatment: str,
    outcome: str,
    *,
    covariates: Sequence[str],
    propensity_model: str | BaseEstimator = "logistic",
    k: int = 1,
    caliper: float | None = 0.2,
    replacement: bool = True,
    target: str = "att",
    n_bootstrap: int = 200,
    random_state: int | None = None,
    layer: str | None = None,
) -> CausalEstimate:
    """Estimate the treatment effect by 1-to-:math:`k` propensity score matching on the logit scale.

    For each treated unit, the :math:`k` nearest control units in logit-propensity space are selected as matches (and vice versa when ``target='ate'``).
    With ``caliper`` set, candidate matches with logit-propensity distance above ``caliper * SD(logit(e))`` are discarded; treated units with no valid match are dropped from the estimate.

    Args:
        edata: Central data object.
        treatment: Column name of the binary (0/1) treatment variable.
        outcome: Column name of the outcome variable.
        covariates: Adjustment set used to fit the propensity model.
            Each entry must refer to a name in ``edata.var_names`` or ``edata.obs.columns``.
        propensity_model: Propensity model specification (see :func:`iptw` for the accepted values).
        k: Number of matches per unit.
        caliper: Maximum logit-propensity distance for a valid match, in units of ``SD(logit(e))``.
            Use ``None`` to disable the caliper.
        replacement: Whether matching is performed with replacement.
        target: ``'att'`` for the average treatment effect on the treated, or ``'ate'`` for the average treatment effect.
        n_bootstrap: Number of bootstrap resamples used for the SE and 95% percentile confidence interval.
            Set to ``0`` to skip uncertainty estimation.
        random_state: Seed for the bootstrap resampler.
        layer: Layer of ``edata`` to draw the var-side variables from.
            If ``None``, ``edata.X`` is used.

    Returns:
        A :class:`~ehrapy.tools.causal.CausalEstimate` whose ``params`` dict contains the propensity scores and the matched-pair indices.
    """
    if target not in {"att", "ate"}:
        raise ValueError(f"target must be 'att' or 'ate'; got {target!r}.")

    design = build_design(edata, treatment=treatment, outcome=outcome, covariates=covariates, layer=layer)
    assert_binary_treatment(design.T, treatment)

    ps, _ = fit_propensity(propensity_model, design.X, design.T, clip=(1e-6, 1 - 1e-6))
    ate, match_info = _ps_match_effect(
        design.T, design.Y, ps, k=k, caliper=caliper, replacement=replacement, target=target
    )

    se: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    if n_bootstrap > 0:

        def _refit(idx: np.ndarray) -> float:
            X_b, T_b, Y_b = design.X[idx], design.T[idx], design.Y[idx]
            if len(np.unique(T_b)) < 2:
                return np.nan
            ps_b, _ = fit_propensity(propensity_model, X_b, T_b, clip=(1e-6, 1 - 1e-6))
            ate_b, _ = _ps_match_effect(T_b, Y_b, ps_b, k=k, caliper=caliper, replacement=replacement, target=target)
            return float(ate_b)

        se, ci_lower, ci_upper = _bootstrap_ate(
            _refit, n=len(design.T), n_bootstrap=n_bootstrap, random_state=random_state
        )

    return CausalEstimate(
        method=f"propensity_score_matching_{target}",
        treatment=treatment,
        outcome=outcome,
        value=float(ate),
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=int(len(design.T)),
        params={
            "propensity_scores": ps,
            "matches": match_info,
            "index": design.index,
            "feature_names": design.feature_names,
        },
    )


def _iptw_weights(T, ps, *, stabilized: bool):
    """Compute IPTW weights from a treatment vector and propensity scores."""
    xp = array_api_compat.array_namespace(T, ps)
    if stabilized:
        p_t = xp.mean(T)
        return T * p_t / ps + (1 - T) * (1 - p_t) / (1 - ps)
    return T / ps + (1 - T) / (1 - ps)


def _weighted_diff_in_means(Y, T, w) -> float:
    """Weighted difference in means (Hájek estimator)."""
    xp = array_api_compat.array_namespace(Y, T, w)
    treated = T == 1
    untreated = ~treated
    mu1 = xp.sum(w[treated] * Y[treated]) / xp.sum(w[treated])
    mu0 = xp.sum(w[untreated] * Y[untreated]) / xp.sum(w[untreated])
    return float(mu1 - mu0)


def _aipw_influence(T, Y, ps, mu1, mu0):
    """Compute the AIPW influence-function values; backend-agnostic."""
    return mu1 - mu0 + (T / ps) * (Y - mu1) - ((1 - T) / (1 - ps)) * (Y - mu0)


def _g_predict(X: np.ndarray, T: np.ndarray, Y: np.ndarray, outcome_model_spec) -> tuple[np.ndarray, np.ndarray]:
    """Fit μ(T, X) and return (μ(1, X), μ(0, X)) for every row of X.

    sklearn currently mandates numpy at the fit boundary, so this helper materialises to numpy.
    """
    model = resolve_outcome_model(outcome_model_spec, y=Y)
    XT = np.column_stack([T, X])
    model.fit(XT, Y if not hasattr(model, "predict_proba") else Y.astype(int))
    X1 = np.column_stack([np.ones_like(T), X])
    X0 = np.column_stack([np.zeros_like(T), X])
    return predict_mean(model, X1), predict_mean(model, X0)


def _logit(p):
    """Logit transform; backend-agnostic."""
    xp = array_api_compat.array_namespace(p)
    p = xp.clip(p, 1e-12, 1 - 1e-12)
    return xp.log(p / (1 - p))


def _ps_match_effect(
    T: np.ndarray,
    Y: np.ndarray,
    ps: np.ndarray,
    *,
    k: int,
    caliper: float | None,
    replacement: bool,
    target: str,
) -> tuple[float, dict]:
    """Compute the matched effect and return (ATE, match diagnostics)."""
    logit_ps = _logit(ps)
    sd = float(np.std(logit_ps, ddof=1))
    caliper_dist = caliper * sd if caliper is not None else None

    treated_idx = np.flatnonzero(T == 1)
    control_idx = np.flatnonzero(T == 0)

    contribs: list[float] = []
    used_treated: list[int] = []
    used_control: list[int] = []
    matched_pairs: list[tuple[int, list[int]]] = []
    dropped = 0

    def _match_one(src: int, pool: np.ndarray) -> list[int] | None:
        dist = np.abs(logit_ps[pool] - logit_ps[src])
        order = np.argsort(dist)
        picks: list[int] = []
        for j in order:
            cand = int(pool[j])
            if caliper_dist is not None and dist[j] > caliper_dist:
                break
            if (not replacement) and cand in picks:
                continue
            picks.append(cand)
            if len(picks) == k:
                break
        return picks if picks else None

    available_controls = list(control_idx)
    for i in treated_idx:
        pool = np.array(available_controls, dtype=int) if not replacement else control_idx
        if len(pool) == 0:
            dropped += 1
            continue
        matches = _match_one(int(i), pool)
        if matches is None:
            dropped += 1
            continue
        contribs.append(float(Y[i] - np.mean(Y[matches])))
        used_treated.append(int(i))
        used_control.extend(matches)
        matched_pairs.append((int(i), matches))
        if not replacement:
            available_controls = [c for c in available_controls if c not in matches]

    if target == "att":
        ate = float(np.mean(contribs)) if contribs else float("nan")
    else:
        # Also match controls to treated for the ATE
        available_treated = list(treated_idx)
        for j in control_idx:
            pool = np.array(available_treated, dtype=int) if not replacement else treated_idx
            if len(pool) == 0:
                dropped += 1
                continue
            matches = _match_one(int(j), pool)
            if matches is None:
                dropped += 1
                continue
            contribs.append(float(np.mean(Y[matches]) - Y[j]))
            used_control.append(int(j))
            used_treated.extend(matches)
            matched_pairs.append((int(j), matches))
            if not replacement:
                available_treated = [t for t in available_treated if t not in matches]
        ate = float(np.mean(contribs)) if contribs else float("nan")

    return ate, {
        "n_matched_pairs": len(matched_pairs),
        "n_dropped": dropped,
        "pairs": matched_pairs,
        "caliper_distance": caliper_dist,
    }
