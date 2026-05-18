"""Meta-learners for the heterogeneous (conditional) treatment effect (CATE).

Each learner fits one or more sklearn-style outcome models and returns the per-observation CATE predictions ``τ(X_i) = E[Y(1) − Y(0) | X = X_i]``.
The ATE summary written to :attr:`~ehrapy.tools.causal.CausalEstimate.value` is ``mean(τ(X))``; the full per-observation CATE vector lives in ``estimate.params['cate']`` and is also written back to ``edata.obs`` under ``key_added`` when supplied.
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


def _store_cate(edata: EHRData, design_index, cate: np.ndarray, key_added: str | None) -> None:
    """Optionally write the CATE vector back to ``edata.obs[key_added]``, NaN where dropped."""
    if key_added is None:
        return
    full = np.full(edata.n_obs, np.nan, dtype=float)
    pos = edata.obs.index.get_indexer(design_index)
    full[pos] = cate
    edata.obs[key_added] = full


@function_2D_only()
def t_learner(
    edata: EHRData,
    treatment: str,
    outcome: str,
    *,
    covariates: Sequence[str],
    outcome_model: str | BaseEstimator = "auto",
    key_added: str | None = None,
    layer: str | None = None,
) -> CausalEstimate:
    """Two-model (T-learner) CATE estimator.

    Fits a separate outcome model on the treated subset (``μ_1``) and on the untreated subset (``μ_0``), then computes ``τ(X_i) = μ_1(X_i) − μ_0(X_i)`` for every row.
    Simple but statistically inefficient when the two groups are imbalanced.

    Args:
        edata: Central data object.
        treatment: Column name of the binary (0/1) treatment variable.
        outcome: Column name of the outcome variable.
        covariates: Adjustment set used for the outcome models.
            Each entry must refer to a name in ``edata.var_names`` or ``edata.obs.columns``.
        outcome_model: Outcome model specification (see :func:`~ehrapy.tl.g_computation` for the accepted values).
        key_added: Optional ``edata.obs`` column name into which the per-observation CATE vector is written.
            Observations dropped during NaN filtering are filled with ``NaN``.
        layer: Layer of ``edata`` to draw the var-side variables from.
            If ``None``, ``edata.X`` is used.

    Returns:
        A :class:`~ehrapy.tools.causal.CausalEstimate` whose ``value`` is the average CATE and whose ``params['cate']`` is the per-observation CATE vector.
    """
    design = build_design(edata, treatment=treatment, outcome=outcome, covariates=covariates, layer=layer)
    assert_binary_treatment(design.T, treatment)

    treated = design.T == 1
    m1 = resolve_outcome_model(outcome_model, y=design.Y[treated])
    m0 = resolve_outcome_model(outcome_model, y=design.Y[~treated])

    m1.fit(design.X[treated], _y_for_fit(m1, design.Y[treated]))
    m0.fit(design.X[~treated], _y_for_fit(m0, design.Y[~treated]))

    mu1 = predict_mean(m1, design.X)
    mu0 = predict_mean(m0, design.X)
    cate = mu1 - mu0
    _store_cate(edata, design.index, cate, key_added)

    return CausalEstimate(
        method="t_learner",
        treatment=treatment,
        outcome=outcome,
        value=float(np.mean(cate)),
        n=int(len(design.T)),
        params={"cate": cate, "mu1": mu1, "mu0": mu0, "index": design.index, "feature_names": design.feature_names},
    )


@function_2D_only()
def s_learner(
    edata: EHRData,
    treatment: str,
    outcome: str,
    *,
    covariates: Sequence[str],
    outcome_model: str | BaseEstimator = "auto",
    key_added: str | None = None,
    layer: str | None = None,
) -> CausalEstimate:
    """Single-model (S-learner) CATE estimator.

    Fits one outcome model ``μ(T, X)`` on all data, then predicts ``τ(X_i) = μ(1, X_i) − μ(0, X_i)``.
    Tends to regularise the treatment effect toward zero when the base learner is heavily regularised, so consider a flexible base learner if you suspect heterogeneity.

    Args:
        edata: Central data object.
        treatment: Column name of the binary (0/1) treatment variable.
        outcome: Column name of the outcome variable.
        covariates: Adjustment set used for the outcome model.
            Each entry must refer to a name in ``edata.var_names`` or ``edata.obs.columns``.
        outcome_model: Outcome model specification (see :func:`~ehrapy.tl.g_computation` for the accepted values).
        key_added: Optional ``edata.obs`` column name into which the per-observation CATE vector is written.
            Observations dropped during NaN filtering are filled with ``NaN``.
        layer: Layer of ``edata`` to draw the var-side variables from.
            If ``None``, ``edata.X`` is used.

    Returns:
        A :class:`~ehrapy.tools.causal.CausalEstimate` whose ``value`` is the average CATE and whose ``params['cate']`` is the per-observation CATE vector.
    """
    design = build_design(edata, treatment=treatment, outcome=outcome, covariates=covariates, layer=layer)
    assert_binary_treatment(design.T, treatment)

    model = resolve_outcome_model(outcome_model, y=design.Y)
    XT = np.column_stack([design.T, design.X])
    model.fit(XT, _y_for_fit(model, design.Y))
    X1 = np.column_stack([np.ones_like(design.T), design.X])
    X0 = np.column_stack([np.zeros_like(design.T), design.X])
    cate = predict_mean(model, X1) - predict_mean(model, X0)
    _store_cate(edata, design.index, cate, key_added)

    return CausalEstimate(
        method="s_learner",
        treatment=treatment,
        outcome=outcome,
        value=float(np.mean(cate)),
        n=int(len(design.T)),
        params={"cate": cate, "index": design.index, "feature_names": design.feature_names},
    )


@function_2D_only()
def x_learner(
    edata: EHRData,
    treatment: str,
    outcome: str,
    *,
    covariates: Sequence[str],
    outcome_model: str | BaseEstimator = "auto",
    propensity_model: str | BaseEstimator = "logistic",
    cate_model: str | BaseEstimator = "auto",
    clip: tuple[float, float] | None = (0.01, 0.99),
    key_added: str | None = None,
    layer: str | None = None,
) -> CausalEstimate:
    """X-learner CATE estimator of Künzel et al. (2019).

    First fits group-specific outcome models ``μ_0`` and ``μ_1`` (like the T-learner), then imputes individual treatment effects on each group's own units::

        D_1 = Y_1 − μ_0(X_1),   D_0 = μ_1(X_0) − Y_0

    Two CATE models ``τ_0`` and ``τ_1`` are fitted on these imputed effects and combined as ``τ(x) = g(x) τ_0(x) + (1 − g(x)) τ_1(x)`` where ``g`` is the propensity score.
    More efficient than the T-learner when treatment groups are imbalanced.

    Args:
        edata: Central data object.
        treatment: Column name of the binary (0/1) treatment variable.
        outcome: Column name of the outcome variable.
        covariates: Adjustment set used for both the outcome and propensity models.
            Each entry must refer to a name in ``edata.var_names`` or ``edata.obs.columns``.
        outcome_model: Outcome model specification for the first-stage ``μ`` models (see :func:`~ehrapy.tl.g_computation` for the accepted values).
        propensity_model: Propensity model specification (see :func:`~ehrapy.tl.iptw` for the accepted values).
        cate_model: Regressor used for the second-stage ``τ`` models.
            Accepts ``'auto'``/``'linear'``/``'gradient_boosting'``/``'random_forest'`` or any sklearn-compatible regressor.
            ``'auto'`` resolves to linear regression.
            Classifiers are rejected because the imputed effects are continuous.
        clip: ``(lo, hi)`` propensity-score clipping range for the combination weight ``g``.
            Use ``None`` to disable clipping.
        key_added: Optional ``edata.obs`` column name into which the per-observation CATE vector is written.
            Observations dropped during NaN filtering are filled with ``NaN``.
        layer: Layer of ``edata`` to draw the var-side variables from.
            If ``None``, ``edata.X`` is used.

    Returns:
        A :class:`~ehrapy.tools.causal.CausalEstimate` whose ``value`` is the average CATE and whose ``params['cate']`` is the per-observation CATE vector.
    """
    design = build_design(edata, treatment=treatment, outcome=outcome, covariates=covariates, layer=layer)
    assert_binary_treatment(design.T, treatment)

    treated = design.T == 1
    untreated = ~treated

    mu0_model = resolve_outcome_model(outcome_model, y=design.Y[untreated])
    mu1_model = resolve_outcome_model(outcome_model, y=design.Y[treated])
    mu0_model.fit(design.X[untreated], _y_for_fit(mu0_model, design.Y[untreated]))
    mu1_model.fit(design.X[treated], _y_for_fit(mu1_model, design.Y[treated]))

    d1 = design.Y[treated] - predict_mean(mu0_model, design.X[treated])
    d0 = predict_mean(mu1_model, design.X[untreated]) - design.Y[untreated]

    # Stage 2 always uses regressors on continuous imputed effects.
    tau1 = _resolve_regressor(cate_model)
    tau0 = _resolve_regressor(cate_model)
    tau1.fit(design.X[treated], d1)
    tau0.fit(design.X[untreated], d0)

    ps, _ = fit_propensity(propensity_model, design.X, design.T, clip=clip)
    cate = ps * tau0.predict(design.X) + (1 - ps) * tau1.predict(design.X)
    _store_cate(edata, design.index, cate, key_added)

    return CausalEstimate(
        method="x_learner",
        treatment=treatment,
        outcome=outcome,
        value=float(np.mean(cate)),
        n=int(len(design.T)),
        params={
            "cate": cate,
            "propensity_scores": ps,
            "index": design.index,
            "feature_names": design.feature_names,
        },
    )


def _resolve_regressor(spec: str | BaseEstimator) -> BaseEstimator:
    """Resolve a regressor for the X-learner stage 2; refuse classifiers."""
    if isinstance(spec, str) and spec == "auto":
        from sklearn.linear_model import LinearRegression

        return LinearRegression()
    if isinstance(spec, str) and spec == "linear":
        from sklearn.linear_model import LinearRegression

        return LinearRegression()
    if isinstance(spec, str) and spec == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor()
    if isinstance(spec, str) and spec == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=200, n_jobs=-1)

    from sklearn.base import clone

    cloned = clone(spec)
    if hasattr(cloned, "predict_proba"):
        raise TypeError(
            f"cate_model must be a regressor; got classifier {type(cloned).__name__}. "
            "The X-learner's stage-2 models predict continuous imputed treatment effects."
        )
    return cloned


def _y_for_fit(model, y: np.ndarray) -> np.ndarray:
    """Cast y to int for classifiers, leave float for regressors."""
    return y.astype(int) if hasattr(model, "predict_proba") else y
