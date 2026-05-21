"""Diagnostics for causal estimators: covariate balance and positivity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_compat
import numpy as np
import pandas as pd

from ehrapy._compat import function_2D_only
from ehrapy.tools.causal._design import Design, _collect_columns, assert_binary_treatment
from ehrapy.tools.causal._models import fit_propensity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData
    from sklearn.base import BaseEstimator


@function_2D_only()
def covariate_balance(
    edata: EHRData,
    treatment: str,
    *,
    covariates: Sequence[str],
    weights: np.ndarray | None = None,
    propensity_model: str | BaseEstimator = "logistic",
    layer: str | None = None,
) -> pd.DataFrame:
    """Report standardised mean differences (SMD) for each covariate, before and after weighting.

    SMD is the standard "love-plot" diagnostic: ``(mean_treated − mean_control) / pooled_SD``.
    Values with ``|SMD| < 0.1`` are conventionally considered balanced.
    The variance ratio is ``Var_treated / Var_control``; values near 1 indicate similar spread between treatment arms.

    When ``weights`` is provided, the "after" columns use the supplied weights (typically the IPTW output stored in ``estimate.params['weights']``).
    When ``weights`` is ``None``, this function fits its own propensity model and computes IPTW weights internally.

    Args:
        edata: Central data object.
        treatment: Column name of the binary (0/1) treatment variable.
        covariates: Adjustment set to evaluate.
            Each entry must refer to a name in ``edata.var_names`` or ``edata.obs.columns``.
        weights: Optional pre-computed IPTW weight vector aligned with ``edata.obs.index``.
            When ``None``, weights are computed internally from a freshly fitted propensity model.
        propensity_model: Propensity model used to compute weights when ``weights`` is ``None`` (see :func:`~ehrapy.tools.iptw` for the accepted values).
        layer: Layer of ``edata`` to draw the var-side variables from.
            If ``None``, ``edata.X`` is used.

    Returns:
        A DataFrame indexed by covariate name with columns ``smd_unweighted``, ``smd_weighted``, ``var_ratio_unweighted``, and ``var_ratio_weighted``.

    Examples:
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2_preprocessed()
        >>> bal = ep.tl.covariate_balance(
        ...     edata,
        ...     "aline_flg",
        ...     covariates=["age", "sofa_first", "sapsi_first"],
        ... )
        >>> print(bal.round(3).to_string())
                     smd_unweighted  smd_weighted  var_ratio_unweighted  var_ratio_weighted
        age                   0.117        -0.044                 0.896               1.018
        sofa_first            0.818        -0.220                 1.135               0.480
        sapsi_first           0.627        -0.157                 1.112               0.781
    """
    design = _build_design_no_outcome(edata, treatment=treatment, covariates=covariates, layer=layer)
    assert_binary_treatment(design.T, treatment)

    if weights is None:
        ps, _ = fit_propensity(propensity_model, design.X, design.T, clip=(0.01, 0.99))
        weights = design.T / ps + (1 - design.T) / (1 - ps)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != edata.n_obs:
            raise ValueError(f"weights has length {len(weights)} but edata has {edata.n_obs} observations.")
        # align to the rows that survived NaN filtering in build_design
        pos = edata.obs.index.get_indexer(design.index)
        weights = weights[pos]

    treated = design.T == 1
    smd_u = _smd(design.X, treated)
    smd_w = _smd(design.X, treated, weights=weights)
    vr_u = _variance_ratio(design.X, treated)
    vr_w = _variance_ratio(design.X, treated, weights=weights)

    return pd.DataFrame(
        {
            "smd_unweighted": smd_u,
            "smd_weighted": smd_w,
            "var_ratio_unweighted": vr_u,
            "var_ratio_weighted": vr_w,
        },
        index=design.feature_names,
    )


@function_2D_only()
def positivity_check(
    edata: EHRData,
    treatment: str,
    *,
    covariates: Sequence[str],
    propensity_model: str | BaseEstimator = "logistic",
    eps: float = 0.05,
    layer: str | None = None,
) -> dict:
    """Diagnose the positivity assumption by inspecting the propensity score distribution.

    Returns summary statistics of the fitted propensity scores by treatment arm together with the fraction of observations whose propensity lies inside ``[eps, 1 − eps]`` (the "common support" region).
    Severe positivity violations show up as bimodal propensity distributions or small support fractions.

    Args:
        edata: Central data object.
        treatment: Column name of the binary (0/1) treatment variable.
        covariates: Adjustment set used to fit the propensity model.
            Each entry must refer to a name in ``edata.var_names`` or ``edata.obs.columns``.
        propensity_model: Propensity model specification (see :func:`~ehrapy.tools.iptw` for the accepted values).
        eps: Lower (and ``1 − eps`` upper) boundary of the common-support interval.
        layer: Layer of ``edata`` to draw the var-side variables from.
            If ``None``, ``edata.X`` is used.

    Returns:
        A dict with keys ``propensity_scores``, ``treatment``, ``index``, ``eps``, ``support_fraction``, ``n_outside_support``, ``summary_treated``, and ``summary_untreated``.

    Examples:
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2_preprocessed()
        >>> info = ep.tl.positivity_check(
        ...     edata,
        ...     "aline_flg",
        ...     covariates=["age", "sofa_first", "sapsi_first"],
        ... )
        >>> print(f"support_fraction={info['support_fraction']:.3f}  n_outside_support={info['n_outside_support']}")
        support_fraction=0.981  n_outside_support=34
    """
    design = _build_design_no_outcome(edata, treatment=treatment, covariates=covariates, layer=layer)
    assert_binary_treatment(design.T, treatment)

    ps, _ = fit_propensity(propensity_model, design.X, design.T, clip=None)
    treated = design.T == 1
    in_support = (ps >= eps) & (ps <= 1 - eps)

    def _summary(p: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(p)),
            "max": float(np.max(p)),
            "mean": float(np.mean(p)),
            "median": float(np.median(p)),
            "p05": float(np.quantile(p, 0.05)),
            "p95": float(np.quantile(p, 0.95)),
        }

    return {
        "propensity_scores": ps,
        "treatment": design.T,
        "index": design.index,
        "eps": eps,
        "support_fraction": float(np.mean(in_support)),
        "n_outside_support": int(np.sum(~in_support)),
        "summary_treated": _summary(ps[treated]),
        "summary_untreated": _summary(ps[~treated]),
    }


# ---------- internals ----------


def _build_design_no_outcome(edata, *, treatment, covariates, layer):
    """Like ``build_design`` but for treatment-balance use cases that don't have an outcome vector."""
    df = _collect_columns(edata, [treatment, *covariates], layer=layer).dropna(axis=0, how="any")
    if len(df) < 2:
        raise ValueError(f"Design has only {len(df)} complete rows; need at least 2.")
    T = pd.to_numeric(df[treatment], errors="raise").to_numpy(dtype=float)
    if covariates:
        cov_encoded = pd.get_dummies(df[list(covariates)], drop_first=True, dummy_na=False).astype(float)
        X = cov_encoded.to_numpy(dtype=float)
        feature_names = list(cov_encoded.columns)
    else:
        X = np.zeros((len(df), 0), dtype=float)
        feature_names = []
    return Design(T=T, Y=T.copy(), X=X, feature_names=feature_names, index=df.index)


def _group_moments(X, *, group_mask, weights):
    """Compute (mean, var) for a group, optionally weighted, in a backend-agnostic way using array_api_compat."""
    xp = array_api_compat.array_namespace(X)
    X_g = X[group_mask]
    if weights is None:
        n = X_g.shape[0]
        if n < 2:
            zeros = xp.zeros(X.shape[1], dtype=X.dtype)
            return zeros, zeros
        m = xp.mean(X_g, axis=0)
        v = xp.sum((X_g - m) ** 2, axis=0) / (n - 1)
        return m, v
    w = weights[group_mask]
    w_sum = xp.sum(w)
    m = xp.sum(w[:, None] * X_g, axis=0) / w_sum
    v = xp.sum(w[:, None] * (X_g - m) ** 2, axis=0) / w_sum
    return m, v


def _smd(X: np.ndarray, treated: np.ndarray, *, weights: np.ndarray | None = None) -> np.ndarray:
    """Standardised mean difference per column, computed via the array API namespace of ``X``."""
    xp = array_api_compat.array_namespace(X)
    m1, v1 = _group_moments(X, group_mask=treated, weights=weights)
    m0, v0 = _group_moments(X, group_mask=~treated, weights=weights)
    pooled = xp.sqrt((v1 + v0) / 2)
    safe = pooled > 0
    return xp.where(safe, (m1 - m0) / xp.where(safe, pooled, xp.ones_like(pooled)), xp.zeros_like(pooled))


def _variance_ratio(X: np.ndarray, treated: np.ndarray, *, weights: np.ndarray | None = None) -> np.ndarray:
    """Variance ratio ``Var(X | T=1) / Var(X | T=0)`` per column, computed via the array API."""
    xp = array_api_compat.array_namespace(X)
    _, v1 = _group_moments(X, group_mask=treated, weights=weights)
    _, v0 = _group_moments(X, group_mask=~treated, weights=weights)
    safe = v0 > 0
    nan = xp.full(v0.shape, float("nan"), dtype=v0.dtype)
    return xp.where(safe, v1 / xp.where(safe, v0, xp.ones_like(v0)), nan)
