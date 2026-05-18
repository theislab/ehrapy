"""Causal inference for EHR data.

This module provides a small, dependency-light set of estimators built directly on top of scikit-learn.
ATE estimators (binary treatment) cover IPTW, parametric g-computation, doubly robust AIPW, and
propensity score matching.
CATE / heterogeneous treatment effect estimation is provided via the T-, S-, and X-learner
meta-learners.
Two diagnostics — covariate balance (love plot) and positivity (propensity overlap) — complete the
toolkit.

All public functions accept an :class:`~ehrdata.EHRData` object and return a
:class:`CausalEstimate` (estimators) or a :class:`~pandas.DataFrame`/``dict`` (diagnostics).
"""

from ehrapy.tools.causal._ate import aipw, g_computation, iptw, propensity_score_matching
from ehrapy.tools.causal._diagnostics import covariate_balance, positivity_check
from ehrapy.tools.causal._estimate import CausalEstimate
from ehrapy.tools.causal._hte import s_learner, t_learner, x_learner

__all__ = [
    "CausalEstimate",
    "aipw",
    "covariate_balance",
    "g_computation",
    "iptw",
    "positivity_check",
    "propensity_score_matching",
    "s_learner",
    "t_learner",
    "x_learner",
]
