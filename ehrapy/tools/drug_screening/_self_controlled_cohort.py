from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Literal

from scipy import stats


@dataclass(frozen=True)
class RateRatioResult:
    """Result container for the exact rate-ratio test used in the original R code."""

    rate_ratio: float
    conf_int: tuple[float, float]
    p_value: float


def rate_ratio_test(
    x: tuple[int, int] | list[int],
    n: tuple[int, int] | list[int],
    rr: float = 1.0,
    alternative: Literal["two.sided", "less", "greater"] = "two.sided",
    conf_level: float = 0.95,
) -> RateRatioResult:
    """Perform the exact rate-ratio test used in the original screening scripts.

    This is a direct translation of ``rate.ratio.test`` from the R workflow in
    ``original/*/screening_drug_*.R``.

    Args:
        x: Event counts as ``[Y, X]`` for the exposed and comparison windows.
        n: Time-at-risk denominators as ``[N, M]`` for the exposed and comparison windows.
        rr: Null-hypothesis rate ratio.
        alternative: Alternative hypothesis.
        conf_level: Confidence level for the interval estimate.

    Returns:
        A dataclass containing the point estimate, confidence interval, and p-value.
    """
    if len(x) != 2 or len(n) != 2:
        raise ValueError("x and n must each contain exactly two values")
    if alternative not in {"two.sided", "less", "greater"}:
        raise ValueError("alternative must be one of 'two.sided', 'less', or 'greater'")

    y, x_count = (int(value) for value in x)
    n_exposed, n_unexposed = (float(value) for value in n)

    if min(y, x_count) < 0:
        raise ValueError("event counts must be non-negative")
    if min(n_exposed, n_unexposed) <= 0:
        raise ValueError("time-at-risk denominators must be positive")

    if x_count == 0:
        rate_ratio = inf if y > 0 else float("nan")
    else:
        rate_ratio = (y / n_exposed) / (x_count / n_unexposed)
    p_rr = (n_exposed * rr) / (n_exposed * rr + n_unexposed)
    total_events = y + x_count

    pval_less = stats.binom.cdf(y, total_events, p_rr)
    pval_greater = stats.binom.sf(y - 1, total_events, p_rr)

    if alternative == "less":
        p_value = float(pval_less)
        conf_int = (
            0.0,
            _beta_to_rate_ratio(
                _beta_upper(y, total_events, 1 - conf_level),
                n_exposed=n_exposed,
                n_unexposed=n_unexposed,
            ),
        )
    elif alternative == "greater":
        p_value = float(pval_greater)
        conf_int = (
            _beta_to_rate_ratio(
                _beta_lower(y, total_events, 1 - conf_level),
                n_exposed=n_exposed,
                n_unexposed=n_unexposed,
            ),
            inf,
        )
    else:
        p_value = float(min(1.0, 2 * min(pval_less, pval_greater)))
        alpha = (1 - conf_level) / 2
        conf_int = (
            _beta_to_rate_ratio(
                _beta_lower(y, total_events, alpha),
                n_exposed=n_exposed,
                n_unexposed=n_unexposed,
            ),
            _beta_to_rate_ratio(
                _beta_upper(y, total_events, alpha),
                n_exposed=n_exposed,
                n_unexposed=n_unexposed,
            ),
        )

    return RateRatioResult(rate_ratio=float(rate_ratio), conf_int=conf_int, p_value=p_value)


def _beta_lower(x_count: int, total: int, alpha: float) -> float:
    if x_count == 0:
        return 0.0
    return float(stats.beta.ppf(alpha, x_count, total - x_count + 1))


def _beta_upper(x_count: int, total: int, alpha: float) -> float:
    if x_count == total:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, x_count + 1, total - x_count))


def _beta_to_rate_ratio(probability: float, *, n_exposed: float, n_unexposed: float) -> float:
    if probability >= 1:
        return inf
    if probability <= 0:
        return 0.0
    return float((probability * n_unexposed) / (n_exposed * (1 - probability)))
