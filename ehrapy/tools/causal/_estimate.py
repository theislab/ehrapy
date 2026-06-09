from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CausalEstimate:
    """Result of a causal effect estimation."""

    #: Name of the estimator that produced this estimate.
    method: str
    #: Treatment variable name.
    treatment: str
    #: Outcome variable name.
    outcome: str
    #: Point estimate of the average treatment effect (ATE).
    value: float
    #: Standard error of the estimate, when available.
    se: float | None = None
    #: Lower bound of the (typically 95%) confidence interval, when available.
    ci_lower: float | None = None
    #: Upper bound of the (typically 95%) confidence interval, when available.
    ci_upper: float | None = None
    #: Number of observations used to compute the estimate.
    n: int | None = None
    #: Estimator-specific metadata such as fitted propensity scores or effective sample size.
    params: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [
            f"CausalEstimate(method={self.method!r}",
            f"treatment={self.treatment!r}",
            f"outcome={self.outcome!r}",
            f"value={self.value:.4f}",
        ]
        if self.se is not None:
            parts.append(f"se={self.se:.4f}")
        if self.ci_lower is not None and self.ci_upper is not None:
            parts.append(f"ci=[{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        if self.n is not None:
            parts.append(f"n={self.n}")
        return ", ".join(parts) + ")"

    def summary(self) -> str:
        """Return a multi-line human-readable summary of the estimate."""
        lines = [
            f"Causal effect of '{self.treatment}' on '{self.outcome}'",
            f"  method: {self.method}",
            f"  ATE:    {self.value:.4f}",
        ]
        if self.se is not None:
            lines.append(f"  SE:     {self.se:.4f}")
        if self.ci_lower is not None and self.ci_upper is not None:
            lines.append(f"  95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        if self.n is not None:
            lines.append(f"  n:      {self.n}")
        return "\n".join(lines)
