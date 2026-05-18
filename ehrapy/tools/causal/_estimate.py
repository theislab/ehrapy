from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CausalEstimate:
    """Result of a causal effect estimation.

    Attributes:
        method: Name of the estimator that produced this estimate.
        treatment: Treatment variable name.
        outcome: Outcome variable name.
        value: Point estimate of the average treatment effect (ATE).
        se: Standard error of the estimate, when available.
        ci_lower: Lower bound of the (typically 95%) confidence interval, when available.
        ci_upper: Upper bound of the (typically 95%) confidence interval, when available.
        n: Number of observations used to compute the estimate.
        params: Estimator-specific metadata such as fitted propensity scores or effective sample size.
    """

    method: str
    treatment: str
    outcome: str
    value: float
    se: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    n: int | None = None
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
