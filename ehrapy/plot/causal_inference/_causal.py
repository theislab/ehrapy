"""HoloViews plots for the causal inference module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from ehrapy.tools.causal import CausalEstimate


_UNWEIGHTED_COLOR = "#1f77b4"
_WEIGHTED_COLOR = "#d62728"
_GUIDE_COLOR = "#404040"


def love_plot(
    balance: pd.DataFrame,
    *,
    threshold: float = 0.1,
    title: str | None = None,
    width: int = 520,
    height: int | None = None,
) -> hv.Overlay:
    """Plot a "love plot" of standardised mean differences before and after weighting.

    Covariates are drawn on the y-axis sorted by their unweighted absolute SMD, with vertical guide lines at ``±threshold`` (commonly 0.1).

    Args:
        balance: Output of :func:`~ehrapy.tools.covariate_balance`, a DataFrame indexed by covariate name with ``smd_unweighted`` and ``smd_weighted`` columns.
        threshold: SMD magnitude used for the dashed guide lines.
        title: Plot title.
            If ``None``, defaults to ``"Covariate balance"``.
        width: Plot width in pixels.
        height: Plot height in pixels.
            If ``None``, ``height`` is set to ``28 * n_covariates + 80``.

    Returns:
        A :class:`holoviews.Overlay` containing the scatter points, connecting lines, and guide lines.
    """
    bal = balance.reindex(balance["smd_unweighted"].abs().sort_values(ascending=True).index)
    n = len(bal)
    if height is None:
        height = max(28 * n + 80, 200)

    covariates = list(bal.index)
    smd_u = bal["smd_unweighted"].to_numpy()
    smd_w = bal["smd_weighted"].to_numpy()

    scatter_u = hv.Scatter((smd_u, covariates), kdims="SMD", vdims="covariate", label="unweighted").opts(
        color=_UNWEIGHTED_COLOR, size=8, tools=["hover"]
    )
    scatter_w = hv.Scatter((smd_w, covariates), kdims="SMD", vdims="covariate", label="weighted").opts(
        color=_WEIGHTED_COLOR, size=8, marker="x", tools=["hover"]
    )
    connectors = hv.Segments(
        (smd_u, covariates, smd_w, covariates),
        kdims=["x0", "y0", "x1", "y1"],
    ).opts(color="grey", alpha=0.4, line_width=1)

    zero = hv.VLine(0).opts(color=_GUIDE_COLOR, line_width=1)
    upper = hv.VLine(threshold).opts(color=_GUIDE_COLOR, line_width=1, line_dash="dashed")
    lower = hv.VLine(-threshold).opts(color=_GUIDE_COLOR, line_width=1, line_dash="dashed")

    overlay = (connectors * scatter_u * scatter_w * zero * upper * lower).opts(
        width=width,
        height=height,
        xlabel="Standardised mean difference",
        ylabel="",
        title=title or "Covariate balance",
        show_legend=True,
        legend_position="bottom_right",
    )
    return overlay


def propensity_overlap(
    positivity: dict,
    *,
    bins: int = 40,
    title: str | None = None,
    width: int = 520,
    height: int = 320,
) -> hv.Overlay:
    """Plot overlapping propensity score histograms for treated and untreated groups.

    Use the dict returned by :func:`~ehrapy.tools.positivity_check`.
    A lack of overlap between the two arms is the visual signature of a positivity violation.

    Args:
        positivity: Output of :func:`~ehrapy.tools.positivity_check`.
        bins: Number of histogram bins per arm.
        title: Plot title.
            If ``None``, defaults to a string describing the support fraction.
        width: Plot width in pixels.
        height: Plot height in pixels.

    Returns:
        A :class:`holoviews.Overlay` containing one histogram per arm and the support-boundary guide lines.
    """
    ps = np.asarray(positivity["propensity_scores"])
    T = np.asarray(positivity["treatment"])
    eps = positivity["eps"]

    counts_u, edges_u = np.histogram(ps[T == 0], bins=bins, density=True)
    counts_t, edges_t = np.histogram(ps[T == 1], bins=bins, density=True)
    hist_u = hv.Histogram((edges_u, counts_u), kdims="propensity_score", vdims="density", label="untreated").opts(
        fill_color=_UNWEIGHTED_COLOR, alpha=0.5, line_alpha=0
    )
    hist_t = hv.Histogram((edges_t, counts_t), kdims="propensity_score", vdims="density", label="treated").opts(
        fill_color=_WEIGHTED_COLOR, alpha=0.5, line_alpha=0
    )

    eps_low = hv.VLine(eps).opts(color=_GUIDE_COLOR, line_width=1, line_dash="dashed")
    eps_high = hv.VLine(1 - eps).opts(color=_GUIDE_COLOR, line_width=1, line_dash="dashed")

    return (hist_u * hist_t * eps_low * eps_high).opts(
        width=width,
        height=height,
        xlabel="Propensity score",
        ylabel="Density",
        title=title or f"Propensity overlap (support fraction: {positivity['support_fraction']:.2f})",
        show_legend=True,
    )


def causal_effect(
    estimate: CausalEstimate,
    *,
    other: dict[str, CausalEstimate] | None = None,
    title: str | None = None,
    width: int = 520,
    height: int = 260,
) -> hv.Overlay:
    """Plot a single causal estimate, or a comparison across estimators, as a forest-style plot.

    With just ``estimate`` the plot is a single point estimate plus 95% confidence interval (when available).
    With ``other`` supplied one row per estimator is drawn so methods can be compared side by side.

    Args:
        estimate: The primary :class:`~ehrapy.tools.CausalEstimate` to display.
        other: Optional mapping ``{label: estimate}`` of additional estimates to plot below the primary one.
        title: Plot title.
            If ``None``, defaults to ``"Causal effect estimate"``.
        width: Plot width in pixels.
        height: Plot height in pixels.

    Returns:
        A :class:`holoviews.Overlay` containing the point estimates, confidence-interval segments, and zero-line.
    """
    items: list[tuple[str, CausalEstimate]] = [(estimate.method, estimate)]
    if other:
        items.extend(other.items())

    labels = [label for label, _ in items]
    values = [est.value for _, est in items]

    points = hv.Scatter((values, labels), kdims="effect", vdims="estimator").opts(
        color="black", size=10, tools=["hover"]
    )

    ci_segments = []
    for label, est in items:
        if est.ci_lower is not None and est.ci_upper is not None:
            ci_segments.append((est.ci_lower, label, est.ci_upper, label))
    if ci_segments:
        segs = hv.Segments(ci_segments, kdims=["x0", "y0", "x1", "y1"]).opts(color="black", line_width=2)
    else:
        segs = hv.Segments([]).opts(color="black", line_width=2)

    zero = hv.VLine(0).opts(color="grey", line_width=1, line_dash="dashed")

    return (segs * points * zero).opts(
        width=width,
        height=height,
        xlabel=f"Estimated effect of '{estimate.treatment}' on '{estimate.outcome}'",
        ylabel="",
        title=title or "Causal effect estimate",
    )
