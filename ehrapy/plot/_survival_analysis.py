from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

import ehrapy as ep

if TYPE_CHECKING:
    from collections.abc import Sequence
    from xmlrpc.client import Boolean

    from anndata import AnnData
    from lifelines import KaplanMeierFitter
    from matplotlib.axes import Axes
    from statsmodels.regression.linear_model import RegressionResults


def ols(
    adata: AnnData | None = None,
    x: str | None = None,
    y: str | None = None,
    scatter_plot: Boolean | None = True,
    ols_results: list[RegressionResults] | None = None,
    ols_color: list[str] | None | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    lines: list[tuple[ndarray | float, ndarray | float]] | None = None,
    lines_color: list[str] | None | None = None,
    lines_style: list[str] | None | None = None,
    lines_label: list[str] | None | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show: bool | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    **kwds,
):
    """Plots a Ordinary Least Squares (OLS) Model result, scatter plot, and line plot.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        x: x coordinate, for scatter plotting.
        y: y coordinate, for scatter plotting.
        scatter_plot: If True, show scatter plot. Defaults to True.
        ols_results: List of RegressionResults from ehrapy.tl.ols. Example: [result_1, result_2]
        ols_color: List of colors for each ols_results. Example: ['red', 'blue'].
        xlabel: The x-axis label text.
        ylabel: The y-axis label text.
        figsize: Width, height in inches. Defaults to None.
        lines: List of Tuples of (slope, intercept) or (x, y). Plot lines by slope and intercept or data points.
               Example: plot two lines (y = x + 2 and y = 2*x + 1): [(1, 2), (2, 1)]
        lines_color: List of colors for each line. Example: ['red', 'blue']
        lines_style: List of line styles for each line. Example: ['-', '--']
        lines_label: List of line labels for each line. Example: ['Line1', 'Line2']
        xlim: Set the x-axis view limits. Required for only plotting lines using slope and intercept.
        ylim: Set the y-axis view limits. Required for only plotting lines using slope and intercept.
        show: Show the plot, do not return axis.
        ax: A matplotlib axes object. Only works if plotting a single component.
        title: Set the title of the plot.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> co2_lm_result = ep.tl.ols(
        ...     adata, var_names=["pco2_first", "tco2_first"], formula="tco2_first ~ pco2_first", missing="drop"
        ... ).fit()
        >>> ep.pl.ols(
        ...     adata,
        ...     x="pco2_first",
        ...     y="tco2_first",
        ...     ols_results=[co2_lm_result],
        ...     ols_color=["red"],
        ...     xlabel="PCO2",
        ...     ylabel="TCO2",
        ... )

        .. image:: /_static/docstring_previews/ols_plot_1.png

        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> ep.pl.ols(adata, x='pco2_first', y='tco2_first', lines=[(0.25, 10), (0.3, 20)],
        >>>           lines_color=['red', 'blue'], lines_style=['-', ':'], lines_label=['Line1', 'Line2'])

        .. image:: /_static/docstring_previews/ols_plot_2.png

        >>> import ehrapy as ep
        >>> ep.pl.ols(lines=[(0.25, 10), (0.3, 20)], lines_color=['red', 'blue'], lines_style=['-', ':'],
        >>>           lines_label=['Line1', 'Line2'], xlim=(0, 150), ylim=(0, 50))

        .. image:: /_static/docstring_previews/ols_plot_3.png
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if ols_color is None and ols_results is not None:
        ols_color = [None] * len(ols_results)
    if lines_color is None and lines is not None:
        lines_color = [None] * len(lines)
    if lines_style is None and lines is not None:
        lines_style = [None] * len(lines)
    if lines_label is None and lines is not None:
        lines_label = [None] * len(lines)
    if adata is not None and x is not None and y is not None:
        x_processed = np.array(adata[:, x].X).astype(float)
        x_processed = x_processed[~np.isnan(x_processed)]
        if scatter_plot is True:
            ax = ep.pl.scatter(adata, x=x, y=y, show=False, ax=ax, **kwds)
        if ols_results is not None:
            for i, ols_result in enumerate(ols_results):
                ax.plot(x_processed, ols_result.predict(), color=ols_color[i])

    if lines is not None:
        for i, line in enumerate(lines):
            a, b = line
            if np.ndim(a) == 0 and np.ndim(b) == 0:
                line_x = np.array(ax.get_xlim())
                line_y = a * line_x + b
                ax.plot(line_x, line_y, linestyle=lines_style[i], color=lines_color[i], label=lines_label[i])
            else:
                ax.plot(a, b, lines_style[i], color=lines_color[i], label=lines_label[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if lines_label is not None and lines_label[0] is not None:
        plt.legend()

    if not show:
        return ax


def kmf(
    kmfs: Sequence[KaplanMeierFitter] = None,
    ci_alpha: list[float] | None = None,
    ci_force_lines: list[Boolean] | None = None,
    ci_show: list[Boolean] | None = None,
    ci_legend: list[Boolean] | None = None,
    at_risk_counts: list[Boolean] | None = None,
    color: list[str] | None | None = None,
    grid: Boolean | None = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool | None = None,
    title: str | None = None,
):
    """Plots a pretty figure of the Fitted KaplanMeierFitter model

    See https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html

    Args:
        kmfs: Iterables of fitted KaplanMeierFitter objects.
        ci_alpha: The transparency level of the confidence interval. If more than one kmfs, this should be a list. Defaults to 0.3.
        ci_force_lines: Force the confidence intervals to be line plots (versus default shaded areas).
                        If more than one kmfs, this should be a list. Defaults to False .
        ci_show: Show confidence intervals. If more than one kmfs, this should be a list. Defaults to True .
        ci_legend: If ci_force_lines is True, this is a boolean flag to add the lines' labels to the legend.
                   If more than one kmfs, this should be a list. Defaults to False .
        at_risk_counts: Show group sizes at time points. If more than one kmfs, this should be a list. Defaults to False.
        color: List of colors for each kmf. If more than one kmfs, this should be a list.
        grid: If True, plot grid lines.
        xlim: Set the x-axis view limits.
        ylim: Set the y-axis view limits.
        xlabel: The x-axis label text.
        ylabel: The y-axis label text.
        figsize: Width, height in inches. Defaults to None .
        show: Show the plot, do not return axis.
        title: Set the title of the plot.

    Examples:
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> adata = ep.dt.mimic_2(encoded=False)

        # Because in MIMIC-II database, `censor_fl` is censored or death (binary: 0 = death, 1 = censored).
        # While in KaplanMeierFitter, `event_observed` is True if the the death was observed, False if the event was lost (right-censored).
        # So we need to flip `censor_fl` when pass `censor_fl` to KaplanMeierFitter

        >>> adata[:, ["censor_flg"]].X = np.where(adata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> kmf = ep.tl.kmf(adata[:, ["mort_day_censored"]].X, adata[:, ["censor_flg"]].X)
        >>> ep.pl.kmf(
        ...     [kmf], color=["r"], xlim=[0, 700], ylim=[0, 1], xlabel="Days", ylabel="Proportion Survived", show=True
        ... )

        .. image:: /_static/docstring_previews/kmf_plot_1.png

        >>> T = adata[:, ["mort_day_censored"]].X
        >>> E = adata[:, ["censor_flg"]].X
        >>> groups = adata[:, ["service_unit"]].X
        >>> ix1 = groups == "FICU"
        >>> ix2 = groups == "MICU"
        >>> ix3 = groups == "SICU"
        >>> kmf_1 = ep.tl.kmf(T[ix1], E[ix1], label="FICU")
        >>> kmf_2 = ep.tl.kmf(T[ix2], E[ix2], label="MICU")
        >>> kmf_3 = ep.tl.kmf(T[ix3], E[ix3], label="SICU")
        >>> ep.pl.kmf([kmf_1, kmf_2, kmf_3], ci_show=[False,False,False], color=['k','r', 'g'],
        >>>           xlim=[0, 750], ylim=[0, 1], xlabel="Days", ylabel="Proportion Survived")

        .. image:: /_static/docstring_previews/kmf_plot_2.png
    """
    if ci_alpha is None:
        ci_alpha = [0.3] * len(kmfs)
    if ci_force_lines is None:
        ci_force_lines = [False] * len(kmfs)
    if ci_show is None:
        ci_show = [True] * len(kmfs)
    if ci_legend is None:
        ci_legend = [False] * len(kmfs)
    if at_risk_counts is None:
        at_risk_counts = [False] * len(kmfs)
    if color is None:
        color = [None] * len(kmfs)
    plt.figure(figsize=figsize)

    for i, kmf in enumerate(kmfs):
        if i == 0:
            ax = kmf.plot_survival_function(
                ci_alpha=ci_alpha[i],
                ci_force_lines=ci_force_lines[i],
                ci_show=ci_show[i],
                ci_legend=ci_legend[i],
                at_risk_counts=at_risk_counts[i],
                color=color[i],
            )
        else:
            ax = kmf.plot_survival_function(
                ax=ax,
                ci_alpha=ci_alpha[i],
                ci_force_lines=ci_force_lines[i],
                ci_show=ci_show[i],
                ci_legend=ci_legend[i],
                at_risk_counts=at_risk_counts[i],
                color=color[i],
            )
    ax.grid(grid)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    if not show:
        return ax
