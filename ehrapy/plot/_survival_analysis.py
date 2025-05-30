from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from numpy import ndarray

from ehrapy.plot import scatter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
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
) -> Axes | None:
    """Plots an Ordinary Least Squares (OLS) Model result, scatter plot, and line plot.

    Args:
        adata: :class:`~anndata.AnnData` object containing all observations.
        x: x coordinate, for scatter plotting.
        y: y coordinate, for scatter plotting.
        scatter_plot: Whether to show a scatter plot.
        ols_results: List of RegressionResults from ehrapy.tl.ols. Example: [result_1, result_2]
        ols_color: List of colors for each ols_results. Example: ['red', 'blue'].
        xlabel: The x-axis label text.
        ylabel: The y-axis label text.
        figsize: Width, height in inches.
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
        **kwds: Passed to Matplotlib Scatterplot.

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
            ax = scatter(adata, x=x, y=y, show=False, ax=ax, **kwds)
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
    else:
        return None


def kmf(
    kmfs: Sequence[KaplanMeierFitter],
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
) -> Axes | None:
    warnings.warn(
        "This function is deprecated and will be removed in the next release. Use `ep.pl.kaplan_meier` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return kaplan_meier(
        kmfs=kmfs,
        ci_alpha=ci_alpha,
        ci_force_lines=ci_force_lines,
        ci_show=ci_show,
        ci_legend=ci_legend,
        at_risk_counts=at_risk_counts,
        color=color,
        grid=grid,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        show=show,
        title=title,
    )


def kaplan_meier(
    kmfs: Sequence[KaplanMeierFitter],
    *,
    display_survival_statistics: bool = False,
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
) -> Axes | None:
    """Plots a pretty figure of the Fitted KaplanMeierFitter model.

    See https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html

    Args:
        kmfs: Iterables of fitted KaplanMeierFitter objects.
        display_survival_statistics: Whether to show survival statistics in a table below the plot.
        ci_alpha: The transparency level of the confidence interval. If more than one kmfs, this should be a list.
        ci_force_lines: Force the confidence intervals to be line plots (versus default shaded areas).
                        If more than one kmfs, this should be a list.
        ci_show: Show confidence intervals. If more than one kmfs, this should be a list.
        ci_legend: If ci_force_lines is True, this is a boolean flag to add the lines' labels to the legend.
                   If more than one kmfs, this should be a list.
        at_risk_counts: Show group sizes at time points. If more than one kmfs, this should be a list.
        color: List of colors for each kmf. If more than one kmfs, this should be a list.
        grid: If True, plot grid lines.
        xlim: Set the x-axis view limits.
        ylim: Set the y-axis view limits.
        xlabel: The x-axis label text.
        ylabel: The y-axis label text.
        figsize: Width, height in inches.
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
        >>> kmf = ep.tl.kaplan_meier(adata, "mort_day_censored", "censor_flg")
        >>> ep.pl.kaplan_meier(
        ...     [kmf], color=["r"], xlim=[0, 700], ylim=[0, 1], xlabel="Days", ylabel="Proportion Survived", show=True
        ... )

        .. image:: /_static/docstring_previews/kmf_plot_1.png

        >>> groups = adata[:, ["service_unit"]].X
        >>> adata_ficu = adata[groups == "FICU"]
        >>> adata_micu = adata[groups == "MICU"]
        >>> adata_sicu = adata[groups == "SICU"]
        >>> kmf_1 = ep.tl.kaplan_meier(adata_ficu, "mort_day_censored", "censor_flg", label="FICU")
        >>> kmf_2 = ep.tl.kaplan_meier(adata_micu, "mort_day_censored", "censor_flg", label="MICU")
        >>> kmf_3 = ep.tl.kaplan_meier(adata_sicu, "mort_day_censored", "censor_flg", label="SICU")
        >>> ep.pl.kaplan_meier([kmf_1, kmf_2, kmf_3], ci_show=[False,False,False], color=['k','r', 'g'],
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

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    spec = fig.add_gridspec(2, 1) if display_survival_statistics else fig.add_gridspec(1, 1)
    ax = plt.subplot(spec[0, 0])

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
    # Configure plot appearance
    ax.grid(grid)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if display_survival_statistics:
        xticks = [x for x in ax.get_xticks() if x >= 0]
        xticks_space = xticks[1] - xticks[0]
        if xlabel is None:
            xlabel = "Time"

        yticks = np.arange(len(kmfs))

        ax_table = plt.subplot(spec[1, 0])
        ax_table.set_xticks(xticks)
        ax_table.set_xlim(-xticks_space / 2, xticks[-1] + xticks_space / 2)
        ax_table.set_ylim(-1, len(kmfs))
        ax_table.set_yticks(yticks)
        ax_table.set_yticklabels([kmf.label if kmf.label else f"Group {i + 1}" for i, kmf in enumerate(kmfs[::-1])])

        for i, kmf in enumerate(kmfs[::-1]):
            survival_probs = kmf.survival_function_at_times(xticks).values
            for j, prob in enumerate(survival_probs):
                ax_table.text(
                    xticks[j],  # x position
                    yticks[i],  # y position
                    f"{prob:.2f}",  # formatted survival probability
                    ha="center",
                    va="center",
                    bbox={"boxstyle": "round,pad=0.2", "edgecolor": "none", "facecolor": "lightgrey"},
                )

        ax_table.grid(grid)
        ax_table.spines["top"].set_visible(False)
        ax_table.spines["right"].set_visible(False)
        ax_table.spines["bottom"].set_visible(False)
        ax_table.spines["left"].set_visible(False)

    if not show:
        return fig, ax

    else:
        return None


def cox_ph_forestplot(
    adata: AnnData,
    *,
    uns_key: str = "cox_ph",
    labels: Iterable[str] | None = None,
    fig_size: tuple = (10, 10),
    t_adjuster: float = 0.1,
    ecolor: str = "dimgray",
    size: int = 3,
    marker: str = "o",
    decimal: int = 2,
    text_size: int = 12,
    color: str = "k",
    show: bool = None,
    title: str | None = None,
):
    """Generates a forest plot to visualize the coefficients and confidence intervals of a Cox Proportional Hazards model.

    The `adata` object must first be populated using the :func:`~ehrapy.tools.cox_ph` function. This function stores the summary table of the `CoxPHFitter` in the `.uns` attribute of `adata`.
    The summary table is created when the model is fitted using the :func:`~ehrapy.tools.cox_ph` function.
    For more information on the `CoxPHFitter`, see the `Lifelines documentation <https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html>`_.

    Inspired by `zepid.graphics.EffectMeasurePlot <https://readthedocs.org>`_ (zEpid Package, https://pypi.org/project/zepid/).

    Args:
        adata: :class:`~anndata.AnnData` object containing the summary table from the CoxPHFitter. This is stored in the `.uns` attribute, after fitting the model using :func:`~ehrapy.tools.cox_ph`.
        uns_key: Key in `.uns` where :func:`~ehrapy.tools.cox_ph` function stored the summary table. See argument `uns_key` in :func:`~ehrapy.tools.cox_ph`.
        labels: List of labels for each coefficient, default uses the index of the summary ta
        fig_size: Width, height in inches.
        t_adjuster: Adjust the table to the right.
        ecolor: Color of the error bars.
        size: Size of the markers.
        marker: Marker style.
        decimal: Number of decimal places to display.
        text_size: Font size of the text.
        color: Color of the markers.
        show: Show the plot, do not return figure and axis.
        title: Set the title of the plot.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> adata_subset = adata[:, ["mort_day_censored", "censor_flg", "gender_num", "afib_flg", "day_icu_intime_num"]]
        >>> coxph = ep.tl.cox_ph(adata_subset, event_col="censor_flg", duration_col="mort_day_censored")
        >>> ep.pl.cox_ph_forestplot(adata_subset)

        .. image:: /_static/docstring_previews/coxph_forestplot.png

    """
    if uns_key not in adata.uns:
        raise ValueError(f"Key {uns_key} not found in adata.uns. Please provide a valid key.")

    coxph_fitting_summary = adata.uns[
        uns_key
    ]  # pd.Dataframe with columns: coef, exp(coef), se(coef), z, p, lower 0.95, upper 0.95
    auc_col = "coef"

    if labels is None:
        labels = coxph_fitting_summary.index
    tval = []
    ytick = []
    for row_index in range(len(coxph_fitting_summary)):
        if not np.isnan(coxph_fitting_summary[auc_col].iloc[row_index]):
            if (
                (isinstance(coxph_fitting_summary[auc_col].iloc[row_index], float))
                & (isinstance(coxph_fitting_summary["coef lower 95%"].iloc[row_index], float))
                & (isinstance(coxph_fitting_summary["coef upper 95%"].iloc[row_index], float))
            ):
                tval.append(
                    [
                        round(coxph_fitting_summary[auc_col].iloc[row_index], decimal),
                        (
                            "("
                            + str(round(coxph_fitting_summary["coef lower 95%"].iloc[row_index], decimal))
                            + ", "
                            + str(round(coxph_fitting_summary["coef upper 95%"].iloc[row_index], decimal))
                            + ")"
                        ),
                    ]
                )
            else:
                tval.append(
                    [
                        coxph_fitting_summary[auc_col].iloc[row_index],
                        (
                            "("
                            + str(coxph_fitting_summary["coef lower 95%"].iloc[row_index])
                            + ", "
                            + str(coxph_fitting_summary["coef upper 95%"].iloc[row_index])
                            + ")"
                        ),
                    ]
                )
            ytick.append(row_index)
        else:
            tval.append([" ", " "])
            ytick.append(row_index)

    x_axis_upper_bound = round(((pd.to_numeric(coxph_fitting_summary["coef upper 95%"])).max() + 0.1), 2)

    x_axis_lower_bound = round(((pd.to_numeric(coxph_fitting_summary["coef lower 95%"])).min() - 0.1), 1)

    fig = plt.figure(figsize=fig_size)
    gspec = gridspec.GridSpec(1, 6)
    plot = plt.subplot(gspec[0, 0:4])
    table = plt.subplot(gspec[0, 4:])
    plot.set_ylim(-1, (len(coxph_fitting_summary)))  # spacing out y-axis properly

    plot.axvline(1, color="gray", zorder=1)
    lower_diff = coxph_fitting_summary[auc_col] - coxph_fitting_summary["coef lower 95%"]
    upper_diff = coxph_fitting_summary["coef upper 95%"] - coxph_fitting_summary[auc_col]
    plot.errorbar(
        coxph_fitting_summary[auc_col],
        coxph_fitting_summary.index,
        xerr=[lower_diff, upper_diff],
        marker="None",
        zorder=2,
        ecolor=ecolor,
        linewidth=0,
        elinewidth=1,
    )
    # plot markers
    plot.scatter(
        coxph_fitting_summary[auc_col],
        coxph_fitting_summary.index,
        c=color,
        s=(size * 25),
        marker=marker,
        zorder=3,
        edgecolors="None",
    )
    # plot settings
    plot.xaxis.set_ticks_position("bottom")
    plot.yaxis.set_ticks_position("left")
    plot.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    plot.get_xaxis().set_minor_formatter(ticker.NullFormatter())
    plot.set_yticks(ytick)
    plot.set_xlim([x_axis_lower_bound, x_axis_upper_bound])
    plot.set_xticks([x_axis_lower_bound, 1, x_axis_upper_bound])
    plot.set_xticklabels([x_axis_lower_bound, 1, x_axis_upper_bound])
    plot.set_yticklabels(labels)
    plot.tick_params(axis="y", labelsize=text_size)
    plot.yaxis.set_ticks_position("none")
    plot.invert_yaxis()  # invert y-axis to align values properly with table
    tb = table.table(
        cellText=tval, cellLoc="center", loc="right", colLabels=[auc_col, "95% CI"], bbox=[0, t_adjuster, 1, 1]
    )
    table.axis("off")
    tb.auto_set_font_size(False)
    tb.set_fontsize(text_size)
    for _, cell in tb.get_celld().items():
        cell.set_linewidth(0)

    # remove spines
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_visible(False)

    if title:
        plt.title(title)

    if not show:
        return fig, plot

    else:
        return None
