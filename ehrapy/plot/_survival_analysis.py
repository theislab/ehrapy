from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from numpy import ndarray

from ehrapy._compat import use_ehrdata

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any
    from xmlrpc.client import Boolean

    from anndata import AnnData
    from ehrdata import EHRData
    from lifelines import KaplanMeierFitter
    from statsmodels.regression.linear_model import RegressionResults


@use_ehrdata(deprecated_after="1.0.0", edata_None_allowed=True)
def ols(
    edata: EHRData | AnnData | None = None,
    *,
    x: str | None = None,
    y: str | None = None,
    scatter_plot: Boolean | None = True,
    ols_results: list[RegressionResults] | None = None,
    ols_color: list[str | None] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    width: int | None = 600,
    height: int | None = 400,
    lines: list[tuple[ndarray | float, ndarray | float]] | None = None,
    lines_color: list[str | None] | None = None,
    lines_style: list[str | None] | None = None,
    lines_label: list[str | None] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    **kwds,
) -> hv.Scatter | hv.Curve | hv.Overlay | None:
    """Plots an Ordinary Least Squares (OLS) Model result, scatter plot, and line plot.

    Args:
        edata: Central data object.
        x: x coordinate, for scatter plotting.
        y: y coordinate, for scatter plotting.
        scatter_plot: Whether to show a scatter plot.
        ols_results: List of RegressionResults from ehrapy.tl.ols.
        ols_color: List of colors for each ols_results.
        xlabel: The x-axis label text.
        ylabel: The y-axis label text.
        width: Plot width in pixels.
        height: Plot height in pixels.
        lines: List of Tuples of (slope, intercept) or (x, y). Plot lines by slope and intercept or data points.
               Example: plot two lines (y = x + 2 and y = 2*x + 1): [(1, 2), (2, 1)]
        lines_color: List of colors for each line.
        lines_style: List of line styles for each line.
        lines_label: List of line labels for each line.
        xlim: Set the x-axis view limits. Required for only plotting lines using slope and intercept.
        ylim: Set the y-axis view limits. Required for only plotting lines using slope and intercept.
        title: Set the title of the plot.
        **kwds: Passed to HoloViews Scatter element.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> co2_lm_result = ep.tl.ols(
        ...     edata, var_names=["pco2_first", "tco2_first"], formula="tco2_first ~ pco2_first", missing="drop"
        ... ).fit()
        >>> ep.pl.ols(
        ...     edata,
        ...     x="pco2_first",
        ...     y="tco2_first",
        ...     ols_results=[co2_lm_result],
        ...     ols_color=["red"],
        ...     xlabel="PCO2",
        ...     ylabel="TCO2",
        ... )

        .. image:: /_static/docstring_previews/ols_plot.png
    """
    if ols_color is None and ols_results is not None:
        ols_color = [None] * len(ols_results)
    if lines_color is None and lines is not None:
        lines_color = [None] * len(lines)
    if lines_style is None and lines is not None:
        lines_style = [None] * len(lines)
    if lines_label is None and lines is not None:
        lines_label = [None] * len(lines)

    plot = None

    if edata is not None and x is not None and y is not None:
        x_data = np.array(edata[:, x].X).flatten().astype(float)
        y_data = np.array(edata[:, y].X).flatten().astype(float)

        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[mask]
        y_clean = y_data[mask]

        if scatter_plot:
            scatter_opts = {**kwds}
            scatter_opts.setdefault("tools", ["hover"])
            plot = hv.Scatter((x_clean, y_clean), kdims=x, vdims=y).opts(**scatter_opts)

        if ols_results is not None:
            x_sorted = np.sort(x_clean)
            for i, ols_result in enumerate(ols_results):
                y_pred = ols_result.predict(exog={"const": 1, x: x_sorted})
                curve_opts: dict[str, Any] = {"tools": ["hover"]}
                if ols_color[i] is not None:
                    curve_opts["color"] = ols_color[i]
                ols_curve = hv.Curve((x_sorted, y_pred)).opts(**curve_opts)
                plot = ols_curve if plot is None else plot * ols_curve

    if lines is not None:
        if xlim is None and plot is not None:
            x_range = plot.range(x if x else 0)
            xlim = (x_range[0], x_range[1]) if x_range[0] is not None else (0, 1)
        elif xlim is None:
            xlim = (0, 1)

        for i, line in enumerate(lines):
            a, b = line
            if np.ndim(a) == 0 and np.ndim(b) == 0:
                line_x = np.array(xlim)
                line_y = a * line_x + b
            else:
                line_x, line_y = a, b

            curve_opts = {"tools": ["hover"]}
            if lines_color[i] is not None:
                curve_opts["color"] = lines_color[i]
            if lines_style[i] is not None:
                style_map = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dashdot"}
                curve_opts["line_dash"] = style_map.get(lines_style[i], "solid")

            curve_kwargs = {"label": lines_label[i]} if lines_label[i] else {}
            line_curve = hv.Curve((line_x, line_y), **curve_kwargs).opts(**curve_opts)
            plot = line_curve if plot is None else plot * line_curve

    if plot is None:
        return None

    opts_dict: dict[str, Any] = {}
    if width is not None:
        opts_dict["width"] = width
    if height is not None:
        opts_dict["height"] = height
    if xlabel is not None:
        opts_dict["xlabel"] = xlabel
    if ylabel is not None:
        opts_dict["ylabel"] = ylabel
    if title is not None:
        opts_dict["title"] = title
    if xlim is not None:
        opts_dict["xlim"] = xlim
    if ylim is not None:
        opts_dict["ylim"] = ylim

    plot = plot.opts(**opts_dict)

    return plot


def kaplan_meier(
    kmfs: Sequence[KaplanMeierFitter],
    *,
    display_survival_statistics: bool = False,
    ci_alpha: list[float] | None = None,
    ci_force_lines: list[Boolean] | None = None,
    ci_show: list[Boolean] | None = None,
    ci_legend: list[Boolean] | None = None,
    at_risk_counts: list[Boolean] | None = None,
    color: list[str | None] | None = None,
    grid: Boolean | None = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    width: int | None = 600,
    height: int | None = 400,
    title: str | None = None,
) -> hv.Layout | hv.Overlay | hv.Curve | None:
    """Plots a pretty figure of the Fitted KaplanMeierFitter model.

    See also: :class:`~lifelines.fitters.kaplan_meier_fitter.KaplanMeierFitter`.

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
        width: Plot width in pixels.
        height: Plot height in pixels.
        title: Set the title of the plot.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.mimic_2()
        >>> edata[:, ["censor_flg"]].X = np.where(
        ...     edata[:, ["censor_flg"]].X == 0, 1, 0
        ... )  # MIMIC-II uses 0=death while KaplanMeierFitter expects True=death
        >>> kmf = ep.tl.kaplan_meier(edata, "mort_day_censored", "censor_flg")
        >>> ep.pl.kaplan_meier(
        ...     [kmf], color=["r"], xlim=(0, 700), ylim=(0, 1), xlabel="Days", ylabel="Proportion Survived", show=True
        ... )

        .. image:: /_static/docstring_previews/kaplan_meier.png
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

    plot = None

    for i, kmf in enumerate(kmfs):
        sf = kmf.survival_function_
        times = sf.index.values
        survival = sf.iloc[:, 0].values

        label = kmf.label if kmf.label else f"Group {i + 1}"
        curve_opts: dict[str, Any] = {"tools": ["hover"]}
        if color[i] is not None:
            curve_opts["color"] = color[i]

        curve = hv.Curve((times, survival), kdims="Time", vdims="Survival", label=label).opts(**curve_opts)

        if ci_show[i] and hasattr(kmf, "confidence_interval_survival_function_"):
            ci = kmf.confidence_interval_survival_function_
            ci_lower = ci.iloc[:, 0].values
            ci_upper = ci.iloc[:, 1].values

            if ci_force_lines[i]:
                ci_lower_curve = hv.Curve((times, ci_lower)).opts(
                    color=color[i] if color[i] else "gray", alpha=ci_alpha[i], line_dash="dashed"
                )
                ci_upper_curve = hv.Curve((times, ci_upper)).opts(
                    color=color[i] if color[i] else "gray", alpha=ci_alpha[i], line_dash="dashed"
                )
                curve = curve * ci_lower_curve * ci_upper_curve
            else:
                ci_area = hv.Area((times, ci_lower, ci_upper), vdims=["y", "y2"]).opts(
                    color=color[i] if color[i] else "gray", alpha=ci_alpha[i], line_width=0
                )
                curve = ci_area * curve

        plot = curve if plot is None else plot * curve

    if plot is None:
        return None

    opts_dict: dict[str, Any] = {"width": width, "show_grid": grid}
    if xlabel:
        opts_dict["xlabel"] = xlabel
    if ylabel:
        opts_dict["ylabel"] = ylabel
    if title:
        opts_dict["title"] = title
    if xlim:
        opts_dict["xlim"] = xlim
    if ylim:
        opts_dict["ylim"] = ylim
    opts_dict["show_legend"] = True

    plot = plot.opts(**opts_dict)

    if display_survival_statistics:
        if xlim:
            time_points = np.linspace(xlim[0], xlim[1], 10)
        else:
            all_times = np.concatenate([kmf.survival_function_.index.values for kmf in kmfs])
            time_points = np.linspace(all_times.min(), all_times.max(), 10)

        # Create table data in wide format (one row per group, columns are time points)
        table_data: dict[str, list[str]] = {}
        table_data["Group"] = []

        for kmf in kmfs:
            label = kmf.label if kmf.label else "Group"
            table_data["Group"].append(label)
            survival_probs = kmf.survival_function_at_times(time_points).values

            for _, (t, prob) in enumerate(zip(time_points, survival_probs, strict=False)):
                col_name = f"{t:.0f}"
                if col_name not in table_data:
                    table_data[col_name] = []
                table_data[col_name].append(f"{prob:.2f}")

        df = pd.DataFrame(table_data)
        table = hv.Table(df).opts(width=width, height=int(height * 0.4), fit_columns=True)
        plot = (plot + table).cols(1)

    return plot


@use_ehrdata(deprecated_after="1.0.0")
def cox_ph_forestplot(
    edata: EHRData | AnnData,
    *,
    uns_key: str = "cox_ph",
    labels: Iterable[str] | None = None,
    width: int = 1200,
    height: int = 600,
    ecolor: str = "dimgray",
    size: int = 3,
    marker: str = "o",
    decimal: int = 2,
    text_size: int = 12,
    color: str = "k",
    title: str | None = None,
) -> hv.Overlay | None:
    """Generates a forest plot to visualize the coefficients and confidence intervals of a Cox Proportional Hazards model.

    The `edata` object must first be populated using the :func:`~ehrapy.tools.cox_ph` function.
    This function stores the summary table of the `CoxPHFitter` in the `.uns` attribute of `edata`.
    The summary table is created when the model is fitted using the :func:`~ehrapy.tools.cox_ph` function.
    See also: :class:`~lifelines.fitters.coxph_fitter.CoxPHFitter`

    Args:
        edata: Data object containing the summary table from the CoxPHFitter. This is stored in the `.uns` attribute, after fitting the model using :func:`~ehrapy.tools.cox_ph`.
        uns_key: Key in `.uns` where :func:`~ehrapy.tools.cox_ph` function stored the summary table. See argument `uns_key` in :func:`~ehrapy.tools.cox_ph`.
        labels: List of labels for each coefficient, default uses the index of the summary table.
        width: Plot width in pixels.
        height: Plot height in pixels.
        ecolor: Color of the error bars.
        size: Size of the markers.
        marker: Marker style.
        decimal: Number of decimal places to display.
        text_size: Font size of the text.
        color: Color of the markers.
        title: Set the title of the plot.

    Returns:
        HoloViews Overlay with forest plot and text annotations.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()[
        ...     :, ["mort_day_censored", "censor_flg", "gender_num", "afib_flg", "day_icu_intime_num"]
        ... ]
        >>> coxph = ep.tl.cox_ph(edata, event_col="censor_flg", duration_col="mort_day_censored")
        >>> ep.pl.cox_ph_forestplot(edata)

        .. image:: /_static/docstring_previews/coxph_forestplot.png
    """
    if uns_key not in edata.uns:
        raise ValueError(f"Key {uns_key} not found in edata.uns. Please provide a valid key.")

    coxph_fitting_summary = edata.uns[uns_key]
    auc_col = "coef"

    if labels is None:
        labels = list(coxph_fitting_summary.index)

    coefs = coxph_fitting_summary[auc_col].values
    lower = coxph_fitting_summary["coef lower 95%"].values
    upper = coxph_fitting_summary["coef upper 95%"].values
    y_positions = np.arange(len(coxph_fitting_summary))

    x_axis_upper_bound = float(pd.to_numeric(coxph_fitting_summary["coef upper 95%"]).max())
    x_axis_lower_bound = float(pd.to_numeric(coxph_fitting_summary["coef lower 95%"]).min())

    data_range = x_axis_upper_bound - x_axis_lower_bound
    plot_padding = data_range * 0.1
    text_gap = data_range * 0.15
    text_spacing = data_range * 0.4

    plot_x_min = x_axis_lower_bound - plot_padding
    plot_x_max = x_axis_upper_bound + plot_padding
    text_start_x = plot_x_max + text_gap
    ci_text_x = text_start_x + text_spacing

    total_x_max = ci_text_x + (data_range * 0.5)

    error_data = []
    for coef, y, lower_val, upper_val in zip(coefs, y_positions, lower, upper, strict=False):
        if not np.isnan(coef) and not np.isnan(lower_val) and not np.isnan(upper_val):
            error_data.append((coef, y, coef - lower_val, upper_val - coef))

    error_bars = hv.ErrorBars(
        error_data,
        kdims=["Coefficient", "Variable"],
        vdims=["negative_error", "positive_error"],
    ).opts(color=ecolor, line_width=2)

    points = hv.Scatter(
        [(coef, y) for coef, y in zip(coefs, y_positions, strict=False) if not np.isnan(coef)],
        kdims=["Coefficient"],
        vdims=["Variable"],
    ).opts(color=color, size=size * 5, marker=marker, tools=["hover"])

    vline = hv.VLine(1).opts(color="gray", line_width=1)

    text_labels = []
    for coef_val, low_val, upp_val, y_pos in zip(coefs, lower, upper, y_positions, strict=False):
        if not np.isnan(coef_val):
            if isinstance(coef_val, float) and isinstance(low_val, float) and isinstance(upp_val, float):
                coef_text = f"{coef_val:.{decimal}f}"
                ci_text = f"({low_val:.{decimal}f}, {upp_val:.{decimal}f})"
            else:
                coef_text = str(coef_val)
                ci_text = f"({low_val}, {upp_val})"

            text_labels.append((text_start_x, y_pos, coef_text))
            text_labels.append((ci_text_x, y_pos, ci_text))

    labels_overlay = hv.Labels(text_labels, kdims=["x", "y"], vdims=["text"]).opts(
        text_font_size=f"{text_size}pt", text_align="left", text_color="black"
    )

    header_y = len(coxph_fitting_summary) - 0.7
    header_labels = hv.Labels(
        [
            (text_start_x, header_y, "coef"),
            (ci_text_x, header_y, "95% CI"),
        ],
        kdims=["x", "y"],
        vdims=["text"],
    ).opts(text_font_size=f"{text_size + 2}pt", text_font_style="bold", text_align="left", text_color="black")

    forest_plot = (vline * error_bars * points * labels_overlay * header_labels).opts(
        width=width,
        height=height,
        xlim=(plot_x_min, total_x_max),
        ylim=(len(coxph_fitting_summary) - 0.5, -0.5),
        invert_yaxis=True,
        yticks=list(zip(y_positions, labels, strict=False)),
        xlabel="Coefficient",
        ylabel="",
        fontsize={"yticks": text_size},
        show_legend=False,
    )

    if title:
        forest_plot = forest_plot.opts(title=title)

    return forest_plot
