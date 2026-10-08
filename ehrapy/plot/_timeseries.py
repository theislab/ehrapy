from __future__ import annotations

from typing import TYPE_CHECKING, Any

import holoviews as hv
import numpy as np
import pandas as pd

from ehrapy._compat import _materialize, _resolve_axis
from ehrapy.plot._holoviews import load_hv_extensions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData
    from fast_array_utils.types import DaskArray


@load_hv_extensions()
def timeseries(
    edata: EHRData,
    *,
    obs_names: str | int | Sequence[str | int] | None = None,
    var_names: str | Sequence[str] | None = None,
    tem_names: Any | Sequence[Any] | slice | None = None,
    layer: str | None = None,
    overlay: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    width: int | None = 600,
    height: int | None = 400,
    title: str | None = None,
) -> hv.Overlay | hv.Layout:
    """Plot time series from a 3D EHRData object.

    Selection logic:
    obs_names, var_names, tem_names select labels from `edata.obs_names`, `edata.var_names`, `edata.tem.index`.
    Use :class:`slice` (e.g. ``slice(0, 5)``) for positional selection along the axes.

    Args:
        edata: Central data object.
        obs_names: Unique observation identifier(s) to plot.
        var_names: Variable name or list of variable names in `edata.var_names` to plot.
        tem_names: Time indices to plot.
        layer: Layer holding the 3D time series.
            If `None`, `edata.X` is used.
        overlay: Whether to overlay multiple observations in a single plot (True) or create subplots (False).
        xlabel: The x-axis label text.
        ylabel: The y-axis label text.
        width: Plot width in pixels.
        height: Plot height in pixels.
        title: Set the title of the plot.

    Returns:
        HoloViews Overlay (if overlay=True) or Layout (if overlay=False) object representing the time series plot(s).

    Examples:
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> edata = ed.dt.ehrdata_blobs(n_variables=10, n_observations=5, base_timepoints=100)
        >>> ep.pl.timeseries(edata, obs_names="1", var_names=["feature_1", "feature_2"], tem_names=slice(0, 10))

        .. image:: /_static/docstring_previews/timeseries_plot.png
    """
    opts_dict: dict[str, Any] = {}
    if width is not None:
        opts_dict["width"] = width
    if height is not None:
        opts_dict["height"] = height
    if xlabel is not None:
        opts_dict["xlabel"] = xlabel
    if ylabel is not None:
        opts_dict["ylabel"] = ylabel
    opts_dict["shared_axes"] = True
    opts_dict["legend_position"] = "right"

    X = _time_series(edata, layer)
    obs_pos, obs_labels = _resolve_axis(pd.Index(edata.obs_names), obs_names, "obs_names")
    var_pos, var_labels = _resolve_axis(pd.Index(edata.var_names), var_names, "var_names")
    tem_pos, tem_labels = _resolve_axis(pd.Index(edata.tem.index), tem_names, "tem_names")

    if obs_pos.size == 0:
        raise ValueError("No observations selected (obs_names resolved to empty).")
    if var_pos.size == 0:
        raise ValueError("No variables selected (var_names resolved to empty).")
    if tem_pos.size == 0:
        raise ValueError("No timepoints selected (tem_names resolved to empty).")

    (mtx,) = _materialize(X[obs_pos][:, var_pos][:, :, tem_pos])
    timepoints = np.asarray(tem_labels)

    if overlay:
        if len(var_labels) != 1:
            raise ValueError("When overlay=True, only a single var_name can be plotted at a time.")

        k = str(var_labels[0])
        y = np.asarray(mtx[:, 0, :], dtype=float)
        n_obs, n_time = y.shape

        df = pd.DataFrame(
            {
                "time": np.tile(timepoints, n_obs),
                "value": y.ravel(order="C"),
                "series": np.repeat([str(x) for x in obs_labels], n_time),
                "variable": k,
            }
        )

        curves = []
        for series, g in df.groupby("series", sort=False):
            curve = hv.Curve(g, kdims="time", vdims=["value", "variable"], label=series)
            points = hv.Scatter(g, kdims="time", vdims=["value", "variable"]).opts(size=6, tools=["hover"])
            curves.append(curve * points)

        plot = hv.Overlay(curves)

        plot_title = title if title is not None else f"Time series for variable {k}"
        plot = plot.relabel(plot_title).opts(**opts_dict)

        return plot

    # overlay=False: one panel per observation; within each panel overlay variables
    panels = []
    for obs_i, obs_label in enumerate(obs_labels):
        curves = []
        for var_i, var_label in enumerate(var_labels):
            y = np.asarray(mtx[obs_i, var_i, :], dtype=float)
            g = pd.DataFrame({"time": timepoints, "value": y, "variable": str(var_label)})
            curve = hv.Curve(g, kdims="time", vdims=["value", "variable"], label=str(var_label))
            points = hv.Scatter(g, kdims="time", vdims=["value", "variable"]).opts(size=6, tools=["hover"])
            curves.append(curve * points)
        panel = hv.Overlay(curves)

        panel_title = (
            title if (title is not None and len(obs_labels) == 1) else f"Time series for observation {obs_label}"
        )

        panel = panel.relabel(panel_title).opts(**opts_dict)
        panels.append(panel)

    layout = hv.Layout(panels).cols(1)
    return layout


@load_hv_extensions()
def trajectories(
    edata: EHRData,
    *,
    var_names: str | Sequence[str] | None = None,
    groupby: str | None = None,
    tem_names: Any | Sequence[Any] | slice | None = None,
    layer: str | None = None,
    ci: float | None = 0.95,
    xlabel: str | None = None,
    ylabel: str | None = None,
    width: int | None = 600,
    height: int | None = 400,
    title: str | None = None,
) -> hv.Overlay | hv.Layout:
    """Plot the mean of variables over time with a confidence band for every group of observations.

    At every timepoint, the mean and its normal-approximation confidence interval are computed from the observations with a value there.

    Args:
        edata: Central data object.
        var_names: Variable name or list of variable names in `edata.var_names` to plot, each in its own panel.
            If `None`, all variables are plotted.
        groupby: Column of `edata.obs` whose categories each get a trajectory.
            If `None`, all observations form one group.
        tem_names: Time indices to plot.
        layer: Layer holding the 3D time series.
            If `None`, `edata.X` is used.
        ci: Coverage of the confidence band around the mean, or `None` for no band.
        xlabel: The x-axis label text.
        ylabel: The y-axis label text.
        width: Plot width in pixels.
        height: Plot height in pixels.
        title: Set the title of the plot.

    Returns:
        HoloViews Overlay for a single variable, or Layout with one panel per variable.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=5, n_observations=100, base_timepoints=20)
        >>> ep.pl.trajectories(edata, var_names="feature_0", groupby="cluster")

        .. image:: /_static/docstring_previews/trajectories.png
    """
    from scipy.stats import norm

    X = _time_series(edata, layer)
    var_pos, var_labels = _resolve_axis(pd.Index(edata.var_names), var_names, "var_names")
    tem_pos, tem_labels = _resolve_axis(pd.Index(edata.tem.index), tem_names, "tem_names")
    if groupby is not None and groupby not in edata.obs:
        raise KeyError(f"{groupby!r} not found in edata.obs.")
    groups = pd.Series("all", index=edata.obs_names) if groupby is None else edata.obs[groupby]

    (mtx,) = _materialize(X[:, var_pos][:, :, tem_pos])
    mtx = np.asarray(mtx, dtype=float)
    numeric = pd.to_numeric(tem_labels, errors="coerce")
    timepoints = np.arange(len(tem_labels)) if numeric.isna().any() else np.asarray(numeric)
    z = None if ci is None else norm.ppf((1 + ci) / 2)

    options = {"width": width, "height": height, "xlabel": xlabel, "ylabel": ylabel}
    opts_dict: dict[str, Any] = {key: value for key, value in options.items() if value is not None}
    opts_dict["legend_position"] = "right"

    panels = []
    for var_i, var_label in enumerate(var_labels):
        elements = []
        for group, members in groups.groupby(groups, observed=True, sort=True).indices.items():
            values = mtx[members, var_i, :]
            observed = ~np.isnan(values)
            n = observed.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(observed, values, 0).sum(axis=0) / n
                sd = np.sqrt(np.where(observed, (values - mean) ** 2, 0).sum(axis=0) / (n - 1))
                half_width = np.zeros_like(mean) if z is None else z * sd / np.sqrt(n)
            label = f"{group} (n={len(members)})"
            if z is not None:
                elements.append(
                    hv.Area(
                        (timepoints, mean - half_width, mean + half_width),
                        kdims="time",
                        vdims=[str(var_label), f"{var_label} upper"],
                        label=label,
                    ).opts(alpha=0.2, line_width=0)
                )
            elements.append(hv.Curve((timepoints, mean), kdims="time", vdims=str(var_label), label=label))
        panel_title = title if title is not None else f"Mean of {var_label} over time"
        panels.append(hv.Overlay(elements).relabel(panel_title).opts(**opts_dict))

    return panels[0] if len(panels) == 1 else hv.Layout(panels).cols(1)


def _time_series(edata: EHRData, layer: str | None) -> np.ndarray | DaskArray:
    if layer is not None and layer not in edata.layers:
        raise KeyError(f"Layer {layer!r} not found in edata.layers. Available layers: {list(edata.layers)}")
    X = edata.X if layer is None else edata.layers[layer]
    if X.ndim != 3:
        source = "edata.X" if layer is None else f"Layer {layer!r}"
        raise ValueError(f"{source} must be 3D (n_obs, n_vars, n_time), got shape {X.shape}.")
    return X
