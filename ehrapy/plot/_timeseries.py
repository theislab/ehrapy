from __future__ import annotations

from typing import TYPE_CHECKING, Any

import holoviews as hv
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData


def timeseries(
    edata: EHRData,
    *,
    obs_names: str | int | Sequence[str | int] | None = None,
    var_names: str | Sequence[str] | None = None,
    tem_names: Any | Sequence[Any] | slice | None = None,
    layer: str = "tem_data",
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
        layer: layer to use for time series data.
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

    if layer not in edata.layers:
        raise KeyError(f"Layer {layer!r} not found in edata.layers. Available layers: {list(edata.layers)}")
    mtx = np.asarray(edata.layers[layer])
    if mtx.ndim != 3:
        raise ValueError(f"Layer {layer!r} must be 3D (n_obs, n_vars, n_time), got shape {mtx.shape}.")

    obs_pos, obs_labels = _resolve_axis(pd.Index(edata.obs_names), obs_names, "obs_names")
    var_pos, var_labels = _resolve_axis(pd.Index(edata.var_names), var_names, "var_names")
    tem_pos, tem_labels = _resolve_axis(pd.Index(edata.tem.index), tem_names, "tem_names")

    if obs_pos.size == 0:
        raise ValueError("No observations selected (obs_names resolved to empty).")
    if var_pos.size == 0:
        raise ValueError("No variables selected (var_names resolved to empty).")
    if tem_pos.size == 0:
        raise ValueError("No timepoints selected (tem_names resolved to empty).")

    mtx = mtx[np.ix_(obs_pos, var_pos, tem_pos)]
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

        curves = [
            hv.Curve(g, kdims="time", vdims="value", label=series) for series, g in df.groupby("series", sort=False)
        ]
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
            g = pd.DataFrame({"time": timepoints, "value": y})
            curves.append(hv.Curve(g, kdims="time", vdims="value", label=str(var_label)))

        panel = hv.Overlay(curves)

        panel_title = (
            title if (title is not None and len(obs_labels) == 1) else f"Time series for observation {obs_label}"
        )

        panel = panel.relabel(panel_title).opts(**opts_dict)
        panels.append(panel)

    layout = hv.Layout(panels).cols(1)
    return layout


def _resolve_axis(index: pd.Index, names: Any, axis: str) -> tuple[np.ndarray, pd.Index]:
    n = len(index)

    if names is None:
        pos = np.arange(n, dtype=int)
        return pos, index.take(pos)

    if isinstance(names, slice):
        pos = np.arange(n, dtype=int)[names]
        return pos, index.take(pos)

    if isinstance(names, (str, int, np.integer)):
        names_list = [names]
    else:
        names_list = list(names)

    names_list = list(dict.fromkeys(names_list))

    pos = index.get_indexer(names_list)
    if (pos < 0).any():
        missing = [names_list[i] for i, p in enumerate(pos) if p < 0]
        raise KeyError(f"{', '.join(str(x) for x in missing)} not found in edata.{axis}")

    pos = pos.astype(int, copy=False)
    return pos, index.take(pos)
