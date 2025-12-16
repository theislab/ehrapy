from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import holoviews as hv
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData


def plot_timeseries(
    edata: EHRData,
    *,
    obs_names: str | int | Sequence[str | int] | None = None,
    var_names: str | Sequence[str] | None = None,
    layer: str = "tem_data",
    tem_time_key: str | None = None,
    overlay: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    width: int | None = 600,
    height: int | None = 400,
    title: str | None = None,
) -> hv.Overlay | hv.Layout:
    """Plot variable time series either for an observation or for multiple observations from a 3D EHRData layer.

    Selection logic:
        - If obs_names is an int in [0, n_obs), use it as row index.
        - Otherwise, it should match a row name in edata.obs_names.

    Args:
        edata: Central data object.
        obs_names: row index or unique observation identifier(s) to plot.
        var_names: Variable name or list of variable names in `edata.var_names` to plot.
        layer: layer to use for time series data.
        tem_time_key: Key in  `edata.tem` to use as timepoints. If None, use edata.tem as 1D array.
        overlay: Whether to overlay multiple observations in a single plot (True) or create subplots (False).
        xlabel: The x-axis label text.
        ylabel: The y-axis label text.
        width: Plot width in pixels.
        height: Plot height in pixels.
        title: Set the title of the plot.

    Returns:
        HoloViews Overlay (if overlay=True) or Layout (if overlay=False) object representing the time series plot(s).

    Examples:
    >>> edata = ed.dt.ehrdata_blobs(n_variables=10, n_observations=5, base_timepoints=100)
    >>> ep.pl.plot_timeseries(edata, obs_names=1)

    .. image:: /_static/docstring_previews/plot_timeseries.png

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

    mtx = np.asarray(edata.layers[layer])
    if mtx.ndim != 3:
        raise ValueError(f"Layer {layer!r} must be 3D (n_obs, n_vars, n_time), got shape {mtx.shape}.")
    n_obs, _, n_time = mtx.shape

    if tem_time_key is None:
        try:
            timepoints = np.asarray(edata.tem.index).astype(float)
        except (TypeError, ValueError):
            timepoints = np.asarray(edata.tem.index)
    else:
        if tem_time_key not in edata.tem:
            raise KeyError(f"Column {tem_time_key!r} not found in edata.tem.")
        timepoints = np.asarray(edata.tem[tem_time_key])

    if timepoints.shape[0] != n_time:
        raise ValueError(f"Length of timepoints ({timepoints.shape[0]}) does not match n_time ({n_time}).")

    if var_names is None:
        key_list = list(np.asarray(edata.var_names).tolist())
    elif isinstance(var_names, str):
        key_list = [var_names]
    else:
        key_list = list(var_names)

    if not key_list:
        raise ValueError("var_names is empty")

    obs_ids: list[str | int]
    if obs_names is None:
        obs_ids = list(range(n_obs))
    elif isinstance(obs_names, (str, int)):
        obs_ids = [obs_names]
    else:
        obs_ids = list(obs_names)

    if not obs_ids:
        raise ValueError("obs_names is empty")

    all_var_names = np.asarray(edata.var_names)
    var_idx_list: list[int] = []
    for k in key_list:
        matches = np.flatnonzero(all_var_names == k)
        if matches.size == 0:
            raise KeyError(f"Variable {k!r} not found in edata.var_names.")
        var_idx_list.append(int(matches[0]))

    rows: list[dict] = []
    if overlay:
        if len(key_list) != 1:
            raise ValueError("When overlay=True, only a single var_name can be plotted at a time.")
        k = key_list[0]
        v_idx = var_idx_list[0]
        for obs in obs_ids:
            obs_idx, obs_id_info = _resolve_obs(edata, obs, n_obs)
            y = np.asarray(mtx[obs_idx, v_idx, :], dtype=float)
            for t, val in zip(timepoints, y, strict=False):
                rows.append({"time": t, "value": val, "series": str(obs_id_info), "variable": k})
        df = pd.DataFrame(rows)

        curves = []
        for series, g in df.groupby("series", sort=False):
            curves.append(hv.Curve(g, kdims="time", vdims="value", label=series))
        plot = hv.Overlay(curves)

        plot_title = title if title is not None else f"Time series for variable {k!r}"
        plot = plot.relabel(plot_title).opts(**opts_dict)

        return plot

    # overlay=False: one panel per observation; within each panel overlay variables
    panels = []
    for obs in obs_ids:
        obs_idx, obs_id_info = _resolve_obs(edata, obs, n_obs)

        curves = []
        for k, v_idx in zip(key_list, var_idx_list, strict=False):
            y = np.asarray(mtx[obs_idx, v_idx, :], dtype=float)
            g = pd.DataFrame({"time": timepoints, "value": y})
            curves.append(hv.Curve(g, kdims="time", vdims="value", label=str(k)))

        panel = hv.Overlay(curves)

        panel_title = (
            title
            if (title is not None and len(obs_ids) == 1)
            else f"Time series for observation index {obs_idx} ({obs_id_info})"
        )
        panel = panel.relabel(panel_title).opts(**opts_dict)
        panels.append(panel)

    layout = hv.Layout(panels).cols(1)
    return layout


def _resolve_obs(edata: EHRData, obs: str | int, n_obs: int) -> tuple[int, str]:
    """Resolve obs identifier to (row index, obs_id_info) tuple."""
    if isinstance(obs, int) and 0 <= obs < n_obs:
        obs_idx = obs
        obs_info = f"row {obs}"

    elif isinstance(obs, str):
        obs_names = np.asarray(edata.obs_names)
        matches = np.flatnonzero(obs_names == obs)
        if matches.size == 0:
            raise KeyError(f"Observation {obs!r} not found in edata.obs_names.")
        obs_idx = int(matches[0])
        obs_info = f"obs_name={obs!r}"

    return obs_idx, obs_info
