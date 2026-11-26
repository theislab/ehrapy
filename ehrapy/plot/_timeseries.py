from __future__ import annotations

from typing import TYPE_CHECKING, Any

import holoviews as hv
import numpy as np
import pandas as pd

from ehrapy._compat import _materialize
from ehrapy.get._get import _resolve_axis
from ehrapy.plot._holoviews import load_hv_extensions
from ehrapy.tools._timeseries import _obsm_values

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
    obsm_keys: str | Sequence[str] | None = None,
    events: str | Sequence[str] | None = None,
    markers: str | Sequence[str] | None = None,
    time_key: str | None = None,
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

    With one panel per observation, a panel can also show values of every timepoint from `obsm` as steps, such as scores or the predicted risk of a rolling task, and events from `obs` as dashed vertical lines.
    Markers, such as the administrations of a drug or the timepoints marked by :func:`~ehrapy.tools.mark_events`, get a row of ticks each, in a strip below the panel.

    Args:
        edata: Central data object.
        obs_names: Unique observation identifier(s) to plot.
        var_names: Variable name or list of variable names in `edata.var_names` to plot.
            An empty list plots no variables, to show only `obsm_keys`, `events` and `markers`.
        tem_names: Time indices to plot.
        layer: Layer holding the 3D time series.
            If `None`, `edata.X` is used.
        obsm_keys: Keys of `edata.obsm` with a value for every observation and timepoint, such as the scores of :func:`~ehrapy.tools.sofa` and :func:`~ehrapy.tools.kdigo` or the predictions of a rolling task from :func:`~ehrapy.ml.predict`, plotted as steps.
        events: Columns of `edata.obs` with the position of an event along the time axis, 0 for the first timepoint, such as `sepsis3_onset` of :func:`~ehrapy.tools.sepsis3`, plotted as dashed vertical lines.
            An event that is missing or lies outside the plotted timepoints is left out.
        markers: Variables of `layer`, or keys of `edata.obsm` with a value for every observation and timepoint, plotted as a row of ticks each at the timepoints they mark.
            A variable marks the timepoints at which it has a value, a key of `obsm` those at which its value is neither missing nor 0, such as the markers of :func:`~ehrapy.tools.mark_events`.
        time_key: Column of `edata.tem` that places the timepoints on the x-axis, such as hours.
            If `None`, the labels of `edata.tem` are used, as numbers where they are numbers and as positions otherwise.
        overlay: Whether to overlay multiple observations in a single plot (True) or create subplots (False).
            `obsm_keys`, `events` and `markers` need subplots.
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

        A score at every timepoint, the timepoint of its peak and the timepoints at which it is high:

        >>> import numpy as np
        >>> edata.obsm["score"] = np.nanmean(edata.X, axis=1)
        >>> edata.obs["peak"] = np.argmax(edata.obsm["score"][:, :10], axis=1)
        >>> edata.obsm["high"] = edata.obsm["score"] > np.nanmedian(edata.obsm["score"])
        >>> ep.pl.timeseries(
        ...     edata,
        ...     obs_names="1",
        ...     var_names=[],
        ...     obsm_keys="score",
        ...     events="peak",
        ...     markers="high",
        ...     tem_names=slice(0, 10),
        ... )
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

    obsm_keys = [obsm_keys] if isinstance(obsm_keys, str) else list(obsm_keys or [])
    events = [events] if isinstance(events, str) else list(events or [])
    markers = [markers] if isinstance(markers, str) else list(markers or [])
    if overlay and (obsm_keys or events or markers):
        raise ValueError("`obsm_keys`, `events` and `markers` need `overlay=False`.")
    for key in obsm_keys:
        if key not in edata.obsm:
            raise KeyError(f"{key!r} not found in edata.obsm.")
        _obsm_values(edata, key)
    for key in events:
        if key not in edata.obs:
            raise KeyError(f"{key!r} not found in edata.obs.")
    for key in markers:
        if key in edata.var_names and key in edata.obsm:
            raise ValueError(f"{key!r} is both a variable and a key of edata.obsm; rename one of them.")
        if key not in edata.var_names and key not in edata.obsm:
            raise KeyError(f"{key!r} not found in edata.var_names or edata.obsm.")
        if key in edata.obsm:
            _obsm_values(edata, key)
    if time_key is not None and time_key not in edata.tem:
        raise KeyError(f"{time_key!r} not found in edata.tem.")

    X = _time_series(edata, layer)
    obs_pos, obs_labels = _resolve_axis(pd.Index(edata.obs_names), obs_names, "obs_names")
    var_pos, var_labels = _resolve_axis(pd.Index(edata.var_names), var_names, "var_names")
    tem_pos, tem_labels = _resolve_axis(pd.Index(edata.tem.index), tem_names, "tem_names")

    if obs_pos.size == 0:
        raise ValueError("No observations selected (obs_names resolved to empty).")
    if var_pos.size == 0 and not obsm_keys and not markers:
        raise ValueError("No variables selected (var_names resolved to empty), and no obsm_keys or markers.")
    if tem_pos.size == 0:
        raise ValueError("No timepoints selected (tem_names resolved to empty).")

    (mtx,) = _materialize(X[obs_pos][:, var_pos][:, :, tem_pos])
    times = _axis_times(np.asarray(edata.tem.index if time_key is None else edata.tem[time_key]))
    timepoints = times[tem_pos]

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

    marked = _marked(edata, X, markers, obs_pos, tem_pos)

    # overlay=False: one panel per observation, which overlays variables, obsm values and events,
    # followed by a strip with a row of ticks for every marker
    panels = []
    for obs_i, obs_label in enumerate(obs_labels):
        event_times = [_event_time(edata.obs[key].iloc[obs_pos[obs_i]], times, tem_pos) for key in events]
        panel_title = (
            title if (title is not None and len(obs_labels) == 1) else f"Time series for observation {obs_label}"
        )
        if len(var_labels) or obsm_keys:
            curves = []
            values = []
            for var_i, var_label in enumerate(var_labels):
                y = np.asarray(mtx[obs_i, var_i, :], dtype=float)
                values.append(y)
                g = pd.DataFrame({"time": timepoints, "value": y, "variable": str(var_label)})
                curve = hv.Curve(g, kdims="time", vdims=["value", "variable"], label=str(var_label))
                points = hv.Scatter(g, kdims="time", vdims=["value", "variable"]).opts(size=6, tools=["hover"])
                curves.append(curve * points)
            for key in obsm_keys:
                y = np.asarray(_obsm_values(edata, key)[obs_pos[obs_i]], dtype=float)[tem_pos]
                values.append(y)
                g = pd.DataFrame({"time": timepoints, "value": y, "variable": key})
                curve = hv.Curve(g, kdims="time", vdims=["value", "variable"], label=key)
                curve = curve.opts(interpolation="steps-post", tools=["hover"], backend="bokeh")
                curves.append(curve.opts(interpolation="steps-post", backend="matplotlib"))
            finite = np.concatenate(values)[np.isfinite(np.concatenate(values))]
            low, high = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
            for key, x in zip(events, event_times, strict=True):
                if x is not None:
                    line = pd.DataFrame({"time": [x, x], "value": [low, high], "variable": key})
                    curve = hv.Curve(line, kdims="time", vdims=["value", "variable"], label=key)
                    curve = curve.opts(line_dash="dashed", backend="bokeh")
                    curves.append(curve.opts(linestyle="dashed", backend="matplotlib"))
            panels.append(hv.Overlay(curves).relabel(panel_title).opts(**opts_dict))
            panel_title = ""
        if markers:
            panels.append(
                _marker_strip(markers, marked[obs_i], timepoints, event_times, width=width, xlabel=xlabel).relabel(
                    panel_title
                )
            )

    layout = hv.Layout(panels).cols(1)
    return layout


def _axis_times(times: np.ndarray) -> np.ndarray:
    """The times of the x-axis: numbers or dates as they are, other labels as numbers if they all are, else positions."""
    if np.issubdtype(times.dtype, np.number) or np.issubdtype(times.dtype, np.datetime64):
        return times
    numeric = pd.to_numeric(pd.Series(times), errors="coerce")
    return np.arange(len(times)) if numeric.isna().any() else numeric.to_numpy()


def _event_time(position: Any, times: np.ndarray, tem_pos: np.ndarray) -> Any:
    """The time of an event at `position` along `tem`, or `None` if it is missing or outside the plotted timepoints."""
    if pd.isna(position) or not tem_pos.min() <= float(position) <= tem_pos.max():
        return None
    if np.issubdtype(times.dtype, np.number):
        return float(np.interp(float(position), np.arange(len(times)), times))
    return times[int(position)]


def _marked(
    edata: EHRData, X: np.ndarray | DaskArray, markers: list[str], obs_pos: np.ndarray, tem_pos: np.ndarray
) -> np.ndarray:
    """Whether every marker marks a timepoint (observations x markers x timepoints).

    A variable marks the timepoints at which it has a value, a key of `obsm` those at which its value is neither missing nor 0.
    """
    marked = np.zeros((len(obs_pos), len(markers), len(tem_pos)), dtype=bool)
    variables = [(i, edata.var_names.get_loc(key)) for i, key in enumerate(markers) if key in edata.var_names]
    if variables:
        (values,) = _materialize(X[obs_pos][:, [pos for _, pos in variables]][:, :, tem_pos])
        marked[:, [i for i, _ in variables]] = ~np.isnan(np.asarray(values, dtype=float))
    for i, key in enumerate(markers):
        if key not in edata.var_names:
            values = np.asarray(_obsm_values(edata, key)[obs_pos], dtype=float)[:, tem_pos]
            marked[:, i] = ~np.isnan(values) & (values != 0)
    return marked


def _marker_strip(
    markers: list[str],
    marked: np.ndarray,
    timepoints: np.ndarray,
    event_times: list[Any],
    *,
    width: int | None,
    xlabel: str | None,
) -> hv.Overlay:
    """A row of ticks at the marked timepoints of every marker, the first on top, with the events as dashed lines."""
    n = len(markers)
    colors = hv.Cycle().values
    elements = []
    for i, key in enumerate(markers):
        ticks = hv.Spikes(pd.DataFrame({"time": timepoints[marked[i]]}), kdims="time", label=key)
        rows = {"position": n - 1 - i, "spike_length": 0.8, "color": colors[i % len(colors)]}
        elements.append(
            ticks.opts(**rows, line_width=2, tools=["hover"], backend="bokeh").opts(
                **rows, linewidth=2, backend="matplotlib"
            )
        )
    for x in event_times:
        if x is not None:
            line = hv.Curve(pd.DataFrame({"time": [x, x], "row": [-0.2, n]}), kdims="time", vdims="row")
            elements.append(
                line.opts(color="grey", line_dash="dashed", backend="bokeh").opts(
                    color="grey", linestyle="dashed", backend="matplotlib"
                )
            )
    options: dict[str, Any] = {
        "yticks": [(n - 1 - i + 0.4, key) for i, key in enumerate(markers)],
        "ylim": (-0.2, n),
        "ylabel": "",
        "show_legend": False,
    }
    if xlabel is not None:
        options["xlabel"] = xlabel
    height = 40 + 25 * n
    strip = hv.Overlay(elements).opts(
        **options, height=height, **({} if width is None else {"width": width}), backend="bokeh"
    )
    return strip.opts(**options, aspect=(width or 600) / height, backend="matplotlib")


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
