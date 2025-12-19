from __future__ import annotations

from typing import TYPE_CHECKING, Any

import holoviews as hv
import numpy as np
import pandas as pd
from fast_array_utils.conv import to_dense

from ehrapy._compat import choose_hv_backend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData


@choose_hv_backend()
def sankey_diagram(
    edata: EHRData,
    *,
    columns: Sequence[str],
    node_width: int | float = 20,
    node_padding: int | float = 10,
    node_color: str | None = None,
    edge_color: str | None = None,
    label_position: str | None = "right",
    show_values: bool = True,
    title: str | None = None,
    width: int | None = 600,
    height: int | None = 400,
    **kwargs,
) -> hv.Sankey:
    """Create a Sankey diagram of relationships across the flat observation table.

    Args:
        edata: Central data object.
        columns: Column names from `edata.obs` to visualize
        node_width: Width of the nodes in the Sankey diagram.
        node_padding: Padding between nodes in the Sankey diagram.
        node_color: Color of the nodes. If None, default coloring is used.
        edge_color: Color of the edges. If None, default coloring is used.
        label_position: Position of the labels on the nodes. Options are 'left', 'right', 'outer', or 'inner'.
        show_values: Whether to display the values on the edges.
        title: Title of the Sankey diagram.
        width: Width of the Sankey diagram.
        height: Height of the Sankey diagram.
        **kwargs: Additional styling options passed to :class:`holoviews.element.sankey.Sankey`.

    Examples:
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> edata = ed.dt.diabetes_130_fairlearn(columns_obs_only=["gender", "race"])
        >>> ep.pl.sankey_diagram(edata, columns=["gender", "race"])

        .. image:: /_static/docstring_previews/sankey.png
    """
    missing = [c for c in columns if c not in edata.obs.columns]
    if missing:
        raise KeyError(f"columns not found in edata.obs: {missing}")

    if len(columns) < 2:
        raise ValueError("columns must contain at least two obs column names.")

    df = edata.obs[columns]

    # Build links between consecutive columns
    sources, targets, values = [], [], []
    source_levels, target_levels = [], []
    for i in range(len(columns) - 1):
        col_from, col_to = columns[i], columns[i + 1]
        flows = df.groupby([col_from, col_to]).size().reset_index(name="count")
        sources.extend(col_from + ": " + flows[col_from].astype("string"))
        targets.extend(col_to + ": " + flows[col_to].astype("string"))
        values.extend(flows["count"].to_numpy())
        source_levels.extend([col_from] * len(flows))
        target_levels.extend([col_to] * len(flows))

    sankey_df = pd.DataFrame(
        {
            "source": sources,
            "target": targets,
            "value": values,
            "source_level": source_levels,
            "target_level": target_levels,
        }
    )

    sankey = hv.Sankey(sankey_df, kdims=["source", "target"], vdims=["value"])

    opts_dict: dict[str, Any] = {}

    if hv.Store.current_backend == "bokeh":
        if width is not None:
            opts_dict["width"] = width
        if height is not None:
            opts_dict["height"] = height

    if node_width is not None:
        opts_dict["node_width"] = node_width
    if node_padding is not None:
        opts_dict["node_padding"] = node_padding
    if title is not None:
        opts_dict["title"] = title
    if node_color is not None:
        opts_dict["node_color"] = node_color
    if edge_color is not None:
        opts_dict["edge_color"] = edge_color
    if label_position is not None:
        opts_dict["label_position"] = label_position
    if show_values is not None:
        opts_dict["show_values"] = show_values

    opts_dict.update(kwargs)

    sankey = sankey.opts(**opts_dict)
    return sankey


@choose_hv_backend()
def sankey_diagram_time(
    edata: EHRData,
    *,
    var_name: str,
    layer: str,
    state_labels: dict[int, str] | None = None,
    node_width: int | float = 20,
    node_padding: int | float = 10,
    node_color: str | None = None,
    edge_color: str | None = None,
    label_position: str | None = "right",
    show_values: bool = True,
    title: str | None = None,
    width: int | None = 600,
    height: int | None = 400,
    **kwargs,
) -> hv.Sankey:
    """Create a Sankey diagram showing patient state transitions over time.

    Each node represents a state at a specific time point, and flows show the
    number of patients transitioning between states.
    Visualizes how patients transition between different states
    (e.g. disease severity, treatment status) across consecutive time points.

    Args:
        edata: Central data object.
        var_name: Variable name from `edata.var_names` to visualize
        layer: Name of the layer in `edata.layers` containing the feature data to visualize.
        state_labels: Mapping from numeric state values to readable labels.
                    If None, state values will be displayed as strings of their numeric codes (e.g., "0", "1", "2").
        node_width: Width of the nodes in the Sankey diagram.
        node_padding: Padding between nodes in the Sankey diagram.
        node_color: Color of the nodes. If None, default coloring is used.
        edge_color: Color of the edges. If None, default coloring is used.
        label_position: Position of the labels on the nodes. Options are 'left', 'right', 'outer', or 'inner'.
        show_values: Whether to display the values on the edges.
        title: Title of the Sankey diagram.
        width: Width of the Sankey diagram.
        height: Height of the Sankey diagram.
        **kwargs: Additional styling options passed to :class:`holoviews.element.sankey.Sankey`.

    Examples:
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> edata = ed.dt.ehrdata_blobs(base_timepoints=5, n_variables=1, n_observations=5, random_state=59)
        >>> edata.layers["tem_data"] = edata.layers["tem_data"].astype(int)
        >>> state_labels = {-2: "no", -3: "mild", -4: "moderate", -5: "severe", -6: "critical"}
        >>> ep.pl.sankey_diagram_time(
        ...     edata,
        ...     var_name="feature_0",
        ...     layer="tem_data",
        ...     state_labels=state_labels,
        ... )

        .. image:: /_static/docstring_previews/sankey_time.png
    """
    if var_name not in edata.var_names:
        raise KeyError(f"{var_name} not found in edata.var_names.")
    if layer not in edata.layers:
        raise KeyError(f"{layer} not found in edata.layers.")

    flare_data = edata[:, edata.var_names == var_name, :].layers[layer][:, 0, :]
    mtx = to_dense(flare_data, to_cpu_memory=True)
    time_steps = edata.tem.index.tolist()

    if np.issubdtype(mtx.dtype, np.floating):
        flat = mtx.ravel()
        for x in flat:
            if not np.isfinite(x):
                continue
            rx = np.rint(x)
            if not np.isclose(x, rx, rtol=0.0, atol=1e-8):
                raise ValueError(
                    "Sankey requires discrete, binned states. "
                    f"Found non-integer value {float(x)!r}. "
                    "Bin first (e.g. with np.digitize) and pass integer codes."
                )
        states = np.unique(mtx[np.isfinite(mtx)])
    else:
        states = np.unique(mtx)

    observed = {int(s) for s in states}

    if state_labels is None:
        state_labels = {state: str(state) for state in sorted(observed)}

    missing = observed - set(state_labels)

    if missing:
        raise KeyError(f"state_labels missing keys for states: {missing}")

    state_values = sorted(state_labels.keys())
    state_names = [state_labels[val] for val in state_values]

    sources, targets, values = [], [], []
    for t in range(len(time_steps) - 1):
        for s_from_idx, s_from_val in enumerate(state_values):
            for s_to_idx, s_to_val in enumerate(state_values):
                count = np.sum((mtx[:, t] == s_from_val) & (mtx[:, t + 1] == s_to_val))
                if count > 0:
                    source_label = f"{state_names[s_from_idx]} ({time_steps[t]})"
                    target_label = f"{state_names[s_to_idx]} ({time_steps[t + 1]})"
                    sources.append(source_label)
                    targets.append(target_label)
                    values.append(int(count))

    sankey_df = pd.DataFrame({"source": sources, "target": targets, "value": values})

    sankey = hv.Sankey(sankey_df, kdims=["source", "target"], vdims=["value"])

    opts_dict: dict[str, Any] = {}

    if hv.Store.current_backend == "bokeh":
        if width is not None:
            opts_dict["width"] = width
        if height is not None:
            opts_dict["height"] = height

    if node_width is not None:
        opts_dict["node_width"] = node_width
    if node_padding is not None:
        opts_dict["node_padding"] = node_padding
    if title is not None:
        opts_dict["title"] = title
    if node_color is not None:
        opts_dict["node_color"] = node_color
    if edge_color is not None:
        opts_dict["edge_color"] = edge_color
    if label_position is not None:
        opts_dict["label_position"] = label_position
    if show_values is not None:
        opts_dict["show_values"] = show_values

    opts_dict.update(kwargs)

    sankey = sankey.opts(**opts_dict)

    return sankey
