from __future__ import annotations

from typing import TYPE_CHECKING, Any

import holoviews as hv
import numpy as np
import pandas as pd
from holoviews import opts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData


def plot_sankey(
    edata: EHRData,
    *,
    columns: Sequence[str],
    node_width: int | float = 20,
    node_padding: int | float = 10,
    node_color: str = None,
    label_position: str | None = "right",
    show_values: bool = True,
    title: str | None = None,
    width: int | None = 600,
    height: int | None = 400,
) -> hv.Sankey:
    """Create a Sankey diagram showing relationships across observation columns.

    Args:
        edata : Central data object.
        columns : Column names from `edata.obs` to visualize
        node_width : Width of the nodes in the Sankey diagram.
        node_padding : Padding between nodes in the Sankey diagram.
        node_color : Color of the nodes. If None, default coloring is used.
        edge_color : Color of the edges. If None, default coloring is used.
        label_position : Position of the labels on the nodes. Options are 'left', 'right', 'top', 'bottom', or 'center'.
        show_values : Whether to display the values on the edges.
        title : Title of the Sankey diagram.
        width : Width of the Sankey diagram.
        height : Height of the Sankey diagram.


    Returns:
        holoviews.Sankey

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.diabetes_130_fairlearn(columns_obs_only=["gender", "race"])
        >>> ep.pl.plot_sankey(edata, columns=["gender", "race"])
    """
    if hv.Store.current_backend is None:
        raise RuntimeError(
            "No holoviews backend selected. "
            ":func:`holoviews.extension` with ``matplotlib`` or ``bokeh`` must be called before using this function."
        )
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
    if label_position is not None:
        opts_dict["label_position"] = label_position
    if show_values is not None:
        opts_dict["show_values"] = show_values

    sankey = sankey.opts(**opts_dict)
    return sankey


def plot_sankey_time(
    edata: EHRData,
    *,
    columns: Sequence[str],
    layer: str,
    state_labels: dict[int, str] | None = None,
    node_width: int | float = 20,
    node_padding: int | float = 10,
    node_color: str = None,
    edge_color: str = None,
    label_position: str | None = "right",
    show_values: bool = True,
    title: str | None = None,
    width: int | None = 600,
    height: int | None = 400,
) -> hv.Sankey:
    """Create a Sankey diagram showing patient state transitions over time.

    Each node represents a state at a specific time point, and flows show the
    number of patients transitioning between states.
    Visualizes how patients transition between different states
    (e.g. disease severity, treatment status) across consecutive time points.

    Args:
        edata: Central data object.
        columns: Variable name from  `edata.var_names` to visualize
        layer: Name of the layer in `edata.layers` containing the feature data to visualize.
        state_labels: Mapping from numeric state values to readable labels.
                    If None, state values will be displayed as strings of their numeric codes (e.g., "0", "1", "2").
        node_width : Width of the nodes in the Sankey diagram.
        node_padding : Padding between nodes in the Sankey diagram.
        node_color : Color of the nodes. If None, default coloring is used.
        edge_color : Color of the edges. If None, default coloring is used.
        label_position : Position of the labels on the nodes. Options are 'left', 'right', 'outer', or 'inner'.
        show_values : Whether to display the values on the edges.
        title : Title of the Sankey diagram.
        width : Width of the Sankey diagram.
        height : Height of the Sankey diagram.

    Returns:
        holoviews.Sankey

    Examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> import ehrdata as ed
    >>>
    >>> layer = np.array(
    ...     [
    ...         [[1, 0, 1], [0, 1, 0]],  # patient 1: treatment, disease_flare
    ...         [[0, 1, 1], [1, 0, 0]],  # patient 2: treatment, disease_flare
    ...         [[1, 1, 0], [0, 0, 1]],  # patient 3: treatment, disease_flare
    ...     ]
    ... )
    >>>
    >>> edata = ed.EHRData(
    ...     layers={"layer_1": layer},
    ...     obs=pd.DataFrame(index=["patient_1", "patient_2", "patient_3"]),
    ...     var=pd.DataFrame(index=["treatment", "disease_flare"]),
    ...     tem=pd.DataFrame(index=["visit_0", "visit_1", "visit_2"]),
    ... )
    >>>
    >>> plot_sankey_time(edata, columns=["disease_flare"], layer="layer_1", state_labels={0: "no flare", 1: "flare"})
    """
    if hv.Store.current_backend is None:
        raise RuntimeError(
            "No holoviews backend selected. "
            ":func:`holoviews.extension` with ``matplotlib`` or ``bokeh`` must be called before using this function."
        )

    flare_data = edata[:, edata.var_names.isin(columns), :].layers[layer][:, 0, :]

    time_steps = edata.tem.index.tolist()

    if state_labels is None:
        unique_states = np.unique(flare_data)
        unique_states = unique_states[~np.isnan(unique_states)]
        state_labels = {int(state): str(state) for state in unique_states}

    state_values = sorted(state_labels.keys())
    state_names = [state_labels[val] for val in state_values]

    sources, targets, values = [], [], []
    for t in range(len(time_steps) - 1):
        for s_from_idx, s_from_val in enumerate(state_values):
            for s_to_idx, s_to_val in enumerate(state_values):
                count = np.sum((flare_data[:, t] == s_from_val) & (flare_data[:, t + 1] == s_to_val))
                if count > 0:
                    source_label = f"{state_names[s_from_idx]} ({time_steps[t]})"
                    target_label = f"{state_names[s_to_idx]} ({time_steps[t + 1]})"
                    sources.append(source_label)
                    targets.append(target_label)
                    values.append(int(count))

    sankey_df = pd.DataFrame({"source": sources, "target": targets, "value": values})

    sankey = hv.Sankey(sankey_df, kdims=["source", "target"], vdims=["value"])

    opts_dict: dict[str, Any] = {}
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
    if label_position is not None:
        opts_dict["label_position"] = label_position
    if show_values is not None:
        opts_dict["show_values"] = show_values

    sankey = sankey.opts(**opts_dict)

    return sankey
