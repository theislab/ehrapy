from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData
    from matplotlib.axes import Axes


def plot_timeseries(
    edata: EHRData,
    obs_id: str | int | Sequence[str | int],
    keys: str | Sequence[str],
    *,
    layer: str,
    obs_id_key: str | None = None,
    tem_time_key: str | None = None,
    overlay: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    show: bool = True,
) -> Axes | Sequence[Axes] | None:
    """Plot variable time series either for an observation or multiple observations from a 3D EHRData layer.

    Selection logic:
        - If obs_id is an int in [0, n_obs), use it as row index.
        - Otherwise, obs_id_key must be a column name in edata.obs, and
          obs_id is matched against that column.

    Args:
        edata: Central data object.
        obs_id: row index or observation identifier(s) to plot.
        keys: Feature key or list of keys in adata.obsm to plot.
        layer: layer to use for time series data.
        obs_id_key: Column in edata.obs to match obs_id against (if obs_id is not given as row index).
        tem_time_key: Key in edata.tem to use as timepoints. If None, use edata.tem as 1D array.
        overlay: Whether to overlay multiple observations in a single plot (True) or create subplots (False).
        xlabel: The x-axis label text.
        ylabel: The y-axis label text.
        title: Set the title of the plot.
        show: Show the plot, do not return axis.

    Returns:
        Axes object or None

    Examples:
        >>> edata = ed.dt.ehrdata_blobs(
        ...     n_variables=4,
        ...     n_observations=10,
        ...     base_timepoints=100,
        ...     layer=DEFAULT_TEM_LAYER_NAME,
        ... )
        >>> edata.var.index = ["feature1", "feature2", "feature3", "feature4"]
        >>> ep.pl.plot_timeseries(
        ...     edata,
        ...     obs_id=2,
        ...     keys=["feature1", "feature2", "feature3"],
        ...     layer=DEFAULT_TEM_LAYER_NAME,
        ...     tem_time_key="timepoint",
        ... )


    """
    if isinstance(keys, str):
        key_list = [keys]
    else:
        key_list = list(keys)

    if isinstance(obs_id, (str, int)):
        obs_ids = [obs_id]
    else:
        obs_ids = list(obs_id)

    mtx = np.asarray(edata.layers[layer])
    if mtx.ndim != 3:
        raise ValueError(f"Layer {layer!r} must be 3D (n_obs, n_vars, n_time), got shape {mtx.shape}.")
    n_obs, _, n_time = mtx.shape

    if tem_time_key is None:
        timepoints = np.asarray(edata.tem)
    else:
        if tem_time_key not in edata.tem:
            raise KeyError(f"Column {tem_time_key!r} not found in edata.tem.")

        timepoints = np.asarray(edata.tem[tem_time_key])

    if timepoints.ndim != 1:
        raise ValueError(f"timepoints must be 1D, got shape {timepoints.shape}.")
    if timepoints.shape[0] != n_time:
        raise ValueError(f"Length of timepoints ({timepoints.shape[0]}) does not match n_time ({n_time}).")

    var_names = np.asarray(edata.var_names)
    var_idx_list: list[int] = []
    for k in key_list:
        matches = np.flatnonzero(var_names == k)
        if matches.size == 0:
            raise KeyError(f"Variable {k!r} not found in edata.var_names.")
        var_idx_list.append(int(matches[0]))

    if overlay:
        if len(key_list) != 1:
            raise ValueError("When overlay=True, only a single key can be plotted at a time.")
        n_panels = 1
    else:
        n_panels = len(obs_ids)

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(12, 4 * n_panels),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    if overlay:
        ax = axes[0]
        k = key_list[0]
        var_idx = var_idx_list[0]

        for obs in obs_ids:
            obs_idx, obs_id_info = _resolve_obs(obs, obs_id_key, n_obs, edata)
            y = np.asarray(mtx[obs_idx, var_idx, :], dtype=float)
            ax.plot(timepoints, y, marker="o", label=str(obs_id_info))
        ax.set_title(title if title is not None else f"Time series for variable {k!r} for multiple observations")
        ax.set_ylabel(ylabel if ylabel is not None else "Value")
        ax.legend(loc="best")
    else:
        for ax, obs in zip(axes, obs_ids, strict=False):
            obs_idx, obs_id_info = _resolve_obs(obs, obs_id_key, n_obs, edata)

            # plot each variable for this observation
            for k, v_idx in zip(key_list, var_idx_list, strict=False):
                y = np.asarray(mtx[obs_idx, v_idx, :], dtype=float)
                ax.plot(timepoints, y, marker="o", label=str(k))

            panel_title = (
                title
                if (title is not None and n_panels == 1)
                else f"Time series for observation with index {obs_idx} ({obs_id_info})"
            )
            ax.set_title(panel_title)
            ax.set_ylabel(ylabel if ylabel is not None else "Value")
            ax.legend(loc="best")

    axes[-1].set_xlabel(xlabel if xlabel is not None else tem_time_key)

    fig.tight_layout()

    if show:
        plt.show()
        return None
    else:
        if n_panels == 1:
            return axes[0]
        return axes


def _resolve_obs(obs, obs_id_key, n_obs, edata) -> tuple[int, str]:
    """Resolve obs identifier to (row index, obs_id_info) tuple."""
    if isinstance(obs, int) and 0 <= obs < n_obs:
        obs_idx = obs
        obs_id_info = f"row {obs}"

    else:
        if obs_id_key is None:
            raise ValueError("obs_id_key must be given when obs_id is not a valid row index.")
        if obs_id_key not in edata.obs:
            raise KeyError(f"Column {obs_id_key!r} not found in edata.obs.")
        col = np.asarray(edata.obs[obs_id_key])
        obs_mask = col == obs
        candidates = np.flatnonzero(obs_mask)

        if candidates.size == 0:
            raise ValueError(f"No row with {obs_id_key} == {obs!r} found in edata.obs.")
        if candidates.size > 1:
            raise ValueError(
                f"Multiple rows with {obs_id_key} == {obs!r} found. "
                "Either make that column unique or adapt the selection logic."
            )
        obs_idx = int(candidates[0])
        obs_id_info = f"{obs_id_key}={obs!r}"

    return obs_idx, obs_id_info
