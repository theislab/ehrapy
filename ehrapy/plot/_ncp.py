from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import numpy as np

if TYPE_CHECKING:
    from ehrdata import EHRData

# Bokeh Category10 colours – one per component
_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _require_ncp(edata: EHRData, key: str) -> None:
    missing = []
    if key not in edata.uns:
        missing.append(f"edata.uns[{key!r}]")
    if f"X_{key}" not in edata.obsm:
        missing.append(f"edata.obsm['X_{key}']")
    if f"{key}_loadings" not in edata.varm:
        missing.append(f"edata.varm['{key}_loadings']")
    if missing:
        raise KeyError(f"NCP results not found ({missing}). Run `ep.tl.ncp(edata, ...)` first.")


def ncp(
    edata: EHRData,
    *,
    key: str = "ncp",
    n_top: int = 12,
    width: int = 380,
    height: int = 280,
) -> hv.Layout:
    """Plot the factors from a Non-negative CP decomposition.

    For each component, three side-by-side panels are shown:

    - **Temporal profile** – normalised time factor, showing how the risk pattern
      evolves across the time axis.
    - **Top variables** – horizontal bar chart of the ``n_top`` variables with the
      highest loading in this component.
    - **Sample loadings** – histogram of per-observation loadings, revealing whether
      the component is diffuse (broad distribution) or selective (heavy right tail).

    Requires :func:`~ehrapy.tools.ncp` to have been run first.

    Args:
        edata: Central data object containing NCP results.
        key: Key under which NCP results are stored (matches ``key_added`` in
            :func:`~ehrapy.tools.ncp`).
        n_top: Number of top-loaded variables to display per component.
        width: Width of each individual panel in pixels.
        height: Height of each individual panel in pixels.

    Returns:
        HoloViews Layout with ``rank × 3`` panels arranged in three columns.

    Examples:
        >>> import numpy as np
        >>> import ehrdata as ed, ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=8, n_centers=3, n_observations=30, base_timepoints=12)
        >>> ep.tl.ncp(edata, layer="tem_data", rank=3, sigmoid_transform=True)
        >>> ep.pl.ncp(edata, n_top=5)

        .. image:: /_static/docstring_previews/ncp.png
    """
    _require_ncp(edata, key)

    A = np.asarray(edata.obsm[f"X_{key}"])  # (n_obs, rank)
    B = np.asarray(edata.varm[f"{key}_loadings"])  # (n_vars, rank)
    C = np.asarray(edata.uns[key]["temporal_factors"])  # (n_time, rank)
    rank = A.shape[1]
    var_names = list(edata.var_names)

    panels = []
    for r in range(rank):
        col = _PALETTE[r % len(_PALETTE)]

        a_norm = A[:, r] / (A[:, r].max() + 1e-12)
        b_norm = B[:, r] / (B[:, r].max() + 1e-12)
        c_norm = C[:, r] / (C[:, r].max() + 1e-12)

        # ── (1) Temporal profile ──────────────────────────────────────────────
        t = np.arange(len(c_norm))
        curve = hv.Curve(
            (t, c_norm),
            kdims=["Year (relative)"],
            vdims=["Norm. loading"],
        ).opts(
            width=width,
            height=height,
            title=f"C{r + 1}: Temporal profile",
            color=col,
            line_width=2.5,
            tools=["hover"],
        )
        dots = hv.Scatter(
            (t, c_norm),
            kdims=["Year (relative)"],
            vdims=["Norm. loading"],
        ).opts(size=7, color=col, tools=["hover"])
        panels.append(curve * dots)

        # ── (2) Top variables (horizontal bars) ───────────────────────────────
        top_idx = np.argsort(b_norm)[-n_top:]  # lowest → highest
        labels = [var_names[i][:48] for i in top_idx]
        vals = b_norm[top_idx]

        bars = hv.Bars(
            list(zip(labels, vals, strict=False)),
            kdims=["Variable"],
            vdims=["Norm. loading"],
        ).opts(
            width=width + 180,
            height=height,
            title=f"C{r + 1}: Top {n_top} variables",
            color=col,
            invert_axes=True,
            tools=["hover"],
            xrotation=0,
        )
        panels.append(bars)

        # ── (3) Sample-loading histogram ──────────────────────────────────────
        counts, edges = np.histogram(a_norm, bins=40)
        hist = hv.Histogram(
            (edges, counts),
            kdims=["Norm. loading"],
            vdims=["# observations"],
        ).opts(
            width=width,
            height=height,
            title=f"C{r + 1}: Sample loadings",
            color=col,
            tools=["hover"],
        )
        panels.append(hist)

    return hv.Layout(panels).cols(3)


def ncp_cluster_trajectories(
    edata: EHRData,
    *,
    layer: str,
    cluster_key: str,
    key: str = "ncp",
    n_top_diseases: int = 5,
    sigmoid_transform: bool = False,
    width: int = 520,
    height: int = 300,
) -> hv.Layout:
    """Plot mean disease-risk trajectories per cluster, guided by NCP loadings.

    For each cluster defined by ``cluster_key`` in ``edata.obs``:

    1. The dominant NCP component is identified (highest mean sample loading for
       that cluster).
    2. The ``n_top_diseases`` variables with the highest loading in that component
       are selected.
    3. Mean probability trajectories over the time axis are plotted for those
       variables, averaged across all observations in the cluster.

    Requires :func:`~ehrapy.tools.ncp` to have been run first.

    Args:
        edata: Central data object.
        layer: Key of the 3D layer holding the raw values
            (shape ``n_obs × n_vars × n_time``).
        cluster_key: Column in ``edata.obs`` with cluster/group labels.
        key: Key under which NCP results are stored.
        n_top_diseases: Number of top variables to show per cluster.
        sigmoid_transform: Apply sigmoid to the layer values before averaging.
            Set to ``True`` when the layer stores raw logits.
        width: Width of each panel in pixels.
        height: Height of each panel in pixels.

    Returns:
        HoloViews Layout with one panel per cluster, arranged in two columns.

    Examples:
        >>> import numpy as np
        >>> import ehrdata as ed, ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=8, n_centers=3, n_observations=30, base_timepoints=12)
        >>> ep.tl.ncp(edata, layer="tem_data", rank=3, sigmoid_transform=True)
        >>> ep.pl.ncp_cluster_trajectories(edata, layer="tem_data", cluster_key="cluster")

        .. image:: /_static/docstring_previews/ncp_cluster_trajectories.png
    """
    _require_ncp(edata, key)
    if cluster_key not in edata.obs:
        raise KeyError(f"Cluster key {cluster_key!r} not found in edata.obs.")
    if layer not in edata.layers:
        raise KeyError(f"Layer {layer!r} not found in edata.layers.")

    tensor = np.asarray(edata.layers[layer], dtype=np.float64)
    if tensor.ndim != 3:
        raise ValueError(f"Layer {layer!r} must be 3D, got shape {tensor.shape}.")

    if sigmoid_transform:
        from scipy.special import expit

        tensor = expit(tensor)

    A = np.asarray(edata.obsm[f"X_{key}"])  # (n_obs, rank)
    B = np.asarray(edata.varm[f"{key}_loadings"])  # (n_vars, rank)
    var_names = list(edata.var_names)
    n_time = tensor.shape[2]
    clusters = edata.obs[cluster_key]

    panels = []
    for cluster_id in sorted(clusters.unique()):
        mask = (clusters == cluster_id).values
        cluster_tensor = tensor[mask]  # (N_cluster, n_vars, n_time)

        # dominant NCP component for this cluster
        avg_loadings = A[mask].mean(axis=0)
        primary_comp = int(np.argmax(avg_loadings))
        _PALETTE[primary_comp % len(_PALETTE)]

        top_f_idx = np.argsort(B[:, primary_comp])[-n_top_diseases:]

        curves = []
        for f_idx in top_f_idx:
            avg_risk = cluster_tensor[:, f_idx, :].mean(axis=0)  # (n_time,)
            label = var_names[f_idx][:40]
            curve = hv.Curve(
                (np.arange(n_time), avg_risk),
                kdims=["Year (relative)"],
                vdims=["Mean probability"],
                label=label,
            ).opts(line_width=2, tools=["hover"])
            dots = hv.Scatter(
                (np.arange(n_time), avg_risk),
                kdims=["Year (relative)"],
                vdims=["Mean probability"],
                label=label,
            ).opts(size=5, tools=["hover"])
            curves.append(curve * dots)

        n_cluster = int(mask.sum())
        panel = hv.Overlay(curves).opts(
            width=width,
            height=height,
            legend_position="right",
            title=(f"Cluster {cluster_id}  (n={n_cluster}, dominant component={primary_comp + 1})"),
        )
        panels.append(panel)

    return hv.Layout(panels).cols(2)
