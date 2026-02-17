from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import holoviews as hv
import numpy as np
import pandas as pd

import ehrapy as ep

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData


def plot_variable_correlations(
    edata: EHRData,
    *,
    layer: str,
    var_names: Sequence[str] | None = None,
    method: Literal["spearman", "pearson", "kendall"] = "pearson",
    agg: Literal["mean", "last", "first"] = "mean",
    correction_method: Literal["bonferroni", "fdr_bh", "fdr_tsbh", "holm", "none"] = "bonferroni",
    alpha: float = 0.05,
    width: int = 600,
    height: int = 600,
    cmap: str = "RdBu_r",
    show_values: bool = True,
    title: str | None = None,
) -> hv.HeatMap | hv.Overlay:
    """Plot variable correlations with heatmap.

    Computes a correlation matrix (Pearson or Spearman) for the selected variables
    from the given layer. If the layer contains a time dimension, values are first
    aggregated per variable across time. Cells are annotated with the correlation
    coefficient; an asterisk marks statistically significant correlations after
    correction.

    Args:
        edata: Central data object.
        layer: Layer to extract data from.
        var_names: List of variable names to compute correlation of. If None, uses all numeric variables.
        method: Correlation method: "spearman", "kendall" or "pearson".
        agg: How to aggregate time dimension: "mean", "last" or "first".
        correction_method: Multiple testing correction method:
            -   "bonferroni": conservative Bonferroni correction
            -   "fdr_bh": Benjamini Hochberg FDR
            -   "fdr_tsbh": two-stage Benjamini-Hochberg, better calibrated when many variables are truly correlated
            -   "holm": Holm-Bonferroni
            -   "none": no correction
        alpha: Significance threshold after correction.
        width: Plot width in pixels.
        height: Plot height in pixels.
        cmap: Colormap for the heatmap.
        show_values: If True, display correlation values on cells.
        title: Set the title of the plot.

    Returns:
        HoloViews HeatMap object.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=10, n_centers=5, n_observations=200, base_timepoints=3)
        >>> ep.pl.plot_variable_correlations(
        ...     edata, layer="tem_data", method="pearson", agg="mean", correction_method="fdr_bh", width=700
        ... )

        .. image:: /_static/docstring_previews/variable_correlations_heatmap.png
    """
    corr_df, _, sig_df = ep.tl.compute_variable_correlations(
        edata=edata,
        layer=layer,
        var_names=var_names,
        method=method,
        agg=agg,
        correction_method=correction_method,
        alpha=alpha,
    )
    variables = corr_df.columns.to_list()
    heatmap_data = []

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            corr = corr_df.loc[var1, var2]
            is_sig = sig_df.loc[var1, var2]

            if np.isnan(corr):
                label = "N/A"
                corr = 0
            else:
                label = f"{corr:.2f}" + ("*" if is_sig and i != j else "")
            heatmap_data.append(
                {
                    "variable1": var1,
                    "variable2": var2,
                    "correlation": corr,
                    "significant": is_sig,
                    "label": label,
                }
            )

    heatmap_df = pd.DataFrame(heatmap_data)

    if title is None:
        title = f"{method.capitalize()} Correlation Matrix "
        if correction_method != "none":
            title += f"(correction method: {correction_method}, alpha={alpha})"

    heatmap = hv.HeatMap(heatmap_df, kdims=["variable1", "variable2"], vdims=["correlation", "label"])
    heatmap = heatmap.opts(
        width=width,
        height=height,
        cmap=cmap,
        clim=(-1, 1),
        colorbar=True,
        title=title,
        xrotation=45,
        toolbar="above",
        fontscale=1.2,
        xlabel="",
        ylabel="",
    )
    if show_values:
        labels = hv.Labels(heatmap_df, kdims=["variable1", "variable2"], vdims="label").opts(
            text_font_size="10pt",
            text_color="black",
            text_align="center",
        )

        overlay = (heatmap * labels).opts(
            width=width,
            height=height,
        )
        return overlay

    return heatmap


def plot_variable_dependencies(
    edata: EHRData,
    *,
    layer: str,
    var_names: Sequence[str] | None = None,
    method: Literal["spearman", "pearson", "kendall"] = "pearson",
    agg: Literal["mean", "last", "first"] = "mean",
    correction_method: Literal["bonferroni", "fdr_bh", "fdr_tsbh", "holm", "none"] = "bonferroni",
    alpha: float = 0.05,
    min_correlation: float = 0.3,
    only_significant: bool = True,
    width: int = 600,
    height: int = 600,
    cmap: str = "RdBu_r",
    title: str | None = None,
) -> hv.Chord:
    """Plot correlation dependencies as a chord diagram.

    Computes pairwise correlations between selected variables from layer and
    visualizes them as a chord diagram. If the layer contains a time dimension,
    values are aggregated per variable before correlation is computed.

    Args:
        edata: Central data object.
        layer: Layer to extract data from.
        var_names: List of variable names to compute correlation of. If None, uses all numeric variables.
        method: Correlation method: "spearman", "kendall" or "pearson".
        agg: How to aggregate time dimension: "mean", "last" or "first".
        correction_method: Multiple testing correction method:
            -   "bonferroni": conservative Bonferroni correction
            -   "fdr_bh": Benjamini Hochberg FDR
            -   "fdr_tsbh": two-stage Benjamini-Hochberg, better calibrated when many variables are truly correlated
            -   "holm": Holm-Bonferroni
            -   "none": no correction
        alpha: Significance threshold after correction.
        min_correlation: Minimum absolute correlation to show a chord.
        only_significant: If True, only show significant correlations.
        width: Plot width in pixels.
        height: Plot height in pixels.
        cmap: Colormap for the chord diagram.
        title: Set the title of the plot.

    Returns:
            HoloViews Chord diagram object.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=10, n_centers=5, n_observations=200, base_timepoints=3)
        >>> ep.pl.plot_variable_dependencies(
        ...     edata, layer="tem_data", method="pearson", agg="mean", correction_method="fdr_bh"
        ... )

        .. image:: /_static/docstring_previews/variable_dependencies_chord.png

    """
    corr_df, _, sig_df = ep.tl.compute_variable_correlations(
        edata=edata,
        layer=layer,
        var_names=var_names,
        method=method,
        agg=agg,
        correction_method=correction_method,
        alpha=alpha,
    )

    if not 0 <= min_correlation <= 1:
        raise ValueError(f"min_correlation must be between 0 and 1, got {min_correlation}")

    edges = []
    variables = corr_df.columns.to_list()
    var_to_idx = {var: idx for idx, var in enumerate(variables)}

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i < j:
                corr = corr_df.loc[var1, var2]
                is_sig = sig_df.loc[var1, var2]

                if np.isnan(corr):
                    continue
                if only_significant and not is_sig:
                    continue
                if abs(corr) < min_correlation:
                    continue

                edges.append(
                    {"source": var_to_idx[var1], "target": var_to_idx[var2], "value": abs(corr), "correlation": corr}
                )
    if len(edges) == 0:
        raise ValueError(
            f"No correlations meet criteria (minimum correlation to plot = {min_correlation},"
            f"Try lowering min_correlation or setting only_significant=False."
        )

    edges_df = pd.DataFrame(edges)
    nodes_df = pd.DataFrame({"index": range(len(variables)), "name": variables})

    if title is None:
        title = f"{method.capitalize()} Correlation Chord Diagram "
        if correction_method != "none":
            title += f"({correction_method}, alpha={alpha})"

    chord = hv.Chord((edges_df, hv.Dataset(nodes_df, "index")))
    chord = chord.opts(
        width=width,
        height=height,
        node_color="index",
        edge_color="value",
        labels="name",
        node_size=15,
        title=title,
        cmap=cmap,
    )
    return chord
