from __future__ import annotations

from typing import TYPE_CHECKING

import missingno as msno
import numpy as np
import pandas as pd
from array_api_compat import array_namespace
from fast_array_utils.conv import to_dense

from ehrapy._compat import _materialize
from ehrapy.preprocessing._missing_data import _missing_mask

if TYPE_CHECKING:
    from ehrdata import EHRData
    from matplotlib.axes import Axes


def _plotted_columns(edata: EHRData, *, categoricals: bool) -> pd.Index:
    return edata.var_names if categoricals else edata.var_names[~edata.var_names.str.startswith("ehrapycat")]


def _nullity_df(edata: EHRData, *, layer: str | None, categoricals: bool) -> pd.DataFrame:
    """The plotted variables as a DataFrame that is NaN where values are missing, because missingno only reads nullity."""
    X = edata.X if layer is None else edata.layers[layer]
    columns = _plotted_columns(edata, categoricals=categoricals)
    missing = to_dense(_missing_mask(X[:, edata.var_names.get_indexer(columns)]), to_cpu_memory=True)
    if missing.ndim == 3:
        missing = np.moveaxis(missing, 1, 2).reshape(-1, len(columns))
    return pd.DataFrame(np.where(missing, np.float32(np.nan), np.float32(0)), columns=columns)


def _observed_over_time(
    edata: EHRData,
    *,
    layer: str | None,
    categoricals: bool,
    figsize: tuple,
    color: tuple,
    fontsize: float,
    labels: bool,
    label_rotation: float,
) -> Axes:
    """Shade the percentage of observations with a value for every timepoint (row) and variable (column)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    X = edata.X if layer is None else edata.layers[layer]
    columns = _plotted_columns(edata, categoricals=categoricals)
    observed = ~_missing_mask(X[:, edata.var_names.get_indexer(columns)])
    xp = array_namespace(observed)
    (percent,) = _materialize(100 * xp.mean(xp.astype(observed, xp.float64), axis=0))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = LinearSegmentedColormap.from_list("observed", ["white", color])
    image = ax.imshow(np.asarray(percent).T, aspect="auto", interpolation="none", cmap=cmap, vmin=0, vmax=100)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(columns)), columns if labels else [], rotation=label_rotation, ha="left", fontsize=fontsize)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    tem = edata.tem.index
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda position, _: str(tem[int(position)]) if 0 <= position < len(tem) else "")
    )
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_ylabel("time", fontsize=fontsize)
    fig.colorbar(image, ax=ax).set_label("observed (%)", fontsize=fontsize)
    return ax


def missing_values_matrix(
    edata: EHRData,
    *,
    filter: str | None = None,
    max_cols: int = 0,
    max_percentage: float = 0,
    sort: str | None = None,
    figsize: tuple = (25, 10),
    width_ratios: tuple = (15, 1),
    color: tuple = (0.25, 0.25, 0.25),
    fontsize: float = 16,
    labels: bool = True,
    label_rotation: float = 45,
    sparkline: bool = True,
    categoricals: bool = False,
    layer: str | None = None,
):  # pragma: no cover
    """A matrix visualization of the nullity of the given data object.

    For 3D data, the matrix shows the percentage of observations with a value of every variable at every timepoint, and `filter`, `max_cols`, `max_percentage`, `sort`, `width_ratios` and `sparkline` do not apply.

    Args:
        edata: Central data object.
        filter: The filter to apply to the matrix. Should be one of "top", "bottom", or None.
        max_cols: The max number of columns from the data object to include.
        max_percentage: The max percentage fill of the columns from the data object.
        sort: The row sort order to apply. Can be "ascending", "descending", or None.
        figsize: The size of the figure to display.
        width_ratios: The ratio of the width of the matrix to the width of the sparkline.
        color: The color of the filled columns.
        fontsize: The figure's font size.
        labels: Whether or not to display the column names.
        label_rotation: What angle to rotate the text labels to.
        sparkline: Whether or not to display the sparkline.
        categoricals: Whether to include "ehrapycat" columns to the plot.
        layer: The layer to use.

    Returns:
        The plot axis.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pl.missing_values_matrix(edata, filter="bottom", max_cols=15, max_percentage=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_matrix.png
    """
    if (edata.X if layer is None else edata.layers[layer]).ndim == 3:
        return _observed_over_time(
            edata,
            layer=layer,
            categoricals=categoricals,
            figsize=figsize,
            color=color,
            fontsize=fontsize,
            labels=labels,
            label_rotation=label_rotation,
        )
    df = _nullity_df(edata, layer=layer, categoricals=categoricals)
    return msno.matrix(
        df,
        filter,
        max_cols,
        max_percentage,
        sort,
        figsize,
        width_ratios,
        color,
        fontsize,
        labels,
        label_rotation,
        sparkline,
    )


def missing_values_barplot(
    edata: EHRData,
    *,
    log: bool = False,
    filter: str | None = None,
    max_cols: int = 0,
    max_percentage: float = 0,
    sort: str | None = None,
    figsize: tuple | None = None,
    color: str = "dimgray",
    fontsize: float = 16,
    labels: str | None = None,
    label_rotation: float = 45,
    orientation: str | None = None,
    categoricals: bool = False,
    layer: str | None = None,
):  # pragma: no cover
    """A bar chart visualization of the nullity of the given data object.

    For 3D data, every timepoint of every observation counts as one row.

    Args:
        edata: Central data object.
        log: Whether to display a logarithmic plot.
        filter: The filter to apply to the barplot. Should be one of "top", "bottom", or None.
        max_cols: The max number of columns from the data object to include.
        max_percentage: The max percentage fill of the columns from the data object.
        sort: The row sort order to apply. Can be "ascending", "descending", or None.
        figsize: The size of the figure to display.
        color: The color of the filled columns.
        fontsize: The figure's font size.
        labels: Whether to display the column names.
        label_rotation: What angle to rotate the text labels to.
        orientation: The way the bar plot is oriented.
        categoricals: Whether to include "ehrapycat" columns to the plot.
        layer: The layer to use.

    Returns:
        The plot axis.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pl.missing_values_barplot(edata, filter="bottom", max_cols=15, max_percentage=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_barplot.png
    """
    df = _nullity_df(edata, layer=layer, categoricals=categoricals)
    return msno.bar(
        df,
        figsize,
        fontsize,
        labels,
        label_rotation,
        log,
        color,
        filter,
        max_cols,
        max_percentage,
        sort,
        orientation,
    )


def missing_values_heatmap(
    edata: EHRData,
    *,
    filter: str | None = None,
    max_cols: int = 0,
    max_percentage: float = 0,
    sort: str | None = None,
    figsize: tuple = (20, 12),
    fontsize: float = 16,
    labels: bool = True,
    label_rotation: float = 45,
    cmap: str = "RdBu",
    vmin: int = -1,
    vmax: int = 1,
    cbar: bool = True,
    categoricals: bool = False,
    layer: str | None = None,
):  # pragma: no cover
    """Presents a `seaborn` heatmap visualization of nullity correlation in the given data object.

    Note that this visualization has no special support for large datasets. For those, try the dendrogram instead.
    For 3D data, every timepoint of every observation counts as one row.

    Args:
        edata: Central data object.
        filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None.
        max_cols: The max number of columns from the data object to include.
        max_percentage: The max percentage fill of the columns from the data object.
        sort: The row sort order to apply. Can be "ascending", "descending", or None.
        figsize: The size of the figure to display.
        fontsize: The figure's font size.
        labels: Whether or not to display the column names.
        label_rotation: What angle to rotate the text labels to.
        cmap: What `matplotlib` colormap to use.
        vmin: The normalized colormap threshold.
        vmax: The normalized colormap threshold.
        cbar: Whether to draw a colorbar.
        categoricals: Whether to include "ehrapycat" columns to the plot.
        layer: The layer to use.

    Returns:
        The plot axis.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pl.missing_values_heatmap(edata, filter="bottom", max_cols=15, max_percentage=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_heatmap.png
    """
    df = _nullity_df(edata, layer=layer, categoricals=categoricals)
    return msno.heatmap(
        df,
        filter,
        max_cols,
        max_percentage,
        sort,
        figsize,
        fontsize,
        labels,
        label_rotation,
        cmap,
        vmin,
        vmax,
        cbar,
    )


def missing_values_dendrogram(
    edata: EHRData,
    *,
    method: str = "average",
    filter: str | None = None,
    max_cols: int = 0,
    max_percentage: float = 0,
    orientation: str | None = None,
    figsize: tuple | None = None,
    fontsize: float = 16,
    label_rotation: float = 45,
    categoricals: bool = False,
    layer: str | None = None,
):
    """Fits a `scipy` hierarchical clustering algorithm and visualizes the results as a `scipy` dendrogram.

    For 3D data, every timepoint of every observation counts as one row.

    The default vertical display will fit up to 50 columns. If more than 50 columns are specified and orientation is
    left unspecified the dendrogram will automatically swap to a horizontal display to fit the additional variables.

    Args:
        edata: Central data object.
        method: The distance measure being used for clustering. This parameter is passed to `scipy.hierarchy`.
        filter: The filter to apply to the dendrogram. Should be one of "top", "bottom", or None.
        max_cols: The max number of columns from the data object to include.
        max_percentage: The max percentage fill of the columns from the data object.
        figsize: The size of the figure to display.
        fontsize: The figure's font size.
        orientation: The way the dendrogram is oriented.
        label_rotation: What angle to rotate the text labels to. .
        categoricals: Whether to include "ehrapycat" columns to the plot.
        layer: The layer to use.

    Returns:
        The plot axis.

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pl.missing_values_dendrogram(edata, filter="bottom", max_cols=15, max_percentage=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_dendrogram.png
    """
    df = _nullity_df(edata, layer=layer, categoricals=categoricals)
    return msno.dendrogram(df, method, filter, max_cols, max_percentage, orientation, figsize, fontsize, label_rotation)
