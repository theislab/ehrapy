from __future__ import annotations

from typing import TYPE_CHECKING

import missingno as msno

from ehrapy.anndata import anndata_ext as ae

if TYPE_CHECKING:
    from anndata import AnnData

# Functionality provided by https://github.com/ResidentMario/missingno


def missing_values_matrix(
    adata: AnnData,
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
):  # pragma: no cover
    """A matrix visualization of the nullity of the given AnnData object.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        filter: The filter to apply to the matrix. Should be one of "top", "bottom", or None. Defaults to None .
        max_cols: The max number of columns from the AnnData object to include.
        max_percentage: The max percentage fill of the columns from the AnnData object.
        sort: The row sort order to apply. Can be "ascending", "descending", or None.
        figsize: The size of the figure to display.
        width_ratios: The ratio of the width of the matrix to the width of the sparkline.
        color: The color of the filled columns.
        fontsize: The figure's font size.
        labels: Whether or not to display the column names.
        label_rotation: What angle to rotate the text labels to.
        sparkline: Whether or not to display the sparkline.
        categoricals: Whether to include "ehrapycat" columns to the plot.

    Returns:
        The plot axis.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.data.mimic_2(encoded=True)
        >>> ep.pl.missing_values_matrix(adata, filter="bottom", max_cols=15, max_percentage=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_matrix.png
    """
    df = ae.anndata_to_df(adata)

    if not categoricals:
        non_categorical_columns = [col for col in df if not col.startswith("ehrapycat")]
        return msno.matrix(
            df[non_categorical_columns],
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
    else:
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
    adata: AnnData,
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
):  # pragma: no cover
    """A bar chart visualization of the nullity of the given AnnData object.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        log: Whether or not to display a logarithmic plot.
        filter: The filter to apply to the barplot. Should be one of "top", "bottom", or None. Defaults to None .
        max_cols: The max number of columns from the AnnData object to include.
        max_percentage: The max percentage fill of the columns from the AnnData object.
        sort: The row sort order to apply. Can be "ascending", "descending", or None.
        figsize: The size of the figure to display.
        color: The color of the filled columns.
        fontsize: The figure's font size.
        labels: Whether or not to display the column names.
        label_rotation: What angle to rotate the text labels to.
        orientation: The way the bar plot is oriented.
        categoricals: Whether to include "ehrapycat" columns to the plot.

    Returns:
        The plot axis.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.data.mimic_2(encoded=True)
        >>> ep.pl.missing_values_barplot(adata, filter="bottom", max_cols=15, max_percentage=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_barplot.png
    """
    df = ae.anndata_to_df(adata)

    if not categoricals:
        non_categorical_columns = [col for col in df if not col.startswith("ehrapycat")]
        return msno.bar(
            df[non_categorical_columns],
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
    else:
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
    adata: AnnData,
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
):  # pragma: no cover
    """Presents a `seaborn` heatmap visualization of nullity correlation in the given AnnData object.

    Note that this visualization has no special support for large datasets. For those, try the dendrogram instead.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None. Defaults to None .
        max_cols: The max number of columns from the AnnData object to include.
        max_percentage: The max percentage fill of the columns from the AnnData object.
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

    Returns:
        The plot axis.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.data.mimic_2(encoded=True)
        >>> ep.pl.missing_values_heatmap(adata, filter="bottom", max_cols=15, max_percentage=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_heatmap.png
    """
    df = ae.anndata_to_df(adata)

    if not categoricals:
        non_categorical_columns = [col for col in df if not col.startswith("ehrapycat")]
        return msno.heatmap(
            df[non_categorical_columns],
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
    else:
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
    adata: AnnData,
    method: str = "average",
    filter: str | None = None,
    max_cols: int = 0,
    max_percentage: float = 0,
    orientation: str | None = None,
    figsize: tuple | None = None,
    fontsize: float = 16,
    label_rotation: float = 45,
    categoricals: bool = False,
):
    """Fits a `scipy` hierarchical clustering algorithm to the given AnnData object's var and visualizes the results as
    a `scipy` dendrogram.

    The default vertical display will fit up to 50 columns. If more than 50 columns are specified and orientation is
    left unspecified the dendrogram will automatically swap to a horizontal display to fit the additional variables.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        method: The distance measure being used for clustering. This parameter is passed to `scipy.hierarchy`.
        filter: The filter to apply to the dendrogram. Should be one of "top", "bottom", or None. Defaults to None .
        max_cols: The max number of columns from the AnnData object to include.
        max_percentage: The max percentage fill of the columns from the AnnData object.
        figsize: The size of the figure to display.
        fontsize: The figure's font size.
        orientation: The way the dendrogram is oriented.
        label_rotation: What angle to rotate the text labels to. .
        categoricals: Whether to include "ehrapycat" columns to the plot.

    Returns:
        The plot axis.

    Example:
        >>> import ehrapy as ep
        >>> adata = ep.data.mimic_2(encoded=True)
        >>> ep.pl.missing_values_dendrogram(adata, filter="bottom", max_cols=15, max_percentage=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_dendrogram.png
    """
    df = ae.anndata_to_df(adata)

    if not categoricals:
        non_categorical_columns = [col for col in df if not col.startswith("ehrapycat")]
        return msno.dendrogram(
            df[non_categorical_columns],
            method,
            filter,
            max_cols,
            max_percentage,
            orientation,
            figsize,
            fontsize,
            label_rotation,
        )
    else:
        return msno.dendrogram(
            df, method, filter, max_cols, max_percentage, orientation, figsize, fontsize, label_rotation
        )
