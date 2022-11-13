from __future__ import annotations

import missingno as msno
import pandas as pd

# Functionality provided by https://github.com/ResidentMario/missingno

def missing_values_matrix(
    adata: AnnData,
    filter: str | None = None,
    n: int = 0,
    p: float = 0,
    sort: str | None = None,
    figsize: tuple = (25, 10),
    width_ratios: tuple = (15, 1),
    color: tuple = (0.25, 0.25, 0.25),
    fontsize: float = 16,
    labels: bool = True,
    label_rotation: float = 45,
    sparkline: bool = True,
    categoricals: bool = False
):  # pragma: no cover
    """A matrix visualization of the nullity of the given AnnData object.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        filter: The filter to apply to the matrix. Should be one of "top", "bottom", or None (default).
        n: The max number of columns from the AnnData object to include.
        p: The max percentage fill of the columns from the AnnData object.
        sort: The row sort order to apply. Can be "ascending", "descending", or None.
        figsize: The size of the figure to display.
        width_ratios: The ratio of the width of the matrix to the width of the sparkline. Defaults to `(15, 1)`. Does nothing if `sparkline=False`.
        color: The color of the filled columns. Default is `(0.25, 0.25, 0.25)`.
        fontsize: The figure's font size. Default is 16pt.
        labels: Whether or not to display the column names. Defaults to the underlying data labels when there are 50 columns or less, and no labels when there are more than 50 columns.
        label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
        sparkline: Whether or not to display the sparkline. Defaults to True.
        categoricals: Whether to include "ehrapycat" columns to the plot. Defaults to False.

    Returns:
        The plot axis.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pl.missing_values_matrix(adata, filter='bottom', n=15, p=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_matrix.png
    """

    df = pd.DataFrame(adata.X, columns=adata.var_names)

    if not categoricals:
        filter_col = [col for col in df if not col.startswith('ehrapycat')]
        return msno.matrix(df[filter_col], filter, n, p, sort, figsize, width_ratios, color, fontsize, labels,
                           label_rotation,
                           sparkline)
    else:
        return msno.matrix(df, filter, n, p, sort, figsize, width_ratios, color, fontsize, labels,
                           label_rotation,
                           sparkline)


def missing_values_barplot(
    adata: AnnData,
    log: bool = False,
    filter: str | None = None,
    n: int = 0,
    p: float = 0,
    sort: str | None = None,
    figsize: tuple | None = None,
    color: str = 'dimgray',
    fontsize: float = 16,
    labels: str | None = None,
    label_rotation: float = 45,
    orientation: str | None = None,
    categoricals: bool = False
):  # pragma: no cover
    """A bar chart visualization of the nullity of the given AnnData object.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        log: Whether or not to display a logarithmic plot. Defaults to False (linear).
        filter: The filter to apply to the barplot. Should be one of "top", "bottom", or None (default).
        n: The max number of columns from the AnnData object to include.
        p: The max percentage fill of the columns from the AnnData object.
        sort: The row sort order to apply. Can be "ascending", "descending", or None.
        figsize: The size of the figure to display.
        color: The color of the filled columns. Default is `(0.25, 0.25, 0.25)`.
        fontsize: The figure's font size. Default is 16pt.
        labels: Whether or not to display the column names. Defaults to the underlying data labels when there are 50 columns or less, and no labels when there are more than 50 columns.
        label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
        orientation: The way the bar plot is oriented. Defaults to vertical if there are less than or equal to 50 columns and horizontal if there are more.
        categoricals: Whether to include "ehrapycat" columns to the plot. Defaults to False.

    Returns:
        The plot axis.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pl.missing_values_barplot(adata, filter='bottom', n=15, p=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_barplot.png
    """

    df = pd.DataFrame(adata.X, columns=adata.var_names)

    if not categoricals:
        filter_col = [col for col in df if not col.startswith('ehrapycat')]
        return msno.bar(df[filter_col], figsize, fontsize, labels, label_rotation, log, color, filter, n, p, sort,
                        orientation)
    else:
        return msno.bar(df, figsize, fontsize, labels, label_rotation, log, color, filter, n, p, sort,
                        orientation)


def missing_values_heatmap(
    adata: AnnData,
    filter: str | None = None,
    n: int = 0,
    p: float = 0,
    sort: str | None = None,
    figsize: tuple = (20, 12),
    fontsize: float = 16,
    labels: bool = True,
    label_rotation: float = 45,
    cmap: str = 'RdBu',
    vmin: int = -1,
    vmax: int = 1,
    cbar: bool = True,
    categoricals: bool = False
):  # pragma: no cover
    """Presents a `seaborn` heatmap visualization of nullity correlation in the given AnnData object.

    Note that this visualization has no special support for large datasets. For those, try the dendrogram instead.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default).
        n: The max number of columns from the AnnData object to include.
        p: The max percentage fill of the columns from the AnnData object.
        sort: The row sort order to apply. Can be "ascending", "descending", or None.
        figsize: The size of the figure to display.
        fontsize: The figure's font size. Default is 16pt.
        labels: Whether or not to display the column names. Defaults to the underlying data labels when there are 50 columns or less, and no labels when there are more than 50 columns.
        label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
        cmap: What `matplotlib` colormap to use. Defaults to `RdBu`.
        vmin: The normalized colormap threshold. Defaults to -1, e.g. the bottom of the color scale.
        vmax: The normalized colormap threshold. Defaults to 1, e.g. the bottom of the color scale.
        cbar: Whether to draw a colorbar. Defaults to True.
        categoricals: Whether to include "ehrapycat" columns to the plot. Defaults to False.

    Returns:
        The plot axis.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pl.missing_values_heatmap(adata, filter='bottom', n=15, p=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_heatmap.png
    """

    df = pd.DataFrame(adata.X, columns=adata.var_names)

    if not categoricals:
        filter_col = [col for col in df if not col.startswith('ehrapycat')]
        return msno.heatmap(df[filter_col], filter, n, p, sort, figsize, fontsize, labels, label_rotation, cmap, vmin,
                            vmax, cbar)
    else:
        return msno.heatmap(df, filter, n, p, sort, figsize, fontsize, labels, label_rotation, cmap, vmin,
                            vmax, cbar)


def missing_values_dendrogram(
    adata: AnnData,
    method: str = 'average',
    filter: str | None = None,
    n: int = 0,
    p: float = 0,
    orientation: str | None = None,
    figsize: tuple | None = None,
    fontsize: float = 16,
    label_rotation: float = 45,
    categoricals: bool = False
):
    """Fits a `scipy` hierarchical clustering algorithm to the given AnnData object's var and visualizes the results as
    a `scipy` dendrogram.

    The default vertical display will fit up to 50 columns. If more than 50 columns are specified and orientation is
    left unspecified the dendrogram will automatically swap to a horizontal display to fit the additional variables.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        filter: The filter to apply to the dendrogram. Should be one of "top", "bottom", or None (default).
        n: The max number of columns from the AnnData object to include.
        p: The max percentage fill of the columns from the AnnData object.
        figsize: The size of the figure to display.
        fontsize: The figure's font size. Default is 16pt.
        orientation: The way the dendrogram is oriented. Defaults to top-down if there are less than or equal to 50 columns and left-right if there are more.
        label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
        categoricals: Whether to include "ehrapycat" columns to the plot. Defaults to False.

    Returns:
        The plot axis.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pl.missing_values_dendrogram(adata, filter='bottom', n=15, p=0.999)

    Preview:
        .. image:: /_static/docstring_previews/missingno_dendrogram.png
    """

    df = pd.DataFrame(adata.X, columns=adata.var_names)

    if not categoricals:
        filter_col = [col for col in df if not col.startswith('ehrapycat')]
        return msno.dendrogram(df[filter_col], method, filter, n, p, orientation, figsize, fontsize, label_rotation)
    else:
        return msno.dendrogram(df, method, filter, n, p, orientation, figsize, fontsize, label_rotation)
