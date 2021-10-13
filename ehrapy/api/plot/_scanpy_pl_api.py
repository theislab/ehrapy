from typing import Callable, Collection, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import scanpy as sc
from anndata import AnnData
from cycler import Cycler
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap, Normalize
from scanpy.plotting import DotPlot, MatrixPlot, StackedViolin
from scanpy.plotting._utils import _AxesSubplot

from ehrapy.util._doc_util import (
    _doc_params,
    doc_common_groupby_plot_args,
    doc_common_plot_args,
    doc_scatter_basic,
    doc_show_save_ax,
    doc_vboundnorm,
)

_Basis = Literal["pca", "tsne", "umap", "diffmap", "draw_graph_fr"]
_VarNames = Union[str, Sequence[str]]
ColorLike = Union[str, Tuple[float, ...]]
_IGraphLayout = Literal["fa", "fr", "rt", "rt_circular", "drl", "eq_tree", ...]  # type: ignore
_FontWeight = Literal["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
_FontSize = Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]
VBound = Union[str, float, Callable[[Sequence[float]], float]]


@_doc_params(scatter_temp=doc_scatter_basic, show_save_ax=doc_show_save_ax)
def scatter(
    adata: AnnData,
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Union[str, Collection[str]] = None,
    use_raw: Optional[bool] = None,
    layers: Union[str, Collection[str]] = None,
    sort_order: bool = True,
    alpha: Optional[float] = None,
    basis: Optional[_Basis] = None,
    groups: Union[str, Iterable[str]] = None,
    components: Union[str, Collection[str]] = None,
    projection: Literal["2d", "3d"] = "2d",
    legend_loc: str = "right margin",
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight, None] = None,
    legend_fontoutline: float = None,
    color_map: Union[str, Colormap] = None,
    palette: Union[Cycler, ListedColormap, ColorLike, Sequence[ColorLike]] = None,
    frameon: Optional[bool] = None,
    right_margin: Optional[float] = None,
    left_margin: Optional[float] = None,
    size: Union[int, float, None] = None,
    title: Optional[str] = None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    ax: Optional[Axes] = None,
):
    """Scatter plot along observations or variables axes.

    Color the plot using annotations of observations (`.obs`), variables (`.var`) or features (`.var_names`).

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        x: x coordinate
        y: y coordinate
        color: Keys for annotations of observations/patients or features, or a hex color specification, e.g.,
               `'ann1'`, `'#fe57a1'`, or `['ann1', 'ann2']`.
        use_raw: Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.
        layers: Use the `layers` attribute of `adata` if present: specify the layer for `x`, `y` and `color`.
                If `layers` is a string, then it is expanded to `(layers, layers, layers)`.
        basis: String that denotes a plotting tool that computed coordinates.
        {scatter_temp}
        {show_save_ax}

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.
    """
    return sc.pl.scatter(
        adata=adata,
        x=x,
        y=y,
        color=color,
        use_raw=use_raw,
        layers=layers,
        sort_order=sort_order,
        alpha=alpha,
        basis=basis,
        groups=groups,
        components=components,
        projection=projection,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        legend_fontweight=legend_fontweight,
        legend_fontoutline=legend_fontoutline,
        color_map=color_map,
        palette=palette,
        frameon=frameon,
        right_margin=right_margin,
        left_margin=left_margin,
        size=size,
        title=title,
        show=show,
        save=save,
        ax=ax,
    )


@_doc_params(
    vminmax=doc_vboundnorm,
    show_save_ax=doc_show_save_ax,
    common_plot_args=doc_common_plot_args,
)
def heatmap(
    adata: AnnData,
    var_names: Union[_VarNames, Mapping[str, _VarNames]],
    groupby: Union[str, Sequence[str]],
    use_raw: Optional[bool] = None,
    log: bool = False,
    num_categories: int = 7,
    dendrogram: Union[bool, str] = False,
    feature_symbols: Optional[str] = None,
    var_group_positions: Optional[Sequence[Tuple[int, int]]] = None,
    var_group_labels: Optional[Sequence[str]] = None,
    var_group_rotation: Optional[float] = None,
    layer: Optional[str] = None,
    standard_scale: Optional[Literal["var", "obs"]] = None,
    swap_axes: bool = False,
    show_feature_labels: Optional[bool] = None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    figsize: Optional[Tuple[float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    norm: Optional[Normalize] = None,
    **kwds,
):
    """Heatmap of the feature values.

    If `groupby` is given, the heatmap is ordered by the respective group. For
    example, a list of marker genes can be plotted, ordered by clustering. If
    the `groupby` observation annotation is not categorical the observation
    annotation is turned into a categorical by binning the data into the number specified in `num_categories`.

    Args:
        {common_plot_args}
        standard_scale: Whether or not to standardize that dimension between 0 and 1, meaning for each variable or observation,
                        subtract the minimum and divide each by its maximum.
        swap_axes: By default, the x axis contains `var_names` (e.g. genes) and the y axis the `groupby`
                   categories (if any). By setting `swap_axes` then x are the `groupby` categories and y the `var_names`.
        show_feature_labels: By default feature labels are shown when there are 50 or less features. Otherwise the labels are removed.
        {show_save_ax}
        {vminmax}
        **kwds:
            Are passed to :func:`matplotlib.pyplot.imshow`.

    Returns:
        List of :class:`~matplotlib.axes.Axes`
    """
    return sc.pl.heatmap(
        adata=adata,
        var_names=var_names,
        groupby=groupby,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        dendrogram=dendrogram,
        gene_symbols=feature_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        var_group_rotation=var_group_rotation,
        layer=layer,
        standard_scale=standard_scale,
        swap_axes=swap_axes,
        show_gene_labels=show_feature_labels,
        show=show,
        save=save,
        figsize=figsize,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        **kwds,
    )


@_doc_params(
    show_save_ax=doc_show_save_ax,
    common_plot_args=doc_common_plot_args,
    groupby_plots_args=doc_common_groupby_plot_args,
    vminmax=doc_vboundnorm,
)
def dotplot(
    adata: AnnData,
    var_names: Union[_VarNames, Mapping[str, _VarNames]],
    groupby: Union[str, Sequence[str]],
    use_raw: Optional[bool] = None,
    log: bool = False,
    num_categories: int = 7,
    feature_cutoff: float = 0.0,
    mean_only_counts: bool = False,
    cmap: str = "Reds",
    dot_max: Optional[float] = DotPlot.DEFAULT_DOT_MAX,
    dot_min: Optional[float] = DotPlot.DEFAULT_DOT_MIN,
    standard_scale: Optional[Literal["var", "group"]] = None,
    smallest_dot: Optional[float] = DotPlot.DEFAULT_SMALLEST_DOT,
    title: Optional[str] = None,
    colorbar_title: Optional[str] = DotPlot.DEFAULT_COLOR_LEGEND_TITLE,
    size_title: Optional[str] = DotPlot.DEFAULT_SIZE_LEGEND_TITLE,
    figsize: Optional[Tuple[float, float]] = None,
    dendrogram: Union[bool, str] = False,
    feature_symbols: Optional[str] = None,
    var_group_positions: Optional[Sequence[Tuple[int, int]]] = None,
    var_group_labels: Optional[Sequence[str]] = None,
    var_group_rotation: Optional[float] = None,
    layer: Optional[str] = None,
    swap_axes: Optional[bool] = False,
    dot_color_df: Optional[pd.DataFrame] = None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    ax: Optional[_AxesSubplot] = None,
    return_fig: Optional[bool] = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    norm: Optional[Normalize] = None,
    **kwds,
) -> Union[DotPlot, dict, None]:
    """Makes a *dot plot* of the count values of `var_names`.

    For each var_name and each `groupby` category a dot is plotted.
    Each dot represents two values: mean expression within each category
    (visualized by color) and fraction of cells expressing the `var_name` in the
    category (visualized by the size of the dot). If `groupby` is not given,
    the dotplot assumes that all data belongs to a single category.

    .. note::
       A count is used if it is above the specified threshold which is zero by default.

    Args:
        {common_plot_args}
        {groupby_plots_args}
        size_title: Title for the size legend. New line character (\\n) can be used.
        feature_cutoff: Count cutoff that is used for binarizing the counts and
                        determining the fraction of patients having the feature.
                        A feature is only used if its counts are greater than this threshold.
        mean_only_counts: If True, counts are averaged only over the patients having the provided feature.
        dot_max: If none, the maximum dot size is set to the maximum fraction value found
                 (e.g. 0.6). If given, the value should be a number between 0 and 1.
                 All fractions larger than dot_max are clipped to this value.
        dot_min: If none, the minimum dot size is set to 0. If given,
                 the value should be a number between 0 and 1.
                 All fractions smaller than dot_min are clipped to this value.
        smallest_dot: If none, the smallest dot has size 0. All counts with `dot_min` are plotted with this size.
        {show_save_ax}
        {vminmax}
        kwds:
            Are passed to :func:`matplotlib.pyplot.scatter`.

    Returns:
        If `return_fig` is `True`, returns a :class:`~scanpy.pl.DotPlot` object, else if `show` is false, return axes dict
    """
    return sc.pl.dotplot(
        adata=adata,
        var_names=var_names,
        groupby=groupby,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        expression_cutoff=feature_cutoff,
        mean_only_expressed=mean_only_counts,
        cmap=cmap,
        dot_max=dot_max,
        dot_min=dot_min,
        standard_scale=standard_scale,
        smallest_dot=smallest_dot,
        title=title,
        colorbar_title=colorbar_title,
        size_title=size_title,
        figsize=figsize,
        dendrogram=dendrogram,
        gene_symbols=feature_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        var_group_rotation=var_group_rotation,
        layer=layer,
        swap_axes=swap_axes,
        dot_color_df=dot_color_df,
        show=show,
        save=save,
        ax=ax,
        return_fig=return_fig,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        **kwds,
    )


@_doc_params(show_save_ax=doc_show_save_ax, common_plot_args=doc_common_plot_args)
def tracksplot(
    adata: AnnData,
    var_names: Union[_VarNames, Mapping[str, _VarNames]],
    groupby: Union[str, Sequence[str]],
    use_raw: Optional[bool] = None,
    log: bool = False,
    dendrogram: Union[bool, str] = False,
    feature_symbols: Optional[str] = None,
    var_group_positions: Optional[Sequence[Tuple[int, int]]] = None,
    var_group_labels: Optional[Sequence[str]] = None,
    layer: Optional[str] = None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwds,
) -> Optional[Dict[str, List]]:
    """Plots a filled line plot.

    In this type of plot each var_name is plotted as a filled line plot where the
    y values correspond to the var_name values and x is each of the observations. Best results
    are obtained when using raw counts that are not log.
    `groupby` is required to sort and order the values using the respective group and should be a categorical value.

    Args:
        {common_plot_args}
        {show_save_ax}
        **kwds:
            Are passed to :func:`~seaborn.heatmap`.

    Returns:
        A list of :class:`~matplotlib.axes.Axes`.
    """
    return sc.pl.tracksplot(
        adata=adata,
        var_names=var_names,
        groupby=groupby,
        use_raw=use_raw,
        log=log,
        dendrogram=dendrogram,
        gene_symbols=feature_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        layer=layer,
        show=show,
        save=save,
        figsize=figsize,
        **kwds,
    )


def violin(
    adata: AnnData,
    keys: Union[str, Sequence[str]],
    groupby: Optional[str] = None,
    log: bool = False,
    use_raw: Optional[bool] = None,
    stripplot: bool = True,
    jitter: Union[float, bool] = True,
    size: int = 1,
    layer: Optional[str] = None,
    scale: Literal["area", "count", "width"] = "width",
    order: Optional[Sequence[str]] = None,
    multi_panel: Optional[bool] = None,
    xlabel: str = "",
    ylabel: Optional[Union[str, Sequence[str]]] = None,
    rotation: Optional[float] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    **kwds,
):
    """Violin plot.

    Wraps :func:`seaborn.violinplot` for :class:`~anndata.AnnData`.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        keys: Keys for accessing variables of `.var_names` or fields of `.obs`.
        groupby: The key of the observation grouping to consider.
        log: Plot on logarithmic axis.
        use_raw: Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.
        stripplot: Add a stripplot on top of the violin plot. See :func:`~seaborn.stripplot`.
        jitter: Add jitter to the stripplot (only when stripplot is True) See :func:`~seaborn.stripplot`.
        size: Size of the jitter points.
        layer: Name of the AnnData object layer that wants to be plotted. By
               default adata.raw.X is plotted. If `use_raw=False` is set,
               then `adata.X` is plotted. If `layer` is set to a valid layer name,
               then the layer is plotted. `layer` takes precedence over `use_raw`.
        scale: The method used to scale the width of each violin.
               If 'width' (the default), each violin will have the same width.
               If 'area', each violin will have the same area.
               If 'count', a violin’s width corresponds to the number of observations.
        order: Order in which to show the categories.
        multi_panel: Display keys in multiple panels also when `groupby is not None`.
        xlabel: Label of the x axis. Defaults to `groupby` if `rotation` is `None`, otherwise, no label is shown.
        ylabel: Label of the y axis. If `None` and `groupby` is `None`, defaults to `'value'`.
                If `None` and `groubpy` is not `None`, defaults to `keys`.
        rotation: Rotation of xtick labels.
        {show_save_ax}
        **kwds:
            Are passed to :func:`~seaborn.violinplot`.
    Returns:
        A :class:`~matplotlib.axes.Axes` object if `ax` is `None` else `None`.
    """
    return sc.pl.violin(
        adata=adata,
        keys=keys,
        groupby=groupby,
        log=log,
        use_raw=use_raw,
        stripplot=stripplot,
        jitter=jitter,
        size=size,
        layer=layer,
        scale=scale,
        order=order,
        multi_panel=multi_panel,
        xlabel=xlabel,
        ylabel=ylabel,
        rotation=rotation,
        show=show,
        save=save,
        ax=ax,
        **kwds,
    )


@_doc_params(
    show_save_ax=doc_show_save_ax,
    common_plot_args=doc_common_plot_args,
    groupby_plots_args=doc_common_groupby_plot_args,
    vminmax=doc_vboundnorm,
)
def stacked_violin(
    adata: AnnData,
    var_names: Union[_VarNames, Mapping[str, _VarNames]],
    groupby: Union[str, Sequence[str]],
    log: bool = False,
    use_raw: Optional[bool] = None,
    num_categories: int = 7,
    title: Optional[str] = None,
    colorbar_title: Optional[str] = StackedViolin.DEFAULT_COLOR_LEGEND_TITLE,
    figsize: Optional[Tuple[float, float]] = None,
    dendrogram: Union[bool, str] = False,
    gene_symbols: Optional[str] = None,
    var_group_positions: Optional[Sequence[Tuple[int, int]]] = None,
    var_group_labels: Optional[Sequence[str]] = None,
    standard_scale: Optional[Literal["var", "obs"]] = None,
    var_group_rotation: Optional[float] = None,
    layer: Optional[str] = None,
    stripplot: bool = StackedViolin.DEFAULT_STRIPPLOT,
    jitter: Union[float, bool] = StackedViolin.DEFAULT_JITTER,
    size: int = StackedViolin.DEFAULT_JITTER_SIZE,
    scale: Literal["area", "count", "width"] = StackedViolin.DEFAULT_SCALE,
    yticklabels: Optional[bool] = StackedViolin.DEFAULT_PLOT_YTICKLABELS,
    order: Optional[Sequence[str]] = None,
    swap_axes: bool = False,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    return_fig: Optional[bool] = False,
    row_palette: Optional[str] = StackedViolin.DEFAULT_ROW_PALETTE,
    cmap: Optional[str] = StackedViolin.DEFAULT_COLORMAP,
    ax: Optional[_AxesSubplot] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    norm: Optional[Normalize] = None,
    **kwds,
) -> Union[StackedViolin, dict, None]:
    """Stacked violin plots.

    Makes a compact image composed of individual violin plots (from :func:`~seaborn.violinplot`) stacked on top of each other.
    Useful to visualize gene expression per cluster. Wraps :func:`seaborn.violinplot` for :class:`~anndata.AnnData`.

    This function provides a convenient interface to the :class:`~scanpy.pl.StackedViolin` class.
    If you need more flexibility, you should use :class:`~scanpy.pl.StackedViolin` directly.


    Args:
        {common_plot_args}
        {groupby_plots_args}
        stripplot: Add a stripplot on top of the violin plot. See :func:`~seaborn.stripplot`.
        jitter: Add jitter to the stripplot (only when stripplot is True) See :func:`~seaborn.stripplot`.
        size: Size of the jitter points.
        yticklabels: Set to true to view the y tick labels
        order: Order in which to show the categories. Note: if `dendrogram=True`
               the categories order will be given by the dendrogram and `order` will be ignored.
        scale: The method used to scale the width of each violin.
               If 'width' (the default), each violin will have the same width.
               If 'area', each violin will have the same area.
               If 'count', a violin’s width corresponds to the number of observations.
        row_palette: Be default, median values are mapped to the violin color using a
                     color map (see `cmap` argument). Alternatively, a 'row_palette` can
                     be given to color each violin plot row using a different colors.
                     The value should be a valid seaborn or matplotlib palette name (see :func:`~seaborn.color_palette`).
                     Alternatively, a single color name or hex value can be passed, e.g. `'red'` or `'#cc33ff'`.
        {show_save_ax}
        {vminmax}
        kwds:
            Are passed to :func:`~seaborn.violinplot`.

    Returns:
        If `return_fig` is `True`, returns a :class:`~scanpy.pl.StackedViolin` object, else if `show` is false, return axes dict
    """
    return sc.pl.stacked_violin(
        adata=adata,
        var_names=var_names,
        groupby=groupby,
        log=log,
        use_raw=use_raw,
        num_categories=num_categories,
        title=title,
        colorbar_title=colorbar_title,
        figsize=figsize,
        dendrogram=dendrogram,
        gene_symbols=gene_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        standard_scale=standard_scale,
        var_group_rotation=var_group_rotation,
        layer=layer,
        stripplot=stripplot,
        jitter=jitter,
        size=size,
        scale=scale,
        yticklabels=yticklabels,
        order=order,
        swap_axes=swap_axes,
        show=show,
        save=save,
        return_fig=return_fig,
        row_palette=row_palette,
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        **kwds,
    )


@_doc_params(
    show_save_ax=doc_show_save_ax,
    common_plot_args=doc_common_plot_args,
    groupby_plots_args=doc_common_groupby_plot_args,
    vminmax=doc_vboundnorm,
)
def matrixplot(
    adata: AnnData,
    var_names: Union[_VarNames, Mapping[str, _VarNames]],
    groupby: Union[str, Sequence[str]],
    use_raw: Optional[bool] = None,
    log: bool = False,
    num_categories: int = 7,
    figsize: Optional[Tuple[float, float]] = None,
    dendrogram: Union[bool, str] = False,
    title: Optional[str] = None,
    cmap: Optional[str] = MatrixPlot.DEFAULT_COLORMAP,
    colorbar_title: Optional[str] = MatrixPlot.DEFAULT_COLOR_LEGEND_TITLE,
    gene_symbols: Optional[str] = None,
    var_group_positions: Optional[Sequence[Tuple[int, int]]] = None,
    var_group_labels: Optional[Sequence[str]] = None,
    var_group_rotation: Optional[float] = None,
    layer: Optional[str] = None,
    standard_scale: Literal["var", "group"] = None,
    values_df: Optional[pd.DataFrame] = None,
    swap_axes: bool = False,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    ax: Optional[_AxesSubplot] = None,
    return_fig: Optional[bool] = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    norm: Optional[Normalize] = None,
    **kwds,
) -> Union[MatrixPlot, dict, None]:
    """Creates a heatmap of the mean count per group of each var_names.

    This function provides a convenient interface to the :class:`~scanpy.pl.MatrixPlot`
    class. If you need more flexibility, you should use :class:`~scanpy.pl.MatrixPlot` directly.


    Args:
        {common_plot_args}
        {groupby_plots_args}
        {show_save_ax}
        {vminmax}
        kwds:
            Are passed to :func:`matplotlib.pyplot.pcolor`.
    Returns:
        If `return_fig` is `True`, returns a :class:`~scanpy.pl.MatrixPlot` object, else if `show` is false, return axes dict
    """
    return sc.pl.matrixplot(
        adata=adata,
        var_names=var_names,
        groupby=groupby,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        figsize=figsize,
        dendrogram=dendrogram,
        title=title,
        cmap=cmap,
        colorbar_title=colorbar_title,
        gene_symbols=gene_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        var_group_rotation=var_group_rotation,
        layer=layer,
        standard_scale=standard_scale,
        values_df=values_df,
        swap_axes=swap_axes,
        show=show,
        save=save,
        ax=ax,
        return_fig=return_fig,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        **kwds,
    )


@_doc_params(show_save_ax=doc_show_save_ax)
def clustermap(
    adata: AnnData,
    obs_keys: str = None,
    use_raw: Optional[bool] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwds,
):
    """Hierarchically-clustered heatmap.

    Wraps :func:`seaborn.clustermap` for :class:`~anndata.AnnData`.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        obs_keys: Categorical annotation to plot with a different color map. Currently, only a single key is supported.
        use_raw: Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.
        {show_save_ax}
        **kwds:
            Keyword arguments passed to :func:`~seaborn.clustermap`.

    Returns:
        If `show` is `False`, a :class:`~seaborn.ClusterGrid` object (see :func:`~seaborn.clustermap`).
    """
    return sc.pl.clustermap(adata=adata, obs_keys=obs_keys, use_raw=use_raw, show=show, save=save, **kwds)


def ranking(
    adata: AnnData,
    attr: Literal["var", "obs", "uns", "varm", "obsm"],
    keys: Union[str, Sequence[str]],
    dictionary=None,
    indices=None,
    labels=None,
    color="black",
    n_points=30,
    log=False,
    include_lowest=False,
    show=None,
):
    """Plot rankings.

    See, for example, how this is used in pl.pca_ranking.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        attr: The attribute of AnnData that contains the score.
        keys: The scores to look up an array from the attribute of adata.
        dictionary: Optional key dictionary.
        indices: Optional dictionary indices.
        labels: Optional labels.
        color: Optional primary color (default: black).
        n_points: Number of points (default: 30)..
        log: Log object
        include_lowest: Whether to include the lowest points.
        show: Whether to show the plot.

    Returns:
        Returns matplotlib gridspec with access to the axes.
    """
    return sc.pl.ranking(
        adata=adata,
        attr=attr,
        keys=keys,
        dictionary=dictionary,
        indices=indices,
        labels=labels,
        color=color,
        n_points=n_points,
        log=log,
        include_lowest=include_lowest,
        show=show,
    )


@_doc_params(show_save_ax=doc_show_save_ax)
def dendrogram(
    adata: AnnData,
    groupby: str,
    *,
    dendrogram_key: Optional[str] = None,
    orientation: Literal["top", "bottom", "left", "right"] = "top",
    remove_labels: bool = False,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    ax: Optional[Axes] = None,
):
    """Plots a dendrogram of the categories defined in `groupby`.

    See :func:`~ehrapy.tl.dendrogram`.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        groupby: Categorical data column used to create the dendrogram.
        dendrogram_key: Key under with the dendrogram information was stored.
                        By default the dendrogram information is stored under `.uns[f'dendrogram_{{groupby}}']`.
        orientation: Origin of the tree. Will grow into the opposite direction.
        remove_labels: Don’t draw labels. Used e.g. by :func:`scanpy.pl.matrixplot` to annotate matrix columns/rows.
        {show_save_ax}

    Returns:
        :class:`matplotlib.axes.Axes`
    """
    return sc.pl.dendrogram(
        adata=adata,
        groupby=groupby,
        dendrogram_key=dendrogram_key,
        orientation=orientation,
        remove_labels=remove_labels,
        show=show,
        save=save,
        ax=ax,
    )
