from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Collection, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from cycler import Cycler
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure
from scanpy.plotting import DotPlot, MatrixPlot, StackedViolin
from scanpy.plotting._tools.scatterplots import _wraps_plot_scatter
from scanpy.plotting._utils import _AxesSubplot

from ehrapy.util._doc_util import (
    _doc_params,
    doc_adata_color_etc,
    doc_common_groupby_plot_args,
    doc_common_plot_args,
    doc_edges_arrows,
    doc_panels,
    doc_scatter_basic,
    doc_scatter_embedding,
    doc_scatter_spatial,
    doc_show_save_ax,
    doc_vbound_percentile,
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
):  # pragma: no cover
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

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.pl.scatter(adata, x='age', y='icu_los_day', color='icu_los_day')

    Preview:
        .. image:: /_images/docstring_previews/scatter.png
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
):  # pragma: no cover
    """Heatmap of the feature values.

    If `groupby` is given, the heatmap is ordered by the respective group.
    If the `groupby` observation annotation is not categorical the observation
    annotation is turned into a categorical by binning the data into the number specified in `num_categories`.

    Args:
        {common_plot_args}
        standard_scale: Whether or not to standardize that dimension between 0 and 1, meaning for each variable or observation,
                        subtract the minimum and divide each by its maximum.
        swap_axes: By default, the x axis contains `var_names` (e.g. features) and the y axis the `groupby`
                   categories (if any). By setting `swap_axes` then x are the `groupby` categories and y the `var_names`.
        show_feature_labels: By default feature labels are shown when there are 50 or less features. Otherwise the labels are removed.
        {show_save_ax}
        {vminmax}
        **kwds:
            Are passed to :func:`matplotlib.pyplot.imshow`.

    Returns:
        List of :class:`~matplotlib.axes.Axes`

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.pl.heatmap(adata, var_names=['map_1st', 'hr_1st', 'temp_1st', 'spo2_1st',
                'abg_count', 'wbc_first', 'hgb_first', 'platelet_first', 'sodium_first',
                'potassium_first', 'tco2_first', 'chloride_first', 'bun_first',
                'creatinine_first', 'po2_first', 'pco2_first', 'iv_day_1'], groupby="leiden_0_5")

    Preview:
        .. image:: /_images/docstring_previews/heatmap.png
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
) -> Union[DotPlot, dict, None]:  # pragma: no cover
    """Makes a *dot plot* of the count values of `var_names`.

    For each var_name and each `groupby` category a dot is plotted.
    Each dot represents two values: mean expression within each category
    (visualized by color) and fraction of observations expressing the `var_name` in the
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

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.pl.dotplot(adata,
                          var_names=['age', 'gender_num', 'weight_first', 'bmi',
                          'wbc_first', 'hgb_first', 'platelet_first', 'sodium_first',
                          'potassium_first', 'tco2_first', 'chloride_first', 'bun_first',
                          'creatinine_first', 'po2_first', 'pco2_first'],
                           groupby="leiden_0_5")

    Preview:
        .. image:: /_images/docstring_previews/dotplot.png
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
) -> Optional[Dict[str, List]]:  # pragma: no cover
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

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.pl.tracksplot(adata, var_names=['age', 'gender_num', 'weight_first', 'bmi',
                'sapsi_first', 'sofa_first', 'service_num', 'day_icu_intime_num',
                'hour_icu_intime'], groupby="leiden_0_5")

    Preview:
        .. image:: /_images/docstring_previews/tracksplot.png
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
):  # pragma: no cover
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

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.pl.violin(adata, keys=['age'], groupby="leiden_0_5")

    Preview:
        .. image:: /_images/docstring_previews/violin.png
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
) -> Union[StackedViolin, dict, None]:  # pragma: no cover
    """Stacked violin plots.

    Makes a compact image composed of individual violin plots (from :func:`~seaborn.violinplot`) stacked on top of each other.

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

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.pl.stacked_violin(adata, var_names=['icu_los_day',
                'hospital_los_day', 'age', 'gender_num', 'weight_first', 'bmi',
                'sapsi_first', 'sofa_first', 'service_num', 'day_icu_intime_num',
                'hour_icu_intime'], groupby="leiden_0_5")

    Preview:
        .. image:: /_images/docstring_previews/stacked_violin.png
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
) -> Union[MatrixPlot, dict, None]:  # pragma: no cover
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

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.pl.matrixplot(adata, var_names=[
                'abg_count', 'wbc_first', 'hgb_first', 'platelet_first', 'sodium_first',
                'potassium_first', 'tco2_first', 'chloride_first', 'bun_first',
                'creatinine_first', 'po2_first', 'pco2_first', 'iv_day_1'], groupby="leiden_0_5")

    Preview:
        .. image:: /_images/docstring_previews/matrixplot.png
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
):  # pragma: no cover
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

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.pl.clustermap(adata)

    Preview:
        .. image:: /_images/docstring_previews/clustermap.png
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
):  # pragma: no cover
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
        log: Whether logarithmic scale should be used.
        include_lowest: Whether to include the lowest points.
        show: Whether to show the plot.

    Returns:
        Returns matplotlib gridspec with access to the axes.

    Example:
        .. code-block:: python

            import ehrapy as ep

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
):  # pragma: no cover
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

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.pl.dendrogram(adata, groupby="leiden_0_5")

    Preview:
        .. image:: /_images/docstring_previews/dendrogram.png
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


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def pca(
    adata,
    *,
    annotate_var_explained: bool = False,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
) -> Union[Axes, List[Axes], None]:  # pragma: no cover
    """Scatter plot in PCA coordinates.

    Use the parameter `annotate_var_explained` to annotate the explained variance.

    Args:
        {adata_color_etc}
        annotate_var_explained: Whether to only annotate the explained variables.
        {scatter_bulk}
        {show_save_ax}

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.pl.dendrogram(adata, groupby="leiden_0_5")

    Preview:
        .. image:: /_images/docstring_previews/pca.png
    """
    return sc.pl.pca(
        adata=adata,
        annotate_var_explained=annotate_var_explained,
        show=show,
        return_fig=return_fig,
        save=save,
        **kwargs,
    )


def pca_loadings(
    adata: AnnData,
    components: Union[str, Sequence[int], None] = None,
    include_lowest: bool = True,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
):  # pragma: no cover
    """Rank features according to contributions to PCs.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        components: For example, ``'1,2,3'`` means ``[1, 2, 3]``, first, second, third principal component.
        include_lowest: Whether to show the features with both highest and lowest loadings.
        show: Show the plot, do not return axis.
        save: If `True` or a `str`, save the figure. A string is appended to the default filename.
              Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.pp.pca(adata)
            ep.pl.pca_loadings(adata, components='1,2,3')

    Preview:
        .. image:: /_images/docstring_previews/pca_loadings.png
    """
    return sc.pl.pca_loadings(adata=adata, components=components, include_lowest=include_lowest, show=show, save=save)


def pca_variance_ratio(
    adata: AnnData,
    n_pcs: int = 30,
    log: bool = False,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
):  # pragma: no cover
    """Plot the variance ratio.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        n_pcs: Number of PCs to show.
        log: Plot on logarithmic scale..
        show: Show the plot, do not return axis.
        save: If `True` or a `str`, save the figure.
              A string is appended to the default filename.
              Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.pp.pca(adata)
            ep.pl.pca_variance_ratio(adata, n_pcs=8)

    Preview:
        .. image:: /_images/docstring_previews/pca_variance_ratio.png
    """
    return sc.pl.pca_variance_ratio(adata=adata, n_pcs=n_pcs, log=log, show=show, save=save)


@_doc_params(scatter_bulk=doc_scatter_embedding, show_save_ax=doc_show_save_ax)
def pca_overview(adata: AnnData, **params):  # pragma: no cover
    """Plot PCA results.

    The parameters are the ones of the scatter plot. Call pca_ranking separately
    if you want to change the default settings.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        {scatter_bulk}
        {show_save_ax}
        **params: Scatterplot parameters

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.pp.pca(adata)
            ep.pl.pca_overview(adata, components='1,2,3', color="service_unit")

    Preview:
        .. image:: /_images/docstring_previews/pca_overview_1.png

        .. image:: /_images/docstring_previews/pca_overview_2.png

        .. image:: /_images/docstring_previews/pca_overview_3.png
    """
    return sc.pl.pca_overview(adata=adata, **params)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def tsne(adata, **kwargs) -> Union[Axes, List[Axes], None]:  # pragma: no cover
    """Scatter plot in tSNE basis.

    Args:
        {adata_color_etc}
        {edges_arrows}
        {scatter_bulk}
        {show_save_ax}

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.tsne(adata)
            ep.pl.tsne(adata)

        .. image:: /_images/docstring_previews/tsne_1.png

        .. code-block:: python

            ep.pl.tsne(adata, color=["day_icu_intime", "service_unit"], wspace=0.5, title=["Day of ICU admission", "Service unit"])

        .. image:: /_images/docstring_previews/tsne_2.png

        .. code-block:: python

            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.pl.tsne(adata, color=["leiden_0_5"], title="Leiden 0.5")

        .. image:: /_images/docstring_previews/tsne_3.png
    """
    return sc.pl.tsne(adata=adata, **kwargs)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def umap(adata, **kwargs) -> Union[Axes, List[Axes], None]:  # pragma: no cover
    """Scatter plot in UMAP basis.

    Args:
        {adata_color_etc}
        {edges_arrows}
        {scatter_bulk}
        {show_save_ax}

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.umap(adata)
            ep.pl.umap(adata)

        .. image:: /_images/docstring_previews/umap_1.png

        .. code-block:: python

            ep.pl.umap(adata, color=["day_icu_intime", "service_unit"], wspace=0.5, title=["Day of ICU admission", "Service unit"])

        .. image:: /_images/docstring_previews/umap_2.png

        .. code-block:: python

            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.pl.umap(adata, color=["leiden_0_5"], title="Leiden 0.5")

        .. image:: /_images/docstring_previews/umap_3.png
    """
    return sc.pl.umap(adata=adata, **kwargs)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def diffmap(adata, **kwargs) -> Union[Axes, List[Axes], None]:  # pragma: no cover
    """Scatter plot in Diffusion Map basis.

    Args:
        {adata_color_etc}
        {scatter_bulk}
        {show_save_ax}

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.diffmap(adata)
            ep.pl.diffmap(adata, color='day_icu_intime')

    Preview:
        .. image:: /_images/docstring_previews/diffmap.png
    """
    return sc.pl.diffmap(adata=adata, **kwargs)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def draw_graph(
    adata: AnnData, *, layout: Optional[_IGraphLayout] = None, **kwargs
) -> Union[Axes, List[Axes], None]:  # pragma: no cover
    """Scatter plot in graph-drawing basis.

    Args:
        {adata_color_etc}
        layout: One of the :func:`~scanpy.tl.draw_graph` layouts. By default, the last computed layout is used.
        {edges_arrows}
        {scatter_bulk}
        {show_save_ax}

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.tl.paga(adata, groups="leiden_0_5")
            ep.pl.paga(
                adata,
                color=["leiden_0_5", "day_28_flg"],
                cmap=ep.pl.Colormaps.grey_red.value,
                title=["Leiden 0.5", "Died in less than 28 days"],
            )
            ep.tl.draw_graph(adata, init_pos="paga")
            ep.pl.draw_graph(adata, color=["leiden_0_5", "icu_exp_flg"], legend_loc="on data")

    Preview:
        .. image:: /_images/docstring_previews/draw_graph_1.png

        .. image:: /_images/docstring_previews/draw_graph_2.png
    """
    return sc.pl.draw_graph(adata=adata, layout=layout, **kwargs)


class Empty(Enum):
    token = 0


_empty = Empty.token


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    scatter_spatial=doc_scatter_spatial,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def spatial(
    adata,
    *,
    basis: str = "spatial",
    img: Union[np.ndarray, None] = None,
    img_key: Union[str, None, Empty] = _empty,
    library_id: Union[str, Empty] = _empty,
    crop_coord: Tuple[int, int, int, int] = None,
    alpha_img: float = 1.0,
    bw: Optional[bool] = False,
    size: float = 1.0,
    scale_factor: Optional[float] = None,
    spot_size: Optional[float] = None,
    na_color: Optional[ColorLike] = None,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
) -> Union[Axes, List[Axes], None]:  # pragma: no cover
    """Scatter plot in spatial coordinates.

    This function allows overlaying data on top of images.
    Use the parameter `img_key` to see the image in the background
    And the parameter `library_id` to select the image.
    By default, `'hires'` and `'lowres'` are attempted.
    Use `crop_coord`, `alpha_img`, and `bw` to control how it is displayed.
    Use `size` to scale the size of the Visium spots plotted on top.
    As this function is designed to for imaging data, there are two key assumptions
    about how coordinates are handled:

    1. The origin (e.g `(0, 0)`) is at the top left – as is common convention
    with image data.
    2. Coordinates are in the pixel space of the source image, so an equal
    aspect ratio is assumed.
    If your anndata object has a `"spatial"` entry in `.uns`, the `img_key`
    and `library_id` parameters to find values for `img`, `scale_factor`,
    and `spot_size` arguments. Alternatively, these values be passed directly.

    Args:
        {adata_color_etc}
        {scatter_spatial}
        {scatter_bulk}
        {show_save_ax}

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.
    """
    return sc.pl.spatial(
        adata=adata,
        basis=basis,
        img=img,
        img_key=img_key,
        library_id=library_id,
        crop_coord=crop_coord,
        alpha_img=alpha_img,
        bw=bw,
        size=size,
        scale_factor=scale_factor,
        spot_size=spot_size,
        na_color=na_color,
        show=show,
        return_fig=return_fig,
        save=save,
        **kwargs,
    )


@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def embedding(
    adata: AnnData,
    basis: str,
    *,
    color: Union[str, Sequence[str], None] = None,
    feature_symbols: Optional[str] = None,
    use_raw: Optional[bool] = None,
    sort_order: bool = True,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: Union[str, Sequence[float], Sequence[str]] = "grey",
    neighbors_key: Optional[str] = None,
    arrows: bool = False,
    arrows_kwds: Optional[Mapping[str, Any]] = None,
    groups: Optional[str] = None,
    components: Union[str, Sequence[str]] = None,
    layer: Optional[str] = None,
    projection: Literal["2d", "3d"] = "2d",
    scale_factor: Optional[float] = None,
    color_map: Union[Colormap, str, None] = None,
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    size: Union[float, Sequence[float], None] = None,
    frameon: Optional[bool] = None,
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = "bold",
    legend_loc: str = "right margin",
    legend_fontoutline: Optional[int] = None,
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ("black", "white"),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: Optional[float] = None,
    title: Union[str, Sequence[str], None] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs,
) -> Union[Figure, Axes, None]:  # pragma: no cover
    """Scatter plot for user specified embedding basis (e.g. umap, pca, etc)

    Args:
        basis:
        {adata_color_etc}
        {edges_arrows}
        {scatter_bulk}
        {show_save_ax}

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.umap(adata)
            ep.pl.embedding(adata, 'X_umap', color='icu_exp_flg')

    Preview:
        .. image:: /_images/docstring_previews/embedding.png
    """
    return sc.pl.embedding(
        adata=adata,
        basis=basis,
        color=color,
        gene_symbols=feature_symbols,
        use_raw=use_raw,
        sort_order=sort_order,
        edges=edges,
        edges_width=edges_width,
        edges_color=edges_color,
        neighbors_key=neighbors_key,
        arrows=arrows,
        arrows_kwds=arrows_kwds,
        groups=groups,
        components=components,
        layer=layer,
        projection=projection,
        scale_factor=scale_factor,
        color_map=color_map,
        cmap=cmap,
        palette=palette,
        na_color=na_color,
        na_in_legend=na_in_legend,
        size=size,
        frameon=frameon,
        legend_fontsize=legend_fontsize,
        legend_fontweight=legend_fontweight,
        legend_loc=legend_loc,
        legend_fontoutline=legend_fontoutline,
        vmax=vmax,
        vmin=vmin,
        vcenter=vcenter,
        norm=norm,
        add_outline=add_outline,
        outline_width=outline_width,
        outline_color=outline_color,
        ncols=ncols,
        hspace=hspace,
        wspace=wspace,
        title=title,
        show=show,
        save=save,
        ax=ax,
        return_fig=return_fig,
        **kwargs,
    )


@_doc_params(vminmax=doc_vbound_percentile, panels=doc_panels, show_save_ax=doc_show_save_ax)
def embedding_density(
    adata: AnnData,
    basis: str = "umap",  # was positional before 1.4.5
    key: Optional[str] = None,  # was positional before 1.4.5
    groupby: Optional[str] = None,
    group: Optional[Union[str, List[str], None]] = "all",
    color_map: Union[Colormap, str] = "YlOrRd",
    bg_dotsize: Optional[int] = 80,
    fg_dotsize: Optional[int] = 180,
    vmax: Optional[int] = 1,
    vmin: Optional[int] = 0,
    vcenter: Optional[int] = None,
    norm: Optional[Normalize] = None,
    ncols: Optional[int] = 4,
    hspace: Optional[float] = 0.25,
    wspace: Optional[None] = None,
    title: str = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs,
) -> Union[Figure, Axes, None]:  # pragma: no cover
    """Plot the density of observations in an embedding (per condition).

    Plots the gaussian kernel density estimates (over condition) from the `sc.tl.embedding_density()` output.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        basis: The embedding over which the density was calculated.
               This embedded representation should be found in `adata.obsm['X_[basis]']``.
        key: Name of the `.obs` covariate that contains the density estimates. Alternatively, pass `groupby`.
        groupby: Name of the condition used in `tl.embedding_density`. Alternatively, pass `key`.
        group: The category in the categorical observation annotation to be plotted.
               If all categories are to be plotted use group='all' (default), If multiple categories
               want to be plotted use a list (e.g.: ['G1', 'S']. If the overall density wants to be ploted set group to 'None'.
        color_map: Matplolib color map to use for density plotting.
        bg_dotsize: Dot size for background data points not in the `group`.
        fg_dotsize: Dot size for foreground data points in the `group`.
        vmin: The value representing the lower limit of the color scale. Values smaller than vmin are plotted
              with the same color as vmin. vmin can be a number, a string, a function or `None`. If
              vmin is a string and has the format `pN`, this is interpreted as a vmin=percentile(N).
              For example vmin='p1.5' is interpreted as the 1.5 percentile. If vmin is function, then
              vmin is interpreted as the return value of the function over the list of values to plot.
              For example to set vmin tp the mean of the values to plot, `def my_vmin(values): return
              np.mean(values)` and then set `vmin=my_vmin`. If vmin is None (default) an automatic
              minimum value is used as defined by matplotlib `scatter` function. When making multiple
              plots, vmin can be a list of values, one for each plot. For example `vmin=[0.1, 'p1', None, my_vmin]`
        vmax: The value representing the upper limit of the color scale. The format is the same as for `vmin`.
        vcenter: The value representing the center of the color scale. Useful for diverging colormaps.
                 The format is the same as for `vmin`.
                 Example: sc.pl.umap(adata, color='TREM2', vcenter='p50', cmap='RdBu_r')
        ncols: Number of panels per row.
        wspace: Adjust the width of the space between multiple panels.
        hspace: Adjust the height of the space between multiple panels.
        return_fig: Return the matplotlib figure.\
        {show_save_ax}

    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.umap(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.tl.embedding_density(adata, groupby='leiden_0_5', key_added='icu_exp_flg')
            ep.pl.embedding_density(adata, key='icu_exp_flg')

    Preview:
        .. image:: /_images/docstring_previews/embedding_density.png
    """
    return sc.pl.embedding_density(
        adata=adata,
        basis=basis,
        key=key,
        groupby=groupby,
        group=group,
        color_map=color_map,
        bg_dotsize=bg_dotsize,
        fg_dotsize=fg_dotsize,
        vmax=vmax,
        vmin=vmin,
        vcenter=vcenter,
        norm=norm,
        ncols=ncols,
        hspace=hspace,
        wspace=wspace,
        title=title,
        show=show,
        save=save,
        ax=ax,
        return_fig=return_fig,
        **kwargs,
    )


def dpt_groups_pseudotime(
    adata: AnnData,
    color_map: Union[str, Colormap, None] = None,
    palette: Union[Sequence[str], Cycler, None] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
):  # pragma: no cover
    """Plot groups and pseudotime.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        color_map: Matplotlib Colormap
        palette: Matplotlib color Palette
        show: Whether to show the plot.
        save: Whether to save the plot or a path to save the plot.

    Example:
        .. code-block:: python

            import ehrapy as ep

            import numpy as np
            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata, method='gauss')
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.tl.diffmap(adata, n_comps=10)
            adata.uns['iroot'] = np.flatnonzero(adata.obs['leiden_0_5'] == '0')[0]
            ep.tl.dpt(adata, n_branchings=3)
            ep.pl.dpt_groups_pseudotime(adata)

    Preview:
        .. image:: /_images/docstring_previews/dpt_groups_pseudotime.png
    """
    sc.pl.dpt_groups_pseudotime(adata=adata, color_map=color_map, palette=palette, show=show, save=save)


def dpt_timeseries(
    adata: AnnData,
    color_map: Union[str, Colormap] = None,
    as_heatmap: bool = True,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
):  # pragma: no cover
    """Heatmap of pseudotime series.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        color_map: Matplotlib Colormap
        as_heatmap: Whether to render the plot a heatmap
        show: Whether to show the plot.
        save: Whether to save the plot or a path to save the plot.

    Example:
        .. code-block:: python

            import ehrapy as ep

            import numpy as np
            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata, method='gauss')
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.tl.diffmap(adata, n_comps=10)
            adata.uns['iroot'] = np.flatnonzero(adata.obs['leiden_0_5'] == '0')[0]
            ep.tl.dpt(adata, n_branchings=3)
            ep.pl.dpt_timeseries(adata)

    Preview:
        .. image:: /_images/docstring_previews/dpt_timeseries.png
    """
    sc.pl.dpt_timeseries(adata=adata, color_map=color_map, show=show, save=save, as_heatmap=as_heatmap)


def paga(
    adata: AnnData,
    threshold: Optional[float] = None,
    color: Optional[Union[str, Mapping[Union[str, int], Mapping[Any, float]]]] = None,
    layout: Optional[_IGraphLayout] = None,
    layout_kwds: Mapping[str, Any] = MappingProxyType({}),
    init_pos: Optional[np.ndarray] = None,
    root: Union[int, str, Sequence[int], None] = 0,
    labels: Union[str, Sequence[str], Mapping[str, str], None] = None,
    single_component: bool = False,
    solid_edges: str = "connectivities",
    dashed_edges: Optional[str] = None,
    transitions: Optional[str] = None,
    fontsize: Optional[int] = None,
    fontweight: str = "bold",
    fontoutline: Optional[int] = None,
    text_kwds: Mapping[str, Any] = MappingProxyType({}),
    node_size_scale: float = 1.0,
    node_size_power: float = 0.5,
    edge_width_scale: float = 1.0,
    min_edge_width: Optional[float] = None,
    max_edge_width: Optional[float] = None,
    arrowsize: int = 30,
    title: Optional[str] = None,
    left_margin: float = 0.01,
    random_state: Optional[int] = 0,
    pos: Union[np.ndarray, str, Path, None] = None,
    normalize_to_color: bool = False,
    cmap: Union[str, Colormap] = None,
    cax: Optional[Axes] = None,
    cb_kwds: Mapping[str, Any] = MappingProxyType({}),
    frameon: Optional[bool] = None,
    add_pos: bool = True,
    export_to_gexf: bool = False,
    use_raw: bool = True,
    plot: bool = True,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
) -> Union[Axes, List[Axes], None]:  # pragma: no cover
    """Plot the PAGA graph through thresholding low-connectivity edges.

    Compute a coarse-grained layout of the data. Reuse this by passing
    `init_pos='paga'` to :func:`~scanpy.tl.umap` or
    :func:`~scanpy.tl.draw_graph` and obtain embeddings with more meaningful
    global topology [Wolf19]_.
    This uses ForceAtlas2 or igraph's layout algorithms for most layouts [Csardi06]_.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        threshold: Do not draw edges for weights below this threshold. Set to 0 if you want
                   all edges. Discarding low-connectivity edges helps in getting a much clearer picture of the graph.
        color: Feature name or `obs` annotation defining the node colors.
               Also plots the degree of the abstracted graph when
               passing {`'degree_dashed'`, `'degree_solid'`}.
               Can be also used to visualize pie chart at each node in the following form:
               `{<group name or index>: {<color>: <fraction>, ...}, ...}`. If the fractions
               do not sum to 1, a new category called `'rest'` colored grey will be created.
        layout: The node labels. If `None`, this defaults to the group labels stored in
                the categorical for which :func:`~scanpy.tl.paga` has been computed.
        layout_kwds: Keywords for the layout.
        init_pos: Two-column array storing the x and y coordinates for initializing the layout.
        root: If choosing a tree layout, this is the index of the root node or a list
              of root node indices. If this is a non-empty vector then the supplied
              node IDs are used as the roots of the trees (or a single tree if the
              graph is connected). If this is `None` or an empty list, the root
              vertices are automatically calculated based on topological sorting.
        labels: The node labels. If `None`, this defaults to the group labels stored in
                the categorical for which :func:`~scanpy.tl.paga` has been computed.
        single_component: Restrict to largest connected component.
        solid_edges: Key for `.uns['paga']` that specifies the matrix that stores the edges to be drawn solid black.
        dashed_edges: Key for `.uns['paga']` that specifies the matrix that stores the edges
                      to be drawn dashed grey. If `None`, no dashed edges are drawn.
        transitions: Key for `.uns['paga']` that specifies the matrix that stores the
                     arrows, for instance `'transitions_confidence'`.
        fontsize: Font size for node labels.
        fontweight: Weight of the font.
        fontoutline: Width of the white outline around fonts.
        text_kwds: Keywords for :meth:`~matplotlib.axes.Axes.text`.
        node_size_scale: Increase or decrease the size of the nodes.
        node_size_power: The power with which groups sizes influence the radius of the nodes.
        edge_width_scale: Edge with scale in units of `rcParams['lines.linewidth']`.
        min_edge_width: Min width of solid edges.
        max_edge_width: Max width of solid and dashed edges.
        arrowsize: For directed graphs, choose the size of the arrow head head's length and width.
                   See :py:class: `matplotlib.patches.FancyArrowPatch` for attribute `mutation_scale` for more info.
        title: Provide a title.
        left_margin: Margin to the left of the plot.
        random_state: For layouts with random initialization like `'fr'`, change this to use
                      different intial states for the optimization. If `None`, the initial state is not reproducible.
        pos: Two-column array-like storing the x and y coordinates for drawing.
             Otherwise, path to a `.gdf` file that has been exported from Gephi or
             a similar graph visualization software.
        normalize_to_color: Whether to normalize categorical plots to `color` or the underlying grouping.
        cmap: The Matplotlib color map.
        cax: A matplotlib axes object for a potential colorbar.
        cb_kwds: Keyword arguments for :class:`~matplotlib.colorbar.ColorbarBase`, for instance, `ticks`.
        frameon: Draw a frame around the PAGA graph.
        add_pos: Add the positions to `adata.uns['paga']`.
        export_to_gexf: Export to gexf format to be read by graph visualization programs such as Gephi.
        use_raw: Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.
        plot: If `False`, do not create the figure, simply compute the layout.
        ax: Matplotlib Axis object.
        show: Whether to show the plot.
        save: Whether or where to save the plot.

    Returns:
        A :class:`~matplotlib.axes.Axes` object, if `ax` is `None`, else `None`.
        If `return_data`, return the timeseries data in addition to an axes.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.tl.paga(adata, groups="leiden_0_5")
            ep.pl.paga(
                adata,
                color=["leiden_0_5", "day_28_flg"],
                cmap=ep.pl.Colormaps.grey_red.value,
                title=["Leiden 0.5", "Died in less than 28 days"],
            )

    Preview:
        .. image:: /_images/docstring_previews/paga.png
    """
    return sc.pl.paga(
        adata=adata,
        threshold=threshold,
        color=color,
        layout=layout,
        layout_kwds=layout_kwds,
        init_pos=init_pos,
        root=root,
        labels=labels,
        single_component=single_component,
        solid_edges=solid_edges,
        dashed_edges=dashed_edges,
        transitions=transitions,
        fontsize=fontsize,
        fontweight=fontweight,
        fontoutline=fontoutline,
        text_kwds=text_kwds,
        node_size_scale=node_size_scale,
        node_size_power=node_size_power,
        edge_width_scale=edge_width_scale,
        min_edge_width=min_edge_width,
        max_edge_width=max_edge_width,
        arrowsize=arrowsize,
        title=title,
        left_margin=left_margin,
        random_state=random_state,
        pos=pos,
        normalize_to_color=normalize_to_color,
        cmap=cmap,
        cax=cax,
        cb_kwds=cb_kwds,
        frameon=frameon,
        add_pos=add_pos,
        export_to_gexf=export_to_gexf,
        use_raw=use_raw,
        plot=plot,
        show=show,
        save=save,
        ax=ax,
    )


def paga_path(
    adata: AnnData,
    nodes: Sequence[Union[str, int]],
    keys: Sequence[str],
    use_raw: bool = True,
    annotations: Sequence[str] = ("dpt_pseudotime",),
    color_map: Union[str, Colormap, None] = None,
    color_maps_annotations: Mapping[str, Union[str, Colormap]] = MappingProxyType(dict(dpt_pseudotime="Greys")),
    palette_groups: Optional[Sequence[str]] = None,
    n_avg: int = 1,
    groups_key: Optional[str] = None,
    xlim: Tuple[Optional[int], Optional[int]] = (None, None),
    title: Optional[str] = None,
    left_margin=None,
    ytick_fontsize: Optional[int] = None,
    title_fontsize: Optional[int] = None,
    show_node_names: bool = True,
    show_yticks: bool = True,
    show_colorbar: bool = True,
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight, None] = None,
    normalize_to_zero_one: bool = False,
    as_heatmap: bool = True,
    return_data: bool = False,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
) -> Optional[Axes]:  # pragma: no cover
    """Feature changes along paths in the abstracted graph.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        nodes: A path through nodes of the abstracted graph, that is, names or indices
               (within `.categories`) of groups that have been used to run PAGA.
        keys: Either variables in `adata.var_names` or annotations in `adata.obs`. They are plotted using `color_map`.
        use_raw: Use `adata.raw` for retrieving feature values if it has been set.
        annotations: Plot these keys with `color_maps_annotations`. Need to be keys for `adata.obs`.
        color_map: Matplotlib colormap.
        color_maps_annotations: Color maps for plotting the annotations. Keys of the dictionary must appear in `annotations`.
        palette_groups: Usually, use the same `sc.pl.palettes...` as used for coloring the abstracted graph.
        n_avg: Number of data points to include in computation of running average.
        groups_key: Key of the grouping used to run PAGA. If `None`, defaults to `adata.uns['paga']['groups']`.
        xlim: Matplotlib x limit.
        title: Plot title.
        left_margin: Margin to the left of the plot.
        ytick_fontsize: Matplotlib ytick fontsize.
        title_fontsize: Font size of the title.
        show_node_names: Whether to plot the node names on the nodes bar.
        show_yticks: Whether to show the y axis ticks.
        show_colorbar: Whether to show the color bar.
        legend_fontsize: Font size of the legend.
        legend_fontweight: Font weight of the legend.
        normalize_to_zero_one: Shift and scale the running average to [0, 1] per feature.
        as_heatmap: Whether to display the plot as heatmap.
        return_data: Whether to return the timeseries data in addition to the axes if `True`.
        ax: Matplotlib Axis object.
        show: Whether to show the plot.
        save: Whether or where to save the plot.

    Returns:
        A :class:`~matplotlib.axes.Axes` object, if `ax` is `None`, else `None`.
        If `return_data`, return the timeseries data in addition to an axes.
    """
    return sc.pl.paga_path(
        adata=adata,
        nodes=nodes,
        keys=keys,
        use_raw=use_raw,
        annotations=annotations,
        color_map=color_map,
        color_maps_annotations=color_maps_annotations,
        palette_groups=palette_groups,
        n_avg=n_avg,
        groups_key=groups_key,
        xlim=xlim,
        title=title,
        left_margin=left_margin,
        ytick_fontsize=ytick_fontsize,
        title_fontsize=title_fontsize,
        show_node_names=show_node_names,
        show_yticks=show_yticks,
        show_colorbar=show_colorbar,
        legend_fontsize=legend_fontsize,
        legend_fontweight=legend_fontweight,
        normalize_to_zero_one=normalize_to_zero_one,
        as_heatmap=as_heatmap,
        return_data=return_data,
        show=show,
        save=save,
        ax=ax,
    )


def paga_compare(
    adata: AnnData,
    basis=None,
    edges=False,
    color=None,
    alpha=None,
    groups=None,
    components=None,
    projection: Literal["2d", "3d"] = "2d",
    legend_loc="on data",
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = "bold",
    legend_fontoutline=None,
    color_map=None,
    palette=None,
    frameon=False,
    size=None,
    title=None,
    right_margin=None,
    left_margin=0.05,
    show=None,
    save=None,
    title_graph=None,
    groups_graph=None,
    *,
    pos=None,
    **paga_graph_params,
) -> Optional[Axes]:  # pragma: no cover
    """Scatter and PAGA graph side-by-side.

    Consists in a scatter plot and the abstracted graph. See :func:`~ehrapy.pl.paga` for all related parameters.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        basis: String that denotes a plotting tool that computed coordinates.
        edges: Whether to display edges.
        color: Keys for annotations of observations/patients or features, or a hex color specification, e.g.,
               `'ann1'`, `'#fe57a1'`, or `['ann1', 'ann2']`.
        alpha: Alpha value for the image
        groups: Key of the grouping used to run PAGA. If `None`, defaults to `adata.uns['paga']['groups']`.
        components: For example, ``'1,2,3'`` means ``[1, 2, 3]``, first, second, third principal component.
        projection: One of '2d' or '3d'
        legend_loc: Location of the legend.
        legend_fontsize: Font size of the legend.
        legend_fontweight: Font weight of the legend.
        legend_fontoutline: Font outline of the legend.
        color_map: Matplotlib color map.
        palette: Matplotlib color palette.
        frameon: Whether to display the labels frameon.
        size: Size of the plot.
        title: Title of the plot.
        right_margin: Margin to the right of the plot.
        left_margin: Margin to the left of the plot.
        show: Whether to show the plot.
        save: Whether or where to save the plot.
        title_graph: The title of the graph.
        groups_graph: Graph labels.
        pos: Position of the plot.
        **paga_graph_params: Keywords for :func:`~ehrapy.pl.paga` and keywords for :func:`~ehrapy.pl.scatter`.

    Returns:
        Matplotlib axes.
    """
    return sc.pl.paga_compare(
        adata=adata,
        basis=basis,
        edges=edges,
        color=color,
        alpha=alpha,
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
        size=size,
        title=title,
        right_margin=right_margin,
        left_margin=left_margin,
        show=show,
        save=save,
        title_graph=title_graph,
        groups_graph=groups_graph,
        pos=pos**paga_graph_params,
    )


@_doc_params(show_save_ax=doc_show_save_ax)
def rank_features_groups(
    adata: AnnData,
    groups: Union[str, Sequence[str]] = None,
    n_features: int = 20,
    feature_symbols: Optional[str] = None,
    key: Optional[str] = "rank_features_groups",
    fontsize: int = 8,
    ncols: int = 4,
    share_y: bool = True,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
    ax: Optional[Axes] = None,
    **kwds,
):  # pragma: no cover
    """Plot ranking of features.

    Args:
        adata: Annotated data matrix.
        groups: The groups for which to show the feature ranking.
        n_features: The number of features to plot.
        feature_symbols: Key for field in `.var` that stores feature symbols if you do not want to use `.var_names`.
        key: The key of the calculated feature group rankings (default: 'rank_features_groups').
        fontsize: Fontsize for feature names.
        ncols: Number of panels shown per row.
        share_y: Controls if the y-axis of each panels should be shared.
                 But passing `sharey=False`, each panel has its own y-axis range.
        {show_save_ax}

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.15, key_added="leiden_0_5")
            ep.tl.rank_features_groups(adata, groupby="leiden_0_5")
            ep.pl.rank_features_groups(adata, key="rank_features_groups")

    Preview:
        .. image:: /_images/docstring_previews/rank_features_groups.png
    """
    return sc.pl.rank_genes_groups(
        adata=adata,
        groups=groups,
        n_genes=n_features,
        gene_symbols=feature_symbols,
        key=key,
        fontsize=fontsize,
        ncols=ncols,
        sharey=share_y,
        show=show,
        save=save,
        ax=ax,
        **kwds,
    )


@_doc_params(show_save_ax=doc_show_save_ax)
def rank_features_groups_violin(
    adata: AnnData,
    groups: Optional[Sequence[str]] = None,
    n_features: int = 20,
    feature_names: Optional[Iterable[str]] = None,
    feature_symbols: Optional[str] = None,
    key: Optional[str] = None,
    split: bool = True,
    scale: str = "width",
    strip: bool = True,
    jitter: Union[int, float, bool] = True,
    size: int = 1,
    ax: Optional[Axes] = None,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
):  # pragma: no cover
    """Plot ranking of features for all tested comparisons as violin plots.

    Args:
        adata: Annotated data matrix.
        groups: List of group names.
        n_features: Number of features to show. Is ignored if `feature_names` is passed.
        feature_names: List of features to plot. Is only useful if interested in a custom feature list,
                       which is not the result of :func:`ehrapy.tl.rank_features_groups`.
        feature_symbols: Key for field in `.var` that stores feature symbols if you do not want to
                         use `.var_names` displayed in the plot.
        key: The key of the calculated feature group rankings (default: 'rank_features_groups').
        split: Whether to split the violins or not.
        scale: See :func:`~seaborn.violinplot`.
        strip: Show a strip plot on top of the violin plot.
        jitter: If set to 0, no points are drawn. See :func:`~seaborn.stripplot`.
        size: Size of the jitter points.
        {show_save_ax}

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.15, key_added="leiden_0_5")
            ep.tl.rank_features_groups(adata, groupby="leiden_0_5")
            ep.pl.rank_features_groups_violin(adata, key="rank_features_groups", n_features=5)

    Preview:
        .. image:: /_images/docstring_previews/rank_features_groups_violin_1.png

        .. image:: /_images/docstring_previews/rank_features_groups_violin_2.png

        .. image:: /_images/docstring_previews/rank_features_groups_violin_3.png

        .. image:: /_images/docstring_previews/rank_features_groups_violin_4.png
    """
    return sc.pl.rank_genes_groups_violin(
        adata=adata,
        groups=groups,
        n_genes=n_features,
        gene_names=feature_names,
        gene_symbols=feature_symbols,
        use_raw=False,
        key=key,
        split=split,
        scale=scale,
        strip=strip,
        jitter=jitter,
        size=size,
        ax=ax,
        show=show,
        save=save,
    )


@_doc_params(show_save_ax=doc_show_save_ax)
def rank_features_groups_stacked_violin(
    adata: AnnData,
    groups: Union[str, Sequence[str]] = None,
    n_features: Optional[int] = None,
    groupby: Optional[str] = None,
    feature_symbols: Optional[str] = None,
    *,
    var_names: Optional[Union[Sequence[str], Mapping[str, Sequence[str]]]] = None,
    min_logfoldchange: Optional[float] = None,
    key: Optional[str] = None,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
    return_fig: Optional[bool] = False,
    **kwds,
):  # pragma: no cover
    """Plot ranking of genes using stacked_violin plot.

    Args:
        adata: Annotated data matrix.
        groups: List of group names.
        n_features: Number of features to show. Is ignored if `feature_names` is passed.
        groupby: Which key to group the features by.
        feature_symbols: Key for field in `.var` that stores feature symbols if you do not want to
                         use `.var_names` displayed in the plot.
        var_names: Feature names.
        min_logfoldchange: Minimum log fold change to consider.
        key: The key of the calculated feature group rankings (default: 'rank_features_groups').
        show: Whether to show the plot.
        save: Where to save the plot.
        return_fig: Returns :class:`StackedViolin` object. Useful for fine-tuning the plot.
                    Takes precedence over `show=False`.

    Returns:
        If `return_fig` is `True`, returns a :class:`StackedViolin` object,
        else if `show` is false, return axes dict

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.15, key_added="leiden_0_5")
            ep.tl.rank_features_groups(adata, groupby="leiden_0_5")
            ep.pl.rank_features_groups_stacked_violin(adata, key="rank_features_groups", n_features=5)

    Preview:
        .. image:: /_images/docstring_previews/rank_features_groups_stacked_violin.png
    """
    return sc.pl.rank_genes_groups_stacked_violin(
        adata=adata,
        groups=groups,
        n_genes=n_features,
        groupby=groupby,
        gene_symbols=feature_symbols,
        var_names=var_names,
        min_logfoldchange=min_logfoldchange,
        key=key,
        show=show,
        save=save,
        return_fig=return_fig,
        **kwds,
    )


def rank_features_groups_heatmap(
    adata: AnnData,
    groups: Union[str, Sequence[str]] = None,
    n_features: Optional[int] = None,
    groupby: Optional[str] = None,
    feature_symbols: Optional[str] = None,
    var_names: Optional[Union[Sequence[str], Mapping[str, Sequence[str]]]] = None,
    min_logfoldchange: Optional[float] = None,
    key: str = None,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
    **kwds,
):  # pragma: no cover
    """Plot ranking of genes using heatmap plot (see :func:`~ehrapy.pl.heatmap`)

    Args:
        adata: Annotated data matrix.
        groups: List of group names.
        n_features: Number of features to show. Is ignored if `feature_names` is passed.
        groupby: Which key to group the features by.
        feature_symbols: Key for field in `.var` that stores feature symbols if you do not want to
                         use `.var_names` displayed in the plot.
        var_names: Feature names.
        min_logfoldchange: Minimum log fold change to consider.
        key: The key of the calculated feature group rankings (default: 'rank_features_groups').
        show: Whether to show the plot.
        save: Where to save the plot.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.15, key_added="leiden_0_5")
            ep.tl.rank_features_groups(adata, groupby="leiden_0_5")
            ep.pl.rank_features_groups_heatmap(adata, key="rank_features_groups")

    Preview:
        .. image:: /_images/docstring_previews/rank_features_groups_heatmap.png
    """
    return sc.pl.rank_genes_groups_heatmap(
        adata=adata,
        groups=groups,
        n_genes=n_features,
        groupby=groupby,
        gene_symbols=feature_symbols,
        var_names=var_names,
        min_logfoldchange=min_logfoldchange,
        key=key,
        show=show,
        save=save,
        **kwds,
    )


def rank_features_groups_dotplot(
    adata: AnnData,
    groups: Union[str, Sequence[str]] = None,
    n_features: Optional[int] = None,
    groupby: Optional[str] = None,
    values_to_plot: Optional[
        Literal[
            "scores",
            "logfoldchanges",
            "pvals",
            "pvals_adj",
            "log10_pvals",
            "log10_pvals_adj",
        ]
    ] = None,
    var_names: Optional[Union[Sequence[str], Mapping[str, Sequence[str]]]] = None,
    feature_symbols: Optional[str] = None,
    min_logfoldchange: Optional[float] = None,
    key: Optional[str] = None,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
    return_fig: Optional[bool] = False,
    **kwds,
):  # pragma: no cover
    """Plot ranking of genes using dotplot plot (see :func:`~ehrapy.pl.dotplot`)

    Args:
        adata: Annotated data matrix.
        groups: List of group names.
        n_features: Number of features to show. Is ignored if `feature_names` is passed.
        groupby: Which key to group the features by.
        feature_symbols: Key for field in `.var` that stores feature symbols if you do not want to
                         use `.var_names` displayed in the plot.
        values_to_plot: Key to plot. One of 'scores', 'logfoldchanges', 'pvals', 'pvalds_adj',
                        'log10_pvals', 'log10_pvalds_adj'.
        var_names: Feature names.
        min_logfoldchange: Minimum log fold change to consider.
        key: The key of the calculated feature group rankings (default: 'rank_features_groups').
        show: Whether to show the plot.
        save: Where to save the plot.
        return_fig: Returns :class:`StackedViolin` object. Useful for fine-tuning the plot.
                    Takes precedence over `show=False`.

    Returns:
        If `return_fig` is `True`, returns a :class:`StackedViolin` object,
        else if `show` is false, return axes dict

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.tl.rank_features_groups(adata, groupby="leiden_0_5")
            ep.pl.rank_features_groups_dotplot(adata, key="rank_features_groups", groupby="leiden_0_5")

    Preview:
        .. image:: /_images/docstring_previews/rank_features_groups_dotplot.png
    """
    return sc.pl.rank_genes_groups_dotplot(
        adata=adata,
        groups=groups,
        n_genes=n_features,
        groupby=groupby,
        values_to_plot=values_to_plot,
        var_names=var_names,
        gene_symbols=feature_symbols,
        min_logfoldchange=min_logfoldchange,
        key=key,
        show=show,
        save=save,
        return_fig=return_fig,
        **kwds,
    )


def rank_features_groups_matrixplot(
    adata: AnnData,
    groups: Union[str, Sequence[str]] = None,
    n_features: Optional[int] = None,
    groupby: Optional[str] = None,
    values_to_plot: Optional[
        Literal[
            "scores",
            "logfoldchanges",
            "pvals",
            "pvals_adj",
            "log10_pvals",
            "log10_pvals_adj",
        ]
    ] = None,
    var_names: Optional[Union[Sequence[str], Mapping[str, Sequence[str]]]] = None,
    feature_symbols: Optional[str] = None,
    min_logfoldchange: Optional[float] = None,
    key: Optional[str] = None,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
    return_fig: Optional[bool] = False,
    **kwds,
):  # pragma: no cover
    """Plot ranking of genes using matrixplot plot (see :func:`~ehrapy.pl.matrixplot`)

    Args:
        adata: Annotated data matrix.
        groups: List of group names.
        n_features: Number of features to show. Is ignored if `feature_names` is passed.
        groupby: Which key to group the features by.
        feature_symbols: Key for field in `.var` that stores feature symbols if you do not want to
                         use `.var_names` displayed in the plot.
        values_to_plot: Key to plot. One of 'scores', 'logfoldchanges', 'pvals', 'pvalds_adj',
                        'log10_pvals', 'log10_pvalds_adj'.
        var_names: Feature names.
        min_logfoldchange: Minimum log fold change to consider.
        key: The key of the calculated feature group rankings (default: 'rank_features_groups').
        show: Whether to show the plot.
        save: Where to save the plot.
        return_fig: Returns :class:`StackedViolin` object. Useful for fine-tuning the plot.
                    Takes precedence over `show=False`.

    Returns:
        If `return_fig` is `True`, returns a :class:`MatrixPlot` object,
        else if `show` is false, return axes dict

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
            ep.tl.rank_features_groups(adata, groupby="leiden_0_5")
            ep.pl.rank_features_groups_matrixplot(adata, key="rank_features_groups", groupby="leiden_0_5")

    Preview:
        .. image:: /_images/docstring_previews/rank_features_groups_matrixplot.png

    """
    return sc.pl.rank_genes_groups_matrixplot(
        adata=adata,
        groups=groups,
        n_genes=n_features,
        groupby=groupby,
        values_to_plot=values_to_plot,
        var_names=var_names,
        gene_symbols=feature_symbols,
        min_logfoldchange=min_logfoldchange,
        key=key,
        show=show,
        save=save,
        return_fig=return_fig,
        **kwds,
    )


def rank_features_groups_tracksplot(
    adata: AnnData,
    groups: Union[str, Sequence[str]] = None,
    n_features: Optional[int] = None,
    groupby: Optional[str] = None,
    var_names: Optional[Union[Sequence[str], Mapping[str, Sequence[str]]]] = None,
    feature_symbols: Optional[str] = None,
    min_logfoldchange: Optional[float] = None,
    key: Optional[str] = None,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
    **kwds,
):  # pragma: no cover
    """Plot ranking of genes using tracksplot plot (see :func:`~ehrapy.pl.tracksplot`)

    Args:
        adata: Annotated data matrix.
        groups: List of group names.
        n_features: Number of features to show. Is ignored if `feature_names` is passed.
        groupby: Which key to group the features by.
        feature_symbols: Key for field in `.var` that stores feature symbols if you do not want to
                         use `.var_names` displayed in the plot.
        var_names: Feature names.
        min_logfoldchange: Minimum log fold change to consider.
        key: The key of the calculated feature group rankings (default: 'rank_features_groups').
        show: Whether to show the plot.
        save: Where to save the plot.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
            ep.pp.norm_log(adata, offset=1)
            ep.pp.neighbors(adata)
            ep.tl.leiden(adata, resolution=0.15, key_added="leiden_0_5")
            ep.tl.rank_features_groups(adata, groupby="leiden_0_5")
            ep.pl.rank_features_groups_tracksplot(adata, key="rank_features_groups")

    Preview:
        .. image:: /_images/docstring_previews/rank_features_groups_tracksplot.png
    """
    return sc.pl.rank_genes_groups_tracksplot(
        adata=adata,
        groups=groups,
        n_genes=n_features,
        groupby=groupby,
        var_names=var_names,
        feature_symbols=feature_symbols,
        min_logfoldchange=min_logfoldchange,
        key=key,
        show=show,
        save=save,
        **kwds,
    )
