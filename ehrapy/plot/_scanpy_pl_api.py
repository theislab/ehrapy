from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import ehrdata as ed
import scanpy as sc
from fast_array_utils.conv import to_dense
from fast_array_utils.types import DaskArray
from scanpy.plotting import DotPlot, MatrixPlot, StackedViolin

from ehrapy._compat import _as_scanpy_input, _materialize, _raise_if_3D, function_2D_only
from ehrapy._utils_doc import _doc_params, doc_plot_params

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from cycler import Cycler
    from ehrdata import EHRData
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, ListedColormap, Normalize
    from matplotlib.figure import Figure
    from scanpy.plotting._utils import _AxesSubplot
    from seaborn import FacetGrid
    from seaborn.matrix import ClusterGrid

_Basis = Literal["pca", "tsne", "umap", "diffmap", "draw_graph_fr"]
_VarNames = str | Sequence[str]
ColorLike = str | tuple[float, ...]
_IGraphLayout = Literal["fa", "fr", "rt", "rt_circular", "drl", "eq_tree", ...]  # type: ignore
_FontWeight = Literal["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
_FontSize = Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]
VBound = str | float | Callable[[Sequence[float]], float]
_ValuesToPlot = Literal["scores", "logfoldchanges", "pvals", "pvals_adj", "log10_pvals", "log10_pvals_adj"]


@function_2D_only(var_keys=("x", "y", "color"))
@_doc_params(**doc_plot_params)
def scatter(
    edata: EHRData,
    *,
    x: str | None = None,
    y: str | None = None,
    color: ColorLike | Collection[ColorLike] | None = None,
    use_raw: bool | None = None,
    layers: str | Collection[str] | None = None,
    sort_order: bool = True,
    alpha: float | None = None,
    basis: _Basis | None = None,
    groups: str | Iterable[str] | None = None,
    components: str | Collection[str] | None = None,
    projection: Literal["2d", "3d"] = "2d",
    legend_loc: str | None = "right margin",
    legend_fontsize: float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight | None = None,
    legend_fontoutline: float | None = None,
    color_map: str | Colormap | None = None,
    palette: Cycler | ListedColormap | ColorLike | Sequence[ColorLike] | None = None,
    frameon: bool | None = None,
    right_margin: float | None = None,
    left_margin: float | None = None,
    size: float | None = None,
    marker: str | Sequence[str] = ".",
    title: str | Collection[str] | None = None,
    show: bool | None = None,
    ax: Axes | None = None,
) -> Axes | list[Axes] | None:  # pragma: no cover
    """Scatter plot along observations or variables axes.

    Color the plot using annotations of observations (`.obs`), variables (`.var`) or features (`.var_names`).

    Args:
        edata: Central data object.
        x: x coordinate.
        y: y coordinate.
        color: Keys for annotations of observations or features, or a hex color specification, e.g., `'ann1'`, `'#fe57a1'`, or `['ann1', 'ann2']`.
        use_raw: Whether to use `raw` attribute of `edata`.
            Defaults to `True` if `.raw` is present.
        layers: Use the `layers` attribute of `edata` if present: specify the layer for `x`, `y` and `color`.
            If `layers` is a string, then it is expanded to `(layers, layers, layers)`.
        sort_order: {sort_order}
        alpha: Opacity of the points.
        basis: String that denotes a plotting tool that computed coordinates.
        groups: {groups}
        components: {components}
        projection: {projection}
        legend_loc: {legend_loc}
        legend_fontsize: {legend_fontsize}
        legend_fontweight: {legend_fontweight}
        legend_fontoutline: {legend_fontoutline}
        color_map: {color_map}
        palette: {palette}
        frameon: {frameon}
        right_margin: Margin to the right of the plot.
        left_margin: Margin to the left of the plot.
        size: {size}
        marker: {marker}
        title: {panel_title}
        show: {show}
        ax: {ax}

    Returns:
        If `show` is `False`, a :class:`~matplotlib.axes.Axes` or a list of it.

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.pl.scatter(edata, x="age", y="icu_los_day", color="icu_los_day")

    Preview:
        .. image:: /_static/docstring_previews/scatter.png
    """
    scatter_partial = partial(
        sc.pl.scatter,
        x=x,
        y=y,
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
        marker=marker,
        title=title,
        show=show,
        ax=ax,
    )

    return scatter_partial(_as_scanpy_input(edata), color=color)


@function_2D_only()
@_doc_params(**doc_plot_params)
def heatmap(
    edata: EHRData,
    var_names: _VarNames | Mapping[str, _VarNames],
    groupby: str | Sequence[str],
    *,
    use_raw: bool | None = None,
    log: bool = False,
    num_categories: int = 7,
    dendrogram: bool | str = False,
    feature_symbols: str | None = None,
    var_group_positions: Sequence[tuple[int, int]] | None = None,
    var_group_labels: Sequence[str] | None = None,
    var_group_rotation: float | None = None,
    layer: str | None = None,
    standard_scale: Literal["var", "obs"] | None = None,
    swap_axes: bool = False,
    show_feature_labels: bool | None = None,
    show: bool | None = None,
    figsize: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    norm: Normalize | None = None,
    **kwds,
) -> dict[str, Axes] | None:  # pragma: no cover
    """Heatmap of the feature values.

    If `groupby` is given, the heatmap is ordered by the respective group.
    If the `groupby` observation annotation is not categorical the observation annotation is turned into a categorical by binning the data into the number specified in `num_categories`.

    Args:
        edata: Central data object.
        var_names: {var_names}
        groupby: {groupby}
        use_raw: {use_raw}
        log: {log}
        num_categories: {num_categories}
        dendrogram: {dendrogram}
        feature_symbols: {feature_symbols}
        var_group_positions: {var_group_positions}
        var_group_labels: {var_group_labels}
        var_group_rotation: {var_group_rotation}
        layer: {layer}
        standard_scale: Whether or not to standardize that dimension between 0 and 1.
            For each variable or observation, subtract the minimum and divide each by its maximum.
        swap_axes: By default, the x axis contains `var_names` (e.g. features) and the y axis the `groupby` categories (if any).
            By setting `swap_axes` then x are the `groupby` categories and y the `var_names`.
        show_feature_labels: By default feature labels are shown when there are 50 or less features.
            Otherwise the labels are removed.
        show: {show}
        figsize: {figsize}
        vmin: {vmin}
        vmax: {vmax}
        vcenter: {vcenter}
        norm: {norm}
        **kwds: Are passed to :func:`matplotlib.pyplot.imshow`.

    Returns:
        Dict of :class:`~matplotlib.axes.Axes` if `show` is `False`.

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.pl.heatmap(
        ...     edata,
        ...     var_names=[
        ...         "map_1st",
        ...         "hr_1st",
        ...         "temp_1st",
        ...         "spo2_1st",
        ...         "abg_count",
        ...         "wbc_first",
        ...         "hgb_first",
        ...         "platelet_first",
        ...         "sodium_first",
        ...         "potassium_first",
        ...         "tco2_first",
        ...         "chloride_first",
        ...         "bun_first",
        ...         "creatinine_first",
        ...         "po2_first",
        ...         "pco2_first",
        ...         "iv_day_1",
        ...     ],
        ...     groupby="leiden_0_5",
        ... )

    Preview:
        .. image:: /_static/docstring_previews/heatmap.png
    """
    heatmap_partial = partial(
        sc.pl.heatmap,
        var_names=var_names,
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
        figsize=figsize,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        **kwds,
    )

    return heatmap_partial(_as_scanpy_input(edata), groupby=groupby)


@function_2D_only()
@_doc_params(**doc_plot_params)
def dotplot(
    edata: EHRData,
    var_names: _VarNames | Mapping[str, _VarNames],
    groupby: str | Sequence[str],
    *,
    use_raw: bool | None = None,
    log: bool = False,
    num_categories: int = 7,
    categories_order: Sequence[str] | None = None,
    feature_cutoff: float = 0.0,
    mean_only_counts: bool = False,
    standard_scale: Literal["var", "group"] | None = None,
    title: str | None = None,
    colorbar_title: str | None = "Mean value in group",
    size_title: str | None = "Fraction of observations\nin group (%)",
    figsize: tuple[float, float] | None = None,
    dendrogram: bool | str = False,
    feature_symbols: str | None = None,
    var_group_positions: Sequence[tuple[int, int]] | None = None,
    var_group_labels: Sequence[str] | None = None,
    var_group_rotation: float | None = None,
    layer: str | None = None,
    swap_axes: bool = False,
    dot_color_df: pd.DataFrame | None = None,
    show: bool | None = None,
    ax: _AxesSubplot | None = None,
    return_fig: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    norm: Normalize | None = None,
    cmap: Colormap | str | None = "Reds",
    group_colors: Mapping[str, ColorLike] | None = None,
    dot_max: float | None = DotPlot.DEFAULT_DOT_MAX,
    dot_min: float | None = DotPlot.DEFAULT_DOT_MIN,
    smallest_dot: float = DotPlot.DEFAULT_SMALLEST_DOT,
    **kwds,
) -> DotPlot | dict | None:  # pragma: no cover
    r"""Makes a *dot plot* of the count values of `var_names`.

    For each var_name and each `groupby` category a dot is plotted.
    Each dot represents two values: mean value within each category (visualized by color) and fraction of observations with the `var_name` in the category (visualized by the size of the dot).
    If `groupby` is not given, the dotplot assumes that all data belongs to a single category.

    .. note::
       A count is used if it is above the specified threshold which is zero by default.

    Args:
        edata: Central data object.
        var_names: {var_names}
        groupby: {groupby}
        use_raw: {use_raw}
        log: {log}
        num_categories: {num_categories}
        categories_order: {categories_order}
        feature_cutoff: Count cutoff that is used for binarizing the counts and determining the fraction of observations having the feature.
            A feature is only used if its counts are greater than this threshold.
        mean_only_counts: If `True`, counts are averaged only over the observations having the provided feature.
        standard_scale: {standard_scale}
        title: {title}
        colorbar_title: {colorbar_title}
        size_title: Title for the size legend.
            New line character (\\n) can be used.
        figsize: {figsize}
        dendrogram: {dendrogram}
        feature_symbols: {feature_symbols}
        var_group_positions: {var_group_positions}
        var_group_labels: {var_group_labels}
        var_group_rotation: {var_group_rotation}
        layer: {layer}
        swap_axes: {swap_axes}
        dot_color_df: Data frame with the values to color the dots by instead of the mean values, with the groups as rows and the features as columns.
        show: {show}
        ax: {ax}
        return_fig: Returns :class:`~scanpy.pl.DotPlot` object.
            Useful for fine-tuning the plot.
            Takes precedence over `show=False`.
        vmin: {vmin}
        vmax: {vmax}
        vcenter: {vcenter}
        norm: {norm}
        cmap: {cmap}
        group_colors: A mapping of group names to colors, e.g. `{{'FICU': 'blue', 'MICU': '#aa40fc'}}`.
            Colors can be specified as any valid matplotlib color.
            If `group_colors` is used, a colormap is generated from white to the given color for each group.
            If a group is not present in the dictionary, the value of `cmap` is used.
        dot_max: If `None`, the maximum dot size is set to the maximum fraction value found (e.g. 0.6).
            If given, the value should be a number between 0 and 1.
            All fractions larger than dot_max are clipped to this value.
        dot_min: If `None`, the minimum dot size is set to 0.
            If given, the value should be a number between 0 and 1.
            All fractions smaller than dot_min are clipped to this value.
        smallest_dot: All counts with `dot_min` are plotted with this size.
        **kwds: Are passed to :func:`matplotlib.pyplot.scatter`.

    Returns:
        If `return_fig` is `True`, returns a :class:`~scanpy.pl.DotPlot` object, else if `show` is false, return axes dict

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.pl.dotplot(
        ...     edata,
        ...     var_names=[
        ...         "age",
        ...         "gender_num",
        ...         "weight_first",
        ...         "bmi",
        ...         "wbc_first",
        ...         "hgb_first",
        ...         "platelet_first",
        ...         "sodium_first",
        ...         "potassium_first",
        ...         "tco2_first",
        ...         "chloride_first",
        ...         "bun_first",
        ...         "creatinine_first",
        ...         "po2_first",
        ...         "pco2_first",
        ...     ],
        ...     groupby="leiden_0_5",
        ... )

    Preview:
        .. image:: /_static/docstring_previews/dotplot.png
    """
    dotplot_partial = partial(
        sc.pl.dotplot,
        var_names=var_names,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        categories_order=categories_order,
        expression_cutoff=feature_cutoff,
        mean_only_expressed=mean_only_counts,
        standard_scale=standard_scale,
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
        ax=ax,
        return_fig=return_fig,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        cmap=cmap,
        group_colors=group_colors,
        dot_max=dot_max,
        dot_min=dot_min,
        smallest_dot=smallest_dot,
        **kwds,
    )

    return dotplot_partial(_as_scanpy_input(edata), groupby=groupby)


@function_2D_only()
@_doc_params(**doc_plot_params)
def tracksplot(
    edata: EHRData,
    var_names: _VarNames | Mapping[str, _VarNames],
    groupby: str,
    *,
    use_raw: bool | None = None,
    log: bool = False,
    dendrogram: bool | str = False,
    feature_symbols: str | None = None,
    var_group_positions: Sequence[tuple[int, int]] | None = None,
    var_group_labels: Sequence[str] | None = None,
    layer: str | None = None,
    show: bool | None = None,
    figsize: tuple[float, float] | None = None,
) -> dict[str, Axes] | None:  # pragma: no cover
    """Plots a filled line plot.

    In this type of plot each var_name is plotted as a filled line plot where the y values correspond to the var_name values and x is each of the observations.
    Best results are obtained when using raw counts that are not log.
    `groupby` is required to sort and order the values using the respective group and should be a categorical value.

    Args:
        edata: Central data object.
        var_names: {var_names}
        groupby: {groupby}
        use_raw: {use_raw}
        log: {log}
        dendrogram: {dendrogram}
        feature_symbols: {feature_symbols}
        var_group_positions: {var_group_positions}
        var_group_labels: {var_group_labels}
        layer: {layer}
        show: {show}
        figsize: {figsize}

    Returns:
        Dict of :class:`~matplotlib.axes.Axes` if `show` is `False`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.pl.tracksplot(
        ...     edata,
        ...     var_names=[
        ...         "age",
        ...         "gender_num",
        ...         "weight_first",
        ...         "bmi",
        ...         "sapsi_first",
        ...         "sofa_first",
        ...         "service_num",
        ...         "day_icu_intime_num",
        ...         "hour_icu_intime",
        ...     ],
        ...     groupby="leiden_0_5",
        ... )

    Preview:
        .. image:: /_static/docstring_previews/tracksplot.png
    """
    tracksplot_partial = partial(
        sc.pl.tracksplot,
        var_names=var_names,
        use_raw=use_raw,
        log=log,
        dendrogram=dendrogram,
        gene_symbols=feature_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        layer=layer,
        show=show,
        figsize=figsize,
    )

    return tracksplot_partial(_as_scanpy_input(edata), groupby=groupby)


@function_2D_only(var_keys=("keys",))
@_doc_params(**doc_plot_params)
def violin(
    edata: EHRData,
    keys: str | Sequence[str],
    *,
    groupby: str | None = None,
    log: bool = False,
    use_raw: bool | None = None,
    stripplot: bool = True,
    jitter: float | bool = True,
    size: int = 1,
    layer: str | None = None,
    density_norm: Literal["area", "count", "width"] = "width",
    order: Sequence[str] | None = None,
    multi_panel: bool = False,
    xlabel: str = "",
    ylabel: str | Sequence[str] | None = None,
    rotation: float | None = None,
    show: bool | None = None,
    ax: Axes | None = None,
    **kwds,
) -> Axes | FacetGrid | None:  # pragma: no cover
    """Violin plot.

    Wraps :func:`seaborn.violinplot` for :class:`~ehrdata.EHRData`.

    Args:
        edata: Central data object.
        keys: Keys for accessing variables of `.var_names` or fields of `.obs`.
        groupby: {groupby}
        log: {log}
        use_raw: Whether to use `raw` attribute of `edata`.
            Defaults to `True` if `.raw` is present.
        stripplot: Add a stripplot on top of the violin plot.
            See :func:`~seaborn.stripplot`.
        jitter: Add jitter to the stripplot (only when stripplot is True).
            See :func:`~seaborn.stripplot`.
        size: Size of the jitter points.
        layer: {layer}
        density_norm: The method used to scale the width of each violin.
            If 'width' (the default), each violin will have the same width.
            If 'area', each violin will have the same area.
            If 'count', a violin's width corresponds to the number of observations.
        order: Order in which to show the categories.
        multi_panel: Display keys in multiple panels also when `groupby is not None`.
        xlabel: Label of the x axis.
            Defaults to `groupby` if `rotation` is `None`, otherwise, no label is shown.
        ylabel: Label of the y axis.
            If `None` and `groupby` is `None`, defaults to `'value'`.
            If `None` and `groupby` is not `None`, defaults to `keys`.
        rotation: Rotation of xtick labels.
        show: {show}
        ax: {ax}
        **kwds: Are passed to :func:`~seaborn.violinplot`.

    Returns:
        A :class:`~matplotlib.axes.Axes` object if `ax` is `None` else `None`.

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.pl.violin(edata, keys=["age"], groupby="leiden_0_5")

    Preview:
        .. image:: /_static/docstring_previews/violin.png
    """
    violin_partial = partial(
        sc.pl.violin,
        keys=keys,
        log=log,
        use_raw=use_raw,
        stripplot=stripplot,
        jitter=jitter,
        size=size,
        layer=layer,
        density_norm=density_norm,
        order=order,
        multi_panel=multi_panel,
        xlabel=xlabel,
        ylabel=ylabel,
        rotation=rotation,
        show=show,
        ax=ax,
        **kwds,
    )

    return violin_partial(_as_scanpy_input(edata), groupby=groupby)


@function_2D_only()
@_doc_params(**doc_plot_params)
def stacked_violin(
    edata: EHRData,
    var_names: _VarNames | Mapping[str, _VarNames],
    groupby: str | Sequence[str],
    *,
    log: bool = False,
    use_raw: bool | None = None,
    num_categories: int = 7,
    title: str | None = None,
    colorbar_title: str | None = "Median value\n in group",
    figsize: tuple[float, float] | None = None,
    dendrogram: bool | str = False,
    feature_symbols: str | None = None,
    var_group_positions: Sequence[tuple[int, int]] | None = None,
    var_group_labels: Sequence[str] | None = None,
    standard_scale: Literal["var", "group"] | None = None,
    var_group_rotation: float | None = None,
    layer: str | None = None,
    categories_order: Sequence[str] | None = None,
    swap_axes: bool = False,
    show: bool | None = None,
    return_fig: bool = False,
    ax: _AxesSubplot | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    norm: Normalize | None = None,
    cmap: Colormap | str | None = StackedViolin.DEFAULT_COLORMAP,
    stripplot: bool = StackedViolin.DEFAULT_STRIPPLOT,
    jitter: float | bool = StackedViolin.DEFAULT_JITTER,
    size: float = StackedViolin.DEFAULT_JITTER_SIZE,
    row_palette: str | None = StackedViolin.DEFAULT_ROW_PALETTE,
    density_norm: Literal["area", "count", "width"] = StackedViolin.DEFAULT_DENSITY_NORM,
    yticklabels: bool = StackedViolin.DEFAULT_PLOT_YTICKLABELS,
    **kwds,
) -> StackedViolin | dict | None:  # pragma: no cover
    r"""Stacked violin plots.

    Makes a compact image composed of individual violin plots (from :func:`~seaborn.violinplot`) stacked on top of each other.

    This function provides a convenient interface to the :class:`~scanpy.pl.StackedViolin` class.
    If you need more flexibility, use :class:`~scanpy.pl.StackedViolin` directly.

    Args:
        edata: Central data object.
        var_names: {var_names}
        groupby: {groupby}
        log: {log}
        use_raw: {use_raw}
        num_categories: {num_categories}
        title: {title}
        colorbar_title: {colorbar_title}
        figsize: {figsize}
        dendrogram: {dendrogram}
        feature_symbols: {feature_symbols}
        var_group_positions: {var_group_positions}
        var_group_labels: {var_group_labels}
        standard_scale: {standard_scale}
        var_group_rotation: {var_group_rotation}
        layer: {layer}
        categories_order: {categories_order}
        swap_axes: {swap_axes}
        show: {show}
        return_fig: Returns :class:`~scanpy.pl.StackedViolin` object.
            Useful for fine-tuning the plot.
            Takes precedence over `show=False`.
        ax: {ax}
        vmin: {vmin}
        vmax: {vmax}
        vcenter: {vcenter}
        norm: {norm}
        cmap: {cmap}
        stripplot: Add a stripplot on top of the violin plot.
            See :func:`~seaborn.stripplot`.
        jitter: Add jitter to the stripplot (only when stripplot is True).
            See :func:`~seaborn.stripplot`.
        size: Size of the jitter points.
        row_palette: By default, median values are mapped to the violin color using a color map (see `cmap` argument).
            Alternatively, a 'row_palette` can be given to color each violin plot row using a different colors.
            The value should be a valid seaborn or matplotlib palette name (see :func:`~seaborn.color_palette`).
            Alternatively, a single color name or hex value can be passed, e.g. `'red'` or `'#cc33ff'`.
        density_norm: The method used to scale the width of each violin.
            If 'width' (the default), each violin will have the same width.
            If 'area', each violin will have the same area.
            If 'count', a violin's width corresponds to the number of observations.
        yticklabels: Set to `True` to view the y tick labels.
        **kwds: Are passed to :func:`~seaborn.violinplot`.

    Returns:
        If `return_fig` is `True`, returns a :class:`~scanpy.pl.StackedViolin` object, else if `show` is false, return axes dict

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.pl.stacked_violin(
        ...     edata,
        ...     var_names=[
        ...         "icu_los_day",
        ...         "hospital_los_day",
        ...         "age",
        ...         "gender_num",
        ...         "weight_first",
        ...         "bmi",
        ...         "sapsi_first",
        ...         "sofa_first",
        ...         "service_num",
        ...         "day_icu_intime_num",
        ...         "hour_icu_intime",
        ...     ],
        ...     groupby="leiden_0_5",
        ... )

    Preview:
        .. image:: /_static/docstring_previews/stacked_violin.png
    """
    stacked_vio_partial = partial(
        sc.pl.stacked_violin,
        var_names=var_names,
        log=log,
        use_raw=use_raw,
        num_categories=num_categories,
        title=title,
        colorbar_title=colorbar_title,
        figsize=figsize,
        dendrogram=dendrogram,
        gene_symbols=feature_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        standard_scale=standard_scale,
        var_group_rotation=var_group_rotation,
        layer=layer,
        categories_order=categories_order,
        swap_axes=swap_axes,
        show=show,
        return_fig=return_fig,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        cmap=cmap,
        stripplot=stripplot,
        jitter=jitter,
        size=size,
        row_palette=row_palette,
        density_norm=density_norm,
        yticklabels=yticklabels,
        **kwds,
    )

    return stacked_vio_partial(_as_scanpy_input(edata), groupby=groupby)


@function_2D_only()
@_doc_params(**doc_plot_params)
def matrixplot(
    edata: EHRData,
    var_names: _VarNames | Mapping[str, _VarNames],
    groupby: str | Sequence[str],
    *,
    use_raw: bool | None = None,
    log: bool = False,
    num_categories: int = 7,
    categories_order: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    dendrogram: bool | str = False,
    title: str | None = None,
    cmap: Colormap | str | None = MatrixPlot.DEFAULT_COLORMAP,
    colorbar_title: str | None = "Mean value\n in group",
    feature_symbols: str | None = None,
    var_group_positions: Sequence[tuple[int, int]] | None = None,
    var_group_labels: Sequence[str] | None = None,
    var_group_rotation: float | None = None,
    layer: str | None = None,
    standard_scale: Literal["var", "group"] | None = None,
    values_df: pd.DataFrame | None = None,
    swap_axes: bool = False,
    show: bool | None = None,
    ax: _AxesSubplot | None = None,
    return_fig: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    norm: Normalize | None = None,
    **kwds,
) -> MatrixPlot | dict | None:  # pragma: no cover
    """Creates a heatmap of the mean count per group of each var_names.

    This function provides a convenient interface to the :class:`~scanpy.pl.MatrixPlot` class.
    If you need more flexibility, you should use :class:`~scanpy.pl.MatrixPlot` directly.

    Args:
        edata: Central data object.
        var_names: {var_names}
        groupby: {groupby}
        use_raw: {use_raw}
        log: {log}
        num_categories: {num_categories}
        categories_order: {categories_order}
        figsize: {figsize}
        dendrogram: {dendrogram}
        title: {title}
        cmap: {cmap}
        colorbar_title: {colorbar_title}
        feature_symbols: {feature_symbols}
        var_group_positions: {var_group_positions}
        var_group_labels: {var_group_labels}
        var_group_rotation: {var_group_rotation}
        layer: {layer}
        standard_scale: {standard_scale}
        values_df: Data frame with the values to plot instead of the mean values, with the groups as rows and the features as columns.
        swap_axes: {swap_axes}
        show: {show}
        ax: {ax}
        return_fig: Returns :class:`~scanpy.pl.MatrixPlot` object.
            Useful for fine-tuning the plot.
            Takes precedence over `show=False`.
        vmin: {vmin}
        vmax: {vmax}
        vcenter: {vcenter}
        norm: {norm}
        **kwds: Are passed to :func:`matplotlib.pyplot.pcolor`.

    Returns:
        If `return_fig` is `True`, returns a :class:`~scanpy.pl.MatrixPlot` object, else if `show` is false, return axes dict

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.pl.matrixplot(
        ...     edata,
        ...     var_names=[
        ...         "abg_count",
        ...         "wbc_first",
        ...         "hgb_first",
        ...         "platelet_first",
        ...         "sodium_first",
        ...         "potassium_first",
        ...         "tco2_first",
        ...         "chloride_first",
        ...         "bun_first",
        ...         "creatinine_first",
        ...         "po2_first",
        ...         "pco2_first",
        ...         "iv_day_1",
        ...     ],
        ...     groupby="leiden_0_5",
        ... )

    Preview:
        .. image:: /_static/docstring_previews/matrixplot.png
    """
    matrix_partial = partial(
        sc.pl.matrixplot,
        var_names=var_names,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        categories_order=categories_order,
        figsize=figsize,
        dendrogram=dendrogram,
        title=title,
        cmap=cmap,
        colorbar_title=colorbar_title,
        gene_symbols=feature_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        var_group_rotation=var_group_rotation,
        layer=layer,
        standard_scale=standard_scale,
        values_df=values_df,
        swap_axes=swap_axes,
        show=show,
        ax=ax,
        return_fig=return_fig,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        **kwds,
    )

    return matrix_partial(_as_scanpy_input(edata), groupby=groupby)


@function_2D_only()
@_doc_params(**doc_plot_params)
def clustermap(
    edata: EHRData,
    *,
    obs_keys: str | None = None,
    use_raw: bool | None = None,
    show: bool | None = None,
    **kwds,
) -> ClusterGrid | None:  # pragma: no cover
    """Hierarchically-clustered heatmap.

    Wraps :func:`seaborn.clustermap` for :class:`~ehrdata.EHRData`.

    Args:
        edata: Central data object.
        obs_keys: Categorical annotation to plot with a different color map.
            Currently, only a single key is supported.
        use_raw: Whether to use `raw` attribute of `edata`.
            Defaults to `True` if `.raw` is present.
        show: {show}
        **kwds: Keyword arguments passed to :func:`~seaborn.clustermap`.

    Returns:
        If `show` is `False`, a `seaborn.ClusterGrid` object (see :func:`~seaborn.clustermap`).

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.pl.clustermap(edata)

    Preview:
        .. image:: /_static/docstring_previews/clustermap.png
    """
    clustermap_partial = partial(sc.pl.clustermap, use_raw=use_raw, show=show, **kwds)

    return clustermap_partial(_as_scanpy_input(edata), obs_keys=obs_keys)


def ranking(
    edata: EHRData,
    attr: Literal["var", "obs", "uns", "varm", "obsm"],
    keys: str | Sequence[str],
    *,
    dictionary: str | None = None,
    indices: Sequence[int] | None = None,
    labels: str | Sequence[str] | None = None,
    color: ColorLike = "black",
    n_points: int = 30,
    log: bool = False,
    include_lowest: bool = False,
    show: bool | None = None,
):  # pragma: no cover
    """Plot rankings.

    See, for example, how this is used in :func:`~ehrapy.plot.pca_loadings`.

    Args:
        edata: Central data object.
        attr: The attribute of `edata` that contains the score.
        keys: The scores to look up an array from the attribute of `edata`.
        dictionary: Optional key dictionary.
        indices: Optional dictionary indices.
        labels: Optional labels.
        color: Optional primary color.
        n_points: Number of points.
        log: Whether logarithmic scale should be used.
        include_lowest: Whether to include the lowest points.
        show: Whether to show the plot.

    Returns:
        Returns matplotlib gridspec with access to the axes.

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.pca(edata)
        >>> ep.pl.ranking(edata, "varm", "PCs", indices=[0, 1, 2])
    """
    return sc.pl.ranking(
        edata,
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


@_doc_params(**doc_plot_params)
def dendrogram(
    edata: EHRData,
    groupby: str,
    *,
    dendrogram_key: str | None = None,
    orientation: Literal["top", "bottom", "left", "right"] = "top",
    remove_labels: bool = False,
    show: bool | None = None,
    ax: Axes | None = None,
) -> Axes:  # pragma: no cover
    """Plots a dendrogram of the categories defined in `groupby`.

    See :func:`~ehrapy.tools.dendrogram`.

    Args:
        edata: Central data object.
        groupby: Categorical data column used to create the dendrogram.
        dendrogram_key: Key under with the dendrogram information was stored.
            By default the dendrogram information is stored under `.uns[f'dendrogram_{{groupby}}']`.
        orientation: Origin of the tree.
            Will grow into the opposite direction.
        remove_labels: Don't draw labels.
            Used e.g. by :func:`~ehrapy.plot.matrixplot` to annotate matrix columns/rows.
        show: {show}
        ax: {ax}

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.pl.dendrogram(edata, groupby="leiden_0_5")

    Preview:
        .. image:: /_static/docstring_previews/dendrogram.png
    """
    # scanpy computes a missing dendrogram from `.X`
    if (f"dendrogram_{groupby}" if dendrogram_key is None else dendrogram_key) not in edata.uns:
        _raise_if_3D(edata.X, "dendrogram", "edata.X")

    dendrogram_partial = partial(
        sc.pl.dendrogram,
        dendrogram_key=dendrogram_key,
        orientation=orientation,
        remove_labels=remove_labels,
        show=show,
        ax=ax,
    )

    return dendrogram_partial(_as_scanpy_input(edata), groupby=groupby)


@_doc_params(**doc_plot_params)
@function_2D_only(var_keys=("color",))
def pca(
    edata: EHRData,
    *,
    annotate_var_explained: bool = False,
    feature_symbols: str | None = None,
    show: bool | None = None,
    return_fig: bool | None = None,
    **kwargs,
) -> Figure | Axes | list[Axes] | None:  # pragma: no cover
    """Scatter plot in PCA coordinates.

    Use the parameter `annotate_var_explained` to annotate the explained variance.

    Args:
        edata: Central data object.
        annotate_var_explained: Whether to annotate the axis labels with the explained variance ratio of the principal components.
        feature_symbols: {feature_symbols}
        show: {show}
        return_fig: {return_fig}
        **kwargs: {embedding_kwargs}

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.pca(edata)
        >>> ep.pl.pca(edata, color="service_unit")

    Preview:
        .. image:: /_static/docstring_previews/pca.png
    """
    pca_partial = partial(
        sc.pl.pca,
        annotate_var_explained=annotate_var_explained,
        gene_symbols=feature_symbols,
        show=show,
        return_fig=return_fig,
    )

    return pca_partial(_as_scanpy_input(edata), **kwargs)


def pca_loadings(
    edata: EHRData,
    *,
    components: str | Sequence[int] | None = None,
    include_lowest: bool = True,
    n_points: int | None = None,
    show: bool | None = None,
) -> None:  # pragma: no cover
    """Rank features according to contributions to PCs.

    Args:
        edata: Central data object.
        components: For example, ``'1,2,3'`` means ``[1, 2, 3]``, first, second, third principal component.
        include_lowest: Whether to show the features with both highest and lowest loadings.
        n_points: Number of features to plot for each component.
            Defaults to 30 or the number of features if there are fewer.
        show: Show the plot, do not return axis.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.pca(edata)
        >>> ep.pl.pca_loadings(edata, components="1,2,3")

    Preview:
        .. image:: /_static/docstring_previews/pca_loadings.png
    """
    return sc.pl.pca_loadings(edata, components=components, include_lowest=include_lowest, n_points=n_points, show=show)


def pca_variance_ratio(
    edata: EHRData,
    *,
    n_pcs: int = 30,
    log: bool = False,
    show: bool | None = None,
) -> None:  # pragma: no cover
    """Plot the variance ratio.

    Args:
        edata: Central data object.
        n_pcs: Number of PCs to show.
        log: Plot on logarithmic scale.
        show: Show the plot, do not return axis.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.pca(edata)
        >>> ep.pl.pca_variance_ratio(edata, n_pcs=8)

    Preview:
        .. image:: /_static/docstring_previews/pca_variance_ratio.png
    """
    return sc.pl.pca_variance_ratio(edata, n_pcs=n_pcs, log=log, show=show)


@function_2D_only(var_keys=("color",))
@_doc_params(**doc_plot_params)
def pca_overview(edata: EHRData, *, feature_symbols: str | None = None, **params) -> None:  # pragma: no cover
    """Plot PCA results.

    Plots the PCA scatter plot, the loadings and the variance ratio.
    The parameters are the ones of the scatter plot.
    Call :func:`~ehrapy.plot.pca_loadings` separately if you want to change the default settings of the loadings plot.

    Args:
        edata: Central data object.
        feature_symbols: {feature_symbols}
        **params: Keyword arguments of :func:`~ehrapy.plot.pca`, for example `color`, `components` or `show`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.pca(edata)
        >>> ep.pl.pca_overview(edata, components="1,2", color="service_unit")

    Preview:
        .. image:: /_static/docstring_previews/pca_overview_1.png

        .. image:: /_static/docstring_previews/pca_overview_2.png

        .. image:: /_static/docstring_previews/pca_overview_3.png
    """
    return sc.pl.pca_overview(_as_scanpy_input(edata), gene_symbols=feature_symbols, **params)


@function_2D_only(var_keys=("color",))
@_doc_params(**doc_plot_params)
def tsne(
    edata: EHRData, *, feature_symbols: str | None = None, **kwargs
) -> Figure | Axes | list[Axes] | None:  # pragma: no cover
    """Scatter plot in tSNE basis.

    Args:
        edata: Central data object.
        feature_symbols: {feature_symbols}
        **kwargs: {embedding_kwargs}

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.tsne(edata)
        >>> ep.pl.tsne(edata)

        .. image:: /_static/docstring_previews/tsne_1.png

        >>> ep.pl.tsne(
        ...     edata,
        ...     color=["day_icu_intime", "service_unit"],
        ...     wspace=0.5,
        ...     title=["Day of ICU admission", "Service unit"],
        ... )

        .. image:: /_static/docstring_previews/tsne_2.png

        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.pl.tsne(edata, color=["leiden_0_5"], title="Leiden 0.5")

        .. image:: /_static/docstring_previews/tsne_3.png

    """
    return sc.pl.tsne(_as_scanpy_input(edata), gene_symbols=feature_symbols, **kwargs)


@function_2D_only(var_keys=("color",))
@_doc_params(**doc_plot_params)
def umap(
    edata: EHRData, *, feature_symbols: str | None = None, **kwargs
) -> Figure | Axes | list[Axes] | None:  # pragma: no cover
    """Scatter plot in UMAP basis.

    Args:
        edata: Central data object.
        feature_symbols: {feature_symbols}
        **kwargs: {embedding_kwargs}

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.umap(edata)
        >>> ep.pl.umap(edata)

        .. image:: /_static/docstring_previews/umap_1.png

        >>> ep.pl.umap(
        ...     edata,
        ...     color=["day_icu_intime", "service_unit"],
        ...     wspace=0.5,
        ...     title=["Day of ICU admission", "Service unit"],
        ... )

        .. image:: /_static/docstring_previews/umap_2.png

        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.pl.umap(edata, color=["leiden_0_5"], title="Leiden 0.5")

        .. image:: /_static/docstring_previews/umap_3.png
    """
    return sc.pl.umap(_as_scanpy_input(edata), gene_symbols=feature_symbols, **kwargs)


@function_2D_only(var_keys=("color",))
@_doc_params(**doc_plot_params)
def diffmap(
    edata: EHRData, *, feature_symbols: str | None = None, **kwargs
) -> Figure | Axes | list[Axes] | None:  # pragma: no cover
    """Scatter plot in Diffusion Map basis.

    Args:
        edata: Central data object.
        feature_symbols: {feature_symbols}
        **kwargs: {embedding_kwargs}

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.diffmap(edata)
        >>> ep.pl.diffmap(edata, color="day_icu_intime")

    Preview:
        .. image:: /_static/docstring_previews/diffmap.png
    """
    return sc.pl.diffmap(_as_scanpy_input(edata), gene_symbols=feature_symbols, **kwargs)


@function_2D_only(var_keys=("color",))
@_doc_params(**doc_plot_params)
def draw_graph(
    edata: EHRData, *, layout: _IGraphLayout | None = None, feature_symbols: str | None = None, **kwargs
) -> Figure | Axes | list[Axes] | None:  # pragma: no cover
    """Scatter plot in graph-drawing basis.

    Args:
        edata: Central data object.
        layout: One of the :func:`~ehrapy.tools.draw_graph` layouts.
            By default, the last computed layout is used.
        feature_symbols: {feature_symbols}
        **kwargs: {embedding_kwargs}

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.tl.paga(edata, groups="leiden_0_5")
        >>> ep.pl.paga(
        ...     edata,
        ...     color=["leiden_0_5", "day_28_flg"],
        ...     cmap=ep.pl.Colormaps.grey_red.value,
        ...     title=["Leiden 0.5", "Died in less than 28 days"],
        ... )
        >>> ep.tl.draw_graph(edata, init_pos="paga")
        >>> ep.pl.draw_graph(edata, color=["leiden_0_5", "icu_exp_flg"], legend_loc="on data")

    Preview:
        .. image:: /_static/docstring_previews/draw_graph_1.png

        .. image:: /_static/docstring_previews/draw_graph_2.png
    """
    return sc.pl.draw_graph(_as_scanpy_input(edata), layout=layout, gene_symbols=feature_symbols, **kwargs)


@function_2D_only(var_keys=("color",))
@_doc_params(**doc_plot_params)
def embedding(
    edata: EHRData,
    basis: str,
    *,
    color: str | Sequence[str] | None = None,
    mask_obs: np.ndarray | str | None = None,
    feature_symbols: str | None = None,
    use_raw: bool | None = None,
    sort_order: bool = True,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: str | Sequence[float] | Sequence[str] = "grey",
    neighbors_key: str | None = None,
    arrows: bool = False,
    arrows_kwds: Mapping[str, Any] | None = None,
    groups: str | Sequence[str] | None = None,
    components: str | Sequence[str] | None = None,
    dimensions: tuple[int, int] | Sequence[tuple[int, int]] | None = None,
    layer: str | None = None,
    projection: Literal["2d", "3d"] = "2d",
    scale_factor: float | None = None,
    color_map: Colormap | str | None = None,
    cmap: Colormap | str | None = None,
    palette: str | Sequence[str] | Cycler | None = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    size: float | Sequence[float] | None = None,
    frameon: bool | None = None,
    legend_fontsize: float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_loc: str | None = "right margin",
    legend_fontoutline: int | None = None,
    colorbar_loc: Literal["right", "left", "top", "bottom"] | None = "right",
    vmax: VBound | Sequence[VBound] | None = None,
    vmin: VBound | Sequence[VBound] | None = None,
    vcenter: VBound | Sequence[VBound] | None = None,
    norm: Normalize | Sequence[Normalize] | None = None,
    add_outline: bool | None = False,
    outline_width: tuple[float, float] = (0.3, 0.05),
    outline_color: tuple[str, str] = ("black", "white"),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: float | None = None,
    title: str | Sequence[str] | None = None,
    show: bool | None = None,
    ax: Axes | None = None,
    return_fig: bool | None = None,
    marker: str | Sequence[str] = ".",
    **kwargs,
) -> Figure | Axes | list[Axes] | None:  # pragma: no cover
    """Scatter plot for user specified embedding basis (e.g. umap, pca, etc).

    Args:
        edata: Central data object.
        basis: Name of the `obsm` basis to use.
        color: {color}
        mask_obs: A boolean array or a string mask expression to subset observations.
        feature_symbols: {feature_symbols}
        use_raw: Use `.raw` attribute of `edata` for coloring with feature values.
            If `None`, defaults to `True` if `layer` isn't provided and `edata.raw` is present.
        sort_order: {sort_order}
        edges: Show edges.
        edges_width: Width of edges.
        edges_color: Color of edges.
            See :func:`~networkx.drawing.nx_pylab.draw_networkx_edges`.
        neighbors_key: Where to look for neighbors connectivities.
            If not specified, this looks .obsp['connectivities'] for connectivities (default storage place for pp.neighbors).
            If specified, this looks at `.obsp[.uns[neighbors_key]['connectivities_key']]` for connectivities.
        arrows: Show arrows (deprecated in favour of `scvelo.pl.velocity_embedding`).
        arrows_kwds: Passed to :meth:`~matplotlib.axes.Axes.quiver`.
        groups: {groups}
        components: {components}
        dimensions: 0-indexed dimensions of the embedding to plot as integers, e.g. `[(0, 1), (1, 2)]`.
            Unlike `components`, this argument is used in the same way as `color`, e.g. is used to specify a single plot at a time.
            Will eventually replace the `components` argument.
        layer: {layer}
        projection: {projection}
        scale_factor: Scaling factor of the coordinates.
        color_map: {color_map}
        cmap: Alias of `color_map`.
        palette: {palette}
        na_color: Color to use for null or masked values.
            Can be anything matplotlib accepts as a color.
            Used for all points if `color=None`.
        na_in_legend: If there are missing values, whether they get an entry in the legend.
            Currently only implemented for categorical legends.
        size: {size}
        frameon: {frameon}
        legend_fontsize: {legend_fontsize}
        legend_fontweight: {legend_fontweight}
        legend_loc: {legend_loc}
        legend_fontoutline: {legend_fontoutline}
        colorbar_loc: Where to place the colorbar for continuous variables.
            If `None`, no colorbar is added.
        vmax: {vbound_vmax}
        vmin: {vbound_vmin}
        vcenter: {vbound_vcenter}
        norm: {norm}
        add_outline: If set to `True`, this will add a thin border around groups of dots.
            In some situations this can enhance the aesthetics of the resulting image.
        outline_width: Tuple with two width numbers used to adjust the outline.
            The first value is the width of the border color as a fraction of the scatter dot size (default: 0.3).
            The second value is width of the gap color (default: 0.05).
        outline_color: Tuple with two valid color names used to adjust the add_outline.
            The first color is the border color (default: black), while the second color is a gap color between the border color and the scatter dot (default: white).
        ncols: {ncols}
        hspace: {hspace}
        wspace: {wspace}
        title: {panel_title}
        show: {show}
        ax: {ax}
        return_fig: {return_fig}
        marker: {marker}
        **kwargs: Arguments to pass to :func:`matplotlib.pyplot.scatter`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.umap(edata)
        >>> ep.pl.embedding(edata, basis="X_umap", color="icu_exp_flg")

    Preview:
        .. image:: /_static/docstring_previews/embedding.png
    """
    embedding_partial = partial(
        sc.pl.embedding,
        basis=basis,
        mask_obs=mask_obs,
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
        dimensions=dimensions,
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
        colorbar_loc=colorbar_loc,
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
        ax=ax,
        return_fig=return_fig,
        marker=marker,
        **kwargs,
    )

    return embedding_partial(adata=_as_scanpy_input(edata), color=color)


@_doc_params(**doc_plot_params)
def embedding_density(
    edata: EHRData,
    *,
    basis: str = "umap",
    key: str | None = None,
    groupby: str | None = None,
    group: str | Sequence[str] | None = "all",
    color_map: Colormap | str = "YlOrRd",
    bg_dotsize: int | None = 80,
    fg_dotsize: int | None = 180,
    vmax: int | None = 1,
    vmin: int | None = 0,
    vcenter: int | None = None,
    norm: Normalize | None = None,
    ncols: int | None = 4,
    hspace: float | None = 0.25,
    wspace: float | None = None,
    title: str | None = None,
    show: bool | None = None,
    ax: Axes | None = None,
    return_fig: bool | None = None,
    **kwargs,
) -> Figure | Axes | None:  # pragma: no cover
    """Plot the density of observations in an embedding (per condition).

    Plots the gaussian kernel density estimates (over condition) from the :func:`~ehrapy.tools.embedding_density` output.

    Args:
        edata: Central data object.
        basis: The embedding over which the density was calculated.
            This embedded representation should be found in `edata.obsm['X_[basis]']`.
        key: Name of the `.obs` covariate that contains the density estimates.
            Alternatively, pass `groupby`.
        groupby: Name of the condition used in :func:`~ehrapy.tools.embedding_density`.
            Alternatively, pass `key`.
        group: The category in the categorical observation annotation to be plotted.
            If all categories are to be plotted use `group='all'` (default).
            If multiple categories want to be plotted use a list, e.g. `['FICU', 'MICU']`.
            If the overall density wants to be plotted set `group` to `None`.
        color_map: Matplotlib color map to use for density plotting.
        bg_dotsize: Dot size for background data points not in the `group`.
        fg_dotsize: Dot size for foreground data points in the `group`.
        vmax: {vbound_vmax}
        vmin: {vbound_vmin}
        vcenter: {vbound_vcenter}
        norm: {norm}
        ncols: {ncols}
        hspace: {hspace}
        wspace: {wspace}
        title: {panel_title}
        show: {show}
        ax: {ax}
        return_fig: {return_fig}
        **kwargs: Arguments to pass to :func:`matplotlib.pyplot.scatter`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.umap(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.tl.embedding_density(edata, groupby="leiden_0_5")
        >>> ep.pl.embedding_density(edata, key="umap_density_leiden_0_5")

    Preview:
        .. image:: /_static/docstring_previews/embedding_density.png
    """
    return sc.pl.embedding_density(
        edata,
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
        ax=ax,
        return_fig=return_fig,
        **kwargs,
    )


@_doc_params(**doc_plot_params)
def dpt_groups_pseudotime(
    edata: EHRData,
    *,
    color_map: str | Colormap | None = None,
    palette: Sequence[str] | Cycler | None = None,
    show: bool | None = None,
    marker: str | Sequence[str] = ".",
    return_fig: bool = False,
) -> Figure | None:  # pragma: no cover
    """Plot groups and pseudotime.

    Args:
        edata: Central data object.
        color_map: {color_map}
        palette: {palette}
        show: {show}
        marker: {marker}
        return_fig: {return_fig}

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata, method="gauss")
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.tl.diffmap(edata, n_comps=10)
        >>> edata.uns["iroot"] = np.flatnonzero(edata.obs["leiden_0_5"] == "0")[0]
        >>> ep.tl.dpt(edata, n_branchings=3)
        >>> ep.pl.dpt_groups_pseudotime(edata)

    Preview:
        .. image:: /_static/docstring_previews/dpt_groups_pseudotime.png
    """
    return sc.pl.dpt_groups_pseudotime(
        adata=edata, color_map=color_map, palette=palette, show=show, marker=marker, return_fig=return_fig
    )


@function_2D_only()
@_doc_params(**doc_plot_params)
def dpt_timeseries(
    edata: EHRData,
    *,
    color_map: str | Colormap | None = None,
    as_heatmap: bool = True,
    marker: str | Sequence[str] = ".",
    show: bool | None = None,
) -> None:  # pragma: no cover
    """Heatmap of pseudotime series.

    Args:
        edata: Central data object.
        color_map: {color_map}
        as_heatmap: Plot the timeseries as heatmap.
        marker: Marker style if `as_heatmap` is `False`.
            See :mod:`~matplotlib.markers` for details.
        show: {show}

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata, method="gauss")
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.tl.diffmap(edata, n_comps=10)
        >>> edata.uns["iroot"] = np.flatnonzero(edata.obs["leiden_0_5"] == "0")[0]
        >>> ep.tl.dpt(edata, n_branchings=3)
        >>> ep.pl.dpt_timeseries(edata)

    Preview:
        .. image:: /_static/docstring_previews/dpt_timeseries.png
    """
    sc.pl.dpt_timeseries(
        adata=_as_scanpy_input(edata, dense=True), color_map=color_map, as_heatmap=as_heatmap, marker=marker, show=show
    )


@function_2D_only(var_keys=("color",))
def paga(
    edata: EHRData,
    *,
    threshold: float | None = None,
    color: str | Sequence[str] | Mapping[str | int, Mapping[Any, float]] | None = None,
    layout: _IGraphLayout | None = None,
    layout_kwds: Mapping[str, Any] = MappingProxyType({}),
    init_pos: np.ndarray | None = None,
    root: int | str | Sequence[int] | None = 0,
    labels: str | Sequence[str] | Mapping[str, str] | None = None,
    single_component: bool = False,
    solid_edges: str = "connectivities",
    dashed_edges: str | None = None,
    transitions: str | None = None,
    fontsize: int | None = None,
    fontweight: str = "bold",
    fontoutline: int | None = None,
    text_kwds: Mapping[str, Any] = MappingProxyType({}),
    node_size_scale: float = 1.0,
    node_size_power: float = 0.5,
    edge_width_scale: float = 1.0,
    min_edge_width: float | None = None,
    max_edge_width: float | None = None,
    arrowsize: int = 30,
    title: str | Sequence[str] | None = None,
    left_margin: float = 0.01,
    random_state: int | None = 0,
    pos: np.ndarray | str | Path | None = None,
    normalize_to_color: bool = False,
    cmap: str | Colormap | None = None,
    cax: Axes | None = None,
    cb_kwds: Mapping[str, Any] = MappingProxyType({}),
    frameon: bool | None = None,
    add_pos: bool = True,
    export_to_gexf: bool = False,
    use_raw: bool = True,
    plot: bool = True,
    show: bool | None = None,
    ax: Axes | None = None,
) -> Axes | list[Axes] | None:  # pragma: no cover
    """Plot the PAGA graph through thresholding low-connectivity edges.

    Compute a coarse-grained layout of the data.
    Reuse this by passing `init_pos='paga'` to :func:`~ehrapy.tools.umap` or :func:`~ehrapy.tools.draw_graph` and obtain embeddings with more meaningful global topology :cite:p:`Wolf2019`.
    This uses ForceAtlas2 or igraph's layout algorithms for most layouts :cite:p:`Csardi2006`.

    Args:
        edata: Central data object.
        threshold: Do not draw edges for weights below this threshold.
            Set to 0 if you want all edges.
            Discarding low-connectivity edges helps in getting a much clearer picture of the graph.
        color: Feature name or `obs` annotation defining the node colors, or a list of them to plot multiple panels.
            Also plots the degree of the abstracted graph when passing {`'degree_dashed'`, `'degree_solid'`}.
            Can be also used to visualize pie chart at each node in the following form: `{<group name or index>: {<color>: <fraction>, ...}, ...}`.
            If the fractions do not sum to 1, a new category called `'rest'` colored grey will be created.
        layout: Plotting layout that computes positions.
            `'fa'` stands for “ForceAtlas2”, `'fr'` stands for “Fruchterman-Reingold”, `'rt'` stands for “Reingold-Tilford”, `'eq_tree'` stands for “eqally spaced tree”.
            All but `'fa'` and `'eq_tree'` are igraph layouts.
            All other igraph layouts are also permitted.
            See also parameter `pos` and :func:`~ehrapy.tools.draw_graph`.
        layout_kwds: Keywords for the layout.
        init_pos: Two-column array storing the x and y coordinates for initializing the layout.
        root: If choosing a tree layout, this is the index of the root node or a list of root node indices.
            If this is a non-empty vector then the supplied node IDs are used as the roots of the trees (or a single tree if the graph is connected).
            If this is `None` or an empty list, the root vertices are automatically calculated based on topological sorting.
        labels: The node labels.
            If `None`, this defaults to the group labels stored in the categorical for which :func:`~ehrapy.tools.paga` has been computed.
        single_component: Restrict to largest connected component.
        solid_edges: Key for `.uns['paga']` that specifies the matrix that stores the edges to be drawn solid black.
        dashed_edges: Key for `.uns['paga']` that specifies the matrix that stores the edges to be drawn dashed grey.
            If `None`, no dashed edges are drawn.
        transitions: Key for `.uns['paga']` that specifies the matrix that stores the arrows, for instance `'transitions_confidence'`.
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
            See :class:`matplotlib.patches.FancyArrowPatch` for attribute `mutation_scale` for more info.
        title: Provide a title.
        left_margin: Margin to the left of the plot.
        random_state: For layouts with random initialization like `'fr'`, change this to use different intial states for the optimization.
            If `None`, the initial state is not reproducible.
        pos: Two-column array-like storing the x and y coordinates for drawing.
            Otherwise, path to a `.gdf` file that has been exported from Gephi or a similar graph visualization software.
        normalize_to_color: Whether to normalize categorical plots to `color` or the underlying grouping.
        cmap: The Matplotlib color map.
        cax: A matplotlib axes object for a potential colorbar.
        cb_kwds: Keyword arguments for :class:`~matplotlib.colorbar.Colorbar`, for instance, `ticks`.
        frameon: Draw a frame around the PAGA graph.
        add_pos: Add the positions to `edata.uns['paga']`.
        export_to_gexf: Export to gexf format to be read by graph visualization programs such as Gephi.
        use_raw: Whether to use `raw` attribute of `edata` if present.
        plot: If `False`, do not create the figure, simply compute the layout.
        show: Show the plot, do not return axis.
        ax: A matplotlib axes object.

    Returns:
        If `show` is `False`, one or more :class:`~matplotlib.axes.Axes` objects.
        Adds `'pos'` to `edata.uns['paga']` if `add_pos` is `True`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.tl.paga(edata, groups="leiden_0_5")
        >>> ep.pl.paga(
        ...     edata,
        ...     color=["leiden_0_5", "day_28_flg"],
        ...     cmap=ep.pl.Colormaps.grey_red.value,
        ...     title=["Leiden 0.5", "Died in less than 28 days"],
        ... )

    Preview:
        .. image:: /_static/docstring_previews/paga.png
    """
    return sc.pl.paga(
        adata=_as_scanpy_input(edata),
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
        ax=ax,
    )


@function_2D_only(var_keys=("keys",))
def paga_path(
    edata: EHRData,
    nodes: Sequence[str | int],
    keys: Sequence[str],
    *,
    use_raw: bool = True,
    annotations: Sequence[str] = ("dpt_pseudotime",),
    color_map: str | Colormap | None = None,
    color_maps_annotations: Mapping[str, str | Colormap] = MappingProxyType({"dpt_pseudotime": "Greys"}),
    palette_groups: Sequence[str] | None = None,
    n_avg: int = 1,
    groups_key: str | None = None,
    xlim: tuple[int | None, int | None] = (None, None),
    title: str | None = None,
    left_margin: float | None = None,
    ytick_fontsize: int | None = None,
    title_fontsize: int | None = None,
    show_node_names: bool = True,
    show_yticks: bool = True,
    show_colorbar: bool = True,
    legend_fontsize: float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight | None = None,
    normalize_to_zero_one: bool = False,
    as_heatmap: bool = True,
    return_data: bool = False,
    show: bool | None = None,
    ax: Axes | None = None,
) -> tuple[Axes, pd.DataFrame] | Axes | pd.DataFrame | None:  # pragma: no cover
    """Feature changes along paths in the abstracted graph.

    Args:
        edata: Central data object.
        nodes: A path through nodes of the abstracted graph, that is, names or indices (within `.categories`) of groups that have been used to run PAGA.
        keys: Either variables in `edata.var_names` or annotations in `edata.obs`.
            They are plotted using `color_map`.
        use_raw: Use `edata.raw` for retrieving feature values if it has been set.
        annotations: Plot these keys with `color_maps_annotations`.
            Need to be keys for `edata.obs`.
        color_map: Matplotlib colormap.
        color_maps_annotations: Color maps for plotting the annotations.
            Keys of the dictionary must appear in `annotations`.
        palette_groups: Colors of the groups, usually the same palette as used for coloring the abstracted graph.
        n_avg: Number of data points to include in computation of running average.
        groups_key: Key of the grouping used to run PAGA.
            If `None`, defaults to `edata.uns['paga']['groups']`.
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
        as_heatmap: Plot the timeseries as heatmap.
            If not plotting as heatmap, `annotations` have no effect.
        return_data: Whether to return the timeseries data in addition to the axes if `True`.
        show: Show the plot, do not return axis.
        ax: A matplotlib axes object.

    Returns:
        A :class:`~matplotlib.axes.Axes` object, if `ax` is `None`, else `None`.
        If `return_data`, return the timeseries data in addition to an axes.
    """
    if isinstance(edata.X, DaskArray):
        var_names = edata.var_names.intersection(keys)
        (X,) = _materialize(to_dense(edata.X[:, edata.var_names.get_indexer(var_names)]))
        adata = ed.EHRData(X, obs=edata.obs, var=edata.var.loc[var_names])
        adata.uns = edata.uns
        edata = adata
    return sc.pl.paga_path(
        adata=_as_scanpy_input(edata),
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
        ax=ax,
    )


@function_2D_only(var_keys=("color",))
@_doc_params(**doc_plot_params)
def paga_compare(
    edata: EHRData,
    *,
    basis: str | None = None,
    edges: bool = False,
    color: str | Sequence[str] | None = None,
    alpha: float | None = None,
    groups: str | Sequence[str] | None = None,
    components: str | Sequence[str] | None = None,
    projection: Literal["2d", "3d"] = "2d",
    legend_loc: str | None = "on data",
    legend_fontsize: float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_fontoutline: int | None = None,
    color_map: str | Colormap | None = None,
    palette: str | Sequence[str] | Cycler | None = None,
    frameon: bool | None = False,
    size: float | Sequence[float] | None = None,
    title: str | None = None,
    right_margin: float | None = None,
    left_margin: float = 0.05,
    show: bool | None = None,
    title_graph: str | None = None,
    groups_graph: str | Sequence[str] | Mapping[str, str] | None = None,
    pos: np.ndarray | str | Path | None = None,
    **paga_graph_params,
) -> list[Axes] | None:  # pragma: no cover
    """Scatter and PAGA graph side-by-side.

    Consists in a scatter plot and the abstracted graph.
    See :func:`~ehrapy.plot.paga` for all related parameters.

    Args:
        edata: Central data object.
        basis: String that denotes a plotting tool that computed coordinates.
        edges: Whether to display edges.
        color: {color}
        alpha: Opacity of the points.
        groups: {groups}
        components: {components}
        projection: {projection}
        legend_loc: {legend_loc}
        legend_fontsize: {legend_fontsize}
        legend_fontweight: {legend_fontweight}
        legend_fontoutline: {legend_fontoutline}
        color_map: {color_map}
        palette: {palette}
        frameon: {frameon}
        size: {size}
        title: Title of the scatter plot.
        right_margin: Margin to the right of the plot.
        left_margin: Margin to the left of the plot.
        show: {show}
        title_graph: Title of the PAGA graph.
        groups_graph: Node labels of the PAGA graph, passed as `labels` to :func:`~ehrapy.plot.paga`.
        pos: Two-column array-like storing the x and y coordinates of the PAGA nodes.
            Otherwise, path to a `.gdf` file that has been exported from Gephi or a similar graph visualization software.
        **paga_graph_params: Keyword arguments of :func:`~ehrapy.plot.paga`.

    Returns:
        A list of :class:`~matplotlib.axes.Axes` if `show` is `False`.
    """
    return sc.pl.paga_compare(
        adata=_as_scanpy_input(edata),
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
        title_graph=title_graph,
        groups_graph=groups_graph,
        pos=pos,
        **paga_graph_params,
    )


@_doc_params(**doc_plot_params)
def rank_features_groups(
    edata: EHRData,
    *,
    groups: str | Sequence[str] | None = None,
    n_features: int = 20,
    feature_symbols: str | None = None,
    key: str = "rank_features_groups",
    fontsize: int = 8,
    ncols: int = 4,
    share_y: bool = True,
    show: bool | None = None,
    ax: Axes | None = None,
) -> list[Axes] | None:  # pragma: no cover
    """Plot ranking of features.

    Args:
        edata: Central data object.
        groups: {rank_groups}
        n_features: Number of features to show.
        feature_symbols: {feature_symbols}
        key: {rank_key}
        fontsize: Fontsize for feature names.
        ncols: Number of panels shown per row.
        share_y: Controls if the y-axis of each panels should be shared.
            By passing `share_y=False`, each panel has its own y-axis range.
        show: {show}
        ax: {ax}

    Returns:
        List of each group's matplotlib axis or `None` if `show=True`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.15, key_added="leiden_0_5")
        >>> ep.tl.rank_features_groups(edata, groupby="leiden_0_5")
        >>> ep.pl.rank_features_groups(edata)

    Preview:
        .. image:: /_static/docstring_previews/rank_features_groups.png
    """
    return sc.pl.rank_genes_groups(
        adata=edata,
        groups=groups,
        n_genes=n_features,
        gene_symbols=feature_symbols,
        key=key,
        fontsize=fontsize,
        ncols=ncols,
        sharey=share_y,
        show=show,
        ax=ax,
    )


@function_2D_only()
@_doc_params(**doc_plot_params)
def rank_features_groups_violin(
    edata: EHRData,
    *,
    groups: Sequence[str] | None = None,
    n_features: int = 20,
    var_names: Iterable[str] | None = None,
    feature_symbols: str | None = None,
    key: str = "rank_features_groups",
    split: bool = True,
    density_norm: Literal["area", "count", "width"] = "width",
    strip: bool = True,
    jitter: float | bool = True,
    size: int = 1,
    ax: Axes | None = None,
    show: bool | None = None,
) -> list[Axes] | None:  # pragma: no cover
    """Plot ranking of features for all tested comparisons as violin plots.

    Args:
        edata: Central data object.
        groups: {rank_groups}
        n_features: Number of features to show.
            Is ignored if `var_names` is passed.
        var_names: List of features to plot.
            Is only useful if interested in a custom feature list, which is not the result of :func:`~ehrapy.tools.rank_features_groups`.
        feature_symbols: {feature_symbols}
        key: {rank_key}
        split: Whether to split the violins or not.
        density_norm: See :func:`~seaborn.violinplot`.
        strip: Show a strip plot on top of the violin plot.
        jitter: If set to 0, no points are drawn.
            See :func:`~seaborn.stripplot`.
        size: Size of the jitter points.
        ax: {ax}
        show: {show}

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.15, key_added="leiden_0_5")
        >>> ep.tl.rank_features_groups(edata, groupby="leiden_0_5")
        >>> ep.pl.rank_features_groups_violin(edata, n_features=5)

    Preview:
        .. image:: /_static/docstring_previews/rank_features_groups_violin_1.png

        .. image:: /_static/docstring_previews/rank_features_groups_violin_2.png

        .. image:: /_static/docstring_previews/rank_features_groups_violin_3.png

        .. image:: /_static/docstring_previews/rank_features_groups_violin_4.png
    """
    return sc.pl.rank_genes_groups_violin(
        adata=_as_scanpy_input(edata),
        groups=groups,
        n_genes=n_features,
        gene_names=var_names,
        gene_symbols=feature_symbols,
        use_raw=False,
        key=key,
        split=split,
        density_norm=density_norm,
        strip=strip,
        jitter=jitter,
        size=size,
        ax=ax,
        show=show,
    )


@function_2D_only()
@_doc_params(**doc_plot_params)
def rank_features_groups_stacked_violin(
    edata: EHRData,
    *,
    groups: str | Sequence[str] | None = None,
    n_features: int | None = None,
    groupby: str | None = None,
    feature_symbols: str | None = None,
    var_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    min_logfoldchange: float | None = None,
    key: str = "rank_features_groups",
    show: bool | None = None,
    return_fig: bool = False,
    **kwds,
) -> StackedViolin | dict | None:  # pragma: no cover
    """Plot ranking of features using stacked_violin plot.

    Args:
        edata: Central data object.
        groups: {rank_groups}
        n_features: {rank_n_features}
        groupby: {rank_groupby}
        feature_symbols: {feature_symbols}
        var_names: {rank_var_names}
        min_logfoldchange: {min_logfoldchange}
        key: {rank_key}
        show: {show}
        return_fig: Returns :class:`~scanpy.pl.StackedViolin` object.
            Useful for fine-tuning the plot.
            Takes precedence over `show=False`.
        **kwds: Keyword arguments of :func:`scanpy.pl.stacked_violin`.

    Returns:
        If `return_fig` is `True`, returns a :class:`~scanpy.pl.StackedViolin` object, else if `show` is false, return axes dict

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.15, key_added="leiden_0_5")
        >>> ep.tl.rank_features_groups(edata, groupby="leiden_0_5")
        >>> ep.pl.rank_features_groups_stacked_violin(edata, n_features=5)

    Preview:
        .. image:: /_static/docstring_previews/rank_features_groups_stacked_violin.png
    """
    return sc.pl.rank_genes_groups_stacked_violin(
        adata=_as_scanpy_input(edata),
        groups=groups,
        n_genes=n_features,
        groupby=groupby,
        gene_symbols=feature_symbols,
        var_names=var_names,
        min_logfoldchange=min_logfoldchange,
        key=key,
        show=show,
        return_fig=return_fig,
        **kwds,
    )


@function_2D_only()
@_doc_params(**doc_plot_params)
def rank_features_groups_heatmap(
    edata: EHRData,
    *,
    groups: str | Sequence[str] | None = None,
    n_features: int | None = None,
    groupby: str | None = None,
    feature_symbols: str | None = None,
    var_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    min_logfoldchange: float | None = None,
    key: str = "rank_features_groups",
    show: bool | None = None,
    **kwds,
) -> dict[str, Axes] | None:  # pragma: no cover
    """Plot ranking of features using heatmap plot (see :func:`~ehrapy.plot.heatmap`).

    Args:
        edata: Central data object.
        groups: {rank_groups}
        n_features: {rank_n_features}
        groupby: {rank_groupby}
        feature_symbols: {feature_symbols}
        var_names: {rank_var_names}
        min_logfoldchange: {min_logfoldchange}
        key: {rank_key}
        show: {show}
        **kwds: Keyword arguments of :func:`scanpy.pl.heatmap`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.15, key_added="leiden_0_5")
        >>> ep.tl.rank_features_groups(edata, groupby="leiden_0_5")
        >>> ep.pl.rank_features_groups_heatmap(edata)

    Preview:
        .. image:: /_static/docstring_previews/rank_features_groups_heatmap.png
    """
    return sc.pl.rank_genes_groups_heatmap(
        adata=_as_scanpy_input(edata),
        groups=groups,
        n_genes=n_features,
        groupby=groupby,
        gene_symbols=feature_symbols,
        var_names=var_names,
        min_logfoldchange=min_logfoldchange,
        key=key,
        show=show,
        **kwds,
    )


@function_2D_only()
@_doc_params(**doc_plot_params)
def rank_features_groups_dotplot(
    edata: EHRData,
    *,
    groups: str | Sequence[str] | None = None,
    n_features: int | None = None,
    groupby: str | None = None,
    values_to_plot: _ValuesToPlot | None = None,
    var_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    feature_symbols: str | None = None,
    min_logfoldchange: float | None = None,
    key: str = "rank_features_groups",
    show: bool | None = None,
    return_fig: bool = False,
    **kwds,
) -> DotPlot | dict | None:  # pragma: no cover
    """Plot ranking of features using dotplot plot (see :func:`~ehrapy.plot.dotplot`).

    Args:
        edata: Central data object.
        groups: {rank_groups}
        n_features: {rank_n_features}
        groupby: {rank_groupby}
        values_to_plot: {values_to_plot}
        var_names: {rank_var_names}
        feature_symbols: {feature_symbols}
        min_logfoldchange: {min_logfoldchange}
        key: {rank_key}
        show: {show}
        return_fig: Returns :class:`~scanpy.pl.DotPlot` object.
            Useful for fine-tuning the plot.
            Takes precedence over `show=False`.
        **kwds: Keyword arguments of :func:`scanpy.pl.dotplot`.

    Returns:
        If `return_fig` is `True`, returns a :class:`~scanpy.pl.DotPlot` object, else if `show` is false, return axes dict

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.tl.rank_features_groups(edata, groupby="leiden_0_5")
        >>> ep.pl.rank_features_groups_dotplot(edata, groupby="leiden_0_5")

    Preview:
        .. image:: /_static/docstring_previews/rank_features_groups_dotplot.png
    """
    if values_to_plot is None:
        kwds.setdefault("colorbar_title", "Mean value in group")
    kwds.setdefault("size_title", "Fraction of observations\nin group (%)")
    return sc.pl.rank_genes_groups_dotplot(
        adata=_as_scanpy_input(edata),
        groups=groups,
        n_genes=n_features,
        groupby=groupby,
        values_to_plot=values_to_plot,
        var_names=var_names,
        gene_symbols=feature_symbols,
        min_logfoldchange=min_logfoldchange,
        key=key,
        show=show,
        return_fig=return_fig,
        **kwds,
    )


@function_2D_only()
@_doc_params(**doc_plot_params)
def rank_features_groups_matrixplot(
    edata: EHRData,
    *,
    groups: str | Sequence[str] | None = None,
    n_features: int | None = None,
    groupby: str | None = None,
    values_to_plot: _ValuesToPlot | None = None,
    var_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    feature_symbols: str | None = None,
    min_logfoldchange: float | None = None,
    key: str = "rank_features_groups",
    show: bool | None = None,
    return_fig: bool = False,
    **kwds,
) -> MatrixPlot | dict | None:  # pragma: no cover
    """Plot ranking of features using matrixplot plot (see :func:`~ehrapy.plot.matrixplot`).

    Args:
        edata: Central data object.
        groups: {rank_groups}
        n_features: {rank_n_features}
        groupby: {rank_groupby}
        values_to_plot: {values_to_plot}
        var_names: {rank_var_names}
        feature_symbols: {feature_symbols}
        min_logfoldchange: {min_logfoldchange}
        key: {rank_key}
        show: {show}
        return_fig: Returns :class:`~scanpy.pl.MatrixPlot` object.
            Useful for fine-tuning the plot.
            Takes precedence over `show=False`.
        **kwds: Keyword arguments of :func:`scanpy.pl.matrixplot`.

    Returns:
        If `return_fig` is `True`, returns a :class:`~scanpy.pl.MatrixPlot` object, else if `show` is false, return axes dict

    Example:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.5, key_added="leiden_0_5")
        >>> ep.tl.rank_features_groups(edata, groupby="leiden_0_5")
        >>> ep.pl.rank_features_groups_matrixplot(edata, groupby="leiden_0_5")

    Preview:
        .. image:: /_static/docstring_previews/rank_features_groups_matrixplot.png

    """
    return sc.pl.rank_genes_groups_matrixplot(
        adata=_as_scanpy_input(edata),
        groups=groups,
        n_genes=n_features,
        groupby=groupby,
        values_to_plot=values_to_plot,
        var_names=var_names,
        gene_symbols=feature_symbols,
        min_logfoldchange=min_logfoldchange,
        key=key,
        show=show,
        return_fig=return_fig,
        **kwds,
    )


@function_2D_only()
@_doc_params(**doc_plot_params)
def rank_features_groups_tracksplot(
    edata: EHRData,
    *,
    groups: str | Sequence[str] | None = None,
    n_features: int | None = None,
    groupby: str | None = None,
    var_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    feature_symbols: str | None = None,
    min_logfoldchange: float | None = None,
    key: str = "rank_features_groups",
    show: bool | None = None,
    **kwds,
) -> dict[str, Axes] | None:  # pragma: no cover
    """Plot ranking of features using tracksplot plot (see :func:`~ehrapy.plot.tracksplot`).

    Args:
        edata: Central data object.
        groups: {rank_groups}
        n_features: {rank_n_features}
        groupby: {rank_groupby}
        var_names: {rank_var_names}
        feature_symbols: {feature_symbols}
        min_logfoldchange: {min_logfoldchange}
        key: {rank_key}
        show: {show}
        **kwds: Keyword arguments of :func:`scanpy.pl.tracksplot`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata, resolution=0.15, key_added="leiden_0_5")
        >>> ep.tl.rank_features_groups(edata, groupby="leiden_0_5")
        >>> ep.pl.rank_features_groups_tracksplot(edata)

    Preview:
        .. image:: /_static/docstring_previews/rank_features_groups_tracksplot.png
    """
    return sc.pl.rank_genes_groups_tracksplot(
        adata=_as_scanpy_input(edata),
        groups=groups,
        n_genes=n_features,
        groupby=groupby,
        var_names=var_names,
        gene_symbols=feature_symbols,
        min_logfoldchange=min_logfoldchange,
        key=key,
        show=show,
        **kwds,
    )
