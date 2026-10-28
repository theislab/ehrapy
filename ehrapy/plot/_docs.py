from inspect import cleandoc


def _doc_params(**kwds):  # pragma: no cover
    r"""Docstrings should start with "\" in the first line for proper formatting."""

    def dec(obj):
        obj.__orig_doc__ = obj.__doc__
        obj.__doc__ = cleandoc(obj.__doc__).format_map(kwds)
        return obj

    return dec


def _lines(*sentences: str) -> str:
    return "\n        ".join(sentences)


"""\
Shared descriptions of plotting function parameters, referenced as `name: {key}` in the docstring Args.
"""


doc_plot_params = {
    # parameters of the grouped plots (heatmap, dotplot, tracksplot, stacked_violin, matrixplot)
    "var_names": _lines(
        "`var_names` should be a valid subset of `edata.var_names`.",
        "If `var_names` is a mapping, then the key is used as label to group the values (see `var_group_labels`).",
        "The mapping values should be sequences of valid `edata.var_names`.",
        "In this case either coloring or 'brackets' are used for the grouping of var names depending on the plot.",
        "When `var_names` is a mapping, then the `var_group_labels` and `var_group_positions` are set.",
    ),
    "groupby": "The key of the observation grouping to consider.",
    "log": "Plot on logarithmic axis.",
    "num_categories": _lines(
        "Only used if groupby observation is not categorical.",
        "This value determines the number of groups into which the groupby observation should be subdivided.",
    ),
    "categories_order": _lines(
        "Order in which to show the categories.",
        "Note: add_dendrogram or add_totals can change the categories order.",
    ),
    "figsize": _lines(
        "Figure size when `multi_panel=True`.",
        "Otherwise the `rcParams['figure.figsize']` value is used.",
        "Format is (width, height).",
    ),
    "dendrogram": _lines(
        "If `True` or a valid dendrogram key, a dendrogram based on the hierarchical clustering between the `groupby` categories is added.",
        "The dendrogram information is computed using :func:`~ehrapy.tools.dendrogram`.",
        "If :func:`~ehrapy.tools.dendrogram` has not been called previously, the function is called with default parameters.",
    ),
    "var_group_positions": _lines(
        "Use this parameter to highlight groups of `var_names`.",
        "This will draw a 'bracket' or a color block between the given start and end positions.",
        "If the parameter `var_group_labels` is set, the corresponding labels are added on top/left.",
        "E.g. `var_group_positions=[(4,10)]` will add a bracket between the fourth `var_name` and the tenth `var_name`.",
        "By giving more positions, more brackets/color blocks are drawn.",
    ),
    "var_group_labels": "Labels for each of the `var_group_positions` that want to be highlighted.",
    "var_group_rotation": _lines(
        "Label rotation degrees.",
        "By default, labels larger than 4 characters are rotated 90 degrees.",
    ),
    "layer": _lines(
        "Name of the `edata` layer to plot.",
        "By default `edata.X` is plotted.",
    ),
    "title": "Title for the figure.",
    "colorbar_title": _lines("Title for the color bar.", "New line character (\\n) can be used."),
    "cmap": "String denoting matplotlib color map.",
    "standard_scale": _lines(
        "Whether or not to standardize the given dimension between 0 and 1.",
        "For each variable or group, subtract the minimum and divide each by its maximum.",
    ),
    "swap_axes": _lines(
        "By default, the x axis contains `var_names` (e.g. features) and the y axis the `groupby` categories.",
        "By setting `swap_axes` then x are the `groupby` categories and y the `var_names`.",
    ),
    "vmin": _lines(
        "The value representing the lower limit of the color scale.",
        "Values smaller than vmin are plotted with the same color as vmin.",
    ),
    "vmax": _lines(
        "The value representing the upper limit of the color scale.",
        "Values larger than vmax are plotted with the same color as vmax.",
    ),
    "vcenter": _lines("The value representing the center of the color scale.", "Useful for diverging colormaps."),
    "norm": _lines(
        "Custom color normalization object from matplotlib.",
        "See :ref:`matplotlib:colormapnorms` for details.",
    ),
    # parameters of the scatter plots (scatter, embedding and the embedding wrappers, paga_compare)
    "color": "Keys for annotations of observations or features, e.g., `'ann1'` or `['ann1', 'ann2']`.",
    "sort_order": "For continuous annotations used as color parameter, plot data points with higher values on top of others.",
    "groups": _lines(
        "Restrict to a few categories in categorical observation annotation.",
        "The default is not to restrict to any groups.",
    ),
    "components": _lines(
        "For instance, `['1,2', '2,3']`.",
        "To plot all available components use `components='all'`.",
    ),
    "projection": "Projection of plot.",
    "legend_loc": "Location of legend, either `'on data'`, `'right margin'`, `None`, or a valid keyword for the `loc` parameter of :class:`~matplotlib.legend.Legend`.",
    "legend_fontsize": _lines(
        "Numeric size in pt or string describing the size.",
        "See :meth:`~matplotlib.text.Text.set_fontsize`.",
    ),
    "legend_fontweight": _lines(
        "Legend font weight.",
        "A numeric value in range 0-1000 or a string.",
        "Defaults to `'bold'` if `legend_loc == 'on data'`, otherwise to `'normal'`.",
        "See :meth:`~matplotlib.text.Text.set_fontweight`.",
    ),
    "legend_fontoutline": _lines(
        "Line width of the legend font outline in pt.",
        "Draws a white outline using the path effect :class:`~matplotlib.patheffects.withStroke`.",
    ),
    "color_map": _lines(
        "Color map to use for continuous variables.",
        'Can be a name or a :class:`~matplotlib.colors.Colormap` instance (e.g. `"magma"`, `"viridis"` or `mpl.cm.cividis`), see :meth:`~matplotlib.cm.ColormapRegistry.get_cmap`.',
        'If `None`, the value of `mpl.rcParams["image.cmap"]` is used.',
    ),
    "palette": _lines(
        "Colors to use for plotting categorical annotation groups.",
        "The palette can be a valid :class:`~matplotlib.colors.ListedColormap` name (`'Set2'`, `'tab20'`, …), a :class:`~cycler.Cycler` object, a dict mapping categories to colors, or a sequence of colors.",
        "Colors must be valid to matplotlib (see :func:`~matplotlib.colors.is_color_like`).",
        'If `None`, `mpl.rcParams["axes.prop_cycle"]` is used unless the categorical variable already has colors stored in `edata.uns["{var}_colors"]`.',
        'If provided, values of `edata.uns["{var}_colors"]` will be set.',
    ),
    "frameon": _lines(
        "Draw a frame around the scatter plot.",
        "Defaults to `True`.",
    ),
    "size": _lines(
        "Point size.",
        "If `None`, is automatically computed as 120000 / n_obs.",
        "Can be a sequence containing the size for each observation.",
        "The order should be the same as in `edata.obs`.",
    ),
    "marker": _lines("Marker style.", "See :mod:`~matplotlib.markers` for details."),
    "panel_title": "Provide title for panels either as string or list of strings, e.g. `['title1', 'title2', ...]`.",
    "vbound_vmin": _lines(
        "The value representing the lower limit of the color scale.",
        "Values smaller than vmin are plotted with the same color as vmin.",
        "vmin can be a number, a string, a function or `None`.",
        "If vmin is a string and has the format `pN`, this is interpreted as a vmin=percentile(N).",
        "For example vmin='p1.5' is interpreted as the 1.5 percentile.",
        "If vmin is function, then vmin is interpreted as the return value of the function over the list of values to plot.",
        "For example to set vmin to the mean of the values to plot, `def my_vmin(values): return np.mean(values)` and then set `vmin=my_vmin`.",
        "If vmin is None (default) an automatic minimum value is used as defined by matplotlib `scatter` function.",
        "When making multiple plots, vmin can be a list of values, one for each plot.",
        "For example `vmin=[0.1, 'p1', None, my_vmin]`.",
    ),
    "vbound_vmax": _lines(
        "The value representing the upper limit of the color scale.",
        "The format is the same as for `vmin`.",
    ),
    "vbound_vcenter": _lines(
        "The value representing the center of the color scale.",
        "Useful for diverging colormaps.",
        "The format is the same as for `vmin`.",
        "Example: ``ep.pl.umap(edata, color='age', vcenter='p50', cmap='RdBu_r')``.",
    ),
    "ncols": "Number of panels per row.",
    "wspace": "Adjust the width of the space between multiple panels.",
    "hspace": "Adjust the height of the space between multiple panels.",
    "return_fig": "Return the matplotlib figure.",
    "embedding_kwargs": "Keyword arguments of :func:`~ehrapy.plot.embedding`, for example `color`, `layer`, `legend_loc`, `show` or `return_fig`.",
    # parameters of the plots of rank_features_groups results
    "rank_groups": "The groups for which to show the feature ranking.",
    "rank_n_features": _lines(
        "Number of features to show.",
        "This can be a negative number to show the bottom ranked features, e.g. `n_features=-10`.",
        "Mutually exclusive with `var_names`.",
    ),
    "rank_groupby": _lines(
        "The key of the observation grouping to consider.",
        "By default, the `groupby` of :func:`~ehrapy.tools.rank_features_groups` is used.",
        "If `groupby` is not a categorical observation, it is subdivided into `num_categories` (see :func:`~ehrapy.plot.dotplot`).",
    ),
    "rank_var_names": _lines(
        "Features to plot instead of the top ranked ones, either as a list or as a mapping of labels to lists as in :func:`~ehrapy.plot.dotplot`.",
        "Mutually exclusive with `n_features`.",
    ),
    "min_logfoldchange": "Only show features whose log fold change in a group is at least `min_logfoldchange`.",
    "rank_key": "Key in `edata.uns` under which the results of :func:`~ehrapy.tools.rank_features_groups` are stored.",
    "values_to_plot": _lines(
        "Instead of the mean feature value, plot the values computed by :func:`~ehrapy.tools.rank_features_groups`.",
        "The options are `'scores'`, `'logfoldchanges'`, `'pvals'`, `'pvals_adj'`, `'log10_pvals'` and `'log10_pvals_adj'`.",
        "When plotting log fold changes, a divergent colormap is recommended.",
    ),
    "show": "Show the plot, do not return axis.",
    "ax": _lines("A matplotlib axes object.", "Only works if plotting a single component."),
}
