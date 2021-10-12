from typing import Callable, Collection, Iterable, Literal, Optional, Sequence, Tuple, Union

import scanpy as sc
from anndata import AnnData
from cycler import Cycler
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap

from ehrapy.util._doc_util import _doc_params, doc_show_save_ax, doc_scatter_basic

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
