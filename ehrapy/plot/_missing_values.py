from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import holoviews as hv
import numpy as np
import pandas as pd
import scipy.sparse as sp
from array_api_compat import array_namespace, is_lazy_array
from fast_array_utils import stats
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase

from ehrapy._compat import _materialize
from ehrapy.plot._holoviews import load_hv_extensions
from ehrapy.plot._timeseries import _resolve_axis
from ehrapy.preprocessing._missing_data import _missing_mask

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from ehrdata import EHRData
    from fast_array_utils.types import DaskArray

    type Array = np.ndarray | CSBase | DaskArray

# more observations are grouped into this many rows, which keeps the matrix responsive in the browser
_MAX_ROWS = 1000


@load_hv_extensions()
def missing_values_matrix(
    edata: EHRData,
    *,
    var_names: str | Sequence[str] | None = None,
    layer: str | None = None,
    categoricals: bool = False,
    width: int | None = 600,
    height: int | None = 400,
    title: str | None = None,
) -> hv.QuadMesh:
    """Plot the percentage of observed values of every variable as a matrix.

    For 2D data, the rows are the observations, and more than 1000 observations are grouped into 1000 rows of consecutive observations.
    For 3D data, the rows are the timepoints.

    Args:
        edata: Central data object.
        var_names: Variable name or list of variable names in `edata.var_names` to plot.
            If `None`, all variables are plotted.
        layer: The layer to use.
            If `None`, `edata.X` is used.
        categoricals: Whether to include the "ehrapycat" variables of encoded categorical variables if `var_names` is `None`.
        width: Plot width in pixels.
        height: Plot height in pixels.
        title: Set the title of the plot.

    Returns:
        HoloViews QuadMesh with the variables on the x-axis.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pl.missing_values_matrix(edata)

        .. image:: /_static/docstring_previews/missing_values_matrix.png
    """
    missing, columns = _missing(edata, var_names=var_names, layer=layer, categoricals=categoricals)
    if missing.ndim == 3:
        xp = array_namespace(missing)
        (fraction,) = _materialize(xp.mean(xp.astype(missing, xp.float64), axis=0))
        observed, edges, row_label = 100 * (1 - fraction.T), np.arange(missing.shape[2] + 1) - 0.5, "timepoint"
    else:
        edges = np.linspace(0, missing.shape[0], min(missing.shape[0], _MAX_ROWS) + 1).astype(int)
        (counts,) = _materialize(_binned_sum(missing, edges))
        observed, row_label = 100 * (1 - counts / np.diff(edges)[:, None]), "observation"
    return hv.QuadMesh(
        (np.arange(len(columns) + 1) - 0.5, edges, observed), kdims=["variable", row_label], vdims="observed (%)"
    ).opts(
        cmap="Greys",
        clim=(0, 100),
        colorbar=True,
        invert_yaxis=True,
        xticks=list(enumerate(str(column) for column in columns)),
        xrotation=45,
        tools=["hover"],
        **_size(width, height, title),
    )


@load_hv_extensions()
def missing_values_barplot(
    edata: EHRData,
    *,
    var_names: str | Sequence[str] | None = None,
    layer: str | None = None,
    categoricals: bool = False,
    width: int | None = 600,
    height: int | None = 400,
    title: str | None = None,
) -> hv.Bars:
    """Plot the percentage of observed values of every variable as bars.

    For 3D data, every timepoint of every observation counts as one value.

    Args:
        edata: Central data object.
        var_names: Variable name or list of variable names in `edata.var_names` to plot.
            If `None`, all variables are plotted.
        layer: The layer to use.
            If `None`, `edata.X` is used.
        categoricals: Whether to include the "ehrapycat" variables of encoded categorical variables if `var_names` is `None`.
        width: Plot width in pixels.
        height: Plot height in pixels.
        title: Set the title of the plot.

    Returns:
        HoloViews Bars with one bar per variable.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pl.missing_values_barplot(edata)

        .. image:: /_static/docstring_previews/missing_values_barplot.png
    """
    missing, columns = _missing(edata, var_names=var_names, layer=layer, categoricals=categoricals)
    (fraction,) = _materialize(stats.mean(_rows(missing), axis=0, dtype=np.float64))
    table = pd.DataFrame({"variable": np.asarray(columns, dtype=str), "observed (%)": 100 * (1 - fraction)})
    return hv.Bars(table, kdims="variable", vdims="observed (%)").opts(
        ylim=(0, 100), xrotation=45, tools=["hover"], **_size(width, height, title)
    )


@load_hv_extensions()
def missing_values_heatmap(
    edata: EHRData,
    *,
    var_names: str | Sequence[str] | None = None,
    layer: str | None = None,
    categoricals: bool = False,
    width: int | None = 600,
    height: int | None = 500,
    title: str | None = None,
) -> hv.HeatMap:
    """Plot how strongly the missing values of every pair of variables coincide.

    The heatmap shows the correlation of the missing value masks of the variables.
    Variables that are never or always missing are left out.
    For 3D data, every timepoint of every observation counts as one row.

    Args:
        edata: Central data object.
        var_names: Variable name or list of variable names in `edata.var_names` to plot.
            If `None`, all variables are plotted.
        layer: The layer to use.
            If `None`, `edata.X` is used.
        categoricals: Whether to include the "ehrapycat" variables of encoded categorical variables if `var_names` is `None`.
        width: Plot width in pixels.
        height: Plot height in pixels.
        title: Set the title of the plot.

    Returns:
        HoloViews HeatMap of the correlations.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pl.missing_values_heatmap(edata)

        .. image:: /_static/docstring_previews/missing_values_heatmap.png
    """
    missing, columns = _missing(edata, var_names=var_names, layer=layer, categoricals=categoricals)
    rows = _rows(missing)
    n_missing, co_missing = _co_missing(rows)
    fraction = n_missing / rows.shape[0]
    varies = (fraction > 0) & (fraction < 1)
    deviation = np.sqrt(fraction * (1 - fraction))[varies]
    covariance = (co_missing / rows.shape[0] - np.outer(fraction, fraction))[np.ix_(varies, varies)]
    names = np.asarray(columns, dtype=str)[varies]
    table = pd.DataFrame(
        {
            "variable": np.tile(names, len(names)),
            "other variable": np.repeat(names, len(names)),
            "correlation": (covariance / np.outer(deviation, deviation)).ravel(),
        }
    )
    return hv.HeatMap(
        table, kdims=[_ordered("variable", names), _ordered("other variable", names[::-1])], vdims="correlation"
    ).opts(cmap="RdBu_r", clim=(-1, 1), colorbar=True, xrotation=45, tools=["hover"], **_size(width, height, title))


@load_hv_extensions()
def missing_values_dendrogram(
    edata: EHRData,
    *,
    method: str = "average",
    var_names: str | Sequence[str] | None = None,
    layer: str | None = None,
    categoricals: bool = False,
    width: int | None = 600,
    height: int | None = 400,
    title: str | None = None,
) -> hv.Path:
    """Cluster the variables by their missing values and plot the hierarchy as a dendrogram.

    Variables are linked by the Euclidean distance between their missing value masks with :func:`scipy.cluster.hierarchy.linkage`.
    For 3D data, every timepoint of every observation counts as one row.

    Args:
        edata: Central data object.
        method: The linkage method passed to :func:`scipy.cluster.hierarchy.linkage`.
        var_names: Variable name or list of variable names in `edata.var_names` to plot.
            If `None`, all variables are plotted.
        layer: The layer to use.
            If `None`, `edata.X` is used.
        categoricals: Whether to include the "ehrapycat" variables of encoded categorical variables if `var_names` is `None`.
        width: Plot width in pixels.
        height: Plot height in pixels.
        title: Set the title of the plot.

    Returns:
        HoloViews Path of the dendrogram with the variables on the x-axis.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pl.missing_values_dendrogram(edata)

        .. image:: /_static/docstring_previews/missing_values_dendrogram.png
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    missing, columns = _missing(edata, var_names=var_names, layer=layer, categoricals=categoricals)
    n_missing, co_missing = _co_missing(_rows(missing))
    distances = np.sqrt(np.maximum(n_missing[:, None] + n_missing[None, :] - 2 * co_missing, 0))
    linkage = hierarchy.linkage(squareform(distances, checks=False), method=method)
    tree = hierarchy.dendrogram(linkage, no_plot=True, labels=[str(column) for column in columns])
    paths = [np.column_stack([x, y]) for x, y in zip(tree["icoord"], tree["dcoord"], strict=True)]
    ticks = [(5 + 10 * position, label) for position, label in enumerate(tree["ivl"])]
    return hv.Path(paths, kdims=["variable", "distance"]).opts(
        xticks=ticks, xrotation=45, color="black", **_size(width, height, title)
    )


def _missing(
    edata: EHRData, *, var_names: str | Sequence[str] | None, layer: str | None, categoricals: bool
) -> tuple[Array, pd.Index]:
    """The missing value mask of the plotted variables and their names."""
    X = edata.X if layer is None else edata.layers[layer]
    if var_names is not None:
        _, columns = _resolve_axis(pd.Index(edata.var_names), var_names, "var_names")
    else:
        columns = edata.var_names if categoricals else edata.var_names[~edata.var_names.str.startswith("ehrapycat")]
    missing = _missing_mask(X[:, edata.var_names.get_indexer(columns)])
    # the mask is small enough to densify, and sparse dask chunks support neither cumulative sums nor products
    return (to_dense(missing) if is_lazy_array(missing) else missing), columns


def _rows(missing: Array) -> Array:
    """Every timepoint of every observation of a 3D mask as a row."""
    if missing.ndim != 3:
        return missing
    xp = array_namespace(missing)
    return xp.reshape(xp.permute_dims(missing, (0, 2, 1)), (-1, missing.shape[1]))


def _co_missing(rows: Array) -> tuple[np.ndarray, np.ndarray]:
    """The number of missing values of every variable and of every pair of variables."""
    values = rows.astype(np.float64)
    return tuple(_materialize(stats.sum(values, axis=0), to_dense(values.T @ values)))


@singledispatch
def _binned_sum(missing: np.ndarray | DaskArray, edges: np.ndarray) -> np.ndarray | DaskArray:
    """The number of missing values of every variable in the consecutive rows between `edges`."""
    xp = array_namespace(missing)
    cumulative = xp.cumulative_sum(xp.astype(missing, xp.int64), axis=0, include_initial=True)
    return cumulative[edges[1:]] - cumulative[edges[:-1]]


@_binned_sum.register(CSBase)
def _(missing: CSBase, edges: np.ndarray) -> np.ndarray:
    rows = np.arange(missing.shape[0])
    groups = np.repeat(np.arange(len(edges) - 1), np.diff(edges))
    indicator = sp.csr_array((np.ones(len(rows)), (groups, rows)), shape=(len(edges) - 1, len(rows)))
    return to_dense(indicator @ missing.astype(np.float64))


def _ordered(name: str, values: Sequence[Any]) -> hv.Dimension:
    return hv.Dimension(name, values=list(np.asarray(values, dtype=str)))


def _size(width: int | None, height: int | None, title: str | None) -> dict[str, Any]:
    return {
        key: value for key, value in {"width": width, "height": height, "title": title}.items() if value is not None
    }
