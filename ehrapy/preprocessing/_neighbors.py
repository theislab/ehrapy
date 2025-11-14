from __future__ import annotations

from collections.abc import Callable
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scanpy as sc

from ehrapy._compat import use_ehrdata
from ehrapy.core._constants import TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY
from ehrapy.tools.distances.timeseries import timeseries_distance

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anndata import AnnData
    from ehrdata import EHRData
    from scanpy.neighbors import KnnTransformerLike

    from ehrapy._types import AnyRandom, KnownTransformer


_Method = Literal["umap", "gauss"]
_MetricFn = Callable[[np.ndarray, np.ndarray], float]
_MetricSparseCapable = Literal["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]
_MetricTimeSeries = Literal["dtw", "soft_dtw", "gak"]  # these are not yet sparse capable
_MetricScipySpatial = Literal[
    "chebyshev",
    "correlation",
    "mahalanobis",
    "minkowski",
    "seuclidean",
    "sqeuclidean",
]
_Metric = _MetricSparseCapable | _MetricTimeSeries | _MetricScipySpatial


@use_ehrdata(deprecated_after="1.0.0")
def neighbors(
    edata: EHRData | AnnData,
    *,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    use_rep: str | None = None,
    knn: bool = True,
    method: _Method = "umap",
    transformer: KnnTransformerLike | KnownTransformer | None = None,
    metric: _Metric | _MetricFn = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    key_added: str | None = None,
    random_state: AnyRandom = 0,
    copy: bool = False,
) -> EHRData | AnnData | None:  # pragma: no cover
    """Compute a neighborhood graph of observations :cite:p:`McInnes2018`.

    The neighbor search efficiency of this heavily relies on UMAP :cite:p:`McInnes2018`,
    which also provides a method for estimating connectivities of data points - the connectivity of the manifold (`method=='umap'`).
    If `method=='gauss'`, connectivities are computed according to :cite:p:`Coifman2005`, in the adaption of :cite:p:`Haghverdi2016`.

    Args:
        edata: Central data object.
        n_neighbors: The size of local neighborhood (in terms of number of neighboring data points) used for manifold approximation.
                     Larger values result in more global views of the manifold, while smaller values result in more local data being preserved.
                     In general values should be in the range 2 to 100. If `knn` is `True`, number of nearest neighbors to be searched.
                     If `knn` is `False`, a Gaussian kernel width is set to the distance of the `n_neighbors` neighbor.
        n_pcs: Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
        use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid. For time series data (`metric='dtw'`, `'soft_dtw'`, `'gak'`), the key must be a 3D array with shape (n_obs, n_vars, n_timepoints).
                 If `None`, the representation is chosen automatically:
                 For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
                 If 'X_pca' is not present, it's computed with default parameters.
        knn: If `True`, use a hard threshold to restrict the number of neighbors to `n_neighbors`, that is, consider a knn graph.
             Otherwise, use a Gaussian Kernel to assign low weights to neighbors more distant than the `n_neighbors` nearest neighbor.
        random_state: A numpy random seed.
        method: Use 'umap' :cite:p:`McInnes2018` or 'gauss' (Gauss kernel following :cite:p:`Coifman2005` with adaptive width :cite:p:`Haghverdi2016` for computing connectivities.
        metric: A known metric's name or a callable that returns a distance.
            'euclidean' works well for 2D data and 'dtw' for 3D time series data.
        metric_kwds: Options for the metric.
        transformer: Approximate kNN search implementation.

                `None` (the default)
                    Behavior depends on data size.
                    For small data, uses :class:`~sklearn.neighbors.KNeighborsTransformer` with algorithm="brute" for exact kNN, otherwise uses
                    :class:`~pynndescent.pynndescent_.PyNNDescentTransformer` for approximate kNN.
                `'pynndescent'`
                    Uses :class:`~pynndescent.pynndescent_.PyNNDescentTransformer` for approximate kNN.
                `'sklearn'`
                    Uses :class:`~sklearn.neighbors.KNeighborsTransformer` with algorithm="brute" for exact kNN.
        key_added: If not specified, the neighbors data is stored in `.uns['neighbors']`,
                   distances and connectivities are stored in `.obsp['distances']` and `.obsp['connectivities']` respectively.
                   If specified, the neighbors data is added to `.uns[key_added]`, distances are stored in `.obsp[key_added+'_distances']`
                   and connectivities in `.obsp[key_added+'_connectivities']`.
        copy: Determines whether a copy of `edata` is returned.

    Returns:
         Depending on `copy`, updates or returns `edata` with the following;
         See `key_added` parameter description for the storage path of connectivities and distances.

         **connectivities** : sparse matrix of dtype `float32`.
         Weighted adjacency matrix of the neighborhood graph of data points. Weights should be interpreted as connectivities.

         **distances** : sparse matrix of dtype `float32`.
         Instead of decaying weights, this stores distances for each pair of neighbors.
    """
    import ehrapy as ep

    if metric in {"dtw", "soft_dtw", "gak"}:
        if use_rep is None:
            raise ValueError(f"use_rep must be specified if metric is {metric}")
        if use_rep in edata.layers:
            arr = edata.layers[use_rep]
        elif use_rep in edata.obsm:
            arr = edata.obsm[use_rep]
        else:
            raise ValueError(f"use_rep {use_rep} not found in edata.layers or edata.obsm")

        if arr.ndim != 3:
            raise ValueError(
                f"If metric is {metric}, use_rep must be a 3D array with shape (n_obs, n_vars, n_timepoints), but {arr} is ndim={arr.ndim}."
            )

        metric = partial(timeseries_distance, arr=arr, metric=metric)  # type: ignore

        # the metric will use arr, but we need to hide this fact from scanpy;
        # this is a hack to do so. It tricks scanpy's checks for the use_rep shap, while the metric brings along its array       use_rep = None

        use_rep = None

        # For longitudinal data, we pass a mock array `np.arange` with indices as `use_rep` to `scanpy.pp.neighbors`.
        # It will call the metric with the indices as the first two arguments.
        # Scanpy thinks that timeseries_distance computes distance on use_rep,
        # but actually timeseries_distance just takes the indices from use_rep and uses them to index the timeseries data along the obs axis.
        if use_rep is not None:
            raise ValueError(f"use_rep must be None when metric is {metric}")
        edata.obsm[TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY] = np.arange(edata.shape[0])
        use_rep = TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY

    edata_returned = sc.pp.neighbors(
        adata=edata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        knn=knn,
        random_state=random_state,
        method=method,
        transformer=transformer,
        metric=metric,
        metric_kwds=metric_kwds,
        key_added=key_added,
        copy=copy,
    )
    if edata_returned is not None:
        edata_returned.obsm.pop(TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY, None)

    edata.obsm.pop(TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY, None)

    return edata_returned
