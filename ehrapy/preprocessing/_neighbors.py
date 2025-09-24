from __future__ import annotations

from collections.abc import Callable
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scanpy as sc

from ehrapy._compat import function_2D_only, use_ehrdata
from ehrapy.tools.distances.dtw import dtw_distance

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anndata import AnnData
    from ehrdata import EHRData
    from scanpy.neighbors import KnnTransformerLike

    from ehrapy.preprocessing._types import AnyRandom, KnownTransformer


_Method = Literal["umap", "gauss"]
_MetricFn = Callable[[np.ndarray, np.ndarray], float]
_MetricSparseCapable = Literal["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]
_MetricScipySpatial = Literal[
    "dwt",  # not yet sparse capable
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "dice",
    "hamming",
    "jaccard",
    "kulsinski",
    "mahalanobis",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
_Metric = _MetricSparseCapable | _MetricScipySpatial


@function_2D_only()
@use_ehrdata(deprecated_after="1.0.0")
def neighbors(
    edata: EHRData | AnnData,
    *,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    use_rep: str | None = None,
    knn: bool = True,
    random_state: AnyRandom = 0,
    method: _Method = "umap",
    transformer: KnnTransformerLike | KnownTransformer | None = None,
    metric: _Metric | _MetricFn = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    key_added: str | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:  # pragma: no cover
    """Compute a neighborhood graph of observations :cite:p:`McInnes2018`.

    The neighbor search efficiency of this heavily relies on UMAP :cite:p:`McInnes2018`,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`).
    If `method=='gauss'`, connectivities are computed according to :cite:p:`Coifman2005`, in the adaption of :cite:p:`Haghverdi2016`.

    Args:
        edata: Central data object.
        n_neighbors: The size of local neighborhood (in terms of number of neighboring data points) used for manifold approximation.
                     Larger values result in more global views of the manifold, while smaller values result in more local data being preserved.
                     In general values should be in the range 2 to 100. If `knn` is `True`, number of nearest neighbors to be searched.
                     If `knn` is `False`, a Gaussian kernel width is set to the distance of the `n_neighbors` neighbor.
        n_pcs: Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
        use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
                 If `None`, the representation is chosen automatically:
                 For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
                 If 'X_pca' is not present, it's computed with default parameters.
        knn: If `True`, use a hard threshold to restrict the number of neighbors to `n_neighbors`, that is, consider a knn graph.
             Otherwise, use a Gaussian Kernel to assign low weights to neighbors more distant than the `n_neighbors` nearest neighbor.
        random_state: A numpy random seed.
        method: Use 'umap' :cite:p:`McInnes2018` or 'gauss' (Gauss kernel following :cite:p:`Coifman2005` with adaptive width :cite:p:`Haghverdi2016` for computing connectivities.
                Use 'rapids' for the RAPIDS implementation of UMAP (experimental, GPU only).
        metric: A known metric's name or a callable that returns a distance.
        metric_kwds: Options for the metric.
        transformer: Approximate kNN search implementation. Follows the API of :class:`~sklearn.neighbors.KNeighborsTransformer`.
                See scanpy's `knn-transformers tutorial <https://scanpy.readthedocs.io/en/latest/how-to/knn-transformers.html>`_ for more details.
                This tutorial is also valid for ehrapy's `neighbors` function.
                Next to the advanced options from the knn-transformers tutorial, this argument accepts the following basic options:

                `None` (the default)
                    Behavior depends on data size.
                    For small data, uses :class:`~sklearn.neighbors.KNeighborsTransformer` with algorithm="brute" for exact kNN, otherwise uses
                    :class:`~pynndescent.pynndescent_.PyNNDescentTransformer` for approximate kNN.
                `'pynndescent'`
                    Uses :class:`~pynndescent.pynndescent_.PyNNDescentTransformer` for approximate kNN.
                `'sklearn'`
                    Uses :class:`~sklearn.neighbors.KNeighborsTransformer` with algorithm="brute" for exact kNN.
        key_added: If not specified, the neighbors data is stored in .uns['neighbors'],
                   distances and connectivities are stored in .obsp['distances'] and .obsp['connectivities'] respectively.
                   If specified, the neighbors data is added to .uns[key_added], distances are stored in .obsp[key_added+'_distances']
                   and connectivities in .obsp[key_added+'_connectivities'].
        copy: Determines whether a copy of `edata` is returned.

    Returns:
         Depending on `copy`, updates or returns `edata` with the following;
         See `key_added` parameter description for the storage path of connectivities and distances.

         **connectivities** : sparse matrix of dtype `float32`.
         Weighted adjacency matrix of the neighborhood graph of data points. Weights should be interpreted as connectivities.

         **distances** : sparse matrix of dtype `float32`.
         Instead of decaying weights, this stores distances for each pair of neighbors.
    """
    if metric == "dwt":
        if edata.R is None:
            raise ValueError("metric 'dwt' requires edata.R to be set")
        metric = partial(dtw_distance, R=edata.R)

    return sc.pp.neighbors(
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
