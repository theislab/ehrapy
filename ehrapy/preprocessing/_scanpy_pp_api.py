from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from scanpy._utils import AnyRandom
from scipy.sparse import spmatrix


def pca(
    data: Union[AnnData, np.ndarray, spmatrix],
    n_comps: Optional[int] = None,
    zero_center: Optional[bool] = True,
    svd_solver: str = "arpack",
    random_state: AnyRandom = 0,
    return_info: bool = False,
    dtype: str = "float32",
    copy: bool = False,
    chunked: bool = False,
    chunk_size: Optional[int] = None,
) -> Union[AnnData, np.ndarray, spmatrix]:  # pragma: no cover
    """Computes a principal component analysis.

    Computes PCA coordinates, loadings and variance decomposition. Uses the implementation of *scikit-learn*.

    Args:
        data: The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond to observations and columns to features.
        n_comps: Number of principal components to compute.
                 Defaults to 50, or 1 - minimum dimension size of selected representation.
        zero_center: If `True`, compute standard PCA from covariance matrix.
                     If `False`, omit zero-centering variables (uses :class:`~sklearn.decomposition.TruncatedSVD`), which allows to handle sparse input efficiently.
                     Passing `None` decides automatically based on sparseness of the data.
        svd_solver: SVD solver to use:

                    * `'arpack'` (the default) for the ARPACK wrapper in SciPy (:func:`~scipy.sparse.linalg.svds`)

                    * `'randomized'` for the randomized algorithm due to Halko (2009).

                    * `'auto'` chooses automatically depending on the size of the problem.

                    * `'lobpcg'` An alternative SciPy solver.

                    Efficient computation of the principal components of a sparse matrix currently only works with the `'arpack`' or `'lobpcg'` solvers.
        random_state: Change to use different initial states for the optimization. Defaults to 0 .
        return_info: Only relevant when not passing an :class:`~anndata.AnnData`: see “**Returns**”.
        dtype: Numpy data type string to which to convert the result. Defaults to float32 .
        copy: If an :class:`~anndata.AnnData` is passed, determines whether a copy is returned. Is ignored otherwise.
        chunked: If `True`, perform an incremental PCA on segments of `chunk_size`.
                  The incremental PCA automatically zero centers and ignores settings of
                  `random_seed` and `svd_solver`. If `False`, perform a full PCA.
        chunk_size: Number of observations to include in each chunk. Required if `chunked=True` was passed.

    Returns:
        :X_pca: :class:`~scipy.sparse.spmatrix`, :class:`~numpy.ndarray`

        If `data` is array-like and `return_info=False` was passed, this function only returns `X_pca`...

        adata : :class:`~anndata.AnnData`

        …otherwise if `copy=True` it returns or else adds fields to `adata`:

        `.obsm['X_pca']`
        PCA representation of data.

        `.varm['PCs']`
        The principal components containing the loadings.

        `.uns['pca']['variance_ratio']`
        Ratio of explained variance.

        `.uns['pca']['variance']`
        Explained variance, equivalent to the eigenvalues of the covariance matrix.
    """
    return sc.pp.pca(
        data=data,
        n_comps=n_comps,
        zero_center=zero_center,
        svd_solver=svd_solver,
        random_state=random_state,
        return_info=return_info,
        use_highly_variable=False,
        dtype=dtype,
        copy=copy,
        chunked=chunked,
        chunk_size=chunk_size,
    )


def regress_out(
    adata: AnnData,
    keys: Union[str, Sequence[str]],
    n_jobs: Optional[int] = None,
    copy: bool = False,
) -> Optional[AnnData]:  # pragma: no cover
    """Regress out (mostly) unwanted sources of variation.

    Uses simple linear regression. This is inspired by Seurat's `regressOut` function in R [Satija15].
    Note that this function tends to overcorrect in certain circumstances.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        keys: Keys for observation annotation on which to regress on.
        n_jobs: Number of jobs for parallel computation. `None` means using :attr:`scanpy._settings.ScanpyConfig.n_jobs`.
        copy: Determines whether a copy of `adata` is returned.

    Returns:
        Depending on `copy` returns or updates an :class:`~anndata.AnnData` object with the corrected data matrix.
    """
    return sc.pp.regress_out(adata=adata, keys=keys, n_jobs=n_jobs, copy=copy)


def subsample(
    data: Union[AnnData, np.ndarray, spmatrix],
    fraction: Optional[float] = None,
    n_obs: Optional[int] = None,
    random_state: AnyRandom = 0,
    copy: bool = False,
) -> Optional[AnnData]:  # pragma: no cover
    """Subsample to a fraction of the number of observations.

    Args:
        data: The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond to observations (patients) and columns to features.
        fraction: Subsample to this `fraction` of the number of observations.
        n_obs: Subsample to this number of observations.
        random_state: Random seed to change subsampling.
        copy: If an :class:`~anndata.AnnData` is passed, determines whether a copy is returned.

    Returns:
        Returns `X[obs_indices], obs_indices` if data is array-like, otherwise subsamples the passed
        :class:`~anndata.AnnData` (`copy == False`) or returns a subsampled copy of it (`copy == True`).
    """
    return sc.pp.subsample(data=data, fraction=fraction, n_obs=n_obs, random_state=random_state, copy=copy)


def combat(
    adata: AnnData,
    key: str = "batch",
    covariates: Optional[Collection[str]] = None,
    inplace: bool = True,
) -> Union[AnnData, np.ndarray, None]:  # pragma: no cover
    """ComBat function for batch effect correction [Johnson07]_ [Leek12]_ [Pedersen12]_.

    Corrects for batch effects by fitting linear models, gains statistical power via an EB framework where information is borrowed across features.
    This uses the implementation `combat.py`_ [Pedersen12]_.

    .. _combat.py: https://github.com/brentp/combat.py

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        key: Key to a categorical annotation from :attr:`~anndata.AnnData.obs` that will be used for batch effect removal.
        covariates: Additional covariates besides the batch variable such as adjustment variables or biological condition.
                    This parameter refers to the design matrix `X` in Equation 2.1 in [Johnson07]_ and to the `mod` argument in
                    the original combat function in the sva R package.
                    Note that not including covariates may introduce bias or lead to the removal of signal in unbalanced designs.
        inplace: Whether to replace adata.X or to return the corrected data

    Returns:
        Depending on the value of `inplace`, either returns the corrected matrix or modifies `adata.X`.
    """
    return sc.pp.combat(adata=adata, key=key, covariates=covariates, inplace=inplace)


_Method = Literal["umap", "gauss", "rapids"]
_MetricFn = Callable[[np.ndarray, np.ndarray], float]
_MetricSparseCapable = Literal["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]
_MetricScipySpatial = Literal[
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
_Metric = Union[_MetricSparseCapable, _MetricScipySpatial]


def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    knn: bool = True,
    random_state: AnyRandom = 0,
    method: Optional[_Method] = "umap",
    metric: Union[_Metric, _MetricFn] = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    key_added: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:  # pragma: no cover
    """Compute a neighborhood graph of observations [McInnes18]_.

    The neighbor search efficiency of this heavily relies on UMAP [McInnes18]_,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`). If `method=='gauss'`,
    connectivities are computed according to [Coifman05]_, in the adaption of [Haghverdi16]_.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        n_neighbors: The size of local neighborhood (in terms of number of neighboring data points) used for manifold approximation.
                     Larger values result in more global views of the manifold, while smaller values result in more local data being preserved.
                     In general values should be in the range 2 to 100. If `knn` is `True`, number of nearest neighbors to be searched.
                     If `knn` is `False`, a Gaussian kernel width is set to the distance of the `n_neighbors` neighbor.
        n_pcs: Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
        use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
                 If `None`, the representation is chosen automatically:
                 For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
                 If 'X_pca' is not present, it’s computed with default parameters.
        knn: If `True`, use a hard threshold to restrict the number of neighbors to `n_neighbors`, that is, consider a knn graph.
             Otherwise, use a Gaussian Kernel to assign low weights to neighbors more distant than the `n_neighbors` nearest neighbor.
        random_state: A numpy random seed.
        method: Use 'umap' [McInnes18]_ or 'gauss' (Gauss kernel following [Coifman05]_ with adaptive width [Haghverdi16]_) for computing connectivities.
                Use 'rapids' for the RAPIDS implementation of UMAP (experimental, GPU only).
        metric: A known metric’s name or a callable that returns a distance.
        metric_kwds: Options for the metric.
        key_added: If not specified, the neighbors data is stored in .uns['neighbors'],
                   distances and connectivities are stored in .obsp['distances'] and .obsp['connectivities'] respectively.
                   If specified, the neighbors data is added to .uns[key_added], distances are stored in .obsp[key_added+'_distances']
                   and connectivities in .obsp[key_added+'_connectivities'].
        copy: Determines whether a copy of `adata` is returned.

    Returns:
         Depending on `copy`, updates or returns `adata` with the following;
         See `key_added` parameter description for the storage path of connectivities and distances.

         **connectivities** : sparse matrix of dtype `float32`.
         Weighted adjacency matrix of the neighborhood graph of data points. Weights should be interpreted as connectivities.

         **distances** : sparse matrix of dtype `float32`.
         Instead of decaying weights, this stores distances for each pair of neighbors.
    """
    return sc.pp.neighbors(
        adata=adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        knn=knn,
        random_state=random_state,
        method=method,
        metric=metric,
        metric_kwds=metric_kwds,
        key_added=key_added,
        copy=copy,
    )
