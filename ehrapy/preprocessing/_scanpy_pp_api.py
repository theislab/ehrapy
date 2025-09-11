from __future__ import annotations

import warnings
from collections.abc import Callable
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import scanpy as sc
from anndata import AnnData

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from numpy.typing import NDArray
    from scanpy.neighbors import KnnTransformerLike
    from scipy.sparse import spmatrix

    from ehrapy.preprocessing._types import KnownTransformer

AnyRandom: TypeAlias = int | np.random.RandomState | None


def pca(
    data: AnnData | np.ndarray | spmatrix,
    n_comps: int | None = None,
    zero_center: bool | None = True,
    svd_solver: str = "arpack",
    random_state: AnyRandom = 0,
    return_info: bool = False,
    dtype: str = "float32",
    copy: bool = False,
    chunked: bool = False,
    chunk_size: int | None = None,
) -> AnnData | np.ndarray | spmatrix | None:  # pragma: no cover
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
        random_state: Change to use different initial states for the optimization.
        return_info: Only relevant when not passing an :class:`~anndata.AnnData`: see “**Returns**”.
        dtype: Numpy data type string to which to convert the result.
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
    keys: str | Sequence[str],
    n_jobs: int | None = None,
    copy: bool = False,
) -> AnnData | None:  # pragma: no cover
    """Regress out (mostly) unwanted sources of variation.

    Uses simple linear regression. This is inspired by Seurat's `regressOut` function in R [Satija15].
    Note that this function tends to overcorrect in certain circumstances.

    Args:
        adata: :class:`~anndata.AnnData` object containing all observations.
        keys: Keys for observation annotation on which to regress on.
        n_jobs: Number of jobs for parallel computation.
        copy: Determines whether a copy of `adata` is returned.

    Returns:
        Depending on `copy` returns or updates an :class:`~anndata.AnnData` object with the corrected data matrix.
    """
    return sc.pp.regress_out(adata=adata, keys=keys, n_jobs=n_jobs, copy=copy)


def subsample(
    data: AnnData | np.ndarray | spmatrix,
    fraction: float | None = None,
    n_obs: int | None = None,
    random_state: AnyRandom = 0,
    copy: bool = False,
) -> AnnData | None:  # pragma: no cover
    warnings.warn(
        "This function is deprecated and will be removed in the next release. Use ep.pp.sample instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sample(data=data, fraction=fraction, n_obs=n_obs, rng=random_state, copy=copy)


def sample(
    data: AnnData | np.ndarray,
    fraction: float | None = None,
    *,
    n_obs: int | None = None,
    rng: int | None = None,
    balanced: bool = False,
    method: Literal["under", "over"] = "under",
    key: str | None = None,
    copy: bool = False,
    replace: bool = False,
    axis: Literal["obs", 0, "var", 1] = "obs",
    p: str | NDArray[np.bool_] | NDArray[np.floating] | None = None,
) -> AnnData | None | tuple[np.ndarray, np.ndarray ]:  # pragma: no cover
    """Sample a fraction or a number of observations / variables with or without replacement.

    Args:
        data: The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond to observations (patients) and columns to features.
        fraction: Sample to this `fraction` of the number of observations.
        n_obs: Sample to this number of observations.
        rng: Random seed to change subsampling.
        copy: If an :class:`~anndata.AnnData` is passed, determines whether a copy is returned.
        balanced: If `True`, balance the groups in `adata.obs[key]` by under- or over-sampling.
                  Requires `key` to be set. If `False`, simple random sampling is performed.
        method: The sampling method, either "under" for under-sampling or "over" for over-sampling. Only relevant if `balanced=True`.
        key: Key in `adata.obs` to use for balancing the groups. Only relevant if `balanced=True`.
        replace: If `True`, samples are drawn with replacement. Only relevant if `balanced=False`.
        axis: Axis to sample on. Either `obs` / `0` (observations, default) or `var` / `1` (variables).
        p: Drawing probabilities (floats) or mask (bools).
            Either an `axis`-sized array, or the name of a column
            If p is an array of probabilities, it must sum to 1.

    Returns:
        Returns `X[obs_indices], obs_indices` if data is array-like, otherwise subsamples the passed
        :class:`~anndata.AnnData` (`copy == False`) or returns a subsampled copy of it (`copy == True`).
    """
    if balanced:
        if not isinstance(data, AnnData):
            raise TypeError(f"Input data is not an AnnData object: type of {data}, is {type(data)}")

        if key is None:
            raise TypeError("Key must be provided when balanced=True")

        if key not in data.obs.columns:
            raise ValueError(f"Key '{key}' not found in adata.obs. Available keys are: {data.obs.columns.tolist()}")

        labels = data.obs[key].values

        if method == "under" or method == "over":
            sampled_indices, sampled_labels = _random_resample(labels, method=method, random_state=rng)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        if copy:
            return data[sampled_indices].copy()
        else:
            data._inplace_subset_obs(sampled_indices)
            return None
    else:
        return sc.pp.sample(data=data, fraction=fraction, n=n_obs, rng=rng, copy=copy, replace=replace, axis=axis, p=p)


def combat(
    adata: AnnData,
    key: str = "batch",
    covariates: Collection[str] | None = None,
    inplace: bool = True,
) -> AnnData | np.ndarray | None:  # pragma: no cover
    """ComBat function for batch effect correction :cite:p:`Johnson2006`, :cite:p:`Leek2012`, :cite:p:`Pedersen2012`.

    Corrects for batch effects by fitting linear models, gains statistical power via an EB framework where information is borrowed across features.
    This uses the implementation `combat.py`:cite:p:`Pedersen2012`.

    .. _combat.py: https://github.com/brentp/combat.py

    Args:
        adata: :class:`~anndata.AnnData` object containing all observations.
        key: Key to a categorical annotation from :attr:`~anndata.AnnData.obs` that will be used for batch effect removal.
        covariates: Additional covariates besides the batch variable such as adjustment variables or biological condition.
                    This parameter refers to the design matrix `X` in Equation 2.1 in :cite:p:`Johnson2006` and to the `mod` argument in
                    the original combat function in the sva R package.
                    Note that not including covariates may introduce bias or lead to the removal of signal in unbalanced designs.
        inplace: Whether to replace adata.X or to return the corrected data

    Returns:
        Depending on the value of `inplace`, either returns the corrected matrix or modifies `adata.X`.
    """
    return sc.pp.combat(adata=adata, key=key, covariates=covariates, inplace=inplace)


_Method = Literal["umap", "gauss"]
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
_Metric = _MetricSparseCapable | _MetricScipySpatial


def neighbors(
    adata: AnnData,
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
) -> AnnData | None:  # pragma: no cover
    """Compute a neighborhood graph of observations :cite:p:`McInnes2018`.

    The neighbor search efficiency of this heavily relies on UMAP :cite:p:`McInnes2018`,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`). If `method=='gauss'`,
    connectivities are computed according to :cite:p:`Coifman2005`, in the adaption of :cite:p:`Haghverdi2016`.

    Args:
        adata: :class:`~anndata.AnnData` object containing all observations.
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
        transformer: Approximate kNN search implementation. Follows the API of
                :class:`~sklearn.neighbors.KNeighborsTransformer`.
                See scanpy's `knn-transformers tutorial <https://scanpy.readthedocs.io/en/latest/how-to/knn-transformers.html>`_ for more details. This tutorial is also valid for ehrapy's `neighbors` function.
                Next to the advanced options from the knn-transformers tutorial, this argument accepts the following basic options:

                `None` (the default)
                    Behavior depends on data size.
                    For small data, uses :class:`~sklearn.neighbors.KNeighborsTransformer` with algorithm="brute" for exact kNN, otherwise uses
                    :class:`~pynndescent.pynndescent_.PyNNDescentTransformer` for approximate kNN.
                `'pynndescent'`
                    Uses :class:`~pynndescent.pynndescent_.PyNNDescentTransformer` for approximate kNN.
                `'sklearn'`
                    Uses :class:`~sklearn.neighbors.KNeighborsTransformer` with algorithm="brute" for exact kNN.
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
        transformer=transformer,
        metric=metric,
        metric_kwds=metric_kwds,
        key_added=key_added,
        copy=copy,
    )


def _random_resample(
    label: str, target: str = "balanced", method: Literal["under", "over"] = "under", random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to under- or over-sample the data to achieve a balanced dataset.

    Args:
        label: The labels of the data.
        target: The target number of samples for each class. If "balanced", it will balance the classes to the minimum class size.
        method: The sampling method, either "under" for under-sampling or "over" for over-sampling.
        random_state: Random seed.

    Returns:
        A tuple of (sampled_indices, sampled_labels).
    """
    label = np.asarray(label)
    rnd = np.random.default_rng(random_state)
    classes, counts = np.unique(label, return_counts=True)

    if target == "balanced":
        if method == "under":
            target_count = counts.min()
        elif method == "over":
            target_count = counts.max()
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    indices = []

    for c in classes:
        class_idx = np.where(label == c)[0]
        n = len(class_idx)
        if method == "under":
            if n > target_count:
                sampled_idx = rnd.choice(class_idx, size=target_count, replace=False)
                indices.extend(sampled_idx)
            else:
                indices.extend(class_idx)
        elif method == "over":
            if n < target_count:
                sampled_idx = rnd.choice(class_idx, size=target_count, replace=True)
                indices.extend(sampled_idx)
            else:
                indices.extend(class_idx)

    sample_indices = np.array(indices)
    return sample_indices, label[sample_indices]
