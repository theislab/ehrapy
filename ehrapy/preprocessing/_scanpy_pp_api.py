from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import singledispatch
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from ehrdata import EHRData
from ehrdata.core.constants import MISSING_VALUES
from lamin_utils import logger

from ehrapy._compat import function_2D_only, use_ehrdata

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from numpy.typing import NDArray
    from scanpy.neighbors import KnnTransformerLike
    from scipy.sparse import spmatrix

    from ehrapy.preprocessing._types import AnyRandom, CSBase, KnownTransformer, RNGLike, SeedLike


@function_2D_only()
def pca(
    data: EHRData | AnnData | np.ndarray | spmatrix,
    *,
    n_comps: int | None = None,
    zero_center: bool | None = True,
    svd_solver: str = "arpack",
    random_state: AnyRandom = 0,
    return_info: bool = False,
    dtype: str = "float32",
    layer: str | None = None,
    copy: bool = False,
    chunked: bool = False,
    chunk_size: int | None = None,
) -> EHRData | AnnData | np.ndarray | spmatrix | None:  # pragma: no cover
    """Computes a principal component analysis.

    Computes PCA coordinates, loadings and variance decomposition. Uses the implementation of *scikit-learn*.

    Args:
        data: Central data object.
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
        return_info: Only relevant when not passing an :class:`~ehrdata.EHRData`: or :class:`~anndata.AnnData`: see “**Returns**”.
        dtype: Numpy data type string to which to convert the result.
        layer: The layer to operate on.
        copy: If an :class:`~ehrdata.EHRData`: or :class:`~anndata.AnnData`: is passed, determines whether a copy is returned. Is ignored otherwise.
        chunked: If `True`, perform an incremental PCA on segments of `chunk_size`.
                  The incremental PCA automatically zero centers and ignores settings of
                  `random_seed` and `svd_solver`. If `False`, perform a full PCA.
        chunk_size: Number of observations to include in each chunk. Required if `chunked=True` was passed.

    Returns:
        :X_pca: :class:`~scipy.sparse.spmatrix`, :class:`~numpy.ndarray`

        If `data` is array-like and `return_info=False` was passed, this function only returns `X_pca`...

        edata : :class:`~ehrdata.EHRData` or :class:`~anndata.AnnData`

        …otherwise if `copy=True` it returns or else adds fields to `edata`:

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
        layer=layer,
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


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
def regress_out(
    edata: EHRData | AnnData,
    *,
    keys: str | Sequence[str],
    n_jobs: int | None = None,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:  # pragma: no cover
    """Regress out (mostly) unwanted sources of variation.

    Uses simple linear regression. This is inspired by Seurat's `regressOut` function in R [Satija15].
    Note that this function tends to overcorrect in certain circumstances.

    Args:
        edata: Central data object.
        keys: Keys for observation annotation on which to regress on.
        n_jobs: Number of jobs for parallel computation.
        layer: The layer to operate on.
        copy: Determines whether a copy of `adata` is returned.

    Returns:
        Depending on `copy` returns or updates the data object with the corrected data matrix.
    """
    return sc.pp.regress_out(adata=edata, keys=keys, n_jobs=n_jobs, layer=layer, copy=copy)


def subsample(
    data: EHRData | AnnData | np.ndarray | spmatrix,
    *,
    fraction: float | None = None,
    n_obs: int | None = None,
    random_state: AnyRandom = 0,
    copy: bool = False,
) -> EHRData | AnnData | None:  # pragma: no cover
    warnings.warn(
        "This function is deprecated and will be removed in the next release. Use ep.pp.sample instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sample(data=data, fraction=fraction, n_obs=n_obs, rng=random_state, copy=copy)


def sample(
    data: EHRData | AnnData | np.ndarray | CSBase,
    fraction: float | None = None,
    *,
    n_obs: int | None = None,
    rng: RNGLike | SeedLike | None = None,
    balanced: bool = False,
    balanced_method: Literal["RandomUnderSampler", "RandomOverSampler"] = "RandomUnderSampler",
    balanced_key: str | None = None,
    copy: bool = False,
    replace: bool = False,
    axis: Literal["obs", 0, "var", 1] = "obs",
    p: str | NDArray[np.bool_] | NDArray[np.floating] | None = None,
) -> EHRData | AnnData | None | tuple[np.ndarray | CSBase, np.ndarray]:  # pragma: no cover
    """Sample a fraction or a number of observations / variables with or without replacement.

    Args:
        data: Central data object.
        fraction: Sample to this `fraction` of the number of observations.
        n_obs: Sample to this number of observations.
        rng: Random seed.
        copy: If an :class:`~anndata.AnnData` is passed, determines whether a copy is returned.
        balanced: If `True`, balance the groups in `adata.obs[key]` by under- or over-sampling.
                  Requires `key` to be set. If `False`, simple random sampling is performed.
        balanced_method: The sampling method, either "RandomUnderSampler" for under-sampling or "RandomOverSampler" for over-sampling. Only relevant if `balanced=True`.
        balanced_key: Key in `adata.obs` to use for balancing the groups. Only relevant if `balanced=True`.
        replace: If `True`, samples are drawn with replacement. Only relevant if `balanced=False`.
        axis: Axis to sample on. Either `obs` / `0` (observations, default) or `var` / `1` (variables).
        p: Drawing probabilities (floats) or mask (bools).
            Either an `axis`-sized array, or the name of a column
            If p is an array of probabilities, it must sum to 1.

    Returns:
        Returns `X[obs_indices], obs_indices` if data is array-like, otherwise subsamples the passed
        Central data object (`copy == False`) or returns a subsampled copy of it (`copy == True`).

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.diabetes_130_fairlearn(columns_obs_only=["age"])
        >>> edata.obs.age.value_counts()
        age
        'Over 60 years'          68541
        '30-60 years'            30716
        '30 years or younger'     2509
        >>> edata_balanced = ep.pp.sample(
        ...     edata, balanced=True, balanced_method="RandomUnderSampler", balanced_key="age", copy=True
        ... )
        >>> edata_balanced.obs.age.value_counts()
         age
        '30 years or younger'    2509
        '30-60 years'            2509
        'Over 60 years'          2509
    """
    if balanced:
        if balanced_key is None:
            raise TypeError("Key must be provided when balanced=True")

        if isinstance(data, AnnData):
            if balanced_key not in data.obs.columns:
                raise ValueError(
                    f"Key '{balanced_key}' not found in edata.obs. Available keys are: {data.obs.columns.tolist()}"
                )

            labels = data.obs[balanced_key].values

        elif isinstance(data, sp.csr_matrix | sp.csc_matrix) or isinstance(data, np.ndarray):
            labels = np.asarray(balanced_key)
            if labels.shape[0] != data.shape[0]:
                raise ValueError(
                    f"Length of labels ({labels.shape[0]}) does not match number of observations ({data.shape[0]})"
                )

        else:
            raise TypeError("data must be an EHRData, AnnData, numpy array or scipy sparse matrix when balanced=True")

        if balanced_method == "RandomUnderSampler" or balanced_method == "RandomOverSampler":
            sampled_indices, sampled_labels = _random_resample(labels, method=balanced_method, random_state=rng)
        else:
            raise ValueError(f"Unknown sampling method: {balanced_method}")

        if isinstance(data, AnnData):
            if copy:
                return data[sampled_indices].copy()
            else:
                data._inplace_subset_obs(sampled_indices)
                return None
        else:
            return data[sampled_indices], sampled_indices
    else:
        return sc.pp.sample(data=data, fraction=fraction, n=n_obs, rng=rng, copy=copy, replace=replace, axis=axis, p=p)


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
def combat(
    edata: EHRData | AnnData,
    *,
    key: str = "batch",
    covariates: Collection[str] | None = None,
    layer: str | None = None,
    inplace: bool = True,
) -> EHRData | AnnData | np.ndarray | None:  # pragma: no cover
    """ComBat function for batch effect correction :cite:p:`Johnson2006`, :cite:p:`Leek2012`, :cite:p:`Pedersen2012`.

    Corrects for batch effects by fitting linear models, gains statistical power via an EB framework where information is borrowed across features.
    This uses the implementation `combat.py`:cite:p:`Pedersen2012`.

    .. _combat.py: https://github.com/brentp/combat.py

    Args:
        edata: Central data object.
        key: Key to a categorical annotation from `.obs` that will be used for batch effect removal.
        covariates: Additional covariates besides the batch variable such as adjustment variables or biological condition.
                    This parameter refers to the design matrix `X` in Equation 2.1 in :cite:p:`Johnson2006` and to the `mod` argument in
                    the original combat function in the sva R package.
                    Note that not including covariates may introduce bias or lead to the removal of signal in unbalanced designs.
        layer: The layer to operate on.
        inplace: Whether to replace edata.X or to return the corrected data

    Returns:
        Depending on the value of `inplace`, either returns the corrected matrix or modifies `edata.X`.
    """
    # Since scanpy's combat does not support layers, we need to copy the data to the X matrix and then copy the result back to the layer
    if layer is None:
        return sc.pp.combat(adata=edata, key=key, covariates=covariates, inplace=inplace)
    else:
        X = edata.X.copy()
        edata.X = edata.layers[layer].copy()
        if not inplace:
            return sc.pp.combat(adata=edata, key=key, covariates=covariates, inplace=False)
        else:
            sc.pp.combat(adata=edata, key=key, covariates=covariates, inplace=True)
            edata.layers[layer] = edata.X
            edata.X = X
            return None


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
    the connectivity of the manifold (`method=='umap'`). If `method=='gauss'`,
    connectivities are computed according to :cite:p:`Coifman2005`, in the adaption of :cite:p:`Haghverdi2016`.

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
        copy: Determines whether a copy of `edata` is returned.

    Returns:
         Depending on `copy`, updates or returns `edata` with the following;
         See `key_added` parameter description for the storage path of connectivities and distances.

         **connectivities** : sparse matrix of dtype `float32`.
         Weighted adjacency matrix of the neighborhood graph of data points. Weights should be interpreted as connectivities.

         **distances** : sparse matrix of dtype `float32`.
         Instead of decaying weights, this stores distances for each pair of neighbors.
    """
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


def filter_features(
    edata: EHRData,
    *,
    layers: str | Sequence[str] | None = None,
    min_obs: int | None = None,
    max_obs: int | None = None,
    time_mode: Literal["all", "any", "proportion"] = "all",
    prop: float | None = None,
    copy: bool = False,
) -> EHRData | None:  # pragma: no cover
    """Filter features based on number of observations.

    Keep only features which have at least `min_obs` observations
    or/and have at most `max_obs` observations.
    When a longitudinal `EHRData` is passed, filtering can be done across time points.

    Only provide `min_obs` and/or `max_obs` per call.

    Args:
        edata: Central data object.
        layers: layers(s) to use for filtering. If `None` (default), filtering is performed on `.R` for 3D EHRData objects and on `.X` for 2D EHRData objects.
        When multiple layers are provided, a feature passes the filtering only if it satisifies the criteria in every layer.
        min_obs: Minimum number of observations required for a feature to pass filtering.
        max_obs: Maximum number of observations allowed for a feature to pass filtering.
        time_mode: How to combine filtering criteria across the time axis. Options are:
                    * `'all'` (default): The feature must pass the filtering criteria in all time points.
                    * `'any'`: The feature must pass the filtering criteria in at least one time point.
                    * `'proportion'`: The feature must pass the filtering criteria in at least a proportion `prop` of time points. For example, with `prop=0.3`,
                    the feature must pass the filtering criteria in at least 30% of the time points.
        prop: Proportion of time points in which the feature must pass the filtering criteria. Only relevant if `time_mode='proportion'`.
        copy: Determines whether a copy is returned.

    Returns:
        Depending on `copy`, subsets and annotates the passed data object and returns `None`

    Examples:
    >>> import ehrapy as ep
    >>> edata = ed.dt.ehrdata_blobs(n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6)
    >>> edata.R.shape
    (500, 45, 15)
    >>> ep.pp.filter_features(edata, min_obs=185, time_mode="all")
    >>> edata.R.shape
    (500, 18, 15)

    """
    if not isinstance(edata, EHRData):
        raise TypeError("Data object must be an EHRData object")

    data = edata.copy() if copy else edata

    lower_set = min_obs is not None
    upper_set = max_obs is not None

    if not (lower_set or upper_set):
        raise ValueError("You must provide at least one of 'min_obs' and 'max_obs'")

    if time_mode not in {"all", "any", "proportion"}:
        raise ValueError(f"time_mode must be one of 'all', 'any', 'proportion', got {time_mode}")

    if time_mode == "proportion" and (prop is None or not (0 < prop <= 1)):
        raise ValueError("prop must be set to a value between 0 and 1 when time_mode is 'proportion'")

    obs_ax, _var_ax, _time_ax = 0, 1, 2

    threshold_min = min_obs
    threshold_max = max_obs

    if layers is None:
        arr = data.R if data.R is not None else data.X
        if arr is None:
            raise ValueError("Both X and R are None, no data to filter")
        arrs = [arr]
    elif isinstance(layers, str):
        arrs = [data.layers[layers]]
    else:  # when filtering is done across multiple layers
        arrs = [data.layers[layer] for layer in layers]

    layer_masks = []
    first_counts = None
    is_2d_ref = False
    for arr in arrs:
        if arr.ndim == 2:
            arr = arr[:, :, None]
            if first_counts is None:
                is_2d_ref = True
        elif arr.ndim != 3:
            raise ValueError(f"expected a 2D or 3D array, got {arr.shape}")

        missing_mask = np.isin(arr, MISSING_VALUES) | np.isnan(arr)

        present = ~missing_mask
        counts = present.sum(axis=obs_ax)

        if first_counts is None:
            first_counts = counts

        if threshold_max is not None and threshold_min is not None:
            pass_threshold = (threshold_min <= counts) & (counts <= threshold_max)
        elif threshold_min is not None:
            pass_threshold = counts >= threshold_min
        else:
            pass_threshold = counts <= threshold_max

        if time_mode == "all":
            feature_mask = pass_threshold.all(axis=1)
        elif time_mode == "any":
            feature_mask = pass_threshold.any(axis=1)
        elif time_mode == "proportion":
            if prop is None:
                raise ValueError("prop must be set when time_mode is 'proportion'")
            feature_mask = (pass_threshold.sum(axis=1) / pass_threshold.shape[1]) >= prop
        else:
            raise ValueError(f"Unknown time_mode: {time_mode}")

        layer_masks.append(feature_mask)

    final_feature_mask = np.logical_and.reduce(layer_masks)

    number_per_feature = first_counts.sum(axis=1).astype(np.float64)

    n_filtered = int((~final_feature_mask).sum())

    if n_filtered > 0:
        msg = f"filtered out {n_filtered} features that are measured "
        if threshold_min is not None:
            msg += f"less than {threshold_min} counts"
        else:
            msg += f"more than {threshold_max} counts"

        if not is_2d_ref:
            if time_mode == "proportion":
                msg += f" in less than {prop * 100:.1f}% of time points"
            else:
                msg += f" in {time_mode} time points"
        logger.info(msg)

    label = "n_obs" if is_2d_ref else "n_obs_over_time"
    data.var[label] = number_per_feature
    data._inplace_subset_var(final_feature_mask)

    return data if copy else None


def filter_observations(
    edata: EHRData,
    *,
    layers: str | Sequence[str] | None = None,
    min_vars: int | None = None,
    max_vars: int | None = None,
    time_mode: Literal["all", "any", "proportion"] = "all",
    prop: float | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Filter observations based on numbers of variables (features/measurements).

    Keep only observations which have at least `min_vars` variables and/or at most `max_vars` variables.
    When a longitudinal `EHRData` is passed, filtering can be done across time points.

    Only provide `min_vars` and/or `max_vars` per call.

    Args:
        edata: Central data object.
        layers: layers(s) to use for filtering. If `None` (default), filtering is performed on `.R` for 3D EHRData objects and on `.X` for 2D EHRData objects.
        When multiple layers are provided, a feature passes the filtering only if it satisifies the criteria in every layer.
        min_vars: Minimum number of variables required for an observation to pass filtering.
        max_vars: Maximum number of variables allowed for an observation to pass filtering.
        time_mode: How to combine filtering criteria across the time axis. Only relevant if an `EHRData` is passed. Options are:
                    * `'all'` (default): The observation must pass the filtering criteria in all time points.
                    * `'any'`: The observation must pass the filtering criteria in at least one time point.
                    * `'proportion'`: The observation must pass the filtering criteria in at least a proportion `prop` of time points. For example, with `prop=0.3`,
                      the observation must pass the filtering criteria in at least 30% of the time points.
        prop: Proportion of time points in which the observation must pass the filtering criteria. Only relevant if `time_mode='proportion'`.
        copy: Determines whether a copy is returned.

    Returns:
        Depending on `copy`, subsets and annotates the passed data object and returns `None`

    Examples:
    >>> import ehrapy as ep
    >>> edata = ed.dt.ehrdata_blobs(n_variables=45, n_observations=500, base_timepoints=15, missing_values=0.6)
    >>> edata.R.shape
    (500, 45, 15)
    >>> ep.pp.filter_observations(edata, min_vars=10, time_mode="all")
    >>> edata.R.shape
    (477, 45, 15)

    """
    if not isinstance(edata, EHRData):
        raise TypeError("Data object must be an EHRData object")

    data = edata.copy() if copy else edata

    lower_set = min_vars is not None
    upper_set = max_vars is not None

    if not (lower_set or upper_set):
        raise ValueError("You must provide at least one of 'min_vars' and 'max_vars'")
    if time_mode not in {"all", "any", "proportion"}:
        raise ValueError(f"time_mode must be one of 'all', 'any', 'proportion', got {time_mode}")
    if time_mode == "proportion" and (prop is None or not (0 < prop <= 1)):
        raise ValueError("prop must be set to a value between 0 and 1 when time_mode is 'proportion'")

    threshold_min = min_vars
    threshold_max = max_vars

    if layers is None:
        arr = data.R if data.R is not None else data.X
        if arr is None:
            raise ValueError("Both R and X are None, no data to filter")
        arrs = [arr]
    elif isinstance(layers, str):
        arrs = [data.layers[layers]]
    else:
        arrs = [data.layers[layer] for layer in layers]

    layers_obs_masks: list[np.ndarray] = []
    first_number_per_obs: np.ndarray | None = None
    is_2d_ref = False

    for arr in arrs:
        if arr.ndim == 2:
            arr = arr[:, :, None]
            is_2d = True
        elif arr.ndim == 3:
            is_2d = False
        else:
            raise ValueError(f"expected 2D or 3D array, got {arr.shape}")

        missing_mask = np.isin(arr, MISSING_VALUES) | np.isnan(arr)
        present = ~missing_mask

        per_time_vals = present.sum(axis=1).astype(float)

        if first_number_per_obs is None:
            first_number_per_obs = per_time_vals.sum(axis=1).astype(np.float64)
            is_2d_ref = is_2d

        if threshold_min is not None and threshold_max is not None:
            masks_t = (per_time_vals >= float(threshold_min)) & (per_time_vals <= float(threshold_max))
        elif threshold_min is not None:
            masks_t = per_time_vals >= float(threshold_min)
        elif threshold_max is not None:
            masks_t = per_time_vals <= float(threshold_max)

        if time_mode == "all":
            obs_mask = masks_t.all(axis=1)
        elif time_mode == "any":
            obs_mask = masks_t.any(axis=1)
        else:
            obs_mask = masks_t.mean(axis=1) >= float(prop)

        layers_obs_masks.append(obs_mask)

    final_obs_mask = np.logical_and.reduce(layers_obs_masks)

    n_filtered = int((~obs_mask).sum())
    if n_filtered > 0:
        msg = f"filtered out {n_filtered} observations that have"
        if threshold_min is not None:
            msg += f"less than {threshold_min} " + "features"
        else:
            msg += f"more than {threshold_max} " + "features"

        if not is_2d_ref and (arrs[0].ndim == 3 and arrs[0].shape[2] > 1):
            if time_mode == "proportion":
                msg += f" in < {prop * 100:.1f}% of time points"
            else:
                msg += f" in {time_mode} time points"

        logger.info(msg)

    label = "n_vars" if is_2d_ref else "n_vars_over_time"
    data.obs[label] = first_number_per_obs
    data._inplace_subset_obs(final_obs_mask)
    return data if copy else None


def _random_resample(
    label: str | np.ndarray,
    target: str = "balanced",
    method: Literal["RandomUnderSampler", "RandomOverSampler"] = "RandomUnderSampler",
    random_state: RNGLike | SeedLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to under- or over-sample the data to achieve a balanced dataset.

    Args:
        label: The labels of the data.
        target: The target number of samples for each class. If "balanced", it will balance the classes to the minimum class size.
        method: The sampling method, either "RandomUnderSampler" for under-sampling or "RandomOverSampler" for over-sampling.
        random_state: Random seed.

    Returns:
        A tuple of (sampled_indices, sampled_labels).
    """
    label = np.asarray(label)
    if isinstance(random_state, np.random.Generator):
        rnd = random_state
    else:
        rnd = np.random.default_rng(random_state)
    classes, counts = np.unique(label, return_counts=True)

    if target == "balanced":
        if method == "RandomUnderSampler":
            target_count = counts.min()
        elif method == "RandomOverSampler":
            target_count = counts.max()
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    indices = []

    for c in classes:
        class_idx = np.where(label == c)[0]
        n = len(class_idx)
        if method == "RandomUnderSampler":
            if n > target_count:
                sampled_idx = rnd.choice(class_idx, size=target_count, replace=False)
                indices.extend(sampled_idx)
            else:
                indices.extend(class_idx)
        elif method == "RandomOverSampler":
            if n < target_count:
                sampled_idx = rnd.choice(class_idx, size=target_count, replace=True)
                indices.extend(sampled_idx)
            else:
                indices.extend(class_idx)

    sample_indices = np.array(indices)
    return sample_indices, label[sample_indices]
