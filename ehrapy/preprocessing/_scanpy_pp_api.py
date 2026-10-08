from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from ehrdata import EHRData
from numpy.typing import NDArray

from ehrapy._compat import function_2D_only
from ehrapy._types import _empty

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from numpy.typing import DTypeLike, NDArray
    from scipy.sparse import spmatrix

    from ehrapy._types import AnyRandom, CSBase, Empty, RNGLike, SeedLike


@function_2D_only()
def pca(
    edata: EHRData | np.ndarray | spmatrix,
    *,
    n_comps: int | None = None,
    zero_center: bool = True,
    svd_solver: Literal["arpack", "covariance_eigh", "randomized", "auto", "full", "tsqr"] | None = None,
    random_state: AnyRandom = 0,
    mask_var: NDArray[np.bool_] | str | Empty | None = _empty,
    return_info: bool = False,
    dtype: DTypeLike = "float32",
    layer: str | None = None,
    obsm: str | None = None,
    key_added: str | None = None,
    copy: bool = False,
    chunked: bool = False,
    chunk_size: int | None = None,
) -> EHRData | np.ndarray | spmatrix | None:  # pragma: no cover
    """Computes a principal component analysis :cite:p:`Pedregosa2011`.

    Computes PCA coordinates, loadings and variance decomposition.
    Uses the implementation of *scikit-learn*.

    Args:
        edata: Central data object.
        n_comps: Number of principal components to compute.
                 Defaults to 50, or 1 - minimum dimension size of selected representation.
        zero_center: If `True`, compute (or approximate) PCA from covariance matrix.
                     If `False`, perform a truncated SVD instead of PCA.
        svd_solver: SVD solver to use.
                    If `None`, chooses automatically: `'arpack'` for PCA and `'randomized'` for truncated SVD (`zero_center=False`).

                    * `'arpack'` for the ARPACK wrapper in SciPy (:func:`~scipy.sparse.linalg.svds`).

                    * `'covariance_eigh'` for the classic eigendecomposition of the covariance matrix, suited for tall-and-skinny matrices.

                    * `'randomized'` for the randomized algorithm due to Halko (2009).

                    * `'auto'` chooses automatically depending on the size of the problem.

                    * `'full'` for the exact full SVD.

                    * `'tsqr'` for the “tall-and-skinny QR” algorithm, only available for dense *dask* arrays.

                    Efficient computation of the principal components of a sparse matrix currently only works with the `'arpack'` or `'covariance_eigh'` solvers.
        random_state: Change to use different initial states for the optimization.
        mask_var: To run only on a certain set of features given by a boolean array or a string referring to an array in `var`.
                  By default, uses `.var['highly_variable']` if available, else all features.
                  Pass `None` to use all features.
        return_info: Only relevant when not passing an :class:`~ehrdata.EHRData`: see “**Returns**”.
        dtype: Numpy data type string to which to convert the result.
        layer: If provided, which element of `layers` to use for PCA instead of `X`.
        obsm: If provided, which element of `obsm` to use for PCA instead of `X`.
        key_added: If not specified, the embedding is stored in `obsm['X_pca']`, the loadings in `varm['PCs']` and the parameters in `uns['pca']`.
                   If specified, the embedding is stored in `obsm[key_added]`, the loadings in `varm[key_added]` and the parameters in `uns[key_added]`.
        copy: If an :class:`~ehrdata.EHRData`: is passed, determines whether a copy is returned. Is ignored otherwise.
        chunked: If `True`, perform an incremental PCA on segments of `chunk_size`.
                  The incremental PCA automatically zero centers and ignores settings of `zero_center`, `random_state` and `svd_solver`.
                  If `False`, perform a full PCA.
        chunk_size: Number of observations to include in each chunk. Required if `chunked=True` was passed.

    Returns:
        If `edata` is array-like and `return_info=False` was passed,
        this function returns the PCA representation of `edata` as an
        array of the same type as the input array.

        Otherwise, it returns `None` if `copy=False`, else an updated `EHRData` object.
        Sets the following fields:

        `.obsm['X_pca' | key_added]` : :class:`~scipy.sparse.csr_matrix` | :class:`~scipy.sparse.csc_matrix` | :class:`~numpy.ndarray` (shape `(edata.n_obs, n_comps)`)
            PCA representation of data.
        `.varm['PCs' | key_added]` : :class:`~numpy.ndarray` (shape `(edata.n_vars, n_comps)`)
            The principal components containing the loadings when `obsm=None`.
        `.uns['pca' | key_added]['components']` : :class:`~numpy.ndarray` (shape `(edata.obsm[obsm].shape[1], n_comps)`)
            The principal components containing the loadings when `obsm` is passed.
        `.uns['pca' | key_added]['variance_ratio']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
            Ratio of explained variance.
        `.uns['pca' | key_added]['variance']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
            Explained variance, equivalent to the eigenvalues of the
            covariance matrix.
    """
    return sc.pp.pca(
        data=edata,
        n_comps=n_comps,
        layer=layer,
        obsm=obsm,
        zero_center=zero_center,
        svd_solver=svd_solver,
        random_state=random_state,
        return_info=return_info,
        dtype=dtype,
        key_added=key_added,
        copy=copy,
        chunked=chunked,
        chunk_size=chunk_size,
        **({} if mask_var is _empty else {"mask_var": mask_var}),
    )


@function_2D_only()
def regress_out(
    edata: EHRData,
    keys: str | Sequence[str],
    *,
    n_jobs: int | None = None,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | None:  # pragma: no cover
    """Regress out (mostly) unwanted sources of variation.

    Uses simple linear regression.
    This is inspired by Seurat's `regressOut` function in R :cite:p:`Satija2015`.
    Note that this function tends to overcorrect in certain circumstances.

    Args:
        edata: Central data object.
        keys: Keys for observation annotation on which to regress on.
        n_jobs: Number of jobs for parallel computation.
                `None` means using :attr:`scanpy.settings.n_jobs`.
        layer: If provided, which element of `layers` to regress on.
        copy: Determines whether a copy of `edata` is returned.

    Returns:
        Depending on `copy` returns or updates the data object with the corrected data matrix in `X` or `layers[layer]`.
    """
    return sc.pp.regress_out(adata=edata, keys=keys, n_jobs=n_jobs, layer=layer, copy=copy)


def sample(
    edata: EHRData | np.ndarray | CSBase,
    *,
    fraction: float | None = None,
    n_obs: int | None = None,
    rng: RNGLike | SeedLike | None = None,
    balanced: bool = False,
    balanced_method: Literal["RandomUnderSampler", "RandomOverSampler"] = "RandomUnderSampler",
    groupby: str | None = None,
    copy: bool = False,
    replace: bool = False,
    axis: Literal["obs", 0, "var", 1] = "obs",
    p: str | NDArray[np.bool_] | NDArray[np.floating] | None = None,
) -> EHRData | None | tuple[np.ndarray | CSBase, np.ndarray]:  # pragma: no cover
    """Sample a fraction or a number of observations / variables with or without replacement.

    Args:
        edata: Central data object.
        fraction: Sample to this `fraction` of the number of observations or variables (see `axis`).
                  This can be larger than 1.0, if `replace=True`.
        n_obs: Sample to this number of observations or variables (see `axis`).
        rng: Random seed to change subsampling.
        copy: If an :class:`~ehrdata.EHRData` is passed, determines whether a copy is returned.
        balanced: If `True`, balance the groups in `edata.obs[groupby]` by under- or over-sampling.
                  Requires `groupby` to be set. If `False`, simple random sampling is performed.
        balanced_method: The sampling method, either "RandomUnderSampler" for under-sampling or "RandomOverSampler" for over-sampling. Only relevant if `balanced=True`.
        groupby: Key in `edata.obs` to use for balancing the groups. Only relevant if `balanced=True`.
        replace: If `True`, samples are drawn with replacement. Only relevant if `balanced=False`.
        axis: Axis to sample on. Either `obs` / `0` (observations, default) or `var` / `1` (variables).
        p: Drawing probabilities (floats) or mask (bools).
            Either an `axis`-sized array, or the name of a column.
            If `p` is an array of probabilities, it must sum to 1.

    Returns:
        Returns `X[obs_indices], obs_indices` if `edata` is array-like, otherwise subsamples the passed
        Central data object (`copy == False`) or returns a subsampled copy of it (`copy == True`).

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.diabetes_130_fairlearn(columns_obs_only=["age"])
        >>> edata.obs.age.value_counts()
        age
        'Over 60 years'          68541
        '30-60 years'            30716
        '30 years or younger'     2509
        >>> edata_balanced = ep.pp.sample(
        ...     edata, balanced=True, balanced_method="RandomUnderSampler", groupby="age", copy=True
        ... )
        >>> edata_balanced.obs.age.value_counts()
        age
        '30 years or younger'    2509
        '30-60 years'            2509
        'Over 60 years'          2509
    """
    if balanced:
        if groupby is None:
            raise TypeError("groupby must be provided when balanced=True")

        if isinstance(edata, EHRData):
            if groupby not in edata.obs.columns:
                raise ValueError(
                    f"Key '{groupby}' not found in edata.obs. Available keys are: {edata.obs.columns.tolist()}"
                )

            labels = edata.obs[groupby].values

        elif isinstance(edata, sp.csr_matrix | sp.csc_matrix) or isinstance(edata, np.ndarray):
            labels = np.asarray(groupby)
            if labels.shape[0] != edata.shape[0]:
                raise ValueError(
                    f"Length of labels ({labels.shape[0]}) does not match number of observations ({edata.shape[0]})"
                )

        else:
            raise TypeError("edata must be an EHRData, numpy array or scipy sparse matrix when balanced=True")

        if balanced_method == "RandomUnderSampler" or balanced_method == "RandomOverSampler":
            sampled_indices, _ = _random_resample(labels, method=balanced_method, random_state=rng)
        else:
            raise ValueError(f"Unknown sampling method: {balanced_method}")

        if isinstance(edata, EHRData):
            if copy:
                return edata[sampled_indices].copy()
            else:
                edata._inplace_subset_obs(sampled_indices)
                return None
        else:
            return edata[sampled_indices], sampled_indices
    else:
        return sc.pp.sample(data=edata, fraction=fraction, n=n_obs, rng=rng, copy=copy, replace=replace, axis=axis, p=p)


@function_2D_only()
def combat(
    edata: EHRData,
    *,
    batch_key: str = "batch",
    covariates: Collection[str] | None = None,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | None:  # pragma: no cover
    """ComBat function for batch effect correction :cite:p:`Johnson2006`, :cite:p:`Leek2012`, :cite:p:`Pedersen2012`.

    Corrects for batch effects by fitting linear models, gains statistical power via an EB framework where information is borrowed across features.
    This uses the implementation `combat.py`_ :cite:p:`Pedersen2012`.

    .. _combat.py: https://github.com/brentp/combat.py

    Args:
        edata: Central data object.
        batch_key: Key to a categorical annotation from `.obs` that will be used for batch effect removal.
        covariates: Additional covariates besides the batch variable such as adjustment variables or biological condition.
                    This parameter refers to the design matrix `X` in Equation 2.1 in :cite:p:`Johnson2006` and to the `mod` argument in
                    the original combat function in the sva R package.
                    Note that not including covariates may introduce bias or lead to the removal of signal in unbalanced designs.
        layer: The layer to operate on.
        copy: Whether to return a corrected copy of `edata` or to correct it in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object.
    """
    edata = edata.copy() if copy else edata
    # Since scanpy's combat does not support layers, we need to copy the data to the X matrix and then copy the result back to the layer
    if layer is None:
        sc.pp.combat(adata=edata, key=batch_key, covariates=covariates, inplace=True)
    else:
        X = edata.X.copy()
        edata.X = edata.layers[layer].copy()
        sc.pp.combat(adata=edata, key=batch_key, covariates=covariates, inplace=True)
        edata.layers[layer] = edata.X
        edata.X = X

    return edata if copy else None


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
