from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from ehrdata._logger import logger
from ehrdata.core.constants import MISSING_VALUES

from ehrapy._compat import function_2D_only, use_ehrdata

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ehrdata import EHRData
    from numpy.typing import NDArray
    from scipy.sparse import spmatrix

    from ehrapy._types import AnyRandom, CSBase, RNGLike, SeedLike


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
        If `data` is array-like and `return_info=False` was passed,
        this function returns the PCA representation of `data` as an
        array of the same type as the input array.

        Otherwise, it returns `None` if `copy=False`, else an updated `AnnData` object.
        Sets the following fields:

        `.obsm['X_pca' | key_added]` : :class:`~scipy.sparse.csr_matrix` | :class:`~scipy.sparse.csc_matrix` | :class:`~numpy.ndarray` (shape `(adata.n_obs, n_comps)`)
            PCA representation of data.
        `.varm['PCs' | key_added]` : :class:`~numpy.ndarray` (shape `(adata.n_vars, n_comps)`)
            The principal components containing the loadings.
        `.uns['pca' | key_added]['variance_ratio']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
            Ratio of explained variance.
        `.uns['pca' | key_added]['variance']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
            Explained variance, equivalent to the eigenvalues of the
            covariance matrix.
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
            sampled_indices, _ = _random_resample(labels, method=balanced_method, random_state=rng)
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
