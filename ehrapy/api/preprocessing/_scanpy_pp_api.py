from numbers import Number
from typing import Dict, Optional, Sequence, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from scanpy._utils import AnyRandom
from scipy.sparse import spmatrix


def log1p(
    X: Union[AnnData, np.ndarray, spmatrix],
    base: Optional[Number] = None,
    copy: bool = False,
    chunked: bool = None,
    chunk_size: Optional[int] = None,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
) -> Optional[AnnData]:
    """Logarithmize the data matrix.

    Computes :math:`X = \\log(X + 1)`, where :math:`log` denotes the natural logarithm unless a different base is given.

    Args:
        X: The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond to patient observations and columns to features.
        base: Base of the logarithm. Natural logarithm is used by default.
        copy: If an :class:`~anndata.AnnData` is passed, determines whether a copy is returned.
        chunked: Process the data matrix in chunks, which will save memory. Applies only to :class:`~anndata.AnnData`.
        chunk_size: `n_obs` of the chunks to process the data in.
        layer: Entry of layers to tranform.
        obsm: Entry of obsm to transform.

    Returns:
        Returns or updates `data`, depending on `copy`.
    """
    return sc.pp.log1p(X, base=base, copy=copy, chunked=chunked, chunk_size=chunk_size, layer=layer, obsm=obsm)


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
) -> Union[AnnData, np.ndarray, spmatrix]:
    """Computes a principal component analysis.

    Computes PCA coordinates, loadings and variance decomposition. Uses the implementation of *scikit-learn*.

    Args:
        data: The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
        n_comps: Number of principal components to compute. Defaults to 50, or 1 - minimum dimension size of selected representation.
        zero_center: If `True`, compute standard PCA from covariance matrix.
                     If `False`, omit zero-centering variables (uses :class:`~sklearn.decomposition.TruncatedSVD`), which allows to handle sparse input efficiently.
                     Passing `None` decides automatically based on sparseness of the data.
        svd_solver: SVD solver to use:

                    `'arpack'` (the default)
                    for the ARPACK wrapper in SciPy (:func:`~scipy.sparse.linalg.svds`)

                    `'randomized'`
                    for the randomized algorithm due to Halko (2009).

                    `'auto'`
                    chooses automatically depending on the size of the problem.

                    `'lobpcg'`
                    An alternative SciPy solver.

                    Efficient computation of the principal components of a sparse matrix currently only works with the `'arpack`' or `'lobpcg'` solvers.
        random_state: Change to use different initial states for the optimization. (default: 0)
        return_info: Only relevant when not passing an :class:`~anndata.AnnData`: see “**Returns**”.
        dtype: Numpy data type string to which to convert the result. (default: float32)
        copy: If an :class:`~anndata.AnnData` is passed, determines whether a copy is returned. Is ignored otherwise.
        chunked: If `True`, perform an incremental PCA on segments of `chunk_size`.
                  The incremental PCA automatically zero centers and ignores settings of
                  `random_seed` and `svd_solver`. If `False`, perform a full PCA.
        chunk_size: Number of observations to include in each chunk. Required if `chunked=True` was passed.

    Returns:
        X_pca: :class:`~scipy.sparse.spmatrix`, :class:`~numpy.ndarray`
        If `data` is array-like and `return_info=False` was passed, this function only returns `X_pca`.

        adata : anndata.AnnData
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


def normalize_total(
    adata: AnnData,
    target_sum: Optional[float] = None,
    key_added: Optional[str] = None,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    """Normalize findings per patient.

    Normalize each patient by total counts over all features, so that every patient has the same total count after normalization.
    If choosing `target_sum=1e6`, this is CPM normalization as commonly used in RNA-Seq.

    Args:
        adata: The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to observations (patients) and columns to features.
        target_sum: If `None`, after normalization, each observation (patient) has a total count equal to the median of total counts for observations before normalization.
        key_added: Name of the field in `adata.obs` where the normalization factor is stored.
        layer: Layer to normalize instead of `X`. If `None`, `X` is normalized.
        inplace: Whether to update `adata` or return dictionary with normalized copies of `adata.X` and `adata.layers`.
        copy: Whether to modify copied input object. Not compatible with inplace=False.

    Returns:

    """
    return sc.pp.normalize_total(
        adata=adata,
        target_sum=target_sum,
        exclude_highly_expressed=False,
        max_fraction=0.05,
        key_added=key_added,
        layer=layer,
        inplace=inplace,
        copy=copy,
    )


def regress_out(
    adata: AnnData,
    keys: Union[str, Sequence[str]],
    n_jobs: Optional[int] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """Regress out (mostly) unwanted sources of variation.

    Uses simple linear regression. This is inspired by Seurat's `regressOut` function in R [Satija15].
    Note that this function tends to overcorrect in certain circumstances.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations
        keys: Keys for observation annotation on which to regress on.
        n_jobs: Number of jobs for parallel computation. `None` means using :attr:`scanpy._settings.ScanpyConfig.n_jobs`.
        copy: Determines whether a copy of `adata` is returned.

    Returns:
        Depending on `copy` returns or updates an :class:`~anndata.AnnData` object with the corrected data matrix.
    """
    return sc.pp.regress_out(adata=adata, keys=keys, n_jobs=n_jobs, copy=copy)


def scale(
    X: Union[AnnData, spmatrix, np.ndarray],
    zero_center: bool = True,
    max_value: Optional[float] = None,
    copy: bool = False,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
) -> Union[AnnData, spmatrix, np.ndarray]:
    """Scale data to unit variance and zero mean.

    .. note::
        Variables (genes) that do not display any variation (are constant across
        all observations) are retained and (for zero_center==True) set to 0
        during this operation. In the future, they might be set to NaNs.

    Args:
        X: The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond to observations (patients) and columns to features.
        zero_center:
        max_value:
        copy:
        layer:
        obsm:

    Returns:

    """
    return sc.pp.scale(X=X, zero_center=zero_center, max_value=max_value, copy=copy, layer=layer, obsm=obsm)


def subsample(
    data: Union[AnnData, np.ndarray, spmatrix],
    fraction: Optional[float] = None,
    n_obs: Optional[int] = None,
    random_state: AnyRandom = 0,
    copy: bool = False,
) -> Optional[AnnData]:
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
