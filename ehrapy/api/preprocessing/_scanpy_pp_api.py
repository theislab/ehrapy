from numbers import Number
from typing import Optional, Union

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
    svd_solver: str = 'arpack',
    random_state: AnyRandom = 0,
    return_info: bool = False,
    dtype: str = 'float32',
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
    return sc.pp.pca(data=data,
                     n_comps=n_comps,
                     zero_center=zero_center,
                     svd_solver=svd_solver,
                     random_state=random_state,
                     return_info=return_info,
                     use_highly_variable=False,
                     dtype=dtype,
                     copy=copy,
                     chunked=chunked,
                     chunk_size=chunk_size)
