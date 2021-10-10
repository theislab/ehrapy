from numbers import Number
from typing import Optional, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import spmatrix


def log1p(
    X: Union[AnnData, np.ndarray, spmatrix],
    base: Optional[Number] = None,
    copy: bool = False,
    chunked: bool = None,
    chunk_size: Optional[int] = None,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
):
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
    sc.pp.log1p(X, base=base, copy=copy, chunked=chunked, chunk_size=chunk_size, layer=layer, obsm=obsm)
