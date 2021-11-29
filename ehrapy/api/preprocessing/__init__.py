from typing import Dict, Optional, Tuple, Union

from anndata import AnnData

from ehrapy.api.preprocessing._data_imputation import Imputation
from ehrapy.api.preprocessing.encoding import encode, type_overview, undo_encoding
from ehrapy.api.preprocessing.normalization import Normalization


def replace_explicit(
    adata: AnnData,
    replacement: Union[Union[str, int], Dict[str, Union[str, int]], Tuple[str, Union[str, int]]] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """Replaces all missing values in all or the specified columns with the passed value

    Args:
        adata: :class:`~anndata.AnnData` object containing X to impute values in
        copy: Whether to return a copy or act in place

    Returns:
        :class:`~anndata.AnnData` object with imputed X

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_replaced = ep.pp.replace_explicit(adata, replacement=0, copy=True)
    """
    return Imputation.explicit(adata, replacement, copy)


def norm_identity(
    adata: AnnData,
    copy: bool = False,
) -> Optional[AnnData]:
    """Returns the original object without any normalisation

    Created as a template during development. Should be removed before merging.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in
        copy: Whether to return a copy with the normalized data.

    Returns:
        :class:`~anndata.AnnData` object with normalized X

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.norm_identity(adata, copy=True)
    """
    return Normalization.identity(adata, copy)


from ehrapy.api.preprocessing._scanpy_pp_api import *  # noqa: E402,F403
