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
        replacement: Replacement value. Can be one of three possible scenarios:

            value: Specified raw value (str | int)

            Dict: Subset of columns with the specified value ( Dict(str: (str, int)) )

            Tuple: Subset of columns with the specified value per column ( str ,(str, int) )
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


def normalize(adata: AnnData, methods: Union[Dict[str, str], str], copy: bool = False) -> Optional[AnnData]:
    """Normalize numeric variable.

    This function normalizes the numeric variables in an AnnData object.

    Available normalization methods are:

    1. identity (return the un-normalized values)
    2. minmax (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encode using ~ehrapy.preprocessing.encode.encode.
        methods: Methods to use for normalization. Either:

            str: Name of the method to use for all numeric variable

            Dict: A dictionary specifying the method for each numeric variable where keys are variable and values are methods
        copy: Whether to return a copy or act in place

    Returns:
        :class:`~anndata.AnnData` object with normalized X

    Example:
    .. code-block:: python

        import ehrapy.api as ep
        adata = ep.data.mimic_2(encode=True)
        adata_norm = ep.pp.normalize(adata, method="minmax", copy=True)
    """

    return Normalization._normalize(adata, methods, copy)


from ehrapy.api.preprocessing._scanpy_pp_api import *  # noqa: E402,F403
