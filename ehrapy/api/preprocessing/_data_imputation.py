from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def replace_explicit(
    adata: AnnData,
    replacement: str | int | dict[str, str | int] | tuple[str, str | int] = None,
    impute_empty_strings: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Replaces all missing values in all or the specified columns with the passed value

    There are several scenarios to cover:
    1. Replace all missing values with the specified value. ( str | int )
    2. Replace all missing values in a subset of columns with the specified value. ( Dict(str: (str, int)) )
    3. Replace all missing values in a subset of columns with a specified value per column. ( str ,(str, int) )

    Args:
        adata: :class:`~anndata.AnnData` object containing X to impute values in
        replacement: Value to use as replacement and optionally keys to indicate which columns to replace.
        See scenarios above
        impute_empty_strings: Whether to also impute empty strings
        copy: Whether to return a copy with the imputed data.

    Returns:
        :class:`~anndata.AnnData` object with imputed X

    Example:
        .. code-block:: python

        import ehrapy.api as ep
        adata = ep.data.mimic_2(encode=True)
        adata_replaced = ep.pp.replace_explicit(adata, replacement=0, copy=True)
    """
    if copy:
        adata = adata.copy()

    # scenario 1: Replace all missing values with the specified value
    impute_conditions = np.logical_or(pd.isnull(adata.X), adata.X == "")
    if not impute_empty_strings:
        impute_conditions = pd.isnull(adata.X)
    adata.X[impute_conditions] = replacement

    # scenario 2: Replace all missing values in a subset of columns with the specified value
    # TODO

    # scenario 3: Replace all missing values in a subset of columns with a specified value per column
    # TODO

    return adata
