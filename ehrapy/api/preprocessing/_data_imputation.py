from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData


class Imputation:
    """Provides functions to impute missing data based on various criteria."""

    @staticmethod
    def explicit(
        adata: AnnData,
        replacement: Union[Union[str, int], Dict[str, Union[str, int]], Tuple[str, Union[str, int]]] = None,
        impute_empty_strings: bool = True,
        copy: bool = False,
    ) -> Optional[AnnData]:
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
        """
        adata_to_act_on = adata
        if copy:
            adata_copy = adata.copy()
            adata_to_act_on = adata_copy

        # scenario 1: Replace all missing values with the specified value
        impute_conditions = np.logical_or(pd.isnull(adata_to_act_on.X), adata_to_act_on.X == "")
        if not impute_empty_strings:
            impute_conditions = pd.isnull(adata_to_act_on.X)
        adata_to_act_on.X[impute_conditions] = replacement

        # scenario 2: Replace all missing values in a subset of columns with the specified value
        # TODO

        # scenario 3: Replace all missing values in a subset of columns with a specified value per column
        # TODO

        return adata_to_act_on
