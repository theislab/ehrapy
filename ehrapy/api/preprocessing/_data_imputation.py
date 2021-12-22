from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData


class Imputation:
    """Provides functions to impute missing data based on various criteria."""

    @staticmethod
    def explicit(
        adata: AnnData,
        copy: bool = False,
        **kwargs
    ) -> Optional[AnnData]:
        """Replaces all missing values in all or the specified columns with the passed value

        There are two scenarios to cover:
        1. Replace all missing values with the specified value. ( str | int )
        2. Replace all missing values in a subset of columns with a specified value per column. ( str ,(str, int) )

        Args:
            adata: :class:`~anndata.AnnData` object containing X to impute values in
            copy: Whether to return a copy with the imputed data.
            **kwargs: replacement: Value to use as replacement and optionally keys to indicate which columns to replace.
                      impute_empty_strings: Whether to also impute empty strings

        Returns:
            :class:`~anndata.AnnData` object with imputed X
        """
        if copy:
            adata = adata.copy()

        # ensure replacement parameter has been passed when using explicit impute mode
        try:
            replacement = kwargs["replacement"]
        except KeyError:
            raise MissingImputeValuesError("No replacement values were passed. Make sure passing a replacement parameter"
                                           "when using explicit data imputation mode!") from None

        # 1: Replace all missing values with the specified value
        if isinstance(replacement, (int, str)):
            Imputation._replace_explicit(adata.X, kwargs["replacement"], kwargs["impute_empty_strings"])

        # 2: Replace all missing values in a subset of columns with a specified value per column or a default value, when the column is not explicitly named
        elif isinstance(replacement, dict):
            for idx, column_name in enumerate(adata.var_names):
                imputation_value = Imputation._extract_impute_value(replacement, column_name)
                Imputation._replace_explicit(adata.X[:, idx:idx+1], imputation_value, kwargs["impute_empty_strings"])
        else:
            raise ReplacementDatatypeError(f"Type {type(replacement)} is not a valid datatype for replacement parameter. Either use int, str or a dict!")

        return adata

    @staticmethod
    def _replace_explicit(x: np.ndarray, replacement: Union[str, int], impute_empty_strings: str) -> None:
        """Replace one column or whole X with a value where missing values are stored.
        """
        if not impute_empty_strings:
            impute_conditions = pd.isnull(x)
        else:
            impute_conditions = np.logical_or(pd.isnull(x), x == "")
        x[impute_conditions] = replacement

    @staticmethod
    def _extract_impute_value(replacement: Dict[str, Union[str, int]], column_name: str) -> Union[str, int]:
        """Extract the replacement value for a given column in the :class:`~anndata.AnnData` object

        Returns: The value to replace missing values

        """
        # try to get a value for the specific column
        imputation_value = replacement.get(column_name)
        if imputation_value:
            return imputation_value
        # search for a default value in case no value was specified for that column
        imputation_value = replacement.get("default")
        if imputation_value:
            return imputation_value
        else:
            raise MissingImputationValue(f"Could not find a replacement value for column {column_name} since None has been provided and"
                                         f"no default was found!")

    # ===================== Mean Imputation =========================

    @staticmethod
    def mean(
        adata: AnnData,
        copy: bool = False,
        **kwargs
    ) -> Optional[AnnData]:
        """MEAN"""
        pass

    # ===================== KNN Imputation =========================

    @staticmethod
    def knn(
        adata: AnnData,
        replacement: Union[Union[str, int], Dict[str, Union[str, int]], Tuple[str, Union[str, int]]] = None,
        impute_empty_strings: bool = True,
        copy: bool = False,
    ) -> Optional[AnnData]:
        """KNN"""
        pass


class MissingImputeValuesError(Exception):
    pass


class ReplacementDatatypeError(Exception):
    pass


class MissingImputationValue(Exception):
    pass
