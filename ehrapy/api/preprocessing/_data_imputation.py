from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from anndata import AnnData

from ehrapy.api._anndata_util import get_column_indices


def explicit(
    adata: AnnData,
    copy: bool = False,
    **kwargs
) -> AnnData | None:
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
    # ensure replacement parameter has been passed when using explicit impute mode
    try:
        replacement = kwargs["replacement"]
    except KeyError:
        raise MissingImputeValuesError("No replacement values were passed. Make sure passing a replacement parameter"
                                       "when using explicit data imputation mode!") from None

    # 1: Replace all missing values with the specified value
    if isinstance(replacement, (int, str)):
        _replace_explicit(adata.X, kwargs["replacement"], kwargs["impute_empty_strings"])

    # 2: Replace all missing values in a subset of columns with a specified value per column or a default value, when the column is not explicitly named
    elif isinstance(replacement, dict):
        for idx, column_name in enumerate(adata.var_names):
            imputation_value = _extract_impute_value(replacement, column_name)
            _replace_explicit(adata.X[:, idx:idx+1], imputation_value, kwargs["impute_empty_strings"])
    else:
        raise ReplacementDatatypeError(f"Type {type(replacement)} is not a valid datatype for replacement parameter. Either use int, str or a dict!")

    return adata


def _replace_explicit(x: np.ndarray, replacement: str | int, impute_empty_strings: str) -> None:
    """Replace one column or whole X with a value where missing values are stored.
    """
    if not impute_empty_strings:
        impute_conditions = pd.isnull(x)
    else:
        impute_conditions = np.logical_or(pd.isnull(x), x == "")
    x[impute_conditions] = replacement


def _extract_impute_value(replacement: dict[str, str | int], column_name: str) -> str | int:
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


def mean(
    adata: AnnData,
    copy: bool = False,
    **kwargs
) -> AnnData | None:
    """MEAN"""
    pass

# ===================== KNN Imputation =========================


def knn(
    adata: AnnData,
    copy: bool = False,
    **kwargs
) -> AnnData | None:
    """KNN"""
    # KNN impute requires non-numerical data to be encoded
    if not np.issubdtype(adata.X.dtype, np.number):
        raise KNNImputationNonNumericalDataError("Trying to impute on non numerical data using KNN Imputation.\n"
              "KNN only works on numerical data, so encode your non-numerical data first and impute afterwards!")

    n_neighbors = kwargs.get("n_neighbors")
    replacement = kwargs.get("replacement")
    imputer = KNNImputer(n_neighbors=n_neighbors if n_neighbors else 5)
    # only impute some columns using KNN imputation
    if isinstance(replacement, list):
        column_indices = get_column_indices(adata, replacement)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    # impute all columns if None passed
    else:
        adata.X = imputer.fit_transform(adata.X)

    return adata


class MissingImputeValuesError(Exception):
    pass


class ReplacementDatatypeError(Exception):
    pass


class MissingImputationValue(Exception):
    pass


class KNNImputationNonNumericalDataError(Exception):
    pass

