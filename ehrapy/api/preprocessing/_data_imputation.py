from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from ehrapy.api._anndata_util import get_column_indices


def explicit(adata: AnnData, **kwargs) -> AnnData:
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
        raise MissingImputeValuesError(
            "No replacement values were passed. Make sure passing a replacement parameter"
            "when using explicit data imputation mode!"
        ) from None

    # 1: Replace all missing values with the specified value
    if isinstance(replacement, (int, str)):
        _replace_explicit(adata.X, kwargs["replacement"], kwargs["impute_empty_strings"])

    # 2: Replace all missing values in a subset of columns with a specified value per column or a default value, when the column is not explicitly named
    elif isinstance(replacement, dict):
        for idx, column_name in enumerate(adata.var_names):
            imputation_value = _extract_impute_value(replacement, column_name)
            _replace_explicit(adata.X[:, idx : idx + 1], imputation_value, kwargs["impute_empty_strings"])
    else:
        raise ReplacementDatatypeError(
            f"Type {type(replacement)} is not a valid datatype for replacement parameter. Either use int, str or a dict!"
        )

    return adata


def _replace_explicit(x: np.ndarray, replacement: str | int, impute_empty_strings: str) -> None:
    """Replace one column or whole X with a value where missing values are stored."""
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
        raise MissingImputationValue(
            f"Could not find a replacement value for column {column_name} since None has been provided and"
            f"no default was found!"
        )


# ===================== Mean Imputation =========================


def _mean(adata: AnnData, **kwargs) -> AnnData:
    """
    Impute AnnData object using mean imputation. This works only for numerical data.

    Args:
        adata: The AnnData object to use mean Imputation on
        **kwargs: Keyword args
                  replacement: A list of strings indicating which columns to use mean imputation on (if None -> all columns)

    Returns:
            The imputed AnnData object

    """
    replacement = kwargs.get("replacement")
    imputer = SimpleImputer(strategy="mean")
    try:
        # impute a subset of columns
        if isinstance(replacement, list):
            column_indices = get_column_indices(adata, replacement)
            adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
        # impute all columns if None passed
        else:
            adata.X = imputer.fit_transform(adata.X)
    # Imputation using median strategy only works with numerical data
    except ValueError:
        raise ImputeStrategyNotAvailableError(
            "Can only impute numerical data using mean strategy. Try to restrict imputation"
            "to certain columns using replacement parameter or use a different mode."
        )

    return adata


def _median(adata: AnnData, **kwargs) -> AnnData:
    """
    Impute AnnData object using median imputation. This works only for numerical data.

    Args:
        adata: The AnnData object to use median Imputation on
        **kwargs: Keyword args
                  replacement: A list of strings indicating which columns to use median imputation on (if None -> all columns)

    Returns:
            The imputed AnnData object

    """
    replacement = kwargs.get("replacement")
    imputer = SimpleImputer(strategy="median")
    try:
        # impute a subset of columns
        if isinstance(replacement, list):
            column_indices = get_column_indices(adata, replacement)
            adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
        # impute all columns if None passed
        else:
            adata.X = imputer.fit_transform(adata.X)
    # Imputation using median strategy only works with numerical data
    except ValueError:
        raise ImputeStrategyNotAvailableError(
            "Can only impute numerical data using median strategy. Try to restrict imputation"
            "to certain columns using replacement parameter or use a different mode."
        )

    return adata


def _most_frequent(adata: AnnData, **kwargs) -> AnnData:
    """
    Impute AnnData object using most_frequent imputation. This works for both, numerical and non-numerical data.

    Args:
        adata: The AnnData object to use most_frequent Imputation on
        **kwargs: Keyword args
                  replacement: A list of strings indicating which columns to use most_frequent imputation on (if None -> all columns)

    Returns:
            The imputed AnnData object

    """
    replacement = kwargs.get("replacement")
    imputer = SimpleImputer(strategy="most_frequent")
    # impute a subset of columns
    if isinstance(replacement, list):
        column_indices = get_column_indices(adata, replacement)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    # impute all columns if None passed
    else:
        adata.X = imputer.fit_transform(adata.X)

    return adata


# ===================== KNN Imputation =========================


def _knn(adata: AnnData, **kwargs) -> AnnData:
    """Impute data using the KNN-Imputer.
    When using KNN Imputation with mixed data (non-numerical and numerical), encoding using ordinal encoding is required
    since KNN Imputation can only work on numerical data. The encoding itself is just a utility and will be redone once
    imputation ran successfully.

    Args:
        adata: The AnnData object to use KNN Imputation on
        **kwargs: Keyword args
                  replacement: A list of strings indicating which columns to use KNN imputation on (if None -> all columns)

    Returns:
            The imputed (but unencoded) AnnData object
    """
    # numerical only data needs no encoding since KNN Imputation can be applied directly
    if np.issubdtype(adata.X.dtype, np.number):
        _knn_impute(adata, **kwargs)
    else:
        # ordinal encoding is used since non-numerical data can not be imputed using KNN Imputation
        enc = OrdinalEncoder()
        adata.X = enc.fit_transform(adata.X)
        # impute the data using KNN imputation
        _knn_impute(adata, **kwargs)
        # decode ordinal encoding to obtain imputed original data
        adata.X = enc.inverse_transform(adata.X)

    return adata


def _knn_impute(adata: AnnData, **kwargs) -> None:
    """Utility function to impute data using KNN-Imputer"""
    n_neighbors = kwargs.get("n_neighbors")
    replacement = kwargs.get("replacement")
    imputer = KNNImputer(n_neighbors=n_neighbors if n_neighbors else 1)

    if isinstance(replacement, list):
        column_indices = get_column_indices(adata, replacement)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    # impute all columns if None passed
    else:
        adata.X = imputer.fit_transform(adata.X)


def _miss_forest(adata: AnnData, **kwargs) -> AnnData:
    """Impute data using the MissForest stratgey. See https://academic.oup.com/bioinformatics/article/28/1/112/219101.
    This requires the computation of which columns in X contain numerical only (including NaNs) and which mixed data, which could be an
    expensive operation on a larger X.
    Speeding this up could be achieved by passing the "replacement" kwarg and indicate, which column is
    numerical only and which mixed data.

    Args:
        adata: The AnnData object to use MissForest Imputation on
        **kwargs: Keyword args
                  replacement: A list of strings indicating which columns to use MissForest imputation on (if None -> all columns)

    Returns:
            The imputed (but unencoded) AnnData object
    """
    # TODO: passe replacement kwarg to indicate which column is numeric and which is mixed data (for those that should be imputed using MissForest strategy)
    non_num_indices = _get_non_numerical_column_indices(adata.X)
    num_indices = [ind for ind in range(adata.X.shape[1]) if ind not in non_num_indices]
    non_num_indices = list(non_num_indices)
    imp_num = IterativeImputer(estimator=RandomForestRegressor(), initial_strategy="mean", max_iter=1, random_state=0)
    imp_cat = IterativeImputer(
        estimator=RandomForestClassifier(), initial_strategy="most_frequent", max_iter=10, random_state=0
    )

    # encode all non numerical columns
    enc = OrdinalEncoder()
    adata.X[::, non_num_indices] = enc.fit_transform(adata.X[::, non_num_indices])
    # perform the imputation strategy on both, numerical and non numerical data
    adata.X[::, num_indices] = imp_num.fit_transform(adata.X[::, num_indices])
    adata.X[::, non_num_indices] = imp_cat.fit_transform(adata.X[::, non_num_indices])
    # decode ordinal encoding to obtain imputed original data
    adata.X[::, non_num_indices] = enc.inverse_transform(adata.X[::, non_num_indices])

    return adata


def _get_non_numerical_column_indices(X: np.ndarray) -> set:
    """Return indices of columns, that contain at least one non numerical value that is not "Nan"."""
    is_numeric_numpy = np.vectorize(_is_float, otypes=[bool])
    mask = np.apply_along_axis(is_numeric_numpy, 0, X)

    _, column_indices = np.where(~mask)
    non_num_indices = set(column_indices)

    return non_num_indices


def _is_float(val):
    try:
        float(val)
    except ValueError:
        if val is np.nan:
            return True
        return False
    else:
        # required to keep original booleans, otherwise those would be casted into ints; Keep or document?
        if val is not False and val is not True:
            return True
        else:
            return False


class MissingImputeValuesError(Exception):
    pass


class ReplacementDatatypeError(Exception):
    pass


class MissingImputationValue(Exception):
    pass


class ImputeStrategyNotAvailableError(Exception):
    pass
