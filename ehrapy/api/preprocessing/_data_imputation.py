from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from rich import print
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from ehrapy.api.anndata_ext import get_column_indices


def explicit_impute(
    adata: AnnData,
    replacement: (str | int) | (dict[str, str | int]),
    impute_empty_strings: bool = True,
    copy: bool = False,
) -> AnnData:
    """Replaces all missing values in all or the specified columns with the passed value

    There are two scenarios to cover:
    1. Replace all missing values with the specified value.
    2. Replace all missing values in a subset of columns with a specified value per column.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to impute values in.
        replacement: Value to use as replacement or optionally keys to indicate which columns to replace with which value.
        impute_empty_strings: Whether to also impute empty strings.
        copy: Whether to return a copy with the imputed data.

    Returns:
        :class:`~anndata.AnnData` object with imputed X

    Example:
        .. code-block:: python

            import ehrapy.api as ep

            adata = ep.dt.mimic_2(encode=True)
            ep.pp.explicit_impute(adata, replacement=0)
    """
    if copy:
        adata = adata.copy()
    # 1: Replace all missing values with the specified value
    if isinstance(replacement, (int, str)):
        _replace_explicit(adata.X, replacement, impute_empty_strings)

    # 2: Replace all missing values in a subset of columns with a specified value per column or a default value, when the column is not explicitly named
    elif isinstance(replacement, dict):
        for idx, column_name in enumerate(adata.var_names):
            imputation_value = _extract_impute_value(replacement, column_name)
            # only replace if an explicit value got passed or could be extracted from replacement
            if imputation_value:
                _replace_explicit(adata.X[:, idx : idx + 1], imputation_value, impute_empty_strings)
            else:
                print(f"[bold yellow]No replace value passed and found for var [not bold green]{column_name}.")
    else:
        raise ReplacementDatatypeError(
            f"Type {type(replacement)} is not a valid datatype for replacement parameter. Either use int, str or a dict!"
        )

    return adata


def _replace_explicit(x: np.ndarray, replacement: str | int, impute_empty_strings: bool) -> None:
    """Replace one column or whole X with a value where missing values are stored."""
    if not impute_empty_strings:
        impute_conditions = pd.isnull(x)
    else:
        impute_conditions = np.logical_or(pd.isnull(x), x == "")
    x[impute_conditions] = replacement


def _extract_impute_value(replacement: dict[str, str | int], column_name: str) -> str | int:
    """Extract the replacement value for a given column in the :class:`~anndata.AnnData` object

    Returns:
        The value to replace missing values
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
        return None


# ===================== Simple Imputation =========================


def simple_impute(
    adata: AnnData, var_names: list[str] | None = None, strategy: str = "mean", copy: bool = False
) -> AnnData:
    """Impute AnnData object using mean imputation. This works for numerical data only.

    Args:
        adata: The AnnData object to use mean Imputation on
        var_names: A list of var names indicating which columns to use mean imputation on (if None -> all columns)
        strategy: Any of mean/median/most_frequent to indicate which strategy to use for simple imputation
        copy: Whether to return a copy or act in place

    Returns:
        The imputed AnnData object

    Example:
        .. code-block:: python

            import ehrapy.api as ep

            adata = ep.dt.mimic_2(encode=True)
            ep.pp.simple_impute(adata, strategy="median")
    """
    if copy:
        adata = adata.copy()
    # Imputation using median and mean strategy works with numerical data only
    if strategy in {"median", "mean"}:
        try:
            _simple_impute(adata, var_names, strategy)
        except ValueError:
            raise ImputeStrategyNotAvailableError(
                f"Can only impute numerical data using {strategy} strategy. Try to restrict imputation"
                "to certain columns using var_names parameter or use a different mode."
            )
    # most_frequent imputation works with non numerical data as well
    elif strategy == "most_frequent":
        _simple_impute(adata, var_names, strategy)
    # unknown simple imputation strategy
    else:
        raise UnknownImputeStrategyError(
            f"Unknown impute strategy {strategy} for simple Imputation. Choose any of mean, median or most_frequent."
        )

    return adata


def _simple_impute(adata: AnnData, var_names: list[str] | None, strategy: str) -> None:
    imputer = SimpleImputer(strategy=strategy)
    # impute a subset of columns
    if isinstance(var_names, list):
        column_indices = get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    # impute all columns if None passed
    else:
        adata.X = imputer.fit_transform(adata.X)


# ===================== KNN Imputation =========================


def knn_impute(adata: AnnData, var_names: list[str] | None = None, copy: bool = False) -> AnnData:
    """Impute data using the KNN-Imputer.

    When using KNN Imputation with mixed data (non-numerical and numerical), encoding using ordinal encoding is required
    since KNN Imputation can only work on numerical data. The encoding itself is just a utility and will be undone once
    imputation ran successfully.

    Args:
        adata: The AnnData object to use KNN Imputation on
        var_names: A list of var names indicating which columns to use median imputation on (if None -> all columns)
        copy: Whether to return a copy or act in place

    Returns:
        The imputed (but unencoded) AnnData object

    Example:
        .. code-block:: python

            import ehrapy.api as ep

            adata = ep.dt.mimic_2(encode=True)
            ep.pp.knn_impute(adata)
    """
    if copy:
        adata = adata.copy()
    # numerical only data needs no encoding since KNN Imputation can be applied directly
    if np.issubdtype(adata.X.dtype, np.number):
        _knn_impute(adata, var_names)
    else:
        # ordinal encoding is used since non-numerical data can not be imputed using KNN Imputation
        enc = OrdinalEncoder()
        adata.X = enc.fit_transform(adata.X)
        # impute the data using KNN imputation
        _knn_impute(adata, var_names)
        # decode ordinal encoding to obtain imputed original data
        adata.X = enc.inverse_transform(adata.X)

    return adata


def _knn_impute(adata: AnnData, var_names: list[str] | None) -> None:
    """Utility function to impute data using KNN-Imputer"""
    imputer = KNNImputer(n_neighbors=1)

    if isinstance(var_names, list):
        column_indices = get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    # impute all columns if None passed
    else:
        adata.X = imputer.fit_transform(adata.X)


# ======================  MissForest Impuation =======================


def miss_forest_impute(
    adata: AnnData,
    var_names: dict[str, list[str]] | None = None,
    num_initial_strategy: str = "mean",
    max_iter: int = 10,
    random_state: int = 0,
    copy: bool = False,
) -> AnnData:
    """Impute data using the MissForest strategy.

    See https://academic.oup.com/bioinformatics/article/28/1/112/219101.
    This requires the computation of which columns in X contain numerical only (including NaNs)
    and which contain non-numerical data. This is an expensive operation on X with many numerical vars resulting in a long runtime.

    Args:
        adata: The AnnData object to use MissForest Imputation on
        var_names: An optional dict with two keys (numerical and non_numerical) indicating which var contains mixed data and which numerical data only
        copy: Whether to return a copy or act in place

    Returns:
        The imputed (but unencoded) AnnData object

    Example:
        .. code-block:: python

            import ehrapy.api as ep

            adata = ep.dt.mimic_2(encode=True)
            ep.pp.miss_forest_impute(adata)
    """
    if copy:
        adata = adata.copy()
    # var names got passed for faster indices lookup
    if var_names:
        # ensure both keys got passed together
        try:
            non_num_vars = var_names["non_numerical"]
            num_vars = var_names["numerical"]
        except KeyError:
            raise MissForestKeyError(
                "One or both of your keys provided for var_names are unknown. Only "
                "numerical and non_numerical are available!"
            )
        # get the indices from the var names
        non_num_indices = get_column_indices(adata, non_num_vars)
        num_indices = get_column_indices(adata, num_vars)

    # infer non numerical and numerical indices automatically
    else:
        non_num_indices_set = _get_non_numerical_column_indices(adata.X)
        num_indices = [idx for idx in range(adata.X.shape[1]) if idx not in non_num_indices_set]
        non_num_indices = list(non_num_indices_set)

    imp_num = IterativeImputer(
        estimator=ExtraTreesRegressor(),
        initial_strategy=num_initial_strategy,
        max_iter=max_iter,
        random_state=random_state,
    )
    # initial strategy here will not be parametrized since only most_frequent will be applied to non numerical data
    imp_cat = IterativeImputer(
        estimator=RandomForestClassifier(),
        initial_strategy="most_frequent",
        max_iter=max_iter,
        random_state=random_state,
    )

    # encode all non numerical columns
    if non_num_indices:
        enc = OrdinalEncoder()
        adata.X[::, non_num_indices] = enc.fit_transform(adata.X[::, non_num_indices])
    # this step is the most expensive one and might extremely slow down the impute process
    if num_indices:
        adata.X[::, num_indices] = imp_num.fit_transform(adata.X[::, num_indices])
    if non_num_indices:
        adata.X[::, non_num_indices] = imp_cat.fit_transform(adata.X[::, non_num_indices])
        # decode ordinal encoding to obtain imputed original data
        adata.X[::, non_num_indices] = enc.inverse_transform(adata.X[::, non_num_indices])

    return adata


def _get_non_numerical_column_indices(X: np.ndarray) -> set:
    """Return indices of columns, that contain at least one non numerical value that is not "Nan"."""
    is_numeric_numpy = np.vectorize(_is_float_or_nan, otypes=[bool])
    mask = np.apply_along_axis(is_numeric_numpy, 0, X)

    _, column_indices = np.where(~mask)
    non_num_indices = set(column_indices)

    return non_num_indices


def _is_float_or_nan(val):
    """Check whether a given item is a float or np.nan"""
    try:
        float(val)
    except ValueError:
        if val is np.nan:
            return True
        return False
    else:
        if not isinstance(val, bool):
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


class UnknownImputeStrategyError(Exception):
    pass


class MissForestKeyError(Exception):
    pass
