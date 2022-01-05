from typing import Dict, List, Optional, Union

from anndata import AnnData

from ehrapy.api.preprocessing._data_imputation import _explicit, _knn, _mean, _median, _miss_forest, _most_frequent
from ehrapy.api.preprocessing._quality_control import calculate_qc_metrics
from ehrapy.api.preprocessing._scanpy_pp_api import *  # noqa: E402,F403


def mean_impute(adata: AnnData, var_names: Optional[List[str]] = None, copy: bool = False) -> AnnData:
    """
    Impute AnnData object using mean imputation. This works for numerical data only.

    Args:
        adata: The AnnData object to use mean Imputation on
        var_names: A list of var names indicating which columns to use mean imputation on (if None -> all columns)
        copy: Whether to return a copy or act in place

    Returns:
           The imputed AnnData object

    """
    if copy:
        adata = adata.copy()

    return _mean(adata, var_names)


def median_impute(adata: AnnData, var_names: Optional[List[str]] = None, copy: bool = False) -> AnnData:
    """
    Impute AnnData object using median imputation. This works for numerical data only.

    Args:
        adata: The AnnData object to use median Imputation on
        var_names: A list of var names indicating which columns to use median imputation on (if None -> all columns)
        copy: Whether to return a copy or act in place

    Returns:
            The imputed AnnData object

    """
    if copy:
        adata = adata.copy()

    return _median(adata, var_names)


def most_frequent_impute(adata: AnnData, var_names: Optional[List[str]] = None, copy: bool = False) -> AnnData:
    """
    Impute AnnData object using most_frequent imputation. This works for both, numerical and non-numerical data.

    Args:
        adata: The AnnData object to use most_frequent Imputation on
        var_names: A list of var names indicating which columns to use median imputation on (if None -> all columns)
        copy: Whether to return a copy or act in place

    Returns:
            The imputed AnnData object

    """
    if copy:
        adata = adata.copy()

    return _most_frequent(adata, var_names)


def knn_impute(adata: AnnData, var_names: Optional[List[str]] = None, copy: bool = False) -> AnnData:
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
    """
    if copy:
        adata = adata.copy()

    return _knn(adata, var_names)


def miss_forest_impute(adata: AnnData, var_names: Optional[Dict[str, List[str]]] = None, copy: bool = False) -> AnnData:
    """Impute data using the MissForest strategy. See https://academic.oup.com/bioinformatics/article/28/1/112/219101.
    This requires the computation of which columns in X contain numerical only (including NaNs) and which non numerical data, which is an
    expensive operation on X with many numerical vars resulting in a long runtime.

    Args:
        adata: The AnnData object to use MissForest Imputation on
        var_names: An optional dict with two keys (numerical and non_numerical) indicating which var contains mixed data and which numerical data only
        copy: Whether to return a copy or act in place

    Returns:
            The imputed (but unencoded) AnnData object

    """
    if copy:
        adata = adata.copy()

    return _miss_forest(adata, var_names)


def explicit_impute(
    adata: AnnData,
    replacement: Union[Union[str, int], Dict[str, Union[str, int]]],
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
    """
    if copy:
        adata = adata.copy()

    return _explicit(adata, replacement, impute_empty_strings)
