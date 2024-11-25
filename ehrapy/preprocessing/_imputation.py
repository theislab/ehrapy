from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from lamin_utils import logger
from sklearn.experimental import enable_iterative_imputer  # noinspection PyUnresolvedReference
from sklearn.impute import SimpleImputer

from ehrapy import settings
from ehrapy._utils_available import _check_module_importable
from ehrapy._utils_rendering import spinner
from ehrapy.anndata import check_feature_types
from ehrapy.anndata.anndata_ext import get_column_indices

if TYPE_CHECKING:
    from anndata import AnnData


@spinner("Performing explicit impute")
def explicit_impute(
    adata: AnnData,
    replacement: (str | int) | (dict[str, str | int]),
    *,
    impute_empty_strings: bool = True,
    warning_threshold: int = 70,
    copy: bool = False,
) -> AnnData | None:
    """Replaces all missing values in all columns or a subset of columns specified by the user with the passed replacement value.

    There are two scenarios to cover:
    1. Replace all missing values with the specified value.
    2. Replace all missing values in a subset of columns with a specified value per column.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to impute values in.
        replacement: The value to replace missing values with. If a dictionary is provided, the keys represent column
                     names and the values represent replacement values for those columns.
        impute_empty_strings: If True, empty strings are also replaced.
        warning_threshold: Threshold of percentage of missing values to display a warning for.
        copy: If True, returns a modified copy of the original AnnData object. If False, modifies the object in place.

    Returns:
        If copy is True, a modified copy of the original AnnData object with imputed X.
        If copy is False, the original AnnData object is modified in place, and None is returned.

    Examples:
        Replace all missing values in adata with the value 0:

        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.explicit_impute(adata, replacement=0)
    """
    if copy:
        adata = adata.copy()

    if isinstance(replacement, int) or isinstance(replacement, str):
        _warn_imputation_threshold(adata, var_names=list(adata.var_names), threshold=warning_threshold)
    else:
        _warn_imputation_threshold(adata, var_names=replacement.keys(), threshold=warning_threshold)  # type: ignore

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
                logger.warning(f"No replace value passed and found for var [not bold green]{column_name}.")
    else:
        raise ValueError(  # pragma: no cover
            f"Type {type(replacement)} is not a valid datatype for replacement parameter. Either use int, str or a dict!"
        )

    if copy:
        return adata


def _replace_explicit(arr: np.ndarray, replacement: str | int, impute_empty_strings: bool) -> None:
    """Replace one column or whole X with a value where missing values are stored."""
    if not impute_empty_strings:  # pragma: no cover
        impute_conditions = pd.isnull(arr)
    else:
        impute_conditions = np.logical_or(pd.isnull(arr), arr == "")
    arr[impute_conditions] = replacement


def _extract_impute_value(replacement: dict[str, str | int], column_name: str) -> str | int | None:
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
    if imputation_value:  # pragma: no cover
        return imputation_value
    else:
        return None


@spinner("Performing simple impute")
def simple_impute(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    *,
    strategy: Literal["mean", "median", "most_frequent"] = "mean",
    copy: bool = False,
    warning_threshold: int = 70,
) -> AnnData | None:
    """Impute missing values in numerical data using mean/median/most frequent imputation.

    If required and using mean or median strategy, the data needs to be properly encoded as this imputation requires
    numerical data only.

    Args:
        adata: The annotated data matrix to impute missing values on.
        var_names: A list of column names to apply imputation on (if None, impute all columns).
        strategy: Imputation strategy to use. One of {'mean', 'median', 'most_frequent'}.
        warning_threshold: Display a warning message if percentage of missing values exceeds this threshold.
        copy:Whether to return a copy of `adata` or modify it inplace.

    Returns:
        If copy is True, a modified copy of the original AnnData object with imputed X.
        If copy is False, the original AnnData object is modified in place, and None is returned.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.simple_impute(adata, strategy="median")
    """
    if copy:
        adata = adata.copy()

    _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    if strategy in {"median", "mean"}:
        try:
            _simple_impute(adata, var_names, strategy)
        except ValueError:
            raise ValueError(
                f"Can only impute numerical data using {strategy} strategy. Try to restrict imputation "
                "to certain columns using var_names parameter or use a different mode."
            ) from None
    # most_frequent imputation works with non-numerical data as well
    elif strategy == "most_frequent":
        _simple_impute(adata, var_names, strategy)
    else:
        raise ValueError(
            f"Unknown impute strategy {strategy} for simple Imputation. Choose any of mean, median or most_frequent."
        ) from None

    if copy:
        return adata


def _simple_impute(adata: AnnData, var_names: Iterable[str] | None, strategy: str) -> None:
    imputer = SimpleImputer(strategy=strategy)
    if isinstance(var_names, Iterable) and all(isinstance(item, str) for item in var_names):
        column_indices = get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    else:
        adata.X = imputer.fit_transform(adata.X)


@spinner("Performing KNN impute")
@check_feature_types
def knn_impute(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    *,
    n_neighbors: int = 5,
    copy: bool = False,
    backend: Literal["scikit-learn", "faiss"] = "faiss",
    warning_threshold: int = 70,
    backend_kwargs: dict | None = None,
    **kwargs,
) -> AnnData :
    """Imputes missing values in the input AnnData object using K-nearest neighbor imputation.

    If required, the data needs to be properly encoded as this imputation requires numerical data only.

    .. warning::
        Currently, both `n_neighbours` and `n_neighbors` are accepted as parameters for the number of neighbors.
        However, in future versions, only `n_neighbors` will be supported. Please update your code accordingly.


    Args:
        adata: An annotated data matrix containing EHR data.
        var_names: A list of variable names indicating which columns to impute.
                   If `None`, all columns are imputed. Default is `None`.
        n_neighbors: Number of neighbors to use when performing the imputation.
        copy: Whether to perform the imputation on a copy of the original `AnnData` object.
              If `True`, the original object remains unmodified.
        backend: The implementation to use for the KNN imputation.
                 'scikit-learn' is very slow but uses an exact KNN algorithm, whereas 'faiss'
                 is drastically faster but uses an approximation for the KNN graph.
                 In practice, 'faiss' is close enough to the 'scikit-learn' results.
        warning_threshold: Percentage of missing values above which a warning is issued.
        backend_kwargs: Passed to the backend.
                  Pass "mean", "median", or "weighted" for 'strategy' to set the imputation strategy for faiss.
                  See `sklearn.impute.KNNImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html>`_ for more information on the 'scikit-learn' backend.
                  See `fknni.faiss.FaissImputer <https://fknni.readthedocs.io/en/latest/>`_ for more information on the 'faiss' backend.
        kwargs: Gathering keyword arguments of earlier ehrapy versions for backwards compatibility. It is encouraged to use the here listed, current arguments.

    Returns:
        If copy is True, a modified copy of the original AnnData object with imputed X.
        If copy is False, the original AnnData object is modified in place, and None is returned.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.ad.infer_feature_types(adata)
        >>> ep.pp.knn_impute(adata)
    """
    if copy:
        adata = adata.copy()

    _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    if backend not in {"scikit-learn", "faiss"}:
        raise ValueError(f"Unknown backend '{backend}' for KNN imputation. Choose between 'scikit-learn' and 'faiss'.")

    if backend_kwargs is None:
        backend_kwargs = {}

    valid_kwargs = {"n_neighbours"}
    unexpected_kwargs = set(kwargs.keys()) - valid_kwargs

    if unexpected_kwargs:
        raise ValueError(f"Unexpected keyword arguments: {unexpected_kwargs}.")

    if "n_neighbours" in kwargs.keys():
        n_neighbors = kwargs["n_neighbours"]
        warnings.warn(
            "ehrapy will use 'n_neighbors' instead of 'n_neighbours'. Please update your code.",
            DeprecationWarning,
            stacklevel=1,
        )

    if _check_module_importable("sklearnex"):  # pragma: no cover
        from sklearnex import patch_sklearn, unpatch_sklearn

        patch_sklearn()

    try:
        if np.issubdtype(adata.X.dtype, np.number):
            _knn_impute(adata, var_names, n_neighbors, backend=backend, **backend_kwargs)
        else:
            # Raise exception since non-numerical data can not be imputed using KNN Imputation
            raise ValueError(
                "Can only impute numerical data. Try to restrict imputation to certain columns using "
                "var_names parameter or perform an encoding of your data."
            )

    except ValueError as e:
        if "Data matrix has wrong shape" in str(e):
            logger.error("Check that your matrix does not contain any NaN only columns!")
        raise

    if _check_module_importable("sklearnex"):  # pragma: no cover
        unpatch_sklearn()

    if copy:
        return adata


def _knn_impute(
    adata: AnnData,
    var_names: Iterable[str] | None,
    n_neighbors: int,
    backend: Literal["scikit-learn", "faiss"],
    **kwargs,
) -> None:
    if backend == "scikit-learn":
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=n_neighbors, **kwargs)
    else:
        from fknni import FaissImputer

        imputer = FaissImputer(n_neighbors=n_neighbors, **kwargs)

    if isinstance(var_names, Iterable) and all(isinstance(item, str) for item in var_names):
        column_indices = get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
        # this is required since X dtype has to be numerical in order to correctly round floats
        adata.X = adata.X.astype("float64")
    else:
        adata.X = imputer.fit_transform(adata.X)


@spinner("Performing miss-forest impute")
def miss_forest_impute(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    *,
    num_initial_strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
    max_iter: int = 3,
    n_estimators: int = 100,
    random_state: int = 0,
    warning_threshold: int = 70,
    copy: bool = False,
) -> AnnData | None:
    """Impute data using the MissForest strategy.

    This function uses the MissForest strategy to impute missing values in the data matrix of an AnnData object.
    The strategy works by fitting a random forest model on each feature containing missing values,
    and using the trained model to predict the missing values.

    See https://academic.oup.com/bioinformatics/article/28/1/112/219101.

    If required, the data needs to be properly encoded as this imputation requires numerical data only.

    Args:
        adata: The AnnData object to use MissForest Imputation on.
        var_names: Iterable of columns to impute
        num_initial_strategy: The initial strategy to replace all missing numerical values with.
        max_iter: The maximum number of iterations if the stop criterion has not been met yet.
        n_estimators: The number of trees to fit for every missing variable. Has a big effect on the run time.
                      Decrease for faster computations.
        random_state: The random seed for the initialization.
        warning_threshold: Threshold of percentage of missing values to display a warning for.
        copy: Whether to return a copy or act in place.

    Returns:
        If copy is True, a modified copy of the original AnnData object with imputed X.
        If copy is False, the original AnnData object is modified in place, and None is returned.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.miss_forest_impute(adata)
    """
    if copy:
        adata = adata.copy()

    if var_names is None:
        _warn_imputation_threshold(adata, list(adata.var_names), threshold=warning_threshold)
    elif isinstance(var_names, Iterable) and all(isinstance(item, str) for item in var_names):
        _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    if _check_module_importable("sklearnex"):  # pragma: no cover
        from sklearnex import patch_sklearn, unpatch_sklearn

        patch_sklearn()

    from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
    from sklearn.impute import IterativeImputer

    try:
        imp_num = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=settings.n_jobs),
            initial_strategy=num_initial_strategy,
            max_iter=max_iter,
            random_state=random_state,
        )
        # initial strategy here will not be parametrized since only most_frequent will be applied to non numerical data
        IterativeImputer(
            estimator=RandomForestClassifier(n_estimators=n_estimators, n_jobs=settings.n_jobs),
            initial_strategy="most_frequent",
            max_iter=max_iter,
            random_state=random_state,
        )

        if isinstance(var_names, Iterable) and all(isinstance(item, str) for item in var_names): # type: ignore
            num_indices = get_column_indices(adata, var_names)
        else:
            num_indices = get_column_indices(adata, adata.var_names)

        if set(num_indices).issubset(_get_non_numerical_column_indices(adata.X)):
            raise ValueError(
                "Can only impute numerical data. Try to restrict imputation to certain columns using "
                "var_names parameter."
            )

        # this step is the most expensive one and might extremely slow down the impute process
        if num_indices:
            adata.X[::, num_indices] = imp_num.fit_transform(adata.X[::, num_indices])
        else:
            raise ValueError("Cannot find any feature to perform imputation")

    except ValueError as e:
        if "Data matrix has wrong shape" in str(e):
            logger.error("Check that your matrix does not contain any NaN only columns!")
        raise

    if _check_module_importable("sklearnex"):  # pragma: no cover
        unpatch_sklearn()

    if copy:
        return adata


@spinner("Performing mice-forest impute")
@check_feature_types
def mice_forest_impute(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    *,
    warning_threshold: int = 70,
    save_all_iterations_data: bool = True,
    random_state: int | None = None,
    inplace: bool = False,
    iterations: int = 5,
    variable_parameters: dict | None = None,
    verbose: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Impute data using the miceforest.

    See https://github.com/AnotherSamWilson/miceforest
    Fast, memory efficient Multiple Imputation by Chained Equations (MICE) with lightgbm.

    If required, the data needs to be properly encoded as this imputation requires numerical data only.

    Args:
        adata: The AnnData object containing the data to impute.
        var_names: A list of variable names to impute. If None, impute all variables.
        warning_threshold: Threshold of percentage of missing values to display a warning for.
        save_all_iterations_data: Whether to save all imputed values from all iterations or just the latest.
                             Saving all iterations allows for additional plotting, but may take more memory.
        random_state: The random state ensures script reproducibility.
        inplace: If True, modify the input AnnData object in-place and return None.
                 If False, return a copy of the modified AnnData object. Default is False.
        iterations: The number of iterations to run.
        variable_parameters: Model parameters can be specified by variable here.
                             Keys should be variable names or indices, and values should be a dict of parameter which should apply to that variable only.
        verbose: Whether to print information about the imputation process.
        copy: Whether to return a copy of the AnnData object or modify it in-place.

    Returns:
        If copy is True, a modified copy of the original AnnData object with imputed X.
        If copy is False, the original AnnData object is modified in place, and None is returned.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.ad.infer_feature_types(adata)
        >>> ep.pp.mice_forest_impute(adata)
    """
    if copy:
        adata = adata.copy()

    _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    try:
        if np.issubdtype(adata.X.dtype, np.number):
            _miceforest_impute(
                adata,
                var_names,
                save_all_iterations_data,
                random_state,
                inplace,
                iterations,
                variable_parameters,
                verbose,
            )
        else:
            raise ValueError(
                "Can only impute numerical data. Try to restrict imputation to certain columns using "
                "var_names parameter."
            )

    except ValueError as e:
        if "Data matrix has wrong shape" in str(e):
            logger.warning("Check that your matrix does not contain any NaN only columns!")
        raise

    if copy:
        return adata


def _miceforest_impute(
    adata, var_names, save_all_iterations_data, random_state, inplace, iterations, variable_parameters, verbose
) -> None:
    import miceforest as mf

    data_df = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
    data_df = data_df.apply(pd.to_numeric, errors="coerce")

    if isinstance(var_names, Iterable) and all(isinstance(item, str) for item in var_names):
        column_indices = get_column_indices(adata, var_names)
        selected_columns = data_df.iloc[:, column_indices]
        selected_columns = selected_columns.reset_index(drop=True)

        kernel = mf.ImputationKernel(
            selected_columns,
            num_datasets=1,
            save_all_iterations_data=save_all_iterations_data,
            random_state=random_state,
        )

        kernel.mice(iterations=iterations, variable_parameters=variable_parameters or {}, verbose=verbose)
        data_df.iloc[:, column_indices] = kernel.complete_data(dataset=0, inplace=inplace)

    else:
        data_df = data_df.reset_index(drop=True)

        kernel = mf.ImputationKernel(
            data_df, num_datasets=1, save_all_iterations_data=save_all_iterations_data, random_state=random_state
        )

        kernel.mice(iterations=iterations, variable_parameters=variable_parameters or {}, verbose=verbose)
        data_df = kernel.complete_data(dataset=0, inplace=inplace)

    adata.X = data_df.values


def _warn_imputation_threshold(adata: AnnData, var_names: Iterable[str] | None, threshold: int = 75) -> dict[str, int]:
    """Warns the user if the more than $threshold percent had to be imputed.

    Args:
        adata: The AnnData object to check
        var_names: The var names which were imputed.
        threshold: A percentage value from 0 to 100 used as minimum.
    """
    try:
        adata.var["missing_values_pct"]
    except KeyError:
        from ehrapy.preprocessing import qc_metrics

        qc_metrics(adata)
    used_var_names = set(adata.var_names) if var_names is None else set(var_names)

    thresholded_var_names = set(adata.var[adata.var["missing_values_pct"] > threshold].index) & set(used_var_names)

    var_name_to_pct: dict[str, int] = {}
    for var in thresholded_var_names:
        var_name_to_pct[var] = adata.var["missing_values_pct"].loc[var]
        logger.warning(f"Feature '{var}' had more than {var_name_to_pct[var]:.2f}% missing values!")

    return var_name_to_pct


def _get_non_numerical_column_indices(arr: np.ndarray) -> set:
    """Return indices of columns, that contain at least one non-numerical value that is not "Nan"."""

    def _is_float_or_nan(val) -> bool:  # pragma: no cover
        """Check whether a given item is a float or np.nan"""
        try:
            _ = float(val)
            return not isinstance(val, bool)
        except (ValueError, TypeError):
            return False

    def _is_float_or_nan_row(row) -> list[bool]:  # pragma: no cover
        return [_is_float_or_nan(val) for val in row]

    mask = np.apply_along_axis(_is_float_or_nan_row, 0, arr)
    _, column_indices = np.where(~mask)

    return set(column_indices)
