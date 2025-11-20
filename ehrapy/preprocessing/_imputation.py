from __future__ import annotations

import warnings
from collections.abc import Iterable
from functools import singledispatch
from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from ehrdata._logger import logger
from sklearn.experimental import enable_iterative_imputer  # noinspection PyUnresolvedReference
from sklearn.impute import SimpleImputer

from ehrapy import settings
from ehrapy._compat import (
    DaskArray,
    _apply_over_time_axis,
    _raise_array_type_not_implemented,
    function_2D_only,
    use_ehrdata,
)
from ehrapy._progress import spinner
from ehrapy.anndata import _check_feature_types
from ehrapy.anndata._feature_specifications import _infer_numerical_column_indices
from ehrapy.anndata.anndata_ext import _get_var_indices

if TYPE_CHECKING:
    from anndata import AnnData
    from ehrdata import EHRData


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
@spinner("Performing explicit impute")
def explicit_impute(
    edata: EHRData | AnnData,
    replacement: (str | int) | (dict[str, str | int]),
    *,
    layer: str | None = None,
    impute_empty_strings: bool = True,
    warning_threshold: int = 70,
    copy: bool = False,
) -> EHRData | AnnData | None:
    """Replaces all missing values in all columns or a subset of columns specified by the user with the passed replacement value.

    There are two scenarios to cover:
    1. Replace all missing values with the specified value.
    2. Replace all missing values in a subset of columns with a specified value per column.

    Args:
        edata: Central data object.
        replacement: The value to replace missing values with. If a dictionary is provided, the keys represent column
                     names and the values represent replacement values for those columns.
        layer: The layer to impute.
        impute_empty_strings: If True, empty strings are also replaced.
        warning_threshold: Threshold of percentage of missing values to display a warning for.
        copy: If True, returns a modified copy of the original data object. If False, modifies the object in place.

    Returns:
        If copy is True, a modified copy of the original data object with imputed X.
        If copy is False, the original data object is modified in place, and None is returned.

    Examples:
        Replace all missing values in edata with the value 0:

        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.explicit_impute(edata, replacement=0)
    """
    if copy:
        edata = edata.copy()

    X = edata.X if layer is None else edata.layers[layer]

    if isinstance(replacement, int) or isinstance(replacement, str):
        _warn_imputation_threshold(edata, var_names=list(edata.var_names), threshold=warning_threshold)
    else:
        _warn_imputation_threshold(edata, var_names=replacement.keys(), threshold=warning_threshold)  # type: ignore

    # 1: Replace all missing values with the specified value
    if isinstance(replacement, int | str):
        _replace_explicit(X, replacement, impute_empty_strings)

    # 2: Replace all missing values in a subset of columns with a specified value per column or a default value, when the column is not explicitly named
    elif isinstance(replacement, dict):
        for idx, column_name in enumerate(edata.var_names):
            imputation_value = _extract_impute_value(replacement, column_name)
            # only replace if an explicit value got passed or could be extracted from replacement
            if imputation_value:
                X[:, idx : idx + 1] = _replace_explicit(X[:, idx : idx + 1], imputation_value, impute_empty_strings)
            else:
                logger.warning(f"No replace value passed and found for var [not bold green]{column_name}.")
    else:
        raise ValueError(  # pragma: no cover
            f"Type {type(replacement)} is not a valid datatype for replacement parameter. Either use int, str or a dict!"
        )

    if layer is None:
        edata.X = X
    else:
        edata.layers[layer] = X

    return edata if copy else None


@singledispatch
def _replace_explicit(arr, replacement: str | int, impute_empty_strings: bool) -> None:
    _raise_array_type_not_implemented(_replace_explicit, type(arr))


@_replace_explicit.register(np.ndarray)
def _(arr: np.ndarray, replacement: str | int, impute_empty_strings: bool) -> np.ndarray:
    """Replace one column or whole X with a value where missing values are stored."""
    if not impute_empty_strings:  # pragma: no cover
        impute_conditions = pd.isnull(arr)
    else:
        impute_conditions = np.logical_or(pd.isnull(arr), arr == "")
    arr[impute_conditions] = replacement
    return arr


@_replace_explicit.register(DaskArray)
def _(arr: DaskArray, replacement: str | int, impute_empty_strings: bool) -> DaskArray:
    """Replace one column or whole X with a value where missing values are stored."""
    import dask.array as da

    if not impute_empty_strings:  # pragma: no cover
        impute_conditions = da.isnull(arr)
    else:
        impute_conditions = da.logical_or(da.isnull(arr), arr == "")
    arr[impute_conditions] = replacement
    return arr


def _extract_impute_value(replacement: dict[str, str | int], column_name: str) -> str | int | None:
    """Extract the replacement value for a given column in the data object.

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


@singledispatch
def _simple_impute_function(arr, strategy: Literal["mean", "median", "most_frequent"]) -> None:
    _raise_array_type_not_implemented(_simple_impute_function, type(arr))


@_simple_impute_function.register(sp.coo_array)
def _(arr: sp.coo_array, strategy: Literal["mean", "median", "most_frequent"]) -> sp.coo_array:
    _raise_array_type_not_implemented(_simple_impute_function, type(arr))


@_simple_impute_function.register(DaskArray)
@_apply_over_time_axis
def _(arr: DaskArray, strategy: Literal["mean", "median", "most_frequent"]) -> DaskArray:
    import dask_ml.impute

    arr_dtype = arr.dtype
    return dask_ml.impute.SimpleImputer(strategy=strategy).fit_transform(arr.astype(float)).astype(arr_dtype)


@_simple_impute_function.register(sp.csc_array)
@_simple_impute_function.register(sp.csr_array)
@_simple_impute_function.register(np.ndarray)
@_apply_over_time_axis
def _(arr: np.ndarray, strategy: Literal["mean", "median", "most_frequent"]) -> np.ndarray:
    import sklearn

    return sklearn.impute.SimpleImputer(strategy=strategy).fit_transform(arr)


@use_ehrdata(deprecated_after="1.0.0")
def simple_impute(
    edata: EHRData | AnnData,
    var_names: Iterable[str] | None = None,
    *,
    strategy: Literal["mean", "median", "most_frequent"] = "mean",
    warning_threshold: int = 70,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:
    """Impute missing values in numerical data using mean/median/most frequent imputation.

    If required and using mean or median strategy, the data needs to be properly encoded as this imputation requires
    numerical data only.

    Args:
        edata: Central data object.
        var_names: A list of column names to apply imputation on (if None, impute all columns).
        strategy: Imputation strategy to use. One of {'mean', 'median', 'most_frequent'}. If data is a `dask.array.Array`, only 'mean' is supported.
        warning_threshold: Display a warning message if percentage of missing values exceeds this threshold.
        layer: The layer to impute.
        copy: Whether to return a copy of `edata` or modify it inplace.

    Returns:
        If copy is True, a modified copy of the original data object with imputed X.
        If copy is False, the original data object is modified in place, and None is returned.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.simple_impute(edata, strategy="median")
    """
    if copy:
        edata = edata.copy()

    # TODO: warn again if qc_metrics is 3D enabled
    # _warn_imputation_threshold(edata, var_names, threshold=warning_threshold, layer=layer)

    var_indices = _get_var_indices(edata, edata.var_names if var_names is None else var_names)

    if layer is None:
        edata.X[:, var_indices] = _simple_impute_function(edata.X[:, var_indices], strategy)
    else:
        edata.layers[layer][:, var_indices] = _simple_impute_function(edata.layers[layer][:, var_indices], strategy)

    return edata if copy else None


@_check_feature_types
@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
@spinner("Performing KNN impute")
def knn_impute(
    edata: EHRData | AnnData,
    var_names: Iterable[str] | None = None,
    *,
    n_neighbors: int = 5,
    layer: str | None = None,
    copy: bool = False,
    backend: Literal["scikit-learn", "faiss"] = "faiss",
    warning_threshold: int = 70,
    backend_kwargs: dict | None = None,
    **kwargs,
) -> EHRData | AnnData | None:
    """Imputes missing values in the input data object using K-nearest neighbor imputation.

    If required, the data needs to be properly encoded as this imputation requires numerical data only.

    .. warning::
        Currently, both `n_neighbours` and `n_neighbors` are accepted as parameters for the number of neighbors.
        However, in future versions, only `n_neighbors` will be supported. Please update your code accordingly.


    Args:
        edata: Central data object.
        var_names: A list of variable names indicating which columns to impute.
                   If `None`, all columns are imputed. Default is `None`.
        n_neighbors: Number of neighbors to use when performing the imputation.
        layer: The layer to impute.
        copy: Whether to perform the imputation on a copy of the original data object.
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
        If copy is True, a modified copy of the original data object with imputed X.
        If copy is False, the original data object is modified in place, and None is returned.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.ad.infer_feature_types(edata)
        >>> ep.pp.knn_impute(edata)
    """
    if copy:
        edata = edata.copy()

    _warn_imputation_threshold(edata, var_names, threshold=warning_threshold, layer=layer)

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

    if find_spec("sklearnex") is not None:  # pragma: no cover
        from sklearnex import patch_sklearn, unpatch_sklearn

        patch_sklearn()

    _knn_impute(edata, var_names, n_neighbors, backend=backend, layer=layer, **backend_kwargs)

    if find_spec("sklearnex") is not None:  # pragma: no cover
        unpatch_sklearn()

    return edata if copy else None


def _knn_impute(
    edata: EHRData | AnnData,
    var_names: Iterable[str] | None,
    n_neighbors: int,
    layer: str | None,
    backend: Literal["scikit-learn", "faiss"],
    **kwargs,
) -> None:
    if backend == "scikit-learn":
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=n_neighbors, **kwargs)
    else:
        from fknni import FaissImputer

        imputer = FaissImputer(n_neighbors=n_neighbors, **kwargs)

    column_indices = _get_var_indices(edata, edata.var_names if var_names is None else var_names)
    numerical_indices = _infer_numerical_column_indices(
        edata,
    )
    if any(idx not in numerical_indices for idx in column_indices):
        raise ValueError(
            "Can only impute numerical data. Try to restrict imputation to certain columns using "
            "var_names parameter or perform an encoding of your data."
        )
    X = edata.X if layer is None else edata.layers[layer]
    complete_numerical_columns = np.array(numerical_indices)[~np.isnan(X[:, numerical_indices]).any(axis=0)].tolist()
    imputer_data_indices = column_indices + [i for i in complete_numerical_columns if i not in column_indices]
    imputer_x = X[::, imputer_data_indices].astype("float64")

    if layer is None:
        edata.X[::, imputer_data_indices] = imputer.fit_transform(imputer_x)
    else:
        edata.layers[layer][::, imputer_data_indices] = imputer.fit_transform(imputer_x)


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
@spinner("Performing miss-forest impute")
def miss_forest_impute(
    edata: EHRData | AnnData,
    var_names: Iterable[str] | None = None,
    *,
    num_initial_strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
    max_iter: int = 3,
    n_estimators: int = 100,
    random_state: int = 0,
    warning_threshold: int = 70,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:
    """Impute data using the MissForest strategy.

    This function uses the MissForest strategy to impute missing values in the data matrix of an data object.
    The strategy works by fitting a random forest model on each feature containing missing values,
    and using the trained model to predict the missing values.

    See https://academic.oup.com/bioinformatics/article/28/1/112/219101.

    If required, the data needs to be properly encoded as this imputation requires numerical data only.

    Args:
        edata: Central data object.
        var_names: Iterable of columns to impute
        num_initial_strategy: The initial strategy to replace all missing numerical values with.
        max_iter: The maximum number of iterations if the stop criterion has not been met yet.
        n_estimators: The number of trees to fit for every missing variable. Has a big effect on the run time.
                      Decrease for faster computations.
        random_state: The random seed for the initialization.
        warning_threshold: Threshold of percentage of missing values to display a warning for.
        layer: The layer to impute.
        copy: Whether to return a copy or act in place.

    Returns:
        If copy is True, a modified copy of the original data object with imputed X.
        If copy is False, the original data object is modified in place, and None is returned.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata = ep.pp.encode(edata, autodetect=True)
        >>> ep.pp.miss_forest_impute(edata)
    """
    if copy:
        edata = edata.copy()

    if var_names is None:
        _warn_imputation_threshold(edata, list(edata.var_names), threshold=warning_threshold, layer=layer)
    elif isinstance(var_names, Iterable) and all(isinstance(item, str) for item in var_names):
        _warn_imputation_threshold(edata, var_names, threshold=warning_threshold, layer=layer)

    if find_spec("sklearnex") is not None:  # pragma: no cover
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

        if isinstance(var_names, Iterable) and all(isinstance(item, str) for item in var_names):  # type: ignore
            num_indices = _get_var_indices(edata, var_names)
        else:
            num_indices = _get_var_indices(edata, edata.var_names)

        if set(num_indices).issubset(_get_non_numerical_column_indices(edata.X)):
            raise ValueError(
                "Can only impute numerical data. Try to restrict imputation to certain columns using "
                "var_names parameter."
            )

        # this step is the most expensive one and might extremely slow down the impute process
        if num_indices:
            if layer is None:
                edata.X[::, num_indices] = imp_num.fit_transform(edata.X[::, num_indices])
            else:
                edata.layers[layer][::, num_indices] = imp_num.fit_transform(edata.layers[layer][::, num_indices])
        else:
            raise ValueError("Cannot find any feature to perform imputation")

    except ValueError as e:
        if "Data matrix has wrong shape" in str(e):
            logger.error("Check that your matrix does not contain any NaN only columns!")
        raise

    if find_spec("sklearnex") is not None:  # pragma: no cover
        unpatch_sklearn()

    return edata if copy else None


@_check_feature_types
@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
@spinner("Performing mice-forest impute")
def mice_forest_impute(
    edata: EHRData | AnnData,
    var_names: Iterable[str] | None = None,
    *,
    warning_threshold: int = 70,
    save_all_iterations_data: bool = True,
    random_state: int | None = None,
    inplace: bool = False,
    iterations: int = 5,
    variable_parameters: dict | None = None,
    verbose: bool = False,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:
    """Impute data using the miceforest method.

    See https://github.com/AnotherSamWilson/miceforest
    Fast, memory efficient Multiple Imputation by Chained Equations (MICE) with lightgbm.

    If required, the data needs to be properly encoded as this imputation requires numerical data only.

    .. warning::
        This function is not supported on MacOS.

    Args:
        edata: Central data object.
        var_names: A list of variable names to impute. If None, impute all variables.
        warning_threshold: Threshold of percentage of missing values to display a warning for.
        save_all_iterations_data: Whether to save all imputed values from all iterations or just the latest.
                             Saving all iterations allows for additional plotting, but may take more memory.
        random_state: The random state ensures script reproducibility.
        inplace: If True, modify the input data object in-place and return None.
                 If False, return a copy of the modified data object. Default is False.
        iterations: The number of iterations to run.
        variable_parameters: Model parameters can be specified by variable here.
                             Keys should be variable names or indices, and values should be a dict of parameter which should apply to that variable only.
        verbose: Whether to print information about the imputation process.
        layer: The layer to impute.
        copy: Whether to return a copy of the data object or modify it in-place.

    Returns:
        If copy is True, a modified copy of the original data object with imputed X.
        If copy is False, the original data object is modified in place, and None is returned.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata = ep.pp.encode(edata, autodetect=True)
        >>> ep.pp.mice_forest_impute(edata)
    """
    if copy:
        edata = edata.copy()

    _warn_imputation_threshold(edata, var_names, threshold=warning_threshold, layer=layer)

    if any(
        idx not in _infer_numerical_column_indices(edata)
        for idx in _get_var_indices(edata, edata.var_names if var_names is None else var_names)
    ):
        raise ValueError(
            "Can only impute numerical data. Try to restrict imputation to certain columns using "
            "var_names parameter or perform an encoding of your data."
        )
    _miceforest_impute(
        edata,
        var_names,
        save_all_iterations_data,
        random_state,
        inplace,
        iterations,
        variable_parameters,
        verbose,
        layer,
    )

    return edata if copy else None


@singledispatch
def load_dataframe(arr, columns, index):
    _raise_array_type_not_implemented(load_dataframe, type(arr))


@load_dataframe.register
def _(arr: np.ndarray, columns, index):
    return pd.DataFrame(arr, columns=columns, index=index)


def _miceforest_impute(
    edata, var_names, save_all_iterations_data, random_state, inplace, iterations, variable_parameters, verbose, layer
) -> None:
    import miceforest as mf

    data_df = load_dataframe(
        edata.X if layer is None else edata.layers[layer], columns=edata.var_names, index=edata.obs_names
    )
    data_df = data_df.apply(pd.to_numeric, errors="coerce")

    if isinstance(var_names, Iterable) and all(isinstance(item, str) for item in var_names):
        column_indices = _get_var_indices(edata, var_names)
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

    if layer is None:
        edata.X = data_df.values
    else:
        edata.layers[layer] = data_df.values


def _warn_imputation_threshold(
    edata: EHRData | AnnData, var_names: Iterable[str] | None, threshold: int = 75, layer: str | None = None
) -> dict[str, int]:
    """Warns the user if the more than $threshold percent had to be imputed.

    Args:
        edata: The data object to check
        var_names: The var names which were imputed.
        threshold: A percentage value from 0 to 100 used as minimum.
        layer: The layer to check.
    """
    try:
        edata.var["missing_values_pct"]
    except KeyError:
        from ehrapy.preprocessing import qc_metrics

        qc_metrics(edata, layer=layer)
    used_var_names = set(edata.var_names) if var_names is None else set(var_names)

    thresholded_var_names = set(edata.var[edata.var["missing_values_pct"] > threshold].index) & set(used_var_names)

    var_name_to_pct: dict[str, int] = {}
    for var in thresholded_var_names:
        var_name_to_pct[var] = edata.var["missing_values_pct"].loc[var]
        logger.warning(f"Feature '{var}' had more than {var_name_to_pct[var]:.2f}% missing values!")

    return var_name_to_pct


def _get_non_numerical_column_indices(arr: np.ndarray) -> set:
    """Return indices of columns, that contain at least one non-numerical value that is not "Nan"."""

    def _is_float_or_nan(val) -> bool:  # pragma: no cover
        """Check whether a given item is a float or np.nan."""
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
