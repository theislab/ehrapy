from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from rich import print
from rich.progress import Progress, SpinnerColumn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from ehrapy import logging as logg
from ehrapy import settings
from ehrapy.anndata._constants import EHRAPY_TYPE_KEY, NON_NUMERIC_TAG
from ehrapy.anndata.anndata_ext import _get_column_indices
from ehrapy.core._tool_available import _check_module_importable

if TYPE_CHECKING:
    from anndata import AnnData


def explicit_impute(
    adata: AnnData,
    replacement: (str | int) | (dict[str, str | int]),
    impute_empty_strings: bool = True,
    warning_threshold: int = 70,
    copy: bool = False,
) -> AnnData:
    """Replaces all missing values in all columns or a subset of columns specified by the user with the passed replacement value.

    There are two scenarios to cover:
    1. Replace all missing values with the specified value.
    2. Replace all missing values in a subset of columns with a specified value per column.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to impute values in.
        replacement: The value to replace missing values with. If a dictionary is provided, the keys represent column
                     names and the values represent replacement values for those columns.
        impute_empty_strings: If True, empty strings are also replaced. Defaults to True.
        warning_threshold: Threshold of percentage of missing values to display a warning for. Defaults to 30.
        copy: If True, returns a modified copy of the original AnnData object. If False, modifies the object in place.

    Returns:
        If copy is True, a modified copy of the original AnnData object with imputed X.
        If copy is False, the original AnnData object is modified in place.

    Examples:
        Replace all missing values in adata with the value 0:

        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.explicit_impute(adata, replacement=0)
    """
    if copy:  # pragma: no cover
        adata = adata.copy()

    if isinstance(replacement, int) or isinstance(replacement, str):
        _warn_imputation_threshold(adata, var_names=list(adata.var_names), threshold=warning_threshold)
    else:
        _warn_imputation_threshold(adata, var_names=replacement.keys(), threshold=warning_threshold)  # type: ignore

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        refresh_per_second=1500,
    ) as progress:
        progress.add_task("[blue]Running explicit imputation", total=1)
        # 1: Replace all missing values with the specified value
        if isinstance(replacement, (int, str)):
            _replace_explicit(adata.X, replacement, impute_empty_strings)
            logg.debug(f"Imputed missing values in the AnnData object by `{replacement}`")

        # 2: Replace all missing values in a subset of columns with a specified value per column or a default value, when the column is not explicitly named
        elif isinstance(replacement, dict):
            for idx, column_name in enumerate(adata.var_names):
                imputation_value = _extract_impute_value(replacement, column_name)
                # only replace if an explicit value got passed or could be extracted from replacement
                if imputation_value:
                    _replace_explicit(adata.X[:, idx : idx + 1], imputation_value, impute_empty_strings)
                else:
                    print(f"[bold yellow]No replace value passed and found for var [not bold green]{column_name}.")
            logg.debug(
                f"Imputed missing values in columns `{replacement.keys()}` by `{replacement.values()}` respectively."
            )
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
    if imputation_value:  # pragma: no cover
        return imputation_value
    else:
        return None


def simple_impute(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    strategy: Literal["mean", "median", "most_frequent"] = "mean",
    copy: bool = False,
    warning_threshold: int = 70,
) -> AnnData:
    """Impute missing values in numerical data using mean/median/most frequent imputation.

    Args:
        adata: The annotated data matrix to impute missing values on.
        var_names: A list of column names to apply imputation on (if None, impute all columns).
        strategy: Imputation strategy to use. One of {'mean', 'median', 'most_frequent'}.
        warning_threshold: Display a warning message if percentage of missing values exceeds this threshold. Defaults to 30.
        copy:Whether to return a copy of `adata` or modify it inplace. Defaults to False.

    Returns:
        An updated AnnData object with imputed values.

    Raises:
        ValueError:
            If the selected imputation strategy is not applicable to the data.
        ValueError:
            If an unknown imputation strategy is provided.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.simple_impute(adata, strategy="median")
    """
    if copy:
        adata = adata.copy()

    _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        refresh_per_second=1500,
    ) as progress:
        progress.add_task(f"[blue]Running simple imputation with {strategy}", total=1)
        # Imputation using median and mean strategy works with numerical data only
        if strategy in {"median", "mean"}:
            try:
                _simple_impute(adata, var_names, strategy)
                logg.debug(f"Imputed the AnnData object using `{strategy}` Imputation.")
            except ValueError:
                raise ValueError(
                    f"Can only impute numerical data using {strategy} strategy. Try to restrict imputation"
                    "to certain columns using var_names parameter or use a different mode."
                ) from None
        # most_frequent imputation works with non numerical data as well
        elif strategy == "most_frequent":
            _simple_impute(adata, var_names, strategy)
            logg.debug("Imputed the AnnData object using `most_frequent` Imputation.")
        # unknown simple imputation strategy
        else:
            raise ValueError(  # pragma: no cover
                f"Unknown impute strategy {strategy} for simple Imputation. Choose any of mean, median or most_frequent."
            ) from None

    if copy:
        return adata


def _simple_impute(adata: AnnData, var_names: Iterable[str] | None, strategy: str) -> None:
    imputer = SimpleImputer(strategy=strategy)
    if isinstance(var_names, Iterable):
        column_indices = _get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    else:
        adata.X = imputer.fit_transform(adata.X)


def knn_impute(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    n_neighbours: int = 5,
    copy: bool = False,
    warning_threshold: int = 70,
) -> AnnData:
    """Imputes missing values in the input AnnData object using K-nearest neighbor imputation.

    When using KNN Imputation with mixed data (non-numerical and numerical), encoding using ordinal encoding is required
    since KNN Imputation can only work on numerical data. The encoding itself is just a utility and will be undone once
    imputation ran successfully.

    Args:
        adata: An annotated data matrix containing gene expression values.
        var_names: A list of variable names indicating which columns to impute.
                   If `None`, all columns are imputed. Default is `None`.
        n_neighbours: Number of neighbors to use when performing the imputation. Defaults to 5.
        copy: Whether to perform the imputation on a copy of the original `AnnData` object.
              If `True`, the original object remains unmodified. Defaults to `False`.
        warning_threshold: Percentage of missing values above which a warning is issued. Defaults to 30.

    Returns:
        An updated AnnData object with imputed values.

    Raises:
        ValueError: If the input data matrix contains only categorical (non-numeric) values.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.knn_impute(adata)
    """
    if copy:
        adata = adata.copy()

    _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    if _check_module_importable("sklearnex"):  # pragma: no cover
        from sklearnex import patch_sklearn, unpatch_sklearn

        patch_sklearn()
    else:
        print(
            "[bold yellow]scikit-learn-intelex is not available. Install via [blue]pip install scikit-learn-intelex [yellow] for faster imputations."
        )
    try:
        with Progress(
            "[progress.description]{task.description}",
            SpinnerColumn(),
            refresh_per_second=1500,
        ) as progress:
            progress.add_task("[blue]Running KNN imputation", total=1)
            # numerical only data needs no encoding since KNN Imputation can be applied directly
            if np.issubdtype(adata.X.dtype, np.number):
                _knn_impute(adata, var_names, n_neighbours)
            else:
                # ordinal encoding is used since non-numerical data can not be imputed using KNN Imputation
                enc = OrdinalEncoder()
                column_indices = adata.var[EHRAPY_TYPE_KEY] == NON_NUMERIC_TAG
                adata.X[::, column_indices] = enc.fit_transform(adata.X[::, column_indices])
                # impute the data using KNN imputation
                _knn_impute(adata, var_names, n_neighbours)
                # imputing on encoded columns might result in float numbers; those can not be decoded
                # cast them to int to ensure they can be decoded
                adata.X[::, column_indices] = np.rint(adata.X[::, column_indices]).astype(int)
                # knn imputer transforms X dtype to numerical (encoded), but object is needed for decoding
                adata.X = adata.X.astype("object")
                # decode ordinal encoding to obtain imputed original data
                adata.X[::, column_indices] = enc.inverse_transform(adata.X[::, column_indices])
    except ValueError as e:
        if "Data matrix has wrong shape" in str(e):
            print("[bold red]Check that your matrix does not contain any NaN only columns!")
            raise

    if _check_module_importable("sklearnex"):  # pragma: no cover
        unpatch_sklearn()

    if var_names:
        logg.debug(
            f"Imputed the columns `{var_names}` in the AnnData object using kNN Imputation with {n_neighbours} neighbours considered."
        )
    elif not var_names:
        logg.debug(
            f"Imputed the data in the AnnData object using kNN Imputation with {n_neighbours} neighbours considered."
        )

    if copy:
        return adata


def _knn_impute(adata: AnnData, var_names: Iterable[str] | None, n_neighbours: int) -> None:
    """Utility function to impute data using KNN-Imputer"""
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=n_neighbours)

    if isinstance(var_names, Iterable):
        column_indices = _get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
        # this is required since X dtype has to be numerical in order to correctly round floats
        adata.X = adata.X.astype("float64")
    else:
        adata.X = imputer.fit_transform(adata.X)


def miss_forest_impute(
    adata: AnnData,
    var_names: dict[str, list[str]] | list[str] | None = None,
    num_initial_strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
    max_iter: int = 10,
    n_estimators=100,
    random_state: int = 0,
    warning_threshold: int = 70,
    copy: bool = False,
) -> AnnData:
    """Impute data using the MissForest strategy.

    This function uses the MissForest strategy to impute missing values in the data matrix of an AnnData object.
    The strategy works by fitting a random forest model on each feature containing missing values,
    and using the trained model to predict the missing values.

    See https://academic.oup.com/bioinformatics/article/28/1/112/219101.
    This requires the computation of which columns in X contain numerical only (including NaNs) and which contain non-numerical data.

    Args:
        adata: The AnnData object to use MissForest Imputation on.
        var_names: List of columns to impute or a dict with two keys ('numerical' and 'non_numerical') indicating which var
                   contain mixed data and which numerical data only.
        num_initial_strategy: The initial strategy to replace all missing numerical values with. Defaults to 'mean'.
        max_iter: The maximum number of iterations if the stop criterion has not been met yet.
        n_estimators: The number of trees to fit for every missing variable. Has a big effect on the run time.
                      Decrease for faster computations. Defaults to 100.
        random_state: The random seed for the initialization. Defaults to 0.
        warning_threshold: Threshold of percentage of missing values to display a warning for. Defaults to 30 .
        copy: Whether to return a copy or act in place. Defaults to False.

    Returns:
        The imputed (but unencoded) AnnData object.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.miss_forest_impute(adata)
    """
    if copy:  # pragma: no cover
        adata = adata.copy()

    if var_names is None:
        _warn_imputation_threshold(adata, list(adata.var_names), threshold=warning_threshold)
    elif isinstance(var_names, dict):
        _warn_imputation_threshold(adata, var_names.keys(), threshold=warning_threshold)  # type: ignore
    elif isinstance(var_names, list):
        _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    if _check_module_importable("sklearnex"):  # pragma: no cover
        from sklearnex import patch_sklearn, unpatch_sklearn

        patch_sklearn()
    else:
        print(
            "[bold yellow]scikit-learn-intelex is not available. Install via [blue]pip install scikit-learn-intelex [yellow] for faster imputations."
        )

    from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
    from sklearn.impute import IterativeImputer

    try:
        with Progress(
            "[progress.description]{task.description}",
            SpinnerColumn(),
            refresh_per_second=1500,
        ) as progress:
            progress.add_task("[blue]Running MissForest imputation", total=1)

            if settings.n_jobs == 1:  # pragma: no cover
                print(
                    "[bold yellow]The number of jobs is only 1. To decrease the runtime set [blue]ep.settings.n_jobs=-1."
                )

            imp_num = IterativeImputer(
                estimator=ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=settings.n_jobs),
                initial_strategy=num_initial_strategy,
                max_iter=max_iter,
                random_state=random_state,
            )
            # initial strategy here will not be parametrized since only most_frequent will be applied to non numerical data
            imp_cat = IterativeImputer(
                estimator=RandomForestClassifier(n_estimators=n_estimators, n_jobs=settings.n_jobs),
                initial_strategy="most_frequent",
                max_iter=max_iter,
                random_state=random_state,
            )

            if isinstance(var_names, list):
                var_indices = _get_column_indices(adata, var_names)  # type: ignore
                adata.X[::, var_indices] = imp_num.fit_transform(adata.X[::, var_indices])
            elif isinstance(var_names, dict) or var_names is None:
                if var_names:
                    try:
                        non_num_vars = var_names["non_numerical"]
                        num_vars = var_names["numerical"]
                    except KeyError:  # pragma: no cover
                        raise ValueError(
                            "One or both of your keys provided for var_names are unknown. Only "
                            "numerical and non_numerical are available!"
                        ) from None
                    non_num_indices = _get_column_indices(adata, non_num_vars)
                    num_indices = _get_column_indices(adata, num_vars)

                # infer non numerical and numerical indices automatically
                else:
                    non_num_indices_set = _get_non_numerical_column_indices(adata.X)
                    num_indices = [idx for idx in range(adata.X.shape[1]) if idx not in non_num_indices_set]
                    non_num_indices = list(non_num_indices_set)

                # encode all non numerical columns
                if non_num_indices:
                    enc = OrdinalEncoder()
                    adata.X[::, non_num_indices] = enc.fit_transform(adata.X[::, non_num_indices])
                # this step is the most expensive one and might extremely slow down the impute process
                if num_indices:
                    adata.X[::, num_indices] = imp_num.fit_transform(adata.X[::, num_indices])
                if non_num_indices:
                    adata.X[::, non_num_indices] = imp_cat.fit_transform(adata.X[::, non_num_indices])
                    adata.X[::, non_num_indices] = enc.inverse_transform(adata.X[::, non_num_indices])
    except ValueError as e:
        if "Data matrix has wrong shape" in str(e):
            print("[bold red]Check that your matrix does not contain any NaN only columns!")
            raise

    if _check_module_importable("sklearnex"):  # pragma: no cover
        unpatch_sklearn()

    if var_names:
        logg.debug(
            f"Imputed the columns `{var_names}` in the AnnData object with MissForest Imputation using {num_initial_strategy} strategy."
        )
    elif not var_names:
        logg.debug("Imputed the data in the AnnData object using MissForest Imputation.")

    if copy:
        return adata


def soft_impute(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    copy: bool = False,
    warning_threshold: int = 70,
    shrinkage_value: float | None = None,
    convergence_threshold: float = 0.001,
    max_iters: int = 100,
    max_rank: int | None = None,
    n_power_iterations: int = 1,
    init_fill_method: str = "zero",
    min_value: float | None = None,
    max_value: float | None = None,
    normalizer: object | None = None,
    verbose: bool = False,
) -> AnnData:
    """Impute data using the SoftImpute.

    See https://github.com/iskandr/fancyimpute/blob/master/fancyimpute/soft_impute.py
    Matrix completion by iterative soft thresholding of SVD decompositions.

    Args:
        adata: The AnnData object to impute missing values for.
        var_names: A list of var names indicating which columns to impute (if None -> all columns).
        copy: Whether to return a copy or act in place.
        warning_threshold: Threshold of percentage of missing values to display a warning for. Defaults to 30 .
        shrinkage_value : Value by which we shrink singular values on each iteration.
                          If omitted then the default value will be the maximum singular value of the initialized matrix (zeros for missing values) divided by 50.
        convergence_threshold : Minimum ration difference between iterations (as a fraction of the Frobenius norm of the current solution) before stopping.
        max_iters: Maximum number of SVD iterations. Defaults to 100.
        max_rank: Perform a truncated SVD on each iteration with this value as its rank. Defaults to None.
        n_power_iterations: Number of power iterations to perform with randomized SVD. Defaults to 1.
        init_fill_method: How to initialize missing values of data matrix, default is to fill them with zeros.
        min_value: Smallest allowable value in the solution.
        max_value: Largest allowable value in the solution.
        normalizer: Any object (such as BiScaler) with fit() and transform() methods.
        verbose: Print debugging info. Defaults to False.

    Returns:
        The AnnData object with imputed missing values.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.soft_impute(adata)
    """
    if copy:
        adata = adata.copy()

    _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        refresh_per_second=1500,
    ) as progress:
        progress.add_task("[blue]Running SoftImpute", total=1)
        if np.issubdtype(adata.X.dtype, np.number):
            _soft_impute(
                adata,
                var_names,
                shrinkage_value,
                convergence_threshold,
                max_iters,
                max_rank,
                n_power_iterations,
                init_fill_method,
                min_value,
                max_value,
                normalizer,
                verbose,
            )
        else:
            # ordinal encoding is used since non-numerical data can not be imputed using SoftImpute
            enc = OrdinalEncoder()
            column_indices = adata.var[EHRAPY_TYPE_KEY] == NON_NUMERIC_TAG
            adata.X[::, column_indices] = enc.fit_transform(adata.X[::, column_indices])
            # impute the data using SoftImpute
            _soft_impute(
                adata,
                var_names,
                shrinkage_value,
                convergence_threshold,
                max_iters,
                max_rank,
                n_power_iterations,
                init_fill_method,
                min_value,
                max_value,
                normalizer,
                verbose,
            )
            adata.X = adata.X.astype("object")
            # decode ordinal encoding to obtain imputed original data
            adata.X[::, column_indices] = enc.inverse_transform(adata.X[::, column_indices])

    if var_names:
        logg.debug(
            f"Imputed the columns `{var_names}` in the AnnData object using Soft Imputation with shrinkage value of `{shrinkage_value}`."
        )
    elif not var_names:
        logg.debug(
            f"Imputed the data in the AnnData object using Soft Imputation with shrinkage value of `{shrinkage_value}`."
        )

    return adata


def _soft_impute(
    adata: AnnData,
    var_names: Iterable[str] | None,
    shrinkage_value,
    convergence_threshold,
    max_iters,
    max_rank,
    n_power_iterations,
    init_fill_method,
    min_value,
    max_value,
    normalizer,
    verbose,
) -> None:
    """Utility function to impute data using SoftImpute"""
    from fancyimpute import SoftImpute

    imputer = SoftImpute(
        shrinkage_value,
        convergence_threshold,
        max_iters,
        max_rank,
        n_power_iterations,
        init_fill_method,
        min_value,
        max_value,
        normalizer,
        verbose,
    )

    if isinstance(var_names, Iterable):
        column_indices = _get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    else:
        adata.X = imputer.fit_transform(adata.X)


def iterative_svd_impute(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    copy: bool = False,
    warning_threshold: int = 70,
    rank: int = 10,
    convergence_threshold: float = 0.00001,
    max_iters: int = 200,
    gradual_rank_increase: bool = True,
    svd_algorithm: Literal["arpack", "randomized"] = "arpack",
    init_fill_method: Literal["zero", "mean", "median"] = "mean",
    min_value: float | None = None,
    max_value: float | None = None,
    verbose: bool = False,
) -> AnnData:
    """Impute missing values in an AnnData object using the IterativeSVD algorithm.

    The IterativeSVD algorithm is a matrix completion method based on iterative low-rank singular value decomposition (SVD).
    This function can impute missing values for numerical and ordinal-encoded data.

    Args:
        adata: An AnnData object to impute missing values in.
        var_names: A list of var names indicating which columns to impute. If `None`, all columns will be imputed.
                   Defaults to None.
        copy: Whether to return a copy of the AnnData object or act in place. Defaults to False.
        warning_threshold: Threshold of percentage of missing values to display a warning for. Defaults to 30.
        rank: Rank of the SVD decomposition. Defaults to 10.
        convergence_threshold: Convergence threshold for the iterative algorithm.
                               The algorithm stops when the relative difference in
                               Frobenius norm between two iterations is less than `convergence_threshold`.
                               Defaults to 0.00001.
        max_iters: Maximum number of iterations. The algorithm stops after `max_iters` iterations if it does not converge.
                   Defaults to 200.
        gradual_rank_increase: Whether to increase the rank gradually or to use the rank value immediately.
                               Defaults to True.
        svd_algorithm: The SVD algorithm to use. Can be one of {'arpack', 'randomized'}. Defaults to `arpack`.
        init_fill_method: The fill method to use for initializing missing values. Can be one of `{'zero', 'mean', 'median'}`.
                          Defaults to `mean`.
        min_value: The minimum value allowed for the imputed data. Any imputed value less than `min_value` is clipped to `min_value`.
                   Defaults to None.
        max_value: The maximum value allowed for the imputed data. Any imputed value greater than `max_value` is clipped to `max_value`.
                   Defaults to None.
        verbose: Whether to print progress messages during the imputation. Defaults to False.

    Returns:
        An AnnData object with imputed values.

    Raises:
        ValueError:
            If `svd_algorithm` is not one of `{'arpack', 'randomized'}`.
        ValueError:
            If `init_fill_method` is not one of `{'zero', 'mean', 'median'}`.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.iterative_svd_impute(adata)
    """
    if copy:
        adata = adata.copy()

    _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        refresh_per_second=1500,
    ) as progress:
        progress.add_task("[blue]Running IterativeSVD", total=1)
        if np.issubdtype(adata.X.dtype, np.number):
            _iterative_svd_impute(
                adata,
                var_names,
                rank,
                convergence_threshold,
                max_iters,
                gradual_rank_increase,
                svd_algorithm,
                init_fill_method,
                min_value,
                max_value,
                verbose,
            )
        else:
            # ordinal encoding is used since non-numerical data can not be imputed using IterativeSVD
            enc = OrdinalEncoder()
            column_indices = adata.var[EHRAPY_TYPE_KEY] == NON_NUMERIC_TAG
            adata.X[::, column_indices] = enc.fit_transform(adata.X[::, column_indices])
            # impute the data using IterativeSVD
            _iterative_svd_impute(
                adata,
                var_names,
                rank,
                convergence_threshold,
                max_iters,
                gradual_rank_increase,
                svd_algorithm,
                init_fill_method,
                min_value,
                max_value,
                verbose,
            )
            adata.X = adata.X.astype("object")
            # decode ordinal encoding to obtain imputed original data
            adata.X[::, column_indices] = enc.inverse_transform(adata.X[::, column_indices])

    if var_names:
        logg.debug(f"Imputed the columns `{var_names}` in the AnnData object using IterativeSVD Imputation.")
    elif not var_names:
        logg.debug("Imputed the data in the AnnData object using IterativeSVD Imputation.")

    return adata


def _iterative_svd_impute(
    adata,
    var_names,
    rank,
    convergence_threshold,
    max_iters,
    gradual_rank_increase,
    svd_algorithm,
    init_fill_method,
    min_value,
    max_value,
    verbose,
) -> None:
    """Utility function to impute data using IterativeSVD"""
    from fancyimpute import IterativeSVD

    imputer = IterativeSVD(
        rank,
        convergence_threshold,
        max_iters,
        gradual_rank_increase,
        svd_algorithm,
        init_fill_method,
        min_value,
        max_value,
        verbose,
    )

    if isinstance(var_names, Iterable):
        column_indices = _get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    else:
        adata.X = imputer.fit_transform(adata.X)


def matrix_factorization_impute(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    warning_threshold: int = 70,
    rank: int = 40,
    learning_rate: float = 0.01,
    max_iters: int = 50,
    shrinkage_value: float = 0,
    min_value: float | None = None,
    max_value: float | None = None,
    verbose: bool = False,
    copy: bool = False,
) -> AnnData:
    """Impute data using the MatrixFactorization.

    See https://github.com/iskandr/fancyimpute/blob/master/fancyimpute/matrix_factorization.py
    Train a matrix factorization model to predict empty entries in a matrix.

    Args:
        adata: The AnnData object to use MatrixFactorization on.
        var_names: A list of var names indicating which columns to impute (if None -> all columns).
        warning_threshold: Threshold of percentage of missing values to display a warning for. Defaults to 30 .
        rank: Number of latent factors to use in the matrix factorization model.
              It determines the size of the latent feature space that will be used to estimate the missing values.
              A higher rank will allow for more complex relationships between the features, but it can also lead to overfitting.
              Defaults to 40.
        learning_rate: The learning rate is the step size at which the optimization algorithm updates the model parameters during training.
                       A larger learning rate can lead to faster convergence, but if it is set too high, the optimization can become unstable.
                       Defaults to 0.01.
        max_iters: Maximum number of iterations to train the matrix factorization model for.
                   The algorithm stops once this number of iterations is reached, or if convergence is achieved earlier.
                   Defaults to 50.
        shrinkage_value: The shrinkage value is a regularization parameter that controls the amount of shrinkage applied to the estimated values during optimization.
                         This term is added to the loss function and serves to penalize large values in the estimated matrix.
                         A higher shrinkage value can help prevent overfitting, but can also lead to underfitting if set too high.
                         Defaults to 0.
        min_value: The minimum value allowed for the imputed data. Any imputed value less than `min_value` is clipped to `min_value`.
                   Defaults to None.
        max_value: The maximum value allowed for the imputed data. Any imputed value greater than `max_value` is clipped to `max_value`.
                   Defaults to None.
        verbose: Whether or not to printout training progress. Defaults to False.
        copy: Whether to return a copy or act in place. Defaults to False.

    Returns:
        The imputed AnnData object

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.matrix_factorization_impute(adata)
    """
    if copy:
        adata = adata.copy()

    _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        refresh_per_second=1500,
    ) as progress:
        progress.add_task("[blue]Running MatrixFactorization", total=1)
        if np.issubdtype(adata.X.dtype, np.number):
            _matrix_factorization_impute(
                adata,
                var_names,
                rank,
                learning_rate,
                max_iters,
                shrinkage_value,
                min_value,
                max_value,
                verbose,
            )
        else:
            # ordinal encoding is used since non-numerical data can not be imputed using MatrixFactorization
            enc = OrdinalEncoder()
            column_indices = adata.var[EHRAPY_TYPE_KEY] == NON_NUMERIC_TAG
            adata.X[::, column_indices] = enc.fit_transform(adata.X[::, column_indices])
            # impute the data using MatrixFactorization
            _matrix_factorization_impute(
                adata,
                var_names,
                rank,
                learning_rate,
                max_iters,
                shrinkage_value,
                min_value,
                max_value,
                verbose,
            )
            adata.X = adata.X.astype("object")
            adata.X[::, column_indices] = enc.inverse_transform(adata.X[::, column_indices])

    if var_names:
        logg.debug(
            f"Imputed the columns `{var_names}` in the AnnData object using MatrixFactorization Imputation with learning rate `{learning_rate}` and shrinkage value `{shrinkage_value}`."
        )
    elif not var_names:
        logg.debug(
            f"Imputed the data in the AnnData object using MatrixFactorization Imputation with learning rate `{learning_rate}` and shrinkage value `{shrinkage_value}`."
        )

    return adata


def _matrix_factorization_impute(
    adata,
    var_names,
    rank,
    learning_rate,
    max_iters,
    shrinkage_value,
    min_value,
    max_value,
    verbose,
) -> None:
    """Utility function to impute data using MatrixFactorization"""
    from fancyimpute import MatrixFactorization

    imputer = MatrixFactorization(
        rank,
        learning_rate,
        max_iters,
        shrinkage_value,
        min_value,
        max_value,
        verbose,
    )

    if isinstance(var_names, Iterable):
        column_indices = _get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    else:
        adata.X = imputer.fit_transform(adata.X)


def nuclear_norm_minimization_impute(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    warning_threshold: int = 70,
    require_symmetric_solution: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
    error_tolerance: float = 0.0001,
    max_iters: int = 50000,
    verbose: bool = False,
    copy: bool = False,
) -> AnnData:
    """Impute data using the NuclearNormMinimization.

    See https://github.com/iskandr/fancyimpute/blob/master/fancyimpute/nuclear_norm_minimization.py
    Simple implementation of "Exact Matrix Completion via Convex Optimization" by Emmanuel Candes and Benjamin Recht using cvxpy.

    Args:
        adata: The AnnData object to apply NuclearNormMinimization on.
        var_names: Var names indicating which columns to impute (if None -> all columns).
        warning_threshold: Threshold of percentage of missing values to display a warning for. Defaults to 30.
        require_symmetric_solution: Whether to add a symmetry constraint to the convex problem. Defaults to False.
        min_value: Smallest possible imputed value. Defaults to None (no minimum value constraint).
        max_value: Largest possible imputed value. Defaults to None (no maximum value constraint).
        error_tolerance: Degree of error allowed on reconstructed values. Defaults to 0.0001.
        max_iters: Maximum number of iterations for the convex solver. Defaults to 50000.
        verbose: Whether to print debug information. Defaults to False.
        copy: Whether to return a copy of the AnnData object or act in place. Defaults to False (act in place).

    Returns:
        The imputed AnnData object.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.nuclear_norm_minimization_impute(adata)
    """
    if copy:
        adata = adata.copy()

    _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        refresh_per_second=1500,
    ) as progress:
        progress.add_task("[blue]Running NuclearNormMinimization", total=1)
        if np.issubdtype(adata.X.dtype, np.number):
            _nuclear_norm_minimization_impute(
                adata,
                var_names,
                require_symmetric_solution,
                min_value,
                max_value,
                error_tolerance,
                max_iters,
                verbose,
            )
        else:
            # ordinal encoding is used since non-numerical data can not be imputed using NuclearNormMinimization
            enc = OrdinalEncoder()
            column_indices = adata.var[EHRAPY_TYPE_KEY] == NON_NUMERIC_TAG
            adata.X[::, column_indices] = enc.fit_transform(adata.X[::, column_indices])
            # impute the data using NuclearNormMinimization
            _nuclear_norm_minimization_impute(
                adata,
                var_names,
                require_symmetric_solution,
                min_value,
                max_value,
                error_tolerance,
                max_iters,
                verbose,
            )
            adata.X = adata.X.astype("object")
            # decode ordinal encoding to obtain imputed original data
            adata.X[::, column_indices] = enc.inverse_transform(adata.X[::, column_indices])

    if var_names:
        logg.debug(
            f"Imputed the columns `{var_names}` in the AnnData object using NuclearNormMinimization Imputation with error tolerance of `{error_tolerance}`."
        )
    elif not var_names:
        logg.debug(
            f"Imputed the data in the AnnData object using NuclearNormMinimization Imputation with error tolerance of `{error_tolerance}`."
        )

    return adata


def _nuclear_norm_minimization_impute(
    adata,
    var_names,
    require_symmetric_solution,
    min_value,
    max_value,
    error_tolerance,
    max_iters,
    verbose,
) -> None:
    """Utility function to impute data using NuclearNormMinimization"""
    from fancyimpute import NuclearNormMinimization

    imputer = NuclearNormMinimization(
        require_symmetric_solution,
        min_value,
        max_value,
        error_tolerance,
        max_iters,
        verbose,
    )

    if isinstance(var_names, list):
        column_indices = _get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    else:
        adata.X = imputer.fit_transform(adata.X)


def mice_forest_impute(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    warning_threshold: int = 70,
    save_all_iterations: bool = True,
    random_state: int | None = None,
    inplace: bool = False,
    iterations: int = 5,
    variable_parameters: dict | None = None,
    verbose: bool = False,
    copy: bool = False,
) -> AnnData:
    """Impute data using the miceforest.

    See https://github.com/AnotherSamWilson/miceforest
    Fast, memory efficient Multiple Imputation by Chained Equations (MICE) with lightgbm.

    Args:
        adata: The AnnData object containing the data to impute.
        var_names: A list of variable names to impute. If None, impute all variables.
        warning_threshold: Threshold of percentage of missing values to display a warning for.
                           Defaults to 30.
        save_all_iterations: Whether to save all imputed values from all iterations or just the latest.
                             Saving all iterations allows for additional plotting, but may take more memory. Defaults to True.
        random_state: The random state ensures script reproducibility.
                      Defaults to None.
        inplace: If True, modify the input AnnData object in-place and return None.
                 If False, return a copy of the modified AnnData object. Default is False.
        iterations: The number of iterations to run. Defaults to 5.
        variable_parameters: Model parameters can be specified by variable here.
                             Keys should be variable names or indices, and values should be a dict of parameter which should apply to that variable only.
                             Defaults to None.
        verbose: Whether to print information about the imputation process. Defaults to False.
        copy: Whether to return a copy of the AnnData object or modify it in-place. Defaults to False.

    Returns:
        The imputed AnnData object.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.mice_forest_impute(adata)
    """
    if copy:
        adata = adata.copy()

    _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)
    try:
        with Progress(
            "[progress.description]{task.description}",
            SpinnerColumn(),
            refresh_per_second=1500,
        ) as progress:
            progress.add_task("[blue]Running miceforest", total=1)
            if np.issubdtype(adata.X.dtype, np.number):
                _miceforest_impute(
                    adata,
                    var_names,
                    save_all_iterations,
                    random_state,
                    inplace,
                    iterations,
                    variable_parameters,
                    verbose,
                )
            else:
                # ordinal encoding is used since non-numerical data can not be imputed using miceforest
                enc = OrdinalEncoder()
                column_indices = adata.var[EHRAPY_TYPE_KEY] == NON_NUMERIC_TAG
                adata.X[::, column_indices] = enc.fit_transform(adata.X[::, column_indices])
                # impute the data using miceforest
                _miceforest_impute(
                    adata,
                    var_names,
                    save_all_iterations,
                    random_state,
                    inplace,
                    iterations,
                    variable_parameters,
                    verbose,
                )
                adata.X = adata.X.astype("object")
                # decode ordinal encoding to obtain imputed original data
                adata.X[::, column_indices] = enc.inverse_transform(adata.X[::, column_indices])
    except ValueError as e:
        if "Data matrix has wrong shape" in str(e):
            print("[bold red]Check that your matrix does not contain any NaN only columns!")
            raise

    if var_names:
        logg.debug(
            f"Imputed the columns `{var_names}` in the AnnData object using MiceForest Imputation with `{iterations}` iterations."
        )
    elif not var_names:
        logg.debug(
            f"Imputed the data in the AnnData object using MiceForest Imputation with `{iterations}` iterations."
        )

    return adata


def _miceforest_impute(
    adata, var_names, save_all_iterations, random_state, inplace, iterations, variable_parameters, verbose
) -> None:
    """Utility function to impute data using miceforest"""
    import miceforest as mf

    if isinstance(var_names, Iterable):
        column_indices = _get_column_indices(adata, var_names)

        # Create kernel.
        kernel = mf.ImputationKernel(
            adata.X[::, column_indices], datasets=1, save_all_iterations=save_all_iterations, random_state=random_state
        )

        kernel.mice(iterations=iterations, variable_parameters=variable_parameters, verbose=verbose)

        adata.X[::, column_indices] = kernel.complete_data(dataset=0, inplace=inplace)

    else:
        # Create kernel.
        kernel = mf.ImputationKernel(
            adata.X, datasets=1, save_all_iterations=save_all_iterations, random_state=random_state
        )

        kernel.mice(iterations=iterations, variable_parameters=variable_parameters, verbose=verbose)

        adata.X = kernel.complete_data(dataset=0, inplace=inplace)


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
        print("[bold yellow]Quality control metrics missing. Calculating...")
        from ehrapy.preprocessing import qc_metrics

        qc_metrics(adata)
    used_var_names = set(adata.var_names) if var_names is None else set(var_names)

    thresholded_var_names = set(adata.var[adata.var["missing_values_pct"] > threshold].index) & set(used_var_names)

    var_name_to_pct: dict[str, int] = {}
    for var in thresholded_var_names:
        var_name_to_pct[var] = adata.var["missing_values_pct"].loc[var]
        print(
            f"[bold yellow]Feature [blue]{var} [yellow]had more than [blue]{var_name_to_pct[var]:.2f}% [yellow]missing values!"
        )

    return var_name_to_pct


def _get_non_numerical_column_indices(X: np.ndarray) -> set:
    """Return indices of columns, that contain at least one non numerical value that is not "Nan"."""

    def _is_float_or_nan(val):  # pragma: no cover
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

    is_numeric_numpy = np.vectorize(_is_float_or_nan, otypes=[bool])
    mask = np.apply_along_axis(is_numeric_numpy, 0, X)

    _, column_indices = np.where(~mask)
    non_num_indices = set(column_indices)

    return non_num_indices
