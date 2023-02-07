from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from rich import print
from rich.progress import Progress, SpinnerColumn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from ehrapy import logging as logg
from ehrapy import settings
from ehrapy.anndata.anndata_ext import get_column_indices
from ehrapy.core._tool_available import _check_module_importable


def explicit_impute(
    adata: AnnData,
    replacement: (str | int) | (dict[str, str | int]),
    impute_empty_strings: bool = True,
    warning_threshold: int = 30,
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
        warning_threshold: Threshold of percentage of missing values to display a warning for (default: 30)
        copy: Whether to return a copy with the imputed data.

    Returns:
        :class:`~anndata.AnnData` object with imputed X

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encoded=True)
            ep.pp.explicit_impute(adata, replacement=0)
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
            raise ReplacementDatatypeError(  # pragma: no cover
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
    var_names: list[str] | None = None,
    strategy: str = "mean",
    copy: bool = False,
    warning_threshold: int = 30,
) -> AnnData:
    """Impute AnnData object using mean/median/most frequent imputation. This works for numerical data only.

    Args:
        adata: The AnnData object to use mean Imputation on
        var_names: A list of var names indicating which columns to use mean imputation on (if None -> all columns)
        strategy: Any of mean/median/most_frequent to indicate which strategy to use for simple imputation
        warning_threshold: Threshold of percentage of missing values to display a warning for (default: 30)
        copy: Whether to return a copy or act in place

    Returns:
        The imputed AnnData object

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encoded=True)
            ep.pp.simple_impute(adata, strategy="median")
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
                raise ImputeStrategyNotAvailableError(
                    f"Can only impute numerical data using {strategy} strategy. Try to restrict imputation"
                    "to certain columns using var_names parameter or use a different mode."
                )
        # most_frequent imputation works with non numerical data as well
        elif strategy == "most_frequent":
            _simple_impute(adata, var_names, strategy)
            logg.debug("Imputed the AnnData object using `most_frequent` Imputation.")
        # unknown simple imputation strategy
        else:
            raise UnknownImputeStrategyError(  # pragma: no cover
                f"Unknown impute strategy {strategy} for simple Imputation. Choose any of mean, median or most_frequent."
            )

    if copy:
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


def knn_impute(
    adata: AnnData,
    var_names: list[str] | None = None,
    n_neighbours: int = 5,
    copy: bool = False,
    warning_threshold: int = 30,
) -> AnnData:
    """Impute data using the KNN-Imputer.

    When using KNN Imputation with mixed data (non-numerical and numerical), encoding using ordinal encoding is required
    since KNN Imputation can only work on numerical data. The encoding itself is just a utility and will be undone once
    imputation ran successfully.

    Args:
        adata: The AnnData object to use KNN Imputation on
        var_names: A list of var names indicating which columns to use median imputation on (if None -> all columns)
        n_neighbours: Number of neighbours to consider while imputing (default: 5)
        warning_threshold: Threshold of percentage of missing values to display a warning for (default: 30)
        copy: Whether to return a copy or act in place

    Returns:
        The imputed (but unencoded) AnnData object

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encoded=True)
            ep.pp.knn_impute(adata)
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
            column_indices = get_column_indices(adata, adata.uns["non_numerical_columns"])
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


def _knn_impute(adata: AnnData, var_names: list[str] | None, n_neighbours: int) -> None:
    """Utility function to impute data using KNN-Imputer"""
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=n_neighbours)

    if isinstance(var_names, list):
        column_indices = get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
        # this is required since X dtype has to be numerical in order to correctly round floats
        adata.X = adata.X.astype("float64")
    else:
        adata.X = imputer.fit_transform(adata.X)


def miss_forest_impute(
    adata: AnnData,
    var_names: dict[str, list[str]] | list[str] | None = None,
    num_initial_strategy: str = "mean",
    max_iter: int = 10,
    n_estimators=100,
    random_state: int = 0,
    warning_threshold: int = 30,
    copy: bool = False,
) -> AnnData:
    """Impute data using the MissForest strategy.

    See https://academic.oup.com/bioinformatics/article/28/1/112/219101.
    This requires the computation of which columns in X contain numerical only (including NaNs) and which contain non-numerical data.

    Args:
        adata: The AnnData object to use MissForest Imputation on.
        var_names: List of columns to impute or a dict with two keys ('numerical' and 'non_numerical') indicating which var
                   contain mixed data and which numerical data only.
        num_initial_strategy: The initial strategy to replace all missing values with (default: 'mean').
        max_iter: The maximum number of iterations if the stop criterion has not been met yet.
        n_estimators: The number of trees to fit for every missing variable. Has a big effect on the run time.
                      Decrease for faster computations (default: 100).
        random_state: The random seed for the initialization.
        warning_threshold: Threshold of percentage of missing values to display a warning for (default: 30).
        copy: Whether to return a copy or act in place.

    Returns:
        The imputed (but unencoded) AnnData object.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encoded=True)
            ep.pp.miss_forest_impute(adata)
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

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        refresh_per_second=1500,
    ) as progress:
        progress.add_task("[blue]Running MissForest imputation", total=1)

        if settings.n_jobs == 1:  # pragma: no cover
            print("[bold yellow]The number of jobs is only 1. To decrease the runtime set [blue]ep.settings.n_jobs=-1.")

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
            var_indices = get_column_indices(adata, var_names)  # type: ignore
            adata.X[::, var_indices] = imp_num.fit_transform(adata.X[::, var_indices])
        elif isinstance(var_names, dict) or var_names is None:
            if var_names:
                try:
                    non_num_vars = var_names["non_numerical"]
                    num_vars = var_names["numerical"]
                except KeyError:  # pragma: no cover
                    raise MissForestKeyError(
                        "One or both of your keys provided for var_names are unknown. Only "
                        "numerical and non_numerical are available!"
                    )
                non_num_indices = get_column_indices(adata, non_num_vars)
                num_indices = get_column_indices(adata, num_vars)

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
    var_names: list[str] | None = None,
    copy: bool = False,
    warning_threshold: int = 30,
    shrinkage_value: float | None = None,
    convergence_threshold: float = 0.001,
    max_iters: int = 100,
    max_rank: int | None = None,
    n_power_iterations: int = 1,
    init_fill_method: str = "zero",
    min_value: float | None = None,
    max_value: float | None = None,
    normalizer: object | None = None,
    verbose: bool = True,
) -> AnnData:
    """Impute data using the SoftImpute.

    See https://github.com/iskandr/fancyimpute/blob/master/fancyimpute/soft_impute.py
    Matrix completion by iterative soft thresholding of SVD decompositions.

    Args:
        adata: The AnnData object to use SoftImpute on.
        var_names: A list of var names indicating which columns to impute (if None -> all columns).
        copy: Whether to return a copy or act in place.
        warning_threshold: Threshold of percentage of missing values to display a warning for (default: 30).
        shrinkage_value : Value by which we shrink singular values on each iteration. If omitted then the default value will be the maximum singular value of the initialized matrix (zeros for missing values) divided by 50.
        convergence_threshold : Minimum ration difference between iterations (as a fraction of the Frobenius norm of the current solution) before stopping.
        max_iters: Maximum number of SVD iterations.
        max_rank: Perform a truncated SVD on each iteration with this value as its rank.
        n_power_iterations: Number of power iterations to perform with randomized SVD.
        init_fill_method: How to initialize missing values of data matrix, default is to fill them with zeros.
        min_value: Smallest allowable value in the solution.
        max_value: Largest allowable value in the solution.
        normalizer: Any object (such as BiScaler) with fit() and transform() methods.
        verbose: Print debugging info.

    Returns:
        The imputed AnnData object

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encoded=True)
            ep.pp.soft_impute(adata)
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
            column_indices = get_column_indices(adata, adata.uns["non_numerical_columns"])
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
    var_names: list[str] | None,
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

    if isinstance(var_names, list):
        column_indices = get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    else:
        adata.X = imputer.fit_transform(adata.X)


def iterative_svd_impute(
    adata: AnnData,
    var_names: list[str] | None = None,
    copy: bool = False,
    warning_threshold: int = 30,
    rank: int = 10,
    convergence_threshold: float = 0.00001,
    max_iters: int = 200,
    gradual_rank_increase: bool = True,
    svd_algorithm: str = "arpack",
    init_fill_method: str = "zero",
    min_value: float | None = None,
    max_value: float | None = None,
    verbose: bool = True,
) -> AnnData:
    """Impute data using the IterativeSVD.

    See https://github.com/iskandr/fancyimpute/blob/master/fancyimpute/iterative_svd.py
    Matrix completion by iterative low-rank SVD decomposition.

    Args:
        adata: The AnnData object to use IterativeSVD on.
        var_names: A list of var names indicating which columns to impute (if None -> all columns).
        copy: Whether to return a copy or act in place.
        warning_threshold: Threshold of percentage of missing values to display a warning for (default: 30).

    Returns:
        The imputed AnnData object

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encoded=True)
            ep.pp.iterative_svd_impute(adata)
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
            column_indices = get_column_indices(adata, adata.uns["non_numerical_columns"])
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

    if isinstance(var_names, list):
        column_indices = get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    else:
        adata.X = imputer.fit_transform(adata.X)


def matrix_factorization_impute(
    adata: AnnData,
    var_names: list[str] | None = None,
    copy: bool = False,
    warning_threshold: int = 30,
    rank: int = 40,
    learning_rate: float = 0.01,
    max_iters: int = 50,
    shrinkage_value: float = 0,
    min_value: float | None = None,
    max_value: float | None = None,
    verbose: bool = True,
) -> AnnData:
    """Impute data using the MatrixFactorization.

    See https://github.com/iskandr/fancyimpute/blob/master/fancyimpute/matrix_factorization.py
    Train a matrix factorization model to predict empty entries in a matrix.

    Args:
        adata: The AnnData object to use MatrixFactorization on.
        var_names: A list of var names indicating which columns to impute (if None -> all columns).
        copy: Whether to return a copy or act in place.
        warning_threshold: Threshold of percentage of missing values to display a warning for (default: 30).
        rank: Number of latent factors to use in matrix factorization model
        learning_rate: Learning rate for optimizer
        max_iters: Number of max_iters to train for
        shrinkage_value: Regularization term for sgd penalty
        min_value: Smallest possible imputed value
        max_value: Largest possible imputed value
        verbose: Whether or not to printout training progress

    Returns:
        The imputed AnnData object

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encoded=True)
            ep.pp.matrix_factorization_impute(adata)
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
            column_indices = get_column_indices(adata, adata.uns["non_numerical_columns"])
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
            # decode ordinal encoding to obtain imputed original data
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

    if isinstance(var_names, list):
        column_indices = get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    else:
        adata.X = imputer.fit_transform(adata.X)


def nuclear_norm_minimization_impute(
    adata: AnnData,
    var_names: list[str] | None = None,
    copy: bool = False,
    warning_threshold: int = 30,
    require_symmetric_solution: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
    error_tolerance: float = 0.0001,
    max_iters: int = 50000,
    verbose: bool = True,
) -> AnnData:
    """Impute data using the NuclearNormMinimization.

    See https://github.com/iskandr/fancyimpute/blob/master/fancyimpute/nuclear_norm_minimization.py
    Simple implementation of "Exact Matrix Completion via Convex Optimization" by Emmanuel Candes and Benjamin Recht using cvxpy.

    Args:
        adata: The AnnData object to use NuclearNormMinimization on.
        var_names: A list of var names indicating which columns to impute (if None -> all columns).
        copy: Whether to return a copy or act in place.
        warning_threshold: Threshold of percentage of missing values to display a warning for (default: 30).
        require_symmetric_solution: Add symmetry constraint to convex problem
        min_value: Smallest possible imputed value
        max_value: Largest possible imputed value
        error_tolerance: Degree of error allowed on reconstructed values. If omitted then defaults to 0.0001
        max_iters: Maximum number of iterations for the convex solver
        verbose: Print debug info

    Returns:
        The imputed AnnData object

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encoded=True)
            ep.pp.nuclear_norm_minimization_impute(adata)
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
            column_indices = get_column_indices(adata, adata.uns["non_numerical_columns"])
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
        column_indices = get_column_indices(adata, var_names)
        adata.X[::, column_indices] = imputer.fit_transform(adata.X[::, column_indices])
    else:
        adata.X = imputer.fit_transform(adata.X)


def mice_forest_impute(
    adata: AnnData,
    var_names: list[str] | None = None,
    copy: bool = False,
    warning_threshold: int = 30,
    save_all_iterations: bool = True,
    random_state: int | None = None,
    inplace: bool = False,
    iterations: int = 5,
    variable_parameters: dict | None = None,
    verbose: bool = False,
) -> AnnData:
    """Impute data using the miceforest.

    See https://github.com/AnotherSamWilson/miceforest
    Fast, memory efficient Multiple Imputation by Chained Equations (MICE) with lightgbm.

    Args:
        adata: The AnnData object to use miceforest on.
        var_names: A list of var names indicating which columns to impute (if None -> all columns).
        copy: Whether to return a copy or act in place.
        warning_threshold: Threshold of percentage of missing values to display a warning for (default: 30).
        save_all_iterations: Save all the imputation values from all iterations, or just the latest. Saving all iterations allows for additional plotting, but may take more memory.
        random_state: The random_state ensures script reproducibility. It only ensures reproducible results if the same script is called multiple times. It does not guarantee reproducible results at the record level, if a record is imputed multiple different times. If reproducible record-results are desired, a seed must be passed for each record in the random_seed_array parameter.
        inplace: Using inplace=False returns a copy of the completed data. Since the raw data is already stored in kernel.working_data, you can set inplace=True to complete the data without returning a copy.
        iterations: The number of iterations to run.
        variable_parameters: Model parameters can be specified by variable here. Keys should be variable names or indices, and values should be a dict of parameter which should apply to that variable only.
        verbose: Should information about the process be printed.
    Returns:
        The imputed AnnData object

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encoded=True)
            ep.pp.miceforest_impute(adata)
    """
    if copy:
        adata = adata.copy()

    _warn_imputation_threshold(adata, var_names, threshold=warning_threshold)

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        refresh_per_second=1500,
    ) as progress:
        progress.add_task("[blue]Running miceforest", total=1)
        if np.issubdtype(adata.X.dtype, np.number):
            _miceforest_impute(
                adata, var_names, save_all_iterations, random_state, inplace, iterations, variable_parameters, verbose
            )
        else:
            # ordinal encoding is used since non-numerical data can not be imputed using miceforest
            enc = OrdinalEncoder()
            column_indices = get_column_indices(adata, adata.uns["non_numerical_columns"])
            adata.X[::, column_indices] = enc.fit_transform(adata.X[::, column_indices])
            # impute the data using miceforest
            _miceforest_impute(
                adata, var_names, save_all_iterations, random_state, inplace, iterations, variable_parameters, verbose
            )
            adata.X = adata.X.astype("object")
            # decode ordinal encoding to obtain imputed original data
            adata.X[::, column_indices] = enc.inverse_transform(adata.X[::, column_indices])

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

    if isinstance(var_names, list):
        column_indices = get_column_indices(adata, var_names)

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


def _warn_imputation_threshold(adata: AnnData, var_names: list[str] | None, threshold: int = 30) -> dict[str, int]:
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
    is_numeric_numpy = np.vectorize(_is_float_or_nan, otypes=[bool])
    mask = np.apply_along_axis(is_numeric_numpy, 0, X)

    _, column_indices = np.where(~mask)
    non_num_indices = set(column_indices)

    return non_num_indices


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
