from __future__ import annotations

import numpy as np
from anndata import AnnData
from sklearn.preprocessing import maxabs_scale, minmax_scale, power_transform, quantile_transform, robust_scale, scale

from ehrapy.anndata.anndata_ext import (
    assert_numeric_vars,
    get_column_indices,
    get_column_values,
    get_numeric_vars,
    set_numeric_vars,
)
from ehrapy import logger as logg


def scale_norm(adata: AnnData, vars: str | list[str] | None = None, copy: bool = False, **kwargs) -> AnnData | None:
    """Apply scaling normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.scale`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place (default: False)
        **kwargs: Additional arguments passed to :func:`~sklearn.preprocessing.scale`

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.scale_norm(adata, copy=True)
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = get_column_indices(adata, vars)
    var_values = get_column_values(adata, var_idx)

    var_values = scale(var_values, **kwargs)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "scale")

    logg.info(f"Scaling normalization was applied on `X`.")

    return adata


def minmax_norm(adata: AnnData, vars: str | list[str] | None = None, copy: bool = False, **kwargs) -> AnnData | None:
    """Apply min-max normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.minmax_scale`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place (default: False)
        **kwargs: Additional arguments passed to :func:`~sklearn.preprocessing.minmax_scale`

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.minmax_norm(adata, copy=True)
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = get_column_indices(adata, vars)
    var_values = get_column_values(adata, var_idx)

    var_values = minmax_scale(var_values, **kwargs)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "minmax")

    logg.info(f"AnnData's `X` was min-max normalized.")

    return adata


def maxabs_norm(adata: AnnData, vars: str | list[str] | None = None, copy: bool = False) -> AnnData | None:
    """Apply max-abs normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.maxabs_scale`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.maxabs_scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place (default: False)

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.maxabs_norm(adata, copy=True)
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = get_column_indices(adata, vars)
    var_values = get_column_values(adata, var_idx)

    var_values = maxabs_scale(var_values)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "maxabs")

    logg.info(f"AnnData's `X` was max-abs normalized.")

    return adata


def robust_scale_norm(
    adata: AnnData, vars: str | list[str] | None = None, copy: bool = False, **kwargs
) -> AnnData | None:
    """Apply robust scaling normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.robust_scale`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place (default: False)
        **kwargs: Additional arguments passed to :func:`~sklearn.preprocessing.robust_scale`

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.robust_scale_norm(adata, copy=True)
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = get_column_indices(adata, vars)
    var_values = get_column_values(adata, var_idx)

    var_values = robust_scale(var_values, **kwargs)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "robust_scale")

    logg.info(f"Robust scaling normalization was applied on `X`.")

    return adata


def quantile_norm(adata: AnnData, vars: str | list[str] | None = None, copy: bool = False, **kwargs) -> AnnData | None:
    """Apply quantile normalization.

    Functionality is provided by ~sklearn.preprocessing.quantile_transform, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place (default: False)
        **kwargs: Additional arguments passed to :func:`~sklearn.preprocessing.quantile_transform`

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.quantile_norm(adata, copy=True)
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = get_column_indices(adata, vars)
    var_values = get_column_values(adata, var_idx)

    var_values = quantile_transform(var_values, **kwargs)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "quantile")

    logg.info(f"AnnData's `X` was quantile normalized.")

    return adata


def power_norm(adata: AnnData, vars: str | list[str] | None = None, copy: bool = False, **kwargs) -> AnnData | None:
    """Apply power transformation normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.power_transform`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place (default: False)
        **kwargs: Additional arguments passed to :func:`~sklearn.preprocessing.power_transform`

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.power_norm(adata, copy=True)
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = get_column_indices(adata, vars)
    var_values = get_column_values(adata, var_idx)

    var_values = power_transform(var_values, **kwargs)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "power")

    logg.info(f"Power transformation normalization was applied on `X`.")

    return adata


def log_norm(
    adata: AnnData,
    vars: str | list[str] | None = None,
    base: int | float | None = None,
    offset: int | float = 1,
    copy: bool = False,
) -> AnnData | None:
    """Apply log normalization.

    Computes :math:`x = \\log(x + offset)`, where :math:`log` denotes the natural logarithm unless a different base is given and the default :math:`offset` is :math:`1`

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        base: Numeric base for logarithm. If None the natural logarithm is used.
        offset: Offset added to values before computing the logarithm. (default: 1)
        copy: Whether to return a copy or act in place (default: False)

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.log_norm(adata, copy=True)
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = get_column_indices(adata, vars)
    var_values = get_column_values(adata, var_idx)

    if offset == 1:
        np.log1p(var_values, out=var_values)
    else:
        var_values = var_values + offset
        np.log(var_values, out=var_values)

    if base is not None:
        np.divide(var_values, np.log(base), out=var_values)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "log")

    logg.info(f"Log normalization was applied on `X`.")

    return adata


def sqrt_norm(adata: AnnData, vars: str | list[str] | None = None, copy: bool = False) -> AnnData | None:
    """Apply square root normalization.

    Take the square root of all values.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place (default: False)

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.sqrt_norm(adata, copy=True)
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = get_column_indices(adata, vars)
    var_values = get_column_values(adata, var_idx)

    var_values = np.sqrt(var_values)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "sqrt")

    logg.info(f"Square root normalization was applied on `X`.")

    return adata


def _prep_adata_norm(adata: AnnData, copy: bool = False) -> AnnData | None:  # pragma: no cover
    if copy:
        adata = adata.copy()

    if "raw_norm" not in adata.layers.keys():
        adata.layers["raw_norm"] = adata.X.copy()

    return adata


def _record_norm(adata: AnnData, vars_list: list[str], method: str) -> None:
    if "normalization" in adata.uns_keys():
        norm_record = adata.uns["normalization"]
    else:
        norm_record = {}

    for var in vars_list:
        if var in norm_record.keys():
            norm_record[var].append(method)
        else:
            norm_record[var] = [method]

    adata.uns["normalization"] = norm_record

    return None
