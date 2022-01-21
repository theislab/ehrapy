from __future__ import annotations

import numpy as np
from anndata import AnnData
from sklearn.preprocessing import maxabs_scale, minmax_scale, power_transform, quantile_transform, robust_scale, scale

from ehrapy.api.anndata_ext import (
    assert_numeric_vars,
    get_column_indices,
    get_column_values,
    get_numeric_vars,
    set_numeric_vars,
)


def norm_scale(adata: AnnData, vars: list[str] | None = None, copy: bool = False, **kwargs) -> AnnData | None:
    """Apply scaling normalization.

    Functionality is provided by ~sklearn.preprocessing.scale, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place
        **kwargs: Additional arguments passed to ~sklearn.preprocessing.scale

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.norm_scale(adata, copy=True)
    """

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

    return adata


def norm_minmax(adata: AnnData, vars: list[str] | None = None, copy: bool = False, **kwargs) -> AnnData | None:
    """Apply min-max normalization.

    Functionality is provided by ~sklearn.preprocessing.minmax_scale, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place
        **kwargs: Additional arguments passed to ~sklearn.preprocessing.minmax_scale

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.norm_minmax(adata, copy=True)
    """

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

    return adata


def norm_maxabs(adata: AnnData, vars: list[str] | None = None, copy: bool = False) -> AnnData | None:
    """Apply max-abs normalization.

    Functionality is provided by ~sklearn.preprocessing.maxabs_scale, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.maxabs_scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.norm_maxabs(adata, copy=True)
    """

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

    return adata


def norm_robust_scale(adata: AnnData, vars: list[str] | None = None, copy: bool = False, **kwargs) -> AnnData | None:
    """Apply robust scaling normalization.

    Functionality is provided by ~sklearn.preprocessing.robust_scale, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place
        **kwargs: Additional arguments passed to ~sklearn.preprocessing.robust_scale

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.norm_robust_scale(adata, copy=True)
    """

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

    return adata


def norm_quantile(adata: AnnData, vars: list[str] | None = None, copy: bool = False, **kwargs) -> AnnData | None:
    """Apply quantile normalization.

    Functionality is provided by ~sklearn.preprocessing.quantile_transform, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place
        **kwargs: Additional arguments passed to ~sklearn.preprocessing.quantile_transform

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.norm_quantile(adata, copy=True)
    """

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

    return adata


def norm_power(adata: AnnData, vars: list[str] | None = None, copy: bool = False, **kwargs) -> AnnData | None:
    """Apply power transformation normalization.

    Functionality is provided by ~sklearn.preprocessing.power_transform, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place
        **kwargs: Additional arguments passed to ~sklearn.preprocessing.power_transform

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.norm_power(adata, copy=True)
    """

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

    return adata


def norm_log(
    adata: AnnData,
    vars: list[str] | None = None,
    base: int | float | None = None,
    offset: int | float = 1,
    copy: bool = False,
) -> AnnData | None:
    """Apply log normalization.

    Computes :math:`x = \\log(x + offset)`, where :math:`log` denotes the natural logarithm unless a different base is given and the default :math:`offset` is :math:`1`

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        base: Numeric base for logarithm. If None the natural logarithm is used.
        offset: Offset added to values before computing the logarithm. The default is 1.
        copy: Whether to return a copy or act in place
        **kwargs: Additional arguments passed to ~sklearn.preprocessing.power_transform

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.norm_log(adata, copy=True)
    """

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

    return adata


def norm_sqrt(adata: AnnData, vars: list[str] | None = None, copy: bool = False) -> AnnData | None:
    """Apply square root normalization.

    Take the square root of all values.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.norm_sqrt(adata, copy=True)
    """

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

    return adata


def _prep_adata_norm(adata: AnnData, copy: bool = False) -> AnnData | None:

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
