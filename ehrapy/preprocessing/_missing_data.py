from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from array_api_compat import array_namespace
from fast_array_utils.types import CSBase, DaskArray

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from ehrdata import EHRData

    type Array = np.ndarray | DaskArray


def missing_data_mask(
    edata: EHRData,
    *,
    layer: str | None = None,
    mask_values: Iterable[float | int] | None = None,
    key_added: str = "missing_data_mask",
    copy: bool = False,
) -> EHRData | None:
    """Create a boolean mask indicating missing values in the data matrix.

    By default marks ``NaN`` values as missing.
    Optionally also marks user-specified sentinel values (e.g. ``-1``, ``0``, ``999``) as missing.
    The mask is elementwise, so 3D data is masked at every timepoint.

    Args:
        edata: Central data object.
        layer: Layer to use instead of ``edata.X``.
        mask_values: Additional values to treat as missing besides ``NaN``.
        key_added: Key under which the boolean mask is stored in ``edata.layers``.
        copy: If ``True``, return a modified copy; otherwise modify in place.

    Returns:
        ``None`` if ``copy=False``, otherwise the updated data object.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.missing_data_mask(edata)
        >>> edata
        EHRData object with n_obs × n_vars × n_t = 1776 × 46 × 1
            layers: 'missing_data_mask'
            shape of .X: (1776, 46)
            shape of .missing_data_mask: (1776, 46)
    """
    if copy:
        edata = edata.copy()

    X = edata.X if layer is None else edata.layers[layer]
    edata.layers[key_added] = _missing_mask(X, () if mask_values is None else tuple(mask_values))

    return edata if copy else None


@singledispatch
def _missing_mask(X: Array, values: Collection[float | str] = ()) -> Array:
    """Elementwise mask of missing values: NaN (any null value for object arrays) and `values`."""
    xp = array_namespace(X)
    mask = pd.isna(X) if X.dtype == object else xp.isnan(X)
    for value in values:
        mask = mask | (X == value)
    return mask


@_missing_mask.register(DaskArray)
def _(X: DaskArray, values: Collection[float | str] = ()) -> DaskArray:
    return X.map_blocks(_missing_mask, values, dtype=bool, meta=_missing_mask(X._meta, values))


@_missing_mask.register(CSBase)
def _(X: CSBase, values: Collection[float | str] = ()) -> CSBase:
    if 0 in values:
        raise NotImplementedError(
            "missing_data_mask does not support the sentinel value 0 on sparse arrays "
            "because it would mark every implicit zero, which would densify the mask."
        )
    data = np.isnan(X.data) | np.isin(X.data, list(values))
    mask = type(X)((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    mask.eliminate_zeros()
    return mask
