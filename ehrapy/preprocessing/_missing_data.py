from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from ehrapy._compat import (
    DaskArray,
    _raise_array_type_not_implemented,
    use_ehrdata,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from ehrdata import EHRData


@use_ehrdata(deprecated_after="1.0.0")
def missing_data_mask(
    edata: EHRData | AnnData,
    *,
    layer: str | None = None,
    mask_values: Iterable[float | int] | None = None,
    key_added: str = "missing_data_mask",
    copy: bool = False,
) -> EHRData | AnnData | None:
    """Create a boolean mask indicating missing values in the data matrix.

    By default marks ``NaN`` values as missing.
    Optionally also marks user-specified sentinel values (e.g. ``-1``, ``0``, ``999``) as missing.

    The result is stored in ``edata.layers[key_added]``.

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
    """
    if copy:
        edata = edata.copy()

    X = edata.X if layer is None else edata.layers[layer]

    mask = _compute_nan_mask(X)

    if mask_values is not None:
        values = list(mask_values)
        mask = _apply_sentinel_mask(X, mask, values)

    edata.layers[key_added] = mask

    return edata if copy else None


@singledispatch
def _compute_nan_mask(mtx):
    _raise_array_type_not_implemented(_compute_nan_mask, type(mtx))


@_compute_nan_mask.register(np.ndarray)
def _(mtx: np.ndarray) -> np.ndarray:
    return np.isnan(mtx)


@_compute_nan_mask.register(sp.csr_array)
@_compute_nan_mask.register(sp.csc_array)
def _(mtx: sp.csr_array | sp.csc_array) -> np.ndarray:
    return np.isnan(mtx.toarray())


@_compute_nan_mask.register(DaskArray)
def _(mtx: DaskArray) -> np.ndarray:
    import dask.array as da

    return da.isnan(mtx).compute()


# singledispatch dispatches on the FIRST argument, so dispatch on mtx (the data matrix), not mask.
# This ensures sparse and Dask paths are reached.
@singledispatch
def _apply_sentinel_mask(mtx, mask, values):
    _raise_array_type_not_implemented(_apply_sentinel_mask, type(mtx))


@_apply_sentinel_mask.register(np.ndarray)
def _(mtx: np.ndarray, mask: np.ndarray, values: list) -> np.ndarray:
    return mask | np.isin(mtx, values)


@_apply_sentinel_mask.register(sp.csr_array)
@_apply_sentinel_mask.register(sp.csc_array)
def _(mtx: sp.csr_array | sp.csc_array, mask: np.ndarray, values: list) -> np.ndarray:
    return mask | np.isin(mtx.toarray(), values)


@_apply_sentinel_mask.register(DaskArray)
def _(mtx: DaskArray, mask: np.ndarray, values: list) -> np.ndarray:
    import dask.array as da

    return mask | da.isin(mtx, values).compute()
