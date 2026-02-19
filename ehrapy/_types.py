from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.sparse as sp
from fast_array_utils.conv import to_dense


def _optional_type(module: str, name: str) -> type:
    """Import *name* from *module* at runtime, or return a mock class if the package is missing."""
    if find_spec(module.split(".")[0]):
        return getattr(import_module(module), name)
    cls = type(name, (), {})
    cls.__module__ = module
    return cls


if TYPE_CHECKING:
    from cupy import ndarray as CupyArray
    from dask.array.core import Array as DaskArray
    from jax import Array as JaxArray
    from ndonnx import Array as NdonnxArray
    from sparse import SparseArray
    from torch import Tensor as TorchTensor
else:
    DaskArray = _optional_type("dask.array", "Array")
    CupyArray = _optional_type("cupy", "ndarray")
    TorchTensor = _optional_type("torch", "Tensor")
    JaxArray = _optional_type("jax", "Array")
    SparseArray = _optional_type("sparse", "SparseArray")
    NdonnxArray = _optional_type("ndonnx", "Array")

KnownTransformer = Literal["pynndescent", "sklearn"]
CSBase = sp.csr_matrix | sp.csc_matrix
RNGLike = np.random.Generator | np.random.BitGenerator
SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
AnyRandom = int | np.random.RandomState | None

ArrayAPICompliant = np.ndarray | DaskArray | CupyArray | TorchTensor | JaxArray | SparseArray | NdonnxArray
"""Union of array types compatible with the Array API standard via ``array_api_compat``."""

ArrayNamespace = ModuleType
"""An Array API-compatible namespace returned by ``array_api_compat.array_namespace()``."""


def asarray(a):
    """Convert input to a dense NumPy array in CPU memory using fast-array-utils."""
    return to_dense(a, to_cpu_memory=True)


def as_dense_dask_array(a, chunk_size=1000):
    """Convert input to a dense Dask array."""
    import dask.array as da

    return da.from_array(a, chunks=chunk_size)


ARRAY_TYPES_NUMERIC = (
    asarray,
    as_dense_dask_array,
    sp.csr_array,
    sp.csc_array,
)  # add coo_array once supported in AnnData
ARRAY_TYPES_NUMERIC_3D_ABLE = (asarray, as_dense_dask_array)  # add coo_array once supported in AnnData
ARRAY_TYPES_NONNUMERIC = (asarray, as_dense_dask_array)
