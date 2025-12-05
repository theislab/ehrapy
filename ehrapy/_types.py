from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import scipy.sparse as sp
from fast_array_utils.conv import to_dense

from ehrapy._compat import as_dense_dask_array

KnownTransformer = Literal["pynndescent", "sklearn"]
CSBase = sp.csr_matrix | sp.csc_matrix
RNGLike = np.random.Generator | np.random.BitGenerator
SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
AnyRandom = int | np.random.RandomState | None


def asarray(a):
    """Convert input to a dense NumPy array in CPU memory using fast-array-utils."""
    return to_dense(a, to_cpu_memory=True)


ARRAY_TYPES_NUMERIC = (
    asarray,
    as_dense_dask_array,
    sp.csr_array,
    sp.csc_array,
)  # add coo_array once supported in AnnData
ARRAY_TYPES_NUMERIC_3D_ABLE = (asarray, as_dense_dask_array)  # add coo_array once supported in AnnData
ARRAY_TYPES_NONNUMERIC = (asarray, as_dense_dask_array)
