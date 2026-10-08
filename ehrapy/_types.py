from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Literal

import numpy as np
from fast_array_utils.types import CSBase

KnownTransformer = Literal["pynndescent", "sklearn"]
RNGLike = np.random.Generator | np.random.BitGenerator
SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
AnyRandom = int | np.random.RandomState | None


class Empty(Enum):
    token = 0


_empty = Empty.token
