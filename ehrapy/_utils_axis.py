from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _resolve_axis(index: pd.Index, names: Any, axis: str) -> tuple[np.ndarray, pd.Index]:
    n = len(index)

    if names is None:
        pos = np.arange(n, dtype=int)
        return pos, index.take(pos)

    if isinstance(names, slice):
        pos = np.arange(n, dtype=int)[names]
        return pos, index.take(pos)

    if isinstance(names, (str, int, np.integer)):
        names_list = [names]
    else:
        names_list = list(names)

    names_list = list(dict.fromkeys(names_list))

    pos = index.get_indexer(names_list)
    if (pos < 0).any():
        missing = [names_list[i] for i, p in enumerate(pos) if p < 0]
        raise KeyError(f"{', '.join(str(x) for x in missing)} not found in edata.{axis}")

    pos = pos.astype(int, copy=False)
    return pos, index.take(pos)
