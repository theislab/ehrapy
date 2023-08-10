from __future__ import annotations

from typing import Iterable, Literal

from anndata import AnnData


def expand_measurements(
    adata: AnnData,
    agg_col: str,
    layer: str = None,
    var_names: Iterable[str] | None = None,
    measures: Literal["min", "max", "avg"] = None,
) -> AnnData:
    """Expands numerical measurements into minimum, maximum and average values.

    Args:
        adata: AnnData object containing measurements that
        agg_col: Column to aggregate values by.
        layer: Layer to calculate the expanded measurements for. Defaults to None (use X).
        var_names: For which measurements to determine the expanded measurements for. Defaults to None (all numerical measurements).
        measures: Which expanded measurements to calculate. Defaults to None (calculate minimum, maximum and average).

    Returns:
        A new AnnData object with expanded X containing the specified measures as additional columns replacing the original values.
    """
