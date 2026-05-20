from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ehrdata.io import from_pandas, to_pandas

from ehrapy._compat import function_2D_only

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ehrdata import EHRData


@function_2D_only()
def summarize_measurements(
    edata: EHRData,
    *,
    layer: str | None = None,
    var_names: Iterable[str] | None = None,
    statistics: Iterable[Literal["min", "max", "mean"]] | None = ["min", "max", "mean"],
) -> EHRData:
    """Summarizes numerical measurements into minimum, maximum and average values.

    Args:
        edata: Data object containing measurements.
        layer: Layer to calculate the expanded measurements for.
        var_names: For which measurements to determine the expanded measurements for. Defaults to None (all numerical measurements).
        statistics: Which expanded measurements to calculate.
                    If `None`, it calculates minimum, maximum and mean.

    Returns:
        A new data object with expanded `.X` containing the specified statistics as additional columns replacing the original values.
    """
    if var_names is None:
        var_names = edata.var_names

    aggregation_functions = dict.fromkeys(var_names, statistics)

    grouped = to_pandas(edata, layer=layer).groupby(edata.obs.index).agg(aggregation_functions)
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]

    expanded_edata = from_pandas(grouped)

    return expanded_edata
