from __future__ import annotations

from typing import TYPE_CHECKING

from ehrapy.anndata import anndata_to_df, df_to_anndata

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData


def summarize_measurements(
    adata: AnnData,
    layer: str = None,
    var_names: Iterable[str] | None = None,
    statistics: Iterable[str] = None,
) -> AnnData:
    """Summarizes numerical measurements into minimum, maximum and average values.

    Args:
        adata: AnnData object containing measurements that
        layer: Layer to calculate the expanded measurements for. Defaults to None (use X).
        var_names: For which measurements to determine the expanded measurements for. Defaults to None (all numerical measurements).
        statistics: Which expanded measurements to calculate.
                    Possible values are 'min', 'max', 'mean'
                    Defaults to None (calculate minimum, maximum and mean).

    Returns:
        A new AnnData object with expanded X containing the specified statistics as additional columns replacing the original values.
    """
    if var_names is None:
        var_names = adata.var_names

    if statistics is None:
        statistics = ["min", "max", "mean"]

    aggregation_functions = {measurement: statistics for measurement in var_names}

    grouped = anndata_to_df(adata, layer=layer).groupby(adata.obs.index).agg(aggregation_functions)
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]

    expanded_adata = df_to_anndata(grouped)

    return expanded_adata
