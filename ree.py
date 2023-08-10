from typing import Iterable

import numpy as np
from anndata import AnnData
from pandas import DataFrame

from ehrapy.anndata import anndata_to_df, df_to_anndata


def adata_to_expand():
    row_ids = ["pat1", "pat1", "pat1", "pat2", "pat2", "pat3"]
    measurement1 = np.random.choice([0, 1], size=6)
    measurement2 = np.random.uniform(0, 20, size=6)
    measurement3 = np.random.uniform(0, 20, size=6)
    data_dict = {"measurement1": measurement1, "measurement2": measurement2, "measurement3": measurement3}
    data_df = DataFrame(data_dict, index=row_ids)
    adata = AnnData(X=data_df)

    return adata


def expand_measurements(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    statistics: Iterable[str] = None,
) -> AnnData:
    if var_names is None:
        var_names = adata.var_names

    if statistics is None:
        statistics = ["min", "max", "mean"]

    aggregation_functions = {measurement: statistics for measurement in var_names}

    grouped = anndata_to_df(adata).groupby(adata.obs.index).agg(aggregation_functions)
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]

    expanded_adata = df_to_anndata(grouped)

    return expanded_adata


adata = adata_to_expand()
adata_expanded = expand_measurements(adata)
print(adata_expanded.shape)
