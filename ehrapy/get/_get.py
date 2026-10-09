from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ehrdata import EHRData
from scanpy.get import obs_df as scanpy_obs_df
from scanpy.get import rank_genes_groups_df
from scanpy.get import var_df as scanpy_var_df

from ehrapy._compat import _as_scanpy_input, _materialize, function_2D_only

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    import pandas as pd

    from ehrapy.preprocessing._summarize_measurements import Statistic


def obs_df(
    edata: EHRData,
    *,
    keys: Collection[str] = (),
    obsm_keys: Iterable[tuple[str, int]] = (),
    layer: str | None = None,
    statistic: Statistic = "first",
) -> pd.DataFrame:
    """Return values for observations in edata.

    Args:
        edata: Central data object.
        keys: Keys from either `.var_names` or `.obs.columns`.
        obsm_keys: Tuples of `(key from obsm, column index of obsm[key])`.
        layer: Layer of `edata` to use as feature values.
        statistic: Statistic over time that gives one value per observation for variables of 3D data, as in :func:`~ehrapy.preprocessing.summarize_measurements`.
            The default `"first"` is the first non-missing value, the baseline.
            To take the values at one timepoint, select it first, such as `edata[:, :, [6]]`.

    Returns:
        A DataFrame with `edata.obs_names` as index, and values specified by `keys` and `obsm_keys`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ages = ep.get.obs_df(edata, keys=["age"])

        Mean of longitudinal variables over time:

        >>> edata = ed.dt.ehrdata_blobs(n_variables=3, base_timepoints=5)
        >>> means = ep.get.obs_df(edata, keys=["feature_0", "cluster"], statistic="mean")
    """
    edata, layer = _over_time(edata, keys, layer=layer, statistic=statistic)
    return scanpy_obs_df(adata=_as_scanpy_input(edata), keys=keys, obsm_keys=obsm_keys, layer=layer)


def _over_time(
    edata: EHRData,
    keys: Iterable[str | None],
    *,
    layer: str | None,
    statistic: Statistic = "first",
) -> tuple[EHRData, str | None]:
    """`edata` with the variables among `keys` reduced to `statistic` over time if they are 3D, and the layer to read them from."""
    X = edata.X if layer is None else edata.layers[layer]
    selected = edata.var_names.isin([key for key in keys if key is not None])
    if getattr(X, "ndim", 2) != 3 or not selected.any():
        return edata, layer
    from ehrapy.preprocessing._summarize_measurements import _aggregate_time

    (values,) = _materialize(_aggregate_time(X[:, selected], statistic))
    reduced = EHRData(values, obs=edata.obs, var=edata.var[selected], obsm=edata.obsm, obsp=edata.obsp)
    reduced.uns = edata.uns
    return reduced, None


@function_2D_only()
def var_df(
    edata: EHRData,
    *,
    keys: Collection[str] = (),
    varm_keys: Iterable[tuple[str, int]] = (),
    layer: str | None = None,
) -> pd.DataFrame:
    """Return values for features in edata.

    Args:
        edata: Central data object.
        keys: Keys from either `.obs_names`, or `.var.columns`.
        varm_keys: Tuples of `(key from varm, column index of varm[key])`.
        layer: Layer of `edata` to use as feature values.

    Returns:
        A DataFrame with `edata.var_names` as index, and values specified by `keys` and `varm_keys`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> four_patients = ep.get.var_df(edata, keys=["0", "1", "2", "3"])
    """
    return scanpy_var_df(adata=_as_scanpy_input(edata), keys=keys, varm_keys=varm_keys, layer=layer)


def rank_features_groups_df(
    edata: EHRData,
    group: str | Iterable[str] | None,
    *,
    key: str = "rank_features_groups",
    pval_cutoff: float | None = None,
    log2fc_min: float | None = None,
    log2fc_max: float | None = None,
) -> pd.DataFrame:
    """:func:`ehrapy.tools.rank_features_groups` results in the form of a :class:`~pandas.DataFrame`.

    Args:
        edata: Central data object.
        group: Which group (as in :func:`ehrapy.tools.rank_features_groups`'s `groupby` argument)
               to return results from. Can be a list. All groups are returned if groups is `None`.
        key: Key the :func:`ehrapy.tools.rank_features_groups` results were stored under.
        pval_cutoff: Return only adjusted p-values below the cutoff.
        log2fc_min: Minimum logfc to return.
        log2fc_max: Maximum logfc to return.

    Returns:
        A Pandas DataFrame of all rank features groups results.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata = ep.pp.encode(edata, autodetect=True)
        >>> ep.tl.rank_features_groups(edata, groupby="service_unit")
        >>> df = ep.get.rank_features_groups_df(edata, group="FICU")
    """
    return rank_genes_groups_df(
        adata=edata,
        group=group,
        key=key,
        pval_cutoff=pval_cutoff,
        log2fc_min=log2fc_min,
        log2fc_max=log2fc_max,
    )


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
