from __future__ import annotations

from typing import TYPE_CHECKING

from scanpy.get import obs_df as scanpy_obs_df
from scanpy.get import rank_genes_groups_df
from scanpy.get import var_df as scanpy_var_df

from ehrapy._compat import _as_scanpy_input, function_2D_only

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    import pandas as pd
    from ehrdata import EHRData


@function_2D_only(var_keys=("keys",))
def obs_df(
    edata: EHRData,
    *,
    keys: Collection[str] = (),
    obsm_keys: Iterable[tuple[str, int]] = (),
    layer: str | None = None,
    feature_symbols: str | None = None,
) -> pd.DataFrame:
    """Return values for observations in edata.

    Args:
        edata: Central data object.
        keys: Keys from either `.var_names`, `.var[feature_symbols]`, or `.obs.columns`.
            Keys from `.var_names` or `.var[feature_symbols]` require `.X` or `layer` to be 2D, whereas `.obs` columns can be read from 3D data.
        obsm_keys: Tuples of `(key from obsm, column index of obsm[key])`.
        layer: Layer of `edata` to use as feature values.
        feature_symbols: Column of `edata.var` to search for `keys` in.

    Returns:
        A DataFrame with `edata.obs_names` as index, and values specified by `keys` and `obsm_keys`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ages = ep.get.obs_df(edata, keys=["age"])
    """
    return scanpy_obs_df(
        adata=_as_scanpy_input(edata), keys=keys, obsm_keys=obsm_keys, layer=layer, gene_symbols=feature_symbols
    )


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
    feature_symbols: str | None = None,
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
        feature_symbols: Column name in `.var` DataFrame that stores feature symbols.
                         Specifying this will add that column to the returned DataFrame.

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
        gene_symbols=feature_symbols,
    )
