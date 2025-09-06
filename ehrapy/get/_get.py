from collections.abc import Iterable

from anndata import AnnData
from ehrdata import EHRData
from scanpy.get import obs_df as scanpy_obs_df
from scanpy.get import rank_genes_groups_df
from scanpy.get import var_df as scanpy_var_df

from ehrapy._compat import use_ehrdata


@use_ehrdata(deprecated_after="1.0.0")
def obs_df(  # pragma: no cover
    edata: EHRData | AnnData,
    keys: Iterable[str] = (),
    obsm_keys: Iterable[tuple[str, int]] = (),
    *,
    layer: str = None,
    features: str = None,
):
    """Return values for observations in edata.

    Args:
        edata: Central data object.
        keys: Keys from either `.var_names`, `.var[gene_symbols]`, or `.obs.columns`.
        obsm_keys: Tuple of `(key from obsm, column index of obsm[key])`.
        layer: Layer of `edata`.
        features: Column of `edata.var` to search for `keys` in.

    Returns:
        A DataFrame with `edata.obs_names` as index, and values specified by `keys` and `obsm_keys`.

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ages = ep.get.obs_df(edata, keys=["age"])
    """
    return scanpy_obs_df(adata=edata, keys=keys, obsm_keys=obsm_keys, layer=layer, gene_symbols=features)


@use_ehrdata(deprecated_after="1.0.0")
def var_df(  # pragma: no cover
    edata: EHRData | AnnData,
    keys: Iterable[str] = (),
    varm_keys: Iterable[tuple[str, int]] = (),
    *,
    layer: str = None,
):
    """Return values for observations in edata.

    Args:
        edata: Central data object.
        keys: Keys from either `.obs_names`, or `.var.columns`.
        varm_keys: Tuple of `(key from varm, column index of varm[key])`.
        layer: Layer of `edata`.

    Returns:
        A DataFrame with `edata.var_names` as index, and values specified by `keys` and `varm_keys`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> four_patients = ep.get.var_df(edata, keys=["0", "1", "2", "3"])
    """
    return scanpy_var_df(adata=edata, keys=keys, varm_keys=varm_keys, layer=layer)


@use_ehrdata(deprecated_after="1.0.0")
def rank_features_groups_df(
    edata: EHRData | AnnData,
    group: str | Iterable[str],
    *,
    key: str = "rank_features_groups",
    pval_cutoff: float | None = None,
    log2fc_min: float | None = None,
    log2fc_max: float | None = None,
    features: str | None = None,
):
    """:func:`ehrapy.tools.rank_features_groups` results in the form of a :class:`~pandas.DataFrame`.

    Args:
        edata: Central data object.
        group: Which group (as in :func:`ehrapy.tools.rank_features_groups`'s `groupby` argument)
               to return results from. Can be a list. All groups are returned if groups is `None`.
        key: Key differential groups were stored under.
        pval_cutoff: Return only adjusted p-values below the  cutoff.
        log2fc_min: Minimum logfc to return.
        log2fc_max: Maximum logfc to return.
        features: Column name in `.var` DataFrame that stores gene symbols.
                  Specifying this will add that column to the returned DataFrame.

    Returns:
        A Pandas DataFrame of all rank genes groups results.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.tl.rank_features_groups(edata, "service_unit")
        >>> df = ep.get.rank_features_groups_df(edata, group="FICU")
    """
    return rank_genes_groups_df(
        adata=edata,
        group=group,
        key=key,
        pval_cutoff=pval_cutoff,
        log2fc_min=log2fc_min,
        log2fc_max=log2fc_max,
        gene_symbols=features,
    )
