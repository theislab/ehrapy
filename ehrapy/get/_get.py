from collections.abc import Iterable

from anndata import AnnData
from scanpy.get import obs_df as scanpy_obs_df
from scanpy.get import rank_genes_groups_df
from scanpy.get import var_df as scanpy_var_df


def obs_df(  # pragma: no cover
    adata: AnnData,
    keys: Iterable[str] = (),
    obsm_keys: Iterable[tuple[str, int]] = (),
    *,
    layer: str = None,
    features: str = None,
):
    """Return values for observations in adata.

    Args:
        adata: AnnData object to get values from.
        keys: Keys from either `.var_names`, `.var[gene_symbols]`, or `.obs.columns`.
        obsm_keys: Tuple of `(key from obsm, column index of obsm[key])`.
        layer: Layer of `adata`.
        features: Column of `adata.var` to search for `keys` in.

    Returns:
        A DataFrame with `adata.obs_names` as index, and values specified by `keys` and `obsm_keys`.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ages = ep.get.obs_df(adata, keys=["age"])
    """
    return scanpy_obs_df(adata=adata, keys=keys, obsm_keys=obsm_keys, layer=layer, gene_symbols=features)


def var_df(  # pragma: no cover
    adata: AnnData,
    keys: Iterable[str] = (),
    varm_keys: Iterable[tuple[str, int]] = (),
    *,
    layer: str = None,
):
    """Return values for observations in adata.

    Args:
        adata: AnnData object to get values from.
        keys: Keys from either `.obs_names`, or `.var.columns`.
        varm_keys: Tuple of `(key from varm, column index of varm[key])`.
        layer: Layer of `adata`.

    Returns:
        A DataFrame with `adata.var_names` as index, and values specified by `keys` and `varm_keys`.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> four_patients = ep.get.var_df(adata, keys=["0", "1", "2", "3"])
    """
    return scanpy_var_df(adata=adata, keys=keys, varm_keys=varm_keys, layer=layer)


def rank_features_groups_df(
    adata: AnnData,
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
        adata: AnnData object to get values from.
        group: Which group (as in :func:`ehrapy.tools.rank_feature_groups`'s `groupby` argument)
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
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.tl.rank_features_groups(adata, "service_unit")
        >>> df = ep.get.rank_features_groups_df(adata, group="FICU")
    """
    return rank_genes_groups_df(
        adata=adata,
        group=group,
        key=key,
        pval_cutoff=pval_cutoff,
        log2fc_min=log2fc_min,
        log2fc_max=log2fc_max,
        gene_symbols=features,
    )
