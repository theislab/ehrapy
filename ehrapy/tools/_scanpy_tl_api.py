from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import scanpy as sc
from fast_array_utils.conv import to_dense
from scipy.sparse import spmatrix  # noqa

from ehrapy._compat import _as_scanpy_input, _materialize, _shallow_copy, function_2D_only

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ehrdata import EHRData

    from ehrapy._types import AnyRandom


def leiden(
    edata: EHRData,
    *,
    resolution: float = 1,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    random_state: AnyRandom = 0,
    key_added: str = "leiden",
    adjacency: spmatrix | None = None,
    use_weights: bool = True,
    n_iterations: int = -1,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
    **clustering_args,
) -> EHRData | None:  # pragma: no cover
    """Cluster observations into subgroups :cite:p:`Traag2019`.

    Cluster observations using the Leiden algorithm :cite:p:`Traag2019`, an improved version of the Louvain algorithm :cite:p:`Blondel2008`.
    It has been proposed for single-cell analysis by :cite:p:`Levine2015`.
    This requires having run :func:`~ehrapy.preprocessing.neighbors`.
    Uses the ``igraph`` implementation (``flavor="igraph"`` in scanpy); ``leidenalg`` is not supported.

    Args:
        edata: Central data object.
        resolution: A parameter value controlling the coarseness of the clustering. Higher values lead to more clusters.
        restrict_to: Restrict the clustering to the categories within the key for sample annotation, tuple needs to contain `(obs_key, list_of_categories)`.
        random_state: Random seed of the initialization of the optimization.
        key_added: `edata.obs` key under which to add the cluster labels.
        adjacency: Sparse adjacency matrix of the graph, defaults to neighbors connectivities.
        use_weights: If `True`, edge weights from the graph are used in the computation (placing more emphasis on stronger edges).
        n_iterations: How many iterations of the Leiden clustering algorithm to perform.
                      Positive values above 2 define the total number of iterations to perform.
                      -1 has the algorithm run until it reaches its optimal clustering.
                      2 is faster and the default of the underlying igraph implementation.
        neighbors_key: Use neighbors connectivities as adjacency.
                       If not specified, leiden looks .obsp['connectivities'] for connectivities (default storage place for pp.neighbors).
                       If specified, leiden looks .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
        obsp: Use `.obsp[obsp]` as adjacency. You can't specify both `obsp` and `neighbors_key` at the same time.
        copy: Whether to copy `edata` or modify it inplace.
        **clustering_args: Any further arguments passed to :meth:`igraph.Graph.community_leiden`.

    Returns:
        Depending on `copy`, returns or updates `edata` with the following fields.

        `edata.obs[key_added]`
        Array of dim (number of samples) that stores the subgroup id (`'0'`, `'1'`, ...) for each observation.

        `edata.uns[key_added]['params']`
        A dict with the values for the parameters `resolution`, `random_state`, and `n_iterations`.

        `edata.uns[key_added]['modularity']`
        The modularity score of the final clustering.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata)
    """
    try:
        import igraph
    except ImportError as e:
        raise ImportError("`ep.tl.leiden` requires `igraph`. Install with `pip install ehrapy[leiden]`.") from e

    return sc.tl.leiden(
        adata=edata,
        resolution=resolution,
        restrict_to=restrict_to,
        random_state=random_state,
        key_added=key_added,
        adjacency=adjacency,
        use_weights=use_weights,
        n_iterations=n_iterations,
        neighbors_key=neighbors_key,
        obsp=obsp,
        copy=copy,
        flavor="igraph",
        **clustering_args,
    )


@function_2D_only()
def dendrogram(
    edata: EHRData,
    groupby: str | Sequence[str],
    *,
    n_pcs: int | None = None,
    use_rep: str | None = None,
    var_names: Sequence[str] | None = None,
    cor_method: Literal["pearson", "kendall", "spearman"] = "pearson",
    linkage_method: str = "complete",
    optimal_ordering: bool = False,
    key_added: str | None = None,
    copy: bool = False,
) -> EHRData | None:  # pragma: no cover
    """Computes a hierarchical clustering for the given `groupby` categories.

    By default, the PCA representation is used unless `.X` has less than 50 variables.
    Alternatively, a list of `var_names` (e.g. features) can be given.
    Average values of either `var_names` or components are used to compute a correlation matrix.

    The hierarchical clustering can be visualized using :func:`ehrapy.plot.dendrogram` or multiple other visualizations that can include a dendrogram: :func:`~ehrapy.plot.matrixplot`, :func:`~ehrapy.plot.heatmap`, :func:`~ehrapy.plot.dotplot`, and :func:`~ehrapy.plot.stacked_violin`.

    .. note::
        The computation of the hierarchical clustering is based on predefined groups and not per observation.
        The correlation matrix is computed using by default pearson but other methods are available.

    Args:
        edata: Central data object.
        groupby: Key or keys of the observation grouping to compute the hierarchical clustering for.
        n_pcs: Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
        use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
                 If `None`, the representation is chosen automatically:
                 For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
                 If 'X_pca' is not present, it's computed with default parameters or `n_pcs` if present.
        var_names: List of var_names to use for computing the hierarchical clustering.
                   If `var_names` is given, then `use_rep` and `n_pcs` are ignored.
        cor_method: Correlation method to use.
                    Options are 'pearson', 'kendall', and 'spearman'.
        linkage_method: Linkage method to use. See :func:`scipy.cluster.hierarchy.linkage` for more information.
        optimal_ordering: Same as the optimal_ordering argument of :func:`scipy.cluster.hierarchy.linkage`
                          which reorders the linkage matrix so that the distance between successive leaves is minimal.
        key_added: By default, the dendrogram information is added to
                   `.uns[f'dendrogram_{groupby}']`.
                   Notice that the `groupby` information is added to the dendrogram.
        copy: Copy `edata` before computation and return a copy. Otherwise, perform computation in place and return `None`.

    Returns:
        Depending on `copy`, returns or updates `edata` with the dendrogram information in `edata.uns[key_added]`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit"])
        >>> edata = ep.pp.encode(edata, autodetect=True)
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.tl.dendrogram(edata, groupby="service_unit")
        >>> ep.pl.dendrogram(edata, groupby="service_unit")
    """
    edata = edata.copy() if copy else edata
    sc.tl.dendrogram(
        adata=_as_scanpy_input(edata),
        groupby=groupby,
        n_pcs=n_pcs,
        use_rep=use_rep,
        var_names=var_names,
        use_raw=False,
        cor_method=cor_method,
        linkage_method=linkage_method,
        optimal_ordering=optimal_ordering,
        key_added=key_added,
        inplace=True,
    )
    return edata if copy else None


def dpt(
    edata: EHRData,
    *,
    n_dcs: int = 10,
    n_branchings: int = 0,
    min_group_size: float = 0.01,
    allow_kendall_tau_shift: bool = True,
    neighbors_key: str | None = None,
    copy: bool = False,
) -> EHRData | None:  # pragma: no cover
    """Infer progression of observations through geodesic distance along the graph :cite:p:`Haghverdi2016`, :cite:p:`Wolf2019`.

    Reconstruct the progression of a process from snapshot data.
    `Diffusion Pseudotime` has been introduced by :cite:p:`Haghverdi2016` and implemented within Scanpy :cite:p:`Wolf2018`.
    Here, we use a further developed version, which is able to deal with disconnected graphs :cite:p:`Wolf2019` and can be run in a `hierarchical` mode by setting the parameter `n_branchings>1`.
    We recommend, however, to only use :func:`~ehrapy.tools.dpt` for computing pseudotime (`n_branchings=0`) and to detect branchings via :func:`~ehrapy.tools.paga`.
    For pseudotime, you need to annotate your data with a root observation.
    For instance `edata.uns['iroot'] = np.flatnonzero(edata.obs['leiden'] == '0')[0]`.
    This requires to run :func:`~ehrapy.preprocessing.neighbors` and :func:`~ehrapy.tools.diffmap` first.
    In order to reproduce the original implementation of DPT, use `method='gauss'` in :func:`~ehrapy.preprocessing.neighbors`.
    Using the default `method='umap'` only leads to minor quantitative differences, though.

    Args:
        edata: Central data object.
        n_dcs: The number of diffusion components to use.
        n_branchings: Number of branchings to detect.
        min_group_size: During recursive splitting of branches ('dpt groups') for `n_branchings`
                        > 1, do not consider groups that contain less than `min_group_size` data
                        points. If a float, `min_group_size` refers to a fraction of the total number of data points.
        allow_kendall_tau_shift: If a very small branch is detected upon splitting, shift away from
                                 maximum correlation in Kendall tau criterion of :cite:p:`Haghverdi2016` to stabilize the splitting.
        neighbors_key: If not specified, dpt looks `.uns['neighbors']` for neighbors settings
                       and `.obsp['connectivities']`, `.obsp['distances']` for connectivities and
                       distances respectively (default storage places for pp.neighbors).
                       If specified, dpt looks .uns[neighbors_key] for neighbors settings and
                       `.obsp[.uns[neighbors_key]['connectivities_key']]`,
                       `.obsp[.uns[neighbors_key]['distances_key']]` for connectivities and distances respectively.
        copy: Copy instance before computation and return a copy. Otherwise, perform computation in place and return `None`.

    Returns:
        Depending on `copy`, returns or updates `edata` with the following fields.
        If `n_branchings==0`, no field `dpt_groups` will be written.

        * `dpt_pseudotime` : :class:`pandas.Series` (`edata.obs`, dtype `float`)
          Array of dim (number of samples) that stores the pseudotime of each
          observation, that is, the DPT distance with respect to the root observation.
        * `dpt_groups` : :class:`pandas.Series` (`edata.obs`, dtype `category`)
          Array of dim (number of samples) that stores the subgroup id ('0', '1', ...) for each observation.
    """
    return sc.tl.dpt(
        adata=edata,
        n_dcs=n_dcs,
        n_branchings=n_branchings,
        min_group_size=min_group_size,
        allow_kendall_tau_shift=allow_kendall_tau_shift,
        neighbors_key=neighbors_key,
        copy=copy,
    )


def paga(
    edata: EHRData,
    *,
    groups: str | None = None,
    model: Literal["v1.2", "v1.0"] = "v1.2",
    neighbors_key: str | None = None,
    copy: bool = False,
) -> EHRData | None:  # pragma: no cover
    """Mapping out the coarse-grained connectivity structures of complex manifolds :cite:p:`Wolf2019`.

    By quantifying the connectivity of partitions (groups, clusters), partition-based graph abstraction (PAGA) generates a much simpler abstracted graph (*PAGA graph*) of partitions, in which edge weights represent confidence in the presence of connections.
    By thresholding this confidence in :func:`~ehrapy.plot.paga`, a much simpler representation of the manifold data is obtained, which is nonetheless faithful to the topology of the manifold.
    The confidence should be interpreted as the ratio of the actual versus the expected value of connections under the null model of randomly connecting partitions.
    We do not provide a p-value as this null model does not precisely capture what one would consider "connected" in real data, hence it strongly overestimates the expected value.
    See an extensive discussion of this in :cite:p:`Wolf2019`.

    .. note::
        Note that you can use the result of :func:`~ehrapy.plot.paga` in :func:`~ehrapy.tools.umap` and :func:`~ehrapy.tools.draw_graph` via `init_pos='paga'` to get embeddings that are typically more faithful to the global topology.

    Args:
        edata: Central data object.
        groups: Key for categorical in `edata.obs`.
                You can pass your predefined groups by choosing any categorical annotation of observations.
                Default: The first present key of `'leiden'` or `'louvain'`.
        model: The PAGA connectivity model.
        neighbors_key: If not specified, paga looks `.uns['neighbors']` for neighbors settings
                       and `.obsp['connectivities']`, `.obsp['distances']` for connectivities and
                       distances respectively (default storage places for `pp.neighbors`).
                       If specified, paga looks `.uns[neighbors_key]` for neighbors settings and
                       `.obsp[.uns[neighbors_key]['connectivities_key']]`,
                       `.obsp[.uns[neighbors_key]['distances_key']]` for connectivities and distances respectively.
        copy: Copy `edata` before computation and return a copy. Otherwise, perform computation in place and return `None`.

    Returns:
        Depending on `copy`, returns or updates `edata` with the following fields.

        **connectivities** :class:`scipy.sparse.csr_matrix` (`edata.uns['paga']['connectivities']`)
        The full adjacency matrix of the abstracted graph, weights correspond to confidence in the connectivities of partitions.

        **connectivities_tree** :class:`scipy.sparse.csr_matrix` (`edata.uns['paga']['connectivities_tree']`)
        The adjacency matrix of the tree-like subgraph that best explains the topology.

    Notes:
        Together with a random walk-based distance measure (e.g. :func:`ehrapy.tools.dpt`) this generates a partial coordinatization of data useful for exploring and explaining its variation.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime"])
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.leiden(edata)
        >>> ep.tl.paga(edata, groups="leiden")
    """
    return sc.tl.paga(
        adata=edata,
        groups=groups,
        use_rna_velocity=False,
        model=model,
        neighbors_key=neighbors_key,
        copy=copy,
    )


@function_2D_only()
def ingest(
    edata: EHRData,
    edata_ref: EHRData,
    *,
    obs: str | Iterable[str] | None = None,
    embedding_method: str | Iterable[str] = ("umap", "pca"),
    labeling_method: Literal["knn"] = "knn",
    neighbors_key: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | None:  # pragma: no cover
    """Map labels and embeddings from reference data to new data.

    Integrates embeddings and annotations of an `edata` with a reference dataset `edata_ref` through projecting on a PCA (or alternate model) that has been fitted on the reference data.
    The function uses a knn classifier for mapping labels and the UMAP package :cite:p:`McInnes2018` for mapping the embeddings.

    .. note::
        We refer to this *asymmetric* dataset integration as *ingesting* annotations from reference data to new data.
        This is different from learning a joint representation that integrates both datasets in an unbiased way, as CCA (e.g. in Seurat) or a conditional VAE (e.g. in scVI) would do.

    You need to run :func:`~ehrapy.preprocessing.neighbors` on `edata_ref` before passing it.

    Args:
        edata: Central data object.
        edata_ref: The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to observations and columns to features.
                   Variables (`n_vars` and `var_names`) of `edata_ref` should be the same as in `edata`.
                   This is the dataset with labels and embeddings which need to be mapped to `edata`.
        obs: Labels' keys in `edata_ref.obs` which need to be mapped to `edata.obs` (inferred for observation of `edata`).
        embedding_method: Embeddings in `edata_ref` which need to be mapped to `edata`. The only supported values are 'umap' and 'pca'.
        labeling_method: The method to map labels in `edata_ref.obs` to `edata.obs`. The only supported value is 'knn'.
        neighbors_key: If not specified, ingest looks edata_ref.uns['neighbors'] for neighbors settings and edata_ref.obsp['distances'] for
                       distances (default storage places for pp.neighbors). If specified, ingest looks edata_ref.uns[neighbors_key] for
                       neighbors settings and edata_ref.obsp[edata_ref.uns[neighbors_key]['distances_key']] for distances.
        copy: Copy `edata` before computation and return a copy. Otherwise, perform computation in place and return `None`.
        **kwargs: Keyword arguments for the nearest neighbor search used to map the labels in `obs`, namely `k`, `queue_size`, `epsilon` and `random_state`.

    Returns:
        Depending on `copy`, returns or updates `edata` with mapped embeddings and labels in `obsm` and `obs` correspondingly.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2(columns_obs_only=["service_unit"])
        >>> edata = ep.pp.encode(edata, autodetect=True)
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> edata_ref, edata_new = edata[:800].copy(), edata[800:].copy()
        >>> ep.pp.pca(edata_ref)
        >>> ep.pp.neighbors(edata_ref)
        >>> ep.tl.umap(edata_ref)
        >>> ep.tl.ingest(edata_new, edata_ref, obs="service_unit")
    """
    edata = edata.copy() if copy else edata
    X, X_ref = _materialize(to_dense(edata.X), to_dense(edata_ref.X))
    adata = edata if X is edata.X else _shallow_copy(edata, X, {})
    if X_ref is not edata_ref.X:
        edata_ref = _shallow_copy(edata_ref, X_ref, {})
    sc.tl.ingest(
        adata=adata,
        adata_ref=edata_ref,
        obs=obs,
        embedding_method=embedding_method,
        labeling_method=labeling_method,
        neighbors_key=neighbors_key,
        inplace=True,
        **kwargs,
    )
    if adata is not edata:
        edata.obsm.update(adata.obsm)
        for key in [obs] if isinstance(obs, str) else obs or ():
            edata.obs[key] = adata.obs[key]
    return edata if copy else None
