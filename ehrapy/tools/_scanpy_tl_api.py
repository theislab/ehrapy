from types import MappingProxyType
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from leidenalg.VertexPartition import MutableVertexPartition
from scanpy._utils import AnyRandom
from scipy.sparse import spmatrix

from ehrapy.preprocessing._scanpy_pp_api import pca  # noqa: E402,F403,F401


def tsne(
    adata: AnnData,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    perplexity: Union[float, int] = 30,
    early_exaggeration: Union[float, int] = 12,
    learning_rate: Union[float, int] = 1000,
    random_state: AnyRandom = 0,
    n_jobs: Optional[int] = None,
    copy: bool = False,
    metric: str = "euclidean",
) -> Optional[AnnData]:  # pragma: no cover
    """Calculates t-SNE [Maaten08]_ [Amir13]_ [Pedregosa11]_.

    t-distributed stochastic neighborhood embedding (tSNE) [Maaten08]_ has been
    proposed for visualizing complex by [Amir13]_. Here, by default, we use the implementation of *scikit-learn* [Pedregosa11]_.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        n_pcs: Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
        use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
                 If `None`, the representation is chosen automatically:
                 For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
                 If 'X_pca' is not present, it’s computed with default parameters.
        perplexity: The perplexity is related to the number of nearest neighbors that
                    is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity.
                    Consider selecting a value between 5 and 50. The choice is not extremely critical since t-SNE
                    is quite insensitive to this parameter.
        early_exaggeration: Controls how tight natural clusters in the original space are in the
                            embedded space and how much space will be between them. For larger
                            values, the space between natural clusters will be larger in the
                            embedded space. Again, the choice of this parameter is not very
                            critical. If the cost function increases during initial optimization,
                            the early exaggeration factor or the learning rate might be too high.
        learning_rate: Note that the R-package "Rtsne" uses a default of 200.
                       The learning rate can be a critical parameter. It should be
                       between 100 and 1000. If the cost function increases during initial
                       optimization, the early exaggeration factor or the learning rate
                       might be too high. If the cost function gets stuck in a bad local
                       minimum increasing the learning rate helps sometimes.
        random_state: Change this to use different intial states for the optimization.
                      If `None`, the initial state is not reproducible.
        n_jobs: Number of jobs for parallel computation.
                `None` means using :attr:`scanpy._settings.ScanpyConfig.n_jobs`.
        copy: Return a copy instead of writing to `adata`.
        metric: Distance metric calculate neighbors on.

    Returns:
        Depending on `copy`, returns or updates `adata` with the following fields.

        **X_tsne** : `np.ndarray` (`adata.obs`, dtype `float`) tSNE coordinates of data.
    """
    return sc.tl.tsne(
        adata=adata,
        n_pcs=n_pcs,
        use_rep=use_rep,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=n_jobs,
        copy=copy,
        metric=metric,
    )


_InitPos = Literal["paga", "spectral", "random"]


def umap(
    adata: AnnData,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: Optional[int] = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: Union[_InitPos, np.ndarray, None] = "spectral",
    random_state: AnyRandom = 0,
    a: Optional[float] = None,
    b: Optional[float] = None,
    copy: bool = False,
    method: Literal["umap", "rapids"] = "umap",
    neighbors_key: Optional[str] = None,
) -> Optional[AnnData]:  # pragma: no cover
    """Embed the neighborhood graph using UMAP [McInnes18]_.

    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
    technique suitable for visualizing high-dimensional data. Besides tending to
    be faster than tSNE, it optimizes the embedding such that it best reflects
    the topology of the data, which we represent throughout ehrapy using a
    neighborhood graph. tSNE, by contrast, optimizes the distribution of
    nearest-neighbor distances in the embedding such that these best match the
    distribution of distances in the high-dimensional space. We use the
    implementation of `umap-learn <https://github.com/lmcinnes/umap>`__
    [McInnes18]_. For a few comparisons of UMAP with tSNE, see this `preprint
    <https://doi.org/10.1101/298430>`__.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        min_dist: The effective minimum distance between embedded points. Smaller values
                  will result in a more clustered/clumped embedding where nearby points on
                  the manifold are drawn closer together, while larger values will result
                  on a more even dispersal of points. The value should be set relative to
                  the ``spread`` value, which determines the scale at which embedded
                  points will be spread out. The default of in the `umap-learn` package is 0.1.
        spread: The effective scale of embedded points.
                In combination with `min_dist` this determines how clustered/clumped the embedded points are.
        n_components: The number of dimensions of the embedding.
        maxiter: The number of iterations (epochs) of the optimization. Called `n_epochs` in the original UMAP.
        alpha: The initial learning rate for the embedding optimization.
        gamma: Weighting applied to negative samples in low dimensional embedding optimization.
               Values higher than one will result in greater weight being given to negative samples.
        negative_sample_rate: The number of negative edge/1-simplex samples to use per positive
                              edge/1-simplex sample in optimizing the low dimensional embedding.
        init_pos: How to initialize the low dimensional embedding. Called `init` in the original UMAP. Options are:

                  * Any key for `adata.obsm`.

                  * 'paga': positions from :func:`~scanpy.pl.paga`.

                  * 'spectral': use a spectral embedding of the graph.

                  * 'random': assign initial embedding positions at random.

                  * A numpy array of initial embedding positions.
        random_state: Random state for the initialization.

                      * If `int`, `random_state` is the seed used by the random number generator;

                      * If `RandomState` or `Generator`, `random_state` is the random number generator;

                      * If `None`, the random number generator is the `RandomState` instance used by `np.random`.
        a: More specific parameters controlling the embedding.
           If `None` these values are set automatically as determined by `min_dist` and `spread`.
        b: More specific parameters controlling the embedding.
           If `None` these values are set automatically as determined by `min_dist` and `spread`.
        copy: Return a copy instead of writing to adata.
        method: Use the original 'umap' implementation, or 'rapids' (experimental, GPU only)
        neighbors_key: If not specified, umap looks .uns['neighbors'] for neighbors settings
                       and .obsp['connectivities'] for connectivities (default storage places for pp.neighbors).
                       If specified, umap looks .uns[neighbors_key] for neighbors settings and
                       .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.

    Returns:
        Depending on `copy`, returns or updates `adata` with the following fields.

        **X_umap** : `adata.obsm` field UMAP coordinates of data.
    """
    if adata.uns["neighbors"] is None or neighbors_key not in adata.uns:
        return sc.tl.umap(
            adata=adata,
            min_dist=min_dist,
            spread=spread,
            n_components=n_components,
            maxiter=maxiter,
            alpha=alpha,
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            init_pos=init_pos,
            random_state=random_state,
            a=a,
            b=b,
            copy=copy,
            method=method,
            neighbors_key=neighbors_key,
        )
    else:
        raise ValueError(f'.uns["{neighbors_key}"] or .uns["neighbors"] were not found. Run `ep.pp.neighbors` first.')


_LAYOUTS = ("fr", "drl", "kk", "grid_fr", "lgl", "rt", "rt_circular", "fa")
_Layout = Literal[_LAYOUTS]  # type: ignore


def draw_graph(
    adata: AnnData,
    layout: _Layout = "fa",
    init_pos: Union[str, bool, None] = None,
    root: Optional[int] = None,
    random_state: AnyRandom = 0,
    n_jobs: Optional[int] = None,
    adjacency: Optional[spmatrix] = None,
    key_added_ext: Optional[str] = None,
    neighbors_key: Optional[str] = None,
    obsp: Optional[str] = None,
    copy: bool = False,
    **kwds,
) -> Optional[AnnData]:  # pragma: no cover
    """Force-directed graph drawing [Islam11]_ [Jacomy14]_ [Chippada18]_.

    .. _fa2: https://github.com/bhargavchippada/forceatlas2
    .. _Force-directed graph drawing: https://en.wikipedia.org/wiki/Force-directed_graph_drawing
    .. _fruchterman-reingold: http://igraph.org/python/doc/igraph.Graph-class.html#layout_fruchterman_reingold

    An alternative to tSNE that often preserves the topology of the data
    better. This requires to run :func:`~ehrapy.pp.neighbors`, first.
    The default layout ('fa', `ForceAtlas2`) [Jacomy14]_ uses the package `fa2`_
    [Chippada18]_, which can be installed via `pip install fa2`.
    `Force-directed graph drawing`_ describes a class of long-established
    algorithms for visualizing graphs.
    It has been suggested for visualizing single-cell data by [Islam11]_.
    Many other layouts as implemented in igraph [Csardi06]_ are available.
    Similar approaches have been used by [Zunder15]_ or [Weinreb17]_.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        layout: 'fa' (`ForceAtlas2`) or any valid `igraph layout
                <http://igraph.org/c/doc/igraph-Layout.html>`__. Of particular interest
                are 'fr' (Fruchterman Reingold), 'grid_fr' (Grid Fruchterman Reingold,
                faster than 'fr'), 'kk' (Kamadi Kawai', slower than 'fr'), 'lgl' (Large
                Graph, very fast), 'drl' (Distributed Recursive Layout, pretty fast) and
                'rt' (Reingold Tilford tree layout).
        init_pos: `'paga'`/`True`, `None`/`False`, or any valid 2d-`.obsm` key.
                  Use precomputed coordinates for initialization.
                  If `False`/`None` (the default), initialize randomly.
        root: Root for tree layouts.
        random_state: For layouts with random initialization like 'fr', change this to use
                      different intial states for the optimization. If `None`, no seed is set.
        n_jobs: Number of jobs for parallel computation.
                `None` means using :attr:`scanpy._settings.ScanpyConfig.n_jobs`.
        adjacency: Sparse adjacency matrix of the graph, defaults to neighbors connectivities.
        key_added_ext: By default, append `layout`.
        neighbors_key: If not specified, draw_graph looks .obsp['connectivities'] for connectivities
                       (default storage place for pp.neighbors).
                       If specified, draw_graph looks .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
        obsp:  Use .obsp[obsp] as adjacency. You can't specify both `obsp` and `neighbors_key` at the same time.
        copy: Whether to return a copy instead of writing to adata.
        **kwds: Parameters of chosen igraph layout. See e.g. `fruchterman-reingold`_
                [Fruchterman91]_. One of the most important ones is `maxiter`.

    Returns:
          Depending on `copy`, returns or updates `adata` with the following field.

          **X_draw_graph_layout** : `adata.obsm`
          Coordinates of graph layout. E.g. for layout='fa' (the default), the field is called 'X_draw_graph_fa'
    """
    return sc.tl.draw_graph(
        adata=adata,
        layout=layout,
        init_pos=init_pos,
        root=root,
        random_state=random_state,
        n_jobs=n_jobs,
        adjacency=adjacency,
        key_added_ext=key_added_ext,
        neighbors_key=neighbors_key,
        obsp=obsp,
        copy=copy,
        **kwds,
    )


def diffmap(
    adata: AnnData,
    n_comps: int = 15,
    neighbors_key: Optional[str] = None,
    random_state: AnyRandom = 0,
    copy: bool = False,
) -> Optional[AnnData]:  # pragma: no cover
    """Diffusion Maps [Coifman05]_ [Haghverdi15]_ [Wolf18]_.

    Diffusion maps [Coifman05]_ has been proposed for visualizing single-cell
    data by [Haghverdi15]_. The tool uses the adapted Gaussian kernel suggested
    by [Haghverdi16]_ in the implementation of [Wolf18]_.
    The width ("sigma") of the connectivity kernel is implicitly determined by
    the number of neighbors used to compute the single-cell graph in
    :func:`~ehrapy.pp.neighbors`. To reproduce the original implementation
    using a Gaussian kernel, use `method=='gauss'` in
    :func:`~ehrapy.pp.neighbors`. To use an exponential kernel, use the default
    `method=='umap'`. Differences between these options shouldn't usually be dramatic.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        n_comps: The number of dimensions of the representation.
                 neighbors_key: If not specified, diffmap looks .uns['neighbors'] for neighbors settings
                 and .obsp['connectivities'], .obsp['distances'] for connectivities and
                 distances respectively (default storage places for pp.neighbors).
                 If specified, diffmap looks .uns[neighbors_key] for neighbors settings and
                 .obsp[.uns[neighbors_key]['connectivities_key']],
                 .obsp[.uns[neighbors_key]['distances_key']] for connectivities and distances respectively.
        random_state: Random seed for the initialization.
        copy: Whether to return a copy of the :class:`~anndata.AnnData` object.

    Returns:
        Depending on `copy`, returns or updates `adata` with the following fields.

        `X_diffmap` : :class:`numpy.ndarray` (`adata.obsm`)
        Diffusion map representation of data, which is the right eigen basis of the transition matrix with eigenvectors as columns.

        `diffmap_evals` : :class:`numpy.ndarray` (`adata.uns`)
        Array of size (number of eigen vectors). Eigenvalues of transition matrix.
    """
    return sc.tl.diffmap(
        adata=adata, n_comps=n_comps, neighbors_key=neighbors_key, random_state=random_state, copy=copy
    )


def embedding_density(
    adata: AnnData,
    basis: str = "umap",  # was positional before 1.4.5
    groupby: Optional[str] = None,
    key_added: Optional[str] = None,
    components: Union[str, Sequence[str]] = None,
) -> None:  # pragma: no cover
    """Calculate the density of observation in an embedding (per condition).
    Gaussian kernel density estimation is used to calculate the density of
    observations in an embedded space. This can be performed per category over a
    categorical observation annotation. The cell density can be plotted using the
    `sc.pl.embedding_density()` function.
    Note that density values are scaled to be between 0 and 1. Thus, the
    density value at each cell is only comparable to other densities in
    the same condition category.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        basis: The embedding over which the density will be calculated. This embedded
               representation should be found in `adata.obsm['X_[basis]']``.
        groupby: Keys for categorical observation/cell annotation for which densities
                 are calculated per category. Columns with up to ten categories are accepted.
        key_added: Name of the `.obs` covariate that will be added with the density estimates.
        components: The embedding dimensions over which the density should be calculated.
                    This is limited to two components.

    Returns:
        Updates `adata.obs` with an additional field specified by the `key_added`
        parameter. This parameter defaults to `[basis]_density_[groupby]`, where
        where `[basis]` is one of `umap`, `diffmap`, `pca`, `tsne`, or `draw_graph_fa`
        and `[groupby]` denotes the parameter input.
        Updates `adata.uns` with an additional field `[key_added]_params`.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.mimic_2(encoded=True)
            ep.tl.umap(adata)
            ep.tl.embedding_density(adata, basis='umap', groupby='phase')
            ep.pl.embedding_density(adata, basis='umap', key='umap_density_phase', group='G1')
    """
    sc.tl.embedding_density(adata=adata, basis=basis, groupby=groupby, key_added=key_added, components=components)


def leiden(
    adata: AnnData,
    resolution: float = 1,
    restrict_to: Optional[Tuple[str, Sequence[str]]] = None,
    random_state: AnyRandom = 0,
    key_added: str = "leiden",
    adjacency: Optional[spmatrix] = None,
    directed: bool = True,
    use_weights: bool = True,
    n_iterations: int = -1,
    partition_type: Optional[Type[MutableVertexPartition]] = None,
    neighbors_key: Optional[str] = None,
    obsp: Optional[str] = None,
    copy: bool = False,
    **partition_kwargs,
) -> Optional[AnnData]:  # pragma: no cover
    """Cluster observations into subgroups [Traag18]_.

    Cluster observations using the Leiden algorithm [Traag18]_,
    an improved version of the Louvain algorithm [Blondel08]_.
    It has been proposed for single-cell analysis by [Levine15]_.
    This requires having ran :func:`~ehrapy.pp.neighbors` or :func:`~ehrapy.pp.bbknn` first.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        resolution: A parameter value controlling the coarseness of the clustering. Higher values lead to more clusters.
                    Set to `None` if overriding `partition_type` to one that doesn’t accept a `resolution_parameter`.
        restrict_to: Restrict the clustering to the categories within the key for sample
                     annotation, tuple needs to contain `(obs_key, list_of_categories)`.
        random_state: Random seed of the initialization of the optimization.
        key_added: `adata.obs` key under which to add the cluster labels.
        adjacency: Sparse adjacency matrix of the graph, defaults to neighbors connectivities.
        directed: Whether to treat the graph as directed or undirected.
        use_weights: If `True`, edge weights from the graph are used in the computation
                     (placing more emphasis on stronger edges).
        n_iterations: How many iterations of the Leiden clustering algorithm to perform.
                      Positive values above 2 define the total number of iterations to perform,
                      -1 has the algorithm run until it reaches its optimal clustering.
        partition_type: Type of partition to use.
                        Defaults to :class:`~leidenalg.RBConfigurationVertexPartition`.
                        For the available options, consult the documentation for
                        :func:`~leidenalg.find_partition`.
        neighbors_key: Use neighbors connectivities as adjacency.
                       If not specified, leiden looks .obsp['connectivities'] for connectivities
                       (default storage place for pp.neighbors).
                       If specified, leiden looks .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
        obsp: Use .obsp[obsp] as adjacency. You can't specify both `obsp` and `neighbors_key` at the same time.
        copy: Whether to copy `adata` or modify it inplace.
        **partition_kwargs: Any further arguments to pass to `~leidenalg.find_partition`
                            (which in turn passes arguments to the `partition_type`).

    Returns:
        `adata.obs[key_added]`
        Array of dim (number of samples) that stores the subgroup id (`'0'`, `'1'`, ...) for each cell.

        `adata.uns['leiden']['params']`
        A dict with the values for the parameters `resolution`, `random_state`, and `n_iterations`.
    """
    return sc.tl.leiden(
        adata=adata,
        resolution=resolution,
        restrict_to=restrict_to,
        random_state=random_state,
        key_added=key_added,
        adjacency=adjacency,
        directed=directed,
        use_weights=use_weights,
        n_iterations=n_iterations,
        partition_type=partition_type,
        neighbors_key=neighbors_key,
        obsp=obsp,
        copy=copy,
        **partition_kwargs,
    )


def louvain(
    adata: AnnData,
    resolution: Optional[float] = None,
    random_state: AnyRandom = 0,
    restrict_to: Optional[Tuple[str, Sequence[str]]] = None,
    key_added: str = "louvain",
    adjacency: Optional[spmatrix] = None,
    flavor: Literal["vtraag", "igraph", "rapids"] = "vtraag",
    directed: bool = True,
    use_weights: bool = False,
    partition_type: Optional[Type[MutableVertexPartition]] = None,
    partition_kwargs: Mapping[str, Any] = MappingProxyType({}),
    neighbors_key: Optional[str] = None,
    obsp: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:  # pragma: no cover
    """Cluster observations into subgroups [Blondel08]_ [Levine15]_ [Traag17]_.

    Cluster observations using the Louvain algorithm [Blondel08]_ in the implementation of [Traag17]_.
    The Louvain algorithm has been proposed for single-cell analysis by [Levine15]_.
    This requires having ran :func:`~ehrapy.pp.neighbors` or
    :func:`~ehrapy.pp.bbknn` first, or explicitly passing a ``adjacency`` matrix.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        resolution: For the default flavor (``'vtraag'``), you can provide a resolution
                    (higher resolution means finding more and smaller clusters),
                    which defaults to 1.0. See “Time as a resolution parameter” in [Lambiotte09]_.
        random_state: Random seed of the initialization of the optimization.
        restrict_to: Restrict the clustering to the categories within the key for sample
                     annotation, tuple needs to contain ``(obs_key, list_of_categories)``.
        key_added: Key under which to add the cluster labels. (default: ``'louvain'``)
        adjacency: Sparse adjacency matrix of the graph, defaults to neighbors connectivities.
        flavor: Choose between to packages for computing the clustering.
                ``'vtraag'`` is much more powerful, and the default.
        directed: Interpret the ``adjacency`` matrix as directed graph?
        use_weights: Use weights from knn graph.
        partition_type: Type of partition to use. Only a valid argument if ``flavor`` is ``'vtraag'``.
        partition_kwargs: Key word arguments to pass to partitioning, if ``vtraag`` method is being used.
        neighbors_key: Use neighbors connectivities as adjacency.
                       If not specified, louvain looks .obsp['connectivities'] for connectivities
                       (default storage place for pp.neighbors).
                       If specified, louvain looks .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
        obsp: Use .obsp[obsp] as adjacency. You can't specify both `obsp` and `neighbors_key` at the same time.
        copy: Whether to copy `adata` or modify it inplace.

    Returns:
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obs['louvain']`` (:class:`pandas.Series`, dtype ``category``)
        Array of dim (number of samples) that stores the subgroup id (``'0'``, ``'1'``, ...) for each observation.

        :class:`~anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """
    return sc.tl.louvain(
        adata=adata,
        resolution=resolution,
        random_state=random_state,
        restrict_to=restrict_to,
        key_added=key_added,
        adjacency=adjacency,
        flavor=flavor,
        directed=directed,
        use_weights=use_weights,
        partition_type=partition_type,
        partition_kwargs=partition_kwargs,
        neighbors_key=neighbors_key,
        obsp=obsp,
        copy=copy,
    )


def dendrogram(
    adata: AnnData,
    groupby: str,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    var_names: Optional[Sequence[str]] = None,
    cor_method: str = "pearson",
    linkage_method: str = "complete",
    optimal_ordering: bool = False,
    key_added: Optional[str] = None,
    inplace: bool = True,
) -> Optional[Dict[str, Any]]:  # pragma: no cover
    """Computes a hierarchical clustering for the given `groupby` categories.

    By default, the PCA representation is used unless `.X` has less than 50 variables.
    Alternatively, a list of `var_names` (e.g. genes) can be given.
    Average values of either `var_names` or components are used to compute a correlation matrix.

    The hierarchical clustering can be visualized using
    :func:`ehrapy.pl.dendrogram` or multiple other visualizations that can
    include a dendrogram: :func:`~ehrapy.pl.matrixplot`,
    :func:`~ehrapy.pl.heatmap`, :func:`~ehrapy.pl.dotplot`,
    and :func:`~ehrapy.pl.stacked_violin`.

    .. note::
        The computation of the hierarchical clustering is based on predefined
        groups and not per observation. The correlation matrix is computed using by
        default pearson but other methods are available.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        groupby: Key to group by
        n_pcs: Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
        use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
                 If `None`, the representation is chosen automatically:
                 For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
                 If 'X_pca' is not present, it’s computed with default parameters.
        var_names: List of var_names to use for computing the hierarchical clustering.
                   If `var_names` is given, then `use_rep` and `n_pcs` is ignored.
        cor_method: correlation method to use.
                    Options are 'pearson', 'kendall', and 'spearman'
        linkage_method: linkage method to use. See :func:`scipy.cluster.hierarchy.linkage` for more information.
        optimal_ordering: Same as the optimal_ordering argument of :func:`scipy.cluster.hierarchy.linkage`
                          which reorders the linkage matrix so that the distance between successive leaves is minimal.
        key_added: By default, the dendrogram information is added to
                   `.uns[f'dendrogram_{{groupby}}']`.
                   Notice that the `groupby` information is added to the dendrogram.
        inplace: If `True`, adds dendrogram information to `adata.uns[key_added]`,
                 else this function returns the information.
    Returns:
        If `inplace=False`, returns dendrogram information, else `adata.uns[key_added]` is updated with it.

    Example:
        .. code-block:: python

            import ehrapy as ep
            adata = ep.data.mimic_2(encoded=True)
            ep.tl.dendrogram(adata, groupby='service_unit')
            ep.pl.dendrogram(adata)
    """
    return sc.tl.dendrogram(
        adata=adata,
        groupby=groupby,
        n_pcs=n_pcs,
        use_rep=use_rep,
        var_names=var_names,
        use_raw=False,
        cor_method=cor_method,
        linkage_method=linkage_method,
        optimal_ordering=optimal_ordering,
        key_added=key_added,
        inplace=inplace,
    )


def dpt(
    adata: AnnData,
    n_dcs: int = 10,
    n_branchings: int = 0,
    min_group_size: float = 0.01,
    allow_kendall_tau_shift: bool = True,
    neighbors_key: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:  # pragma: no cover
    """Infer progression of observations through geodesic distance along the graph [Haghverdi16]_ [Wolf19]_.

    Reconstruct the progression of a biological process from snapshot
    data. `Diffusion Pseudotime` has been introduced by [Haghverdi16]_ and
    implemented within Scanpy [Wolf18]_. Here, we use a further developed
    version, which is able to deal with disconnected graphs [Wolf19]_ and can
    be run in a `hierarchical` mode by setting the parameter `n_branchings>1`.
    We recommend, however, to only use :func:`~ehrapy.tl.dpt` for computing pseudotime (`n_branchings=0`) and
    to detect branchings via :func:`~scanpy.tl.paga`. For pseudotime, you need
    to annotate your data with a root cell. For instance `adata.uns['iroot'] = np.flatnonzero(adata.obs['cell_types'] == 'Stem')[0]`
    This requires to run :func:`~ehrapy.pp.neighbors`, first. In order to
    reproduce the original implementation of DPT, use `method=='gauss'` in
    this. Using the default `method=='umap'` only leads to minor quantitative differences, though.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        n_dcs: The number of diffusion components to use.
        n_branchings: Number of branchings to detect.
        min_group_size: During recursive splitting of branches ('dpt groups') for `n_branchings`
                        > 1, do not consider groups that contain less than `min_group_size` data
                        points. If a float, `min_group_size` refers to a fraction of the total number of data points.
        allow_kendall_tau_shift: If a very small branch is detected upon splitting, shift away from
                                 maximum correlation in Kendall tau criterion of [Haghverdi16]_ to stabilize the splitting.
        neighbors_key: If not specified, dpt looks `.uns['neighbors']` for neighbors settings
                       and `.obsp['connectivities']`, `.obsp['distances']` for connectivities and
                       distances respectively (default storage places for pp.neighbors).
                       If specified, dpt looks .uns[neighbors_key] for neighbors settings and
                       `.obsp[.uns[neighbors_key]['connectivities_key']]`,
                       `.obsp[.uns[neighbors_key]['distances_key']]` for connectivities and distances respectively.
        copy: Copy instance before computation and return a copy. Otherwise, perform computation in place and return `None`.

    Returns:
        Depending on `copy`, returns or updates `adata` with the following fields.
        If `n_branchings==0`, no field `dpt_groups` will be written.

        * `dpt_pseudotime` : :class:`pandas.Series` (`adata.obs`, dtype `float`)
          Array of dim (number of samples) that stores the pseudotime of each
          observation, that is, the DPT distance with respect to the root observation.
        * `dpt_groups` : :class:`pandas.Series` (`adata.obs`, dtype `category`)
          Array of dim (number of samples) that stores the subgroup id ('0', '1', ...) for each observation.
    """
    return sc.tl.dpt(
        adata=adata,
        n_dcs=n_dcs,
        n_branchings=n_branchings,
        min_group_size=min_group_size,
        allow_kendall_tau_shift=allow_kendall_tau_shift,
        neighbors_key=neighbors_key,
        copy=copy,
    )


def paga(
    adata: AnnData,
    groups: Optional[str] = None,
    use_rna_velocity: bool = False,
    model: Literal["v1.2", "v1.0"] = "v1.2",
    neighbors_key: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:  # pragma: no cover
    """Mapping out the coarse-grained connectivity structures of complex manifolds [Wolf19]_.

    By quantifying the connectivity of partitions (groups, clusters),
    partition-based graph abstraction (PAGA) generates a much
    simpler abstracted graph (*PAGA graph*) of partitions, in which edge weights
    represent confidence in the presence of connections. By tresholding this
    confidence in :func:`~ehrapy.pl.paga`, a much simpler representation of the
    manifold data is obtained, which is nonetheless faithful to the topology of
    the manifold.
    The confidence should be interpreted as the ratio of the actual versus the
    expected value of connections under the null model of randomly connecting
    partitions. We do not provide a p-value as this null model does not
    precisely capture what one would consider "connected" in real data, hence it
    strongly overestimates the expected value. See an extensive discussion of this in [Wolf19]_.

    .. note::
        Note that you can use the result of :func:`~ehrapy.pl.paga` in
        :func:`~ehrapy.tl.umap` and :func:`~ehrapy.tl.draw_graph` via
        `init_pos='paga'` to get embeddings that are typically more faithful to the global topology.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        groups: Key for categorical in `adata.obs`. You can pass your predefined groups
                by choosing any categorical annotation of observations. Default:
                The first present key of `'leiden'` or `'louvain'`.
        model: The PAGA connectivity model.
        neighbors_key: If not specified, paga looks `.uns['neighbors']` for neighbors settings
                       and `.obsp['connectivities']`, `.obsp['distances']` for connectivities and
                       distances respectively (default storage places for `pp.neighbors`).
                       If specified, paga looks `.uns[neighbors_key]` for neighbors settings and
                       `.obsp[.uns[neighbors_key]['connectivities_key']]`,
                       `.obsp[.uns[neighbors_key]['distances_key']]` for connectivities and distances respectively.
        copy: Copy `adata` before computation and return a copy. Otherwise, perform computation in place and return `None`.

    Returns:
        **connectivities** : :class:`numpy.ndarray` (adata.uns['connectivities'])
        The full adjacency matrix of the abstracted graph, weights correspond to confidence in the connectivities of partitions.

       **connectivities_tree** : :class:`scipy.sparse.csr_matrix` (adata.uns['connectivities_tree'])
        The adjacency matrix of the tree-like subgraph that best explains the topology.

    Notes:
    Together with a random walk-based distance measure (e.g. :func:`ehrapy.tl.dpt`)
    this generates a partial coordinatization of data useful for exploring and explaining its variation.
    """
    return sc.tl.paga(
        adata=adata,
        groups=groups,
        use_rna_velocity=use_rna_velocity,
        model=model,
        neighbors_key=neighbors_key,
        copy=copy,
    )


def ingest(
    adata: AnnData,
    adata_ref: AnnData,
    obs: Optional[Union[str, Iterable[str]]] = None,
    embedding_method: Union[str, Iterable[str]] = ("umap", "pca"),
    labeling_method: str = "knn",
    neighbors_key: Optional[str] = None,
    inplace: bool = True,
    **kwargs,
) -> Optional[AnnData]:  # pragma: no cover
    """Map labels and embeddings from reference data to new data.

    Integrates embeddings and annotations of an `adata` with a reference dataset
    `adata_ref` through projecting on a PCA (or alternate model) that has been fitted on the reference data.
    The function uses a knn classifier for mapping labels and the UMAP package [McInnes18]_ for mapping the embeddings.

    .. note::
        We refer to this *asymmetric* dataset integration as *ingesting*
        annotations from reference data to new data. This is different from
        learning a joint representation that integrates both datasets in an
        unbiased way, as CCA (e.g. in Seurat) or a conditional VAE (e.g. in
        scVI) would do.

    You need to run :func:`~ehrapy.pp.neighbors` on `adata_ref` before passing it.

    Args:
        adata: :class:`~anndata.AnnData` object object containing all observations.
        adata_ref: The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to observations and columns to features.
                   Variables (`n_vars` and `var_names`) of `adata_ref` should be the same as in `adata`.
                   This is the dataset with labels and embeddings which need to be mapped to `adata`.
        obs: Labels' keys in `adata_ref.obs` which need to be mapped to `adata.obs` (inferred for observation of `adata`).
        embedding_method: Embeddings in `adata_ref` which need to be mapped to `adata`. The only supported values are 'umap' and 'pca'.
        labeling_method: The method to map labels in `adata_ref.obs` to `adata.obs`. The only supported value is 'knn'.
        neighbors_key: If not specified, ingest looks adata_ref.uns['neighbors'] for neighbors settings and adata_ref.obsp['distances'] for
                       distances (default storage places for pp.neighbors). If specified, ingest looks adata_ref.uns[neighbors_key] for
                       neighbors settings and adata_ref.obsp[adata_ref.uns[neighbors_key]['distances_key']] for distances.
        inplace: Only works if `return_joint=False`.
                 Add labels and embeddings to the passed `adata` (if `True`) or return a copy of `adata` with mapped embeddings and labels.
        **kwargs: Further keyword arguments for the Neighbor calculation

    Returns:
        * if `inplace=False` returns a copy of `adata` with mapped embeddings and labels in `obsm` and `obs` correspondingly
        * if `inplace=True` returns `None` and updates `adata.obsm` and `adata.obs` with mapped embeddings and labels

    Example:
        .. code-block:: python

            import ehrapy as ep

            ep.pp.neighbors(adata_ref)
            ep.tl.umap(adata_ref)
            ep.tl.ingest(adata, adata_ref, obs="service_unit")
    """
    return sc.tl.ingest(
        adata=adata,
        adata_ref=adata_ref,
        obs=obs,
        embedding_method=embedding_method,
        labeling_method=labeling_method,
        neighbors_key=neighbors_key,
        inplace=inplace,
        **kwargs,
    )


_rank_features_groups_method = Optional[Literal["logreg", "t-test", "wilcoxon", "t-test_overestim_var"]]
_corr_method = Literal["benjamini-hochberg", "bonferroni"]


def rank_features_groups(
    adata: AnnData,
    groupby: str,
    groups: Union[Literal["all"], Iterable[str]] = "all",
    reference: str = "rest",
    n_features: Optional[int] = None,
    rankby_abs: bool = False,
    pts: bool = False,
    key_added: Optional[str] = "rank_features_groups",
    copy: bool = False,
    method: _rank_features_groups_method = None,
    corr_method: _corr_method = "benjamini-hochberg",
    tie_correct: bool = False,
    layer: Optional[str] = None,
    **kwds,
) -> None:  # pragma: no cover
    """Rank features for characterizing groups.

    Expects logarithmized data.

    Args:
        adata: Annotated data matrix.
        groupby: The key of the observations grouping to consider.
        groups: Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
                shall be restricted, or `'all'` (default), for all groups.
        reference: If `'rest'`, compare each group to the union of the rest of the group.
                   If a group identifier, compare with respect to this group.
        n_features: The number of features that appear in the returned tables. Defaults to all features.
        rankby_abs: Rank genes by the absolute value of the score, not by the score.
                    The returned scores are never the absolute values.
        pts: Compute the fraction of observations containing the features.
        key_added: The key in `adata.uns` information is saved to.
        copy: Whether to return a copy of the AnnData object.
        method:  The default method is `'t-test'`,
                 `'t-test_overestim_var'` overestimates variance of each group,
                 `'wilcoxon'` uses Wilcoxon rank-sum,
                 `'logreg'` uses logistic regression.
        corr_method:  p-value correction method.
                      Used only for `'t-test'`, `'t-test_overestim_var'`, and `'wilcoxon'`.
        tie_correct: Use tie correction for `'wilcoxon'` scores. Used only for `'wilcoxon'`.
        layer: Key from `adata.layers` whose value will be used to perform tests on.
        **kwds: Are passed to test methods. Currently this affects only parameters that
                are passed to :class:`sklearn.linear_model.LogisticRegression`.
                For instance, you can pass `penalty='l1'` to try to come up with a
                minimal set of genes that are good predictors (sparse solution meaning few non-zero fitted coefficients).

    Returns:
        *names*: structured `np.ndarray` (`.uns['rank_features_groups']`)
                  Structured array to be indexed by group id storing the gene
                  names. Ordered according to scores.
        *scores*: structured `np.ndarray` (`.uns['rank_features_groups']`)
                  Structured array to be indexed by group id storing the z-score
                  underlying the computation of a p-value for each gene for each group.
                  Ordered according to scores.
        *logfoldchanges*: structured `np.ndarray` (`.uns['rank_features_groups']`)
                          Structured array to be indexed by group id storing the log2
                          fold change for each gene for each group. Ordered according to scores.
                          Only provided if method is 't-test' like.
                          Note: this is an approximation calculated from mean-log values.
        *pvals*: structured `np.ndarray` (`.uns['rank_features_groups']`)
                 p-values.
        *pvals_adj* : structured `np.ndarray` (`.uns['rank_features_groups']`)
                      Corrected p-values.
        *pts*: `pandas.DataFrame` (`.uns['rank_features_groups']`)
               Fraction of cells expressing the genes for each group.
        *pts_rest*: `pandas.DataFrame` (`.uns['rank_features_groups']`)
                    Only if `reference` is set to `'rest'`.
                    Fraction of observations from the union of the rest of each group containing the features.

     Example:
        .. code-block:: python

            import ehrapy as ep
            adata = ep.dt.mimic_2(encoded=True)
            ep.tl.rank_features_groups(adata, "service_unit")
            ep.pl.rank_features_groups(adata)
    """
    return sc.tl.rank_genes_groups(
        adata=adata,
        groupby=groupby,
        use_raw=False,
        groups=groups,
        reference=reference,
        n_genes=n_features,
        rankby_abs=rankby_abs,
        pts=pts,
        key_added=key_added,
        copy=copy,
        method=method,
        corr_method=corr_method,
        tie_correct=tie_correct,
        layer=layer,
        **kwds,
    )


def filter_rank_features_groups(
    adata: AnnData,
    key="rank_features_groups",
    groupby=None,
    key_added="rank_features_groups_filtered",
    min_in_group_fraction=0.25,
    min_fold_change=1,
    max_out_group_fraction=0.5,
) -> None:  # pragma: no cover
    """Filters out features based on fold change and fraction of features containing the feature within and outside the `groupby` categories.

    See :func:`~ehrapy.tl.rank_features_groups`.

    Results are stored in `adata.uns[key_added]`
    (default: 'rank_genes_groups_filtered').

    To preserve the original structure of adata.uns['rank_genes_groups'],
    filtered genes are set to `NaN`.

    Args:
        adata: Annotated data matrix.
        key: Key previously added by :func:`~ehrapy.tl.rank_features_groups`
        groupby: The key of the observations grouping to consider.
        key_added: The key in `adata.uns` information is saved to.
        min_in_group_fraction: Minimum in group fraction (default: 0.25).
        min_fold_change: Miniumum fold change (default: 1).
        max_out_group_fraction: Maximum out group fraction (default: 0.5).

    Returns:
        Same output as :func:`ehrapy.tl.rank_features_groups` but with filtered feature names set to `nan`

    Example:
        .. code-block:: python

            import ehrapy as ep
            adata = ep.dt.mimic_2(encoded=True)
            ep.tl.rank_features_groups(adata, "service_unit")
            ep.pl.rank_features_groups(adata)
    """
    return sc.tl.filter_rank_genes_groups(
        adata=adata,
        key=key,
        groupby=groupby,
        use_raw=False,
        key_added=key_added,
        min_in_group_fraction=min_in_group_fraction,
        min_fold_change=min_fold_change,
        max_out_group_fraction=max_out_group_fraction,
    )


_marker_feature_overlap_methods = Literal["overlap_count", "overlap_coef", "jaccard"]


def marker_feature_overlap(
    adata: AnnData,
    reference_markers: Union[Dict[str, set], Dict[str, list]],
    *,
    key: str = "rank_features_groups",
    method: _marker_feature_overlap_methods = "overlap_count",
    normalize: Optional[Literal["reference", "data"]] = None,
    top_n_markers: Optional[int] = None,
    adj_pval_threshold: Optional[float] = None,
    key_added: str = "feature_overlap",
    inplace: bool = False,
):  # pragma: no cover
    """Calculate an overlap score between data-deriven features and provided marker features.

    Marker feature overlap scores can be quoted as overlap counts, overlap
    coefficients, or jaccard indices. The method returns a pandas dataframe
    which can be used to annotate clusters based on feature overlaps.

    Args:
        adata: Annotated data matrix.
        reference_markers: A marker gene dictionary object. Keys should be strings with the
                           cell identity name and values are sets or lists of strings which match format of `adata.var_name`.
        key: The key in `adata.uns` where the rank_features_groups output is stored (default: rank_features_groups).
        method: Method to calculate marker gene overlap. `'overlap_count'` uses the
                intersection of the feature set, `'overlap_coef'` uses the overlap
                coefficient, and `'jaccard'` uses the Jaccard index (default: `overlap_count`).
        normalize: Normalization option for the feature overlap output. This parameter
                   can only be set when `method` is set to `'overlap_count'`. `'reference'`
                   normalizes the data by the total number of marker features given in the
                   reference annotation per group. `'data'` normalizes the data by the
                   total number of marker genes used for each cluster.
        top_n_markers: The number of top data-derived marker genes to use. By default the top
                       100 marker features are used. If `adj_pval_threshold` is set along with
                       `top_n_markers`, then `adj_pval_threshold` is ignored.
        adj_pval_threshold: A significance threshold on the adjusted p-values to select marker features.
                            This can only be used when adjusted p-values are calculated by `ep.tl.rank_features_groups`.
                            If `adj_pval_threshold` is set along with `top_n_markers`, then `adj_pval_threshold` is ignored.
        key_added: Name of the `.uns` field that will contain the marker overlap scores.
        inplace: Return a marker gene dataframe or store it inplace in `adata.uns`.

    Returns:
        A pandas dataframe with the marker gene overlap scores if `inplace=False`.
        For `inplace=True` `adata.uns` is updated with an additional field
        specified by the `key_added` parameter (default = 'marker_gene_overlap').

    Example:
        TODO
    """
    return sc.tl.marker_gene_overlap(
        adata=adata,
        reference_markers=reference_markers,
        key=key,
        method=method,
        normalize=normalize,
        top_n_markers=top_n_markers,
        adj_pval_threshold=adj_pval_threshold,
        key_added=key_added,
        inplace=inplace,
    )
