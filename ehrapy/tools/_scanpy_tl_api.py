from collections.abc import Iterable, Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from leidenalg.VertexPartition import MutableVertexPartition
from scanpy._utils import AnyRandom
from scipy.sparse import spmatrix

from ehrapy.tools import _method_options


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


def umap(
    adata: AnnData,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: Optional[int] = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: Union[_method_options._InitPos, np.ndarray, None] = "spectral",
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


def draw_graph(
    adata: AnnData,
    layout: _method_options._Layout = "fa",
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

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.data.mimic_2(encoded=True)
        >>> ep.tl.umap(adata)
        >>> ep.tl.embedding_density(adata, basis="umap", groupby="phase")
        >>> ep.pl.embedding_density(adata, basis="umap", key="umap_density_phase", group="G1")
    """
    sc.tl.embedding_density(adata=adata, basis=basis, groupby=groupby, key_added=key_added, components=components)


def leiden(
    adata: AnnData,
    resolution: float = 1,
    restrict_to: Optional[tuple[str, Sequence[str]]] = None,
    random_state: AnyRandom = 0,
    key_added: str = "leiden",
    adjacency: Optional[spmatrix] = None,
    directed: bool = True,
    use_weights: bool = True,
    n_iterations: int = -1,
    partition_type: Optional[type[MutableVertexPartition]] = None,
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
) -> Optional[dict[str, Any]]:  # pragma: no cover
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

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.data.mimic_2(encoded=True)
        >>> ep.tl.dendrogram(adata, groupby="service_unit")
        >>> ep.pl.dendrogram(adata)
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
        **connectivities** :class:`numpy.ndarray` (adata.uns['connectivities'])
        The full adjacency matrix of the abstracted graph, weights correspond to confidence in the connectivities of partitions.

       **connectivities_tree** :class:`scipy.sparse.csr_matrix` (adata.uns['connectivities_tree'])
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

    Examples:
        >>> import ehrapy as ep
        >>> ep.pp.neighbors(adata_ref)
        >>> ep.tl.umap(adata_ref)
        >>> ep.tl.ingest(adata, adata_ref, obs="service_unit")
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
