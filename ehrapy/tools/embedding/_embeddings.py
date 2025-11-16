from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from ehrdata import EHRData
from scipy.linalg import svd
from scipy.sparse import spmatrix  # noqa

from ehrapy._compat import _raise_array_type_not_implemented, function_2D_only, use_ehrdata
from ehrapy.anndata._feature_specifications import _detect_feature_type
from ehrapy.core._constants import TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY
from ehrapy.tools import _method_options  # noqa

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

    from ehrapy._types import AnyRandom


# No need for testing 3D; tSNE does not not support layers, and
# and X can only be 2D currently, until this PR is merged: https://github.com/scverse/anndata/pull/1707
@use_ehrdata(deprecated_after="1.0.0")
def tsne(
    edata: EHRData | AnnData,
    *,
    n_pcs: int | None = None,
    use_rep: str | None = None,
    perplexity: float | int = 30,
    early_exaggeration: float | int = 12,
    learning_rate: float | int = 1000,
    random_state: AnyRandom = 0,
    n_jobs: int | None = None,
    copy: bool = False,
    metric: str = "euclidean",
) -> EHRData | EHRData | AnnData | None:  # pragma: no cover
    """Calculates t-SNE :cite:p:`vanDerMaaten2008`, :cite:p:`Amir2013`, and :cite:p:`Pedregosa2011`.

    t-distributed stochastic neighborhood embedding (tSNE) :cite:p:`vanDerMaaten2008` has been
    proposed for visualizing complex by :cite:p:`Amir2013`.

    Args:
        edata: Central data object.
        n_pcs: Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
        use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
                 If `None`, the representation is chosen automatically:
                 For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
                 If 'X_pca' is not present, it's computed with default parameters.
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
        copy: Return a copy instead of writing to `adata`.
        metric: Distance metric to calculate neighbors on.

    Returns:
        Depending on `copy`, returns or updates `edata` with the following fields.

        **X_tsne** : `np.ndarray` (`edata.obs`, dtype `float`) tSNE coordinates of data.
    """
    return sc.tl.tsne(
        adata=edata,
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


@use_ehrdata(deprecated_after="1.0.0")
def umap(
    edata: EHRData | AnnData,
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: Literal["paga", "spectral", "random"] | np.ndarray | None = "spectral",
    random_state: AnyRandom = 0,
    a: float | None = None,
    b: float | None = None,
    copy: bool = False,
    method: Literal["umap", "rapids"] = "umap",
    neighbors_key: str | None = None,
) -> EHRData | AnnData | None:  # pragma: no cover
    """Embed the neighborhood graph using UMAP :cite:p:`McInnes2018`.

    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning technique suitable for visualizing high-dimensional data.
    Besides tending to be faster than tSNE, it optimizes the embedding such that it best reflects
    the topology of the data, which we represent throughout ehrapy using a
    neighborhood graph. tSNE, by contrast, optimizes the distribution of
    nearest-neighbor distances in the embedding such that these best match the
    distribution of distances in the high-dimensional space.
    For a few comparisons of UMAP with tSNE, see this `preprint <https://doi.org/10.1101/298430>`__.

    Args:
        edata: Central data object.
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

                  * Any key for `edata.obsm`.
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
        copy: Return a copy instead of writing to edata.
        method: Use the original 'umap' implementation, or 'rapids' (experimental, GPU only)
        neighbors_key: If not specified, umap looks .uns['neighbors'] for neighbors settings
                       and .obsp['connectivities'] for connectivities (default storage places for pp.neighbors).
                       If specified, umap looks .uns[neighbors_key] for neighbors settings and
                       .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.

    Returns:
        Depending on `copy`, returns or updates `edata` with the following fields.

        **X_umap** : `edata.obsm` field UMAP coordinates of data.
    """
    key_to_check = neighbors_key if neighbors_key is not None else "neighbors"
    if key_to_check not in edata.uns:
        raise ValueError(f"Did not find .uns[{key_to_check!r}]. Please run `ep.pp.neighbors` first.")

    if (
        "use_rep" in edata.uns[key_to_check]["params"]
        and edata.uns[key_to_check]["params"]["use_rep"] == TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY
    ):
        edata.obsm[TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY] = np.zeros(edata.shape[0])

    edata_returned = sc.tl.umap(
        adata=edata,
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

    if edata_returned is not None:
        edata_returned.obsm.pop(TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY, None)
    edata.obsm.pop(TEMPORARY_TIMESERIES_NEIGHBORS_USE_REP_KEY, None)

    return edata_returned


@use_ehrdata(deprecated_after="1.0.0")
def draw_graph(
    edata: EHRData | AnnData,
    *,
    layout: _method_options._Layout = "fa",
    init_pos: str | bool | None = None,
    root: int | None = None,
    random_state: AnyRandom = 0,
    n_jobs: int | None = None,
    adjacency: spmatrix | None = None,
    key_added_ext: str | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
    **kwds,
) -> EHRData | AnnData | None:  # pragma: no cover
    """Force-directed graph drawing :cite:p:`Islam2011`, :cite:p:`Jacomy2014`, and :cite:p:`Chippada2018`.

    .. _fa2: https://github.com/bhargavchippada/forceatlas2
    .. _Force-directed graph drawing: https://en.wikipedia.org/wiki/Force-directed_graph_drawing
    .. _fruchterman-reingold: http://igraph.org/python/doc/igraph.Graph-class.html#layout_fruchterman_reingold

    An alternative to tSNE that often preserves the topology of the data better.
    This requires to run :func:`~ehrapy.preprocessing.neighbors`, first.
    The default layout ('fa', `ForceAtlas2`) :cite:p:`Jacomy2014` uses the package `fa2`_
    :cite:p:`Chippada2018`, which can be installed via `pip install fa2`.
    `Force-directed graph drawing`_ describes a class of long-established algorithms for visualizing graphs.

    Args:
        edata: Central data object.
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
        adjacency: Sparse adjacency matrix of the graph, defaults to neighbors connectivities.
        key_added_ext: By default, append `layout`.
        neighbors_key: If not specified, draw_graph looks .obsp['connectivities'] for connectivities
                       (default storage place for pp.neighbors).
                       If specified, draw_graph looks .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
        obsp:  Use `.obsp[obsp]` as adjacency. You can't specify both `obsp` and `neighbors_key` at the same time.
        copy: Whether to return a copy instead of writing to edata.
        **kwds: Parameters of chosen igraph layout. See e.g. `fruchterman-reingold`_
                :cite:p:`Fruchterman1991`. One of the most important ones is `maxiter`.

    Returns:
          Depending on `copy`, returns or updates `edata` with the following field.

          **X_draw_graph_layout** : `edata.obsm`
          Coordinates of graph layout. E.g. for layout='fa' (the default), the field is called 'X_draw_graph_fa'
    """
    return sc.tl.draw_graph(
        adata=edata,
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


@use_ehrdata(deprecated_after="1.0.0")
def diffmap(
    edata: EHRData | AnnData,
    *,
    n_comps: int = 15,
    neighbors_key: str | None = None,
    random_state: AnyRandom = 0,
    copy: bool = False,
) -> EHRData | AnnData | None:  # pragma: no cover
    """Diffusion Maps :cite:p:`Coifman2005`, :cite:p:`Haghverdi2015`, :cite:p:`Wolf2019`.

    Diffusion maps :cite:p:`Coifman2005` has been proposed for visualizing biomedical data by :cite:p:`Haghverdi2015`.
    The tool uses the adapted Gaussian kernel suggested by :cite:p:`Haghverdi2016` in the implementation of :cite:p:`Wolf2018`.
    The width ("sigma") of the connectivity kernel is implicitly determined by
    the number of neighbors used to compute the graph in :func:`~ehrapy.preprocessing.neighbors`.
    To reproduce the original implementation using a Gaussian kernel, use `method=='gauss'` in :func:`~ehrapy.preprocessing.neighbors`.
    To use an exponential kernel, use the default `method=='umap'`.
    Differences between these options shouldn't usually be dramatic.

    Args:
        edata: Central data object.
        n_comps: The number of dimensions of the representation.
                 neighbors_key: If not specified, diffmap looks .uns['neighbors'] for neighbors settings
                 and .obsp['connectivities'], .obsp['distances'] for connectivities and
                 distances respectively (default storage places for pp.neighbors).
                 If specified, diffmap looks .uns[neighbors_key] for neighbors settings and
                 .obsp[.uns[neighbors_key]['connectivities_key']],
                 .obsp[.uns[neighbors_key]['distances_key']] for connectivities and distances respectively.
        neighbors_key: Key to stored neighbors.
        random_state: Random seed for the initialization.
        copy: Whether to return a copy of the Data object.

    Returns:
        Depending on `copy`, returns or updates `edata` with the following fields.

        `X_diffmap` : :class:`numpy.ndarray` (`edata.obsm`)
        Diffusion map representation of data, which is the right eigen basis of the transition matrix with eigenvectors as columns.

        `diffmap_evals` : :class:`numpy.ndarray` (`edata.uns`)
        Array of size (number of eigen vectors). Eigenvalues of transition matrix.
    """
    return sc.tl.diffmap(
        adata=edata, n_comps=n_comps, neighbors_key=neighbors_key, random_state=random_state, copy=copy
    )


@use_ehrdata(deprecated_after="1.0.0")
def embedding_density(
    edata: EHRData | AnnData,
    *,
    basis: str = "umap",  # was positional before 1.4.5
    groupby: str | None = None,
    key_added: str | None = None,
    components: str | Sequence[str] = None,
) -> None:  # pragma: no cover
    """Calculate the density of observation in an embedding (per condition).

    Gaussian kernel density estimation is used to calculate the density of observations in an embedded space.
    This can be performed per category over a categorical observation annotation.
    The cell density can be plotted using the `sc.pl.embedding_density()` function.
    Note that density values are scaled to be between 0 and 1.
    Thus, the density value at each cell is only comparable to other densities in the same condition category.

    Args:
        edata: Central data object.
        basis: The embedding over which the density will be calculated. This embedded
               representation should be found in `edata.obsm['X_[basis]']`.
        groupby: Keys for categorical observation/cell annotation for which densities
                 are calculated per category. Columns with up to ten categories are accepted.
        key_added: Name of the `.obs` covariate that will be added with the density estimates.
        components: The embedding dimensions over which the density should be calculated.
                    This is limited to two components.

    Returns:
        Updates `edata.obs` with an additional field specified by the `key_added` parameter.
        This parameter defaults to `[basis]_density_[groupby]`,
        where `[basis]` is one of `umap`, `diffmap`, `pca`, `tsne`, or `draw_graph_fa` and `[groupby]` denotes the parameter input.
        Updates `edata.uns` with an additional field `[key_added]_params`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata = ep.pp.encode(edata, autodetect=True)
        >>> ep.pp.simple_impute(edata, strategy="median")
        >>> ep.pp.neighbors(edata)
        >>> ep.tl.umap(edata)
        >>> ep.tl.embedding_density(edata, basis="umap")
        >>> ep.pl.embedding_density(edata, basis="umap")
    """
    sc.tl.embedding_density(adata=edata, basis=basis, groupby=groupby, key_added=key_added, components=components)


@singledispatch
def famd(
    edata: EHRData | np.ndarray,
    *,
    layer: str | None = None,
    n_components: int = 2,
    key_added: str | None = None,
    var_names: Sequence[str] | None = None,
    copy: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict] | EHRData | None:
    """Calculates factors of mixed data.

    FAMD (Factor Analysis of Mixed Data) is a dimensionality reduction technique for datasets containing both quantitative and qualitative variables.
    Roughly, FAMD works as a PCA for quantitative variables and as a multiple correspondence analysis (MCA) for qualitative variables.
    It maximizes the sum of squared correlations with quantitative variables and squared correlation ratios with qualitative variables,
    treating both types equally with each variable's contribution bounded by 1.
    The method produces factor scores for individuals, correlation circles for quantitative variables, and category centroids for qualitative variables.

    Args:
        edata: The EHRData object (n_obs × n_vars × n_timesteps) containing mixed data types.
        layer: The layer to perform the computation on.
        n_components: Number of dimensions to retain in the reduced space. Must be less than min(n_obs, n_vars).
        key_added: Key under which to store the results in `.obsm` and `.uns`. Defaults to 'famd'.
        var_names: Names of the input variables (features).
            Used to generate interpretable feature names in the output (e.g., 'age' vs 'var_0', 'sex_M' vs 'var_1_M').
            If None, defaults to 'var_0', 'var_1', etc. Automatically extracted from `.var_names`.
        copy: Whether to return a copy or modify the object inplace.

    Examples:
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> edata = ed.dt.ehrdata_blobs(n_observations=100, n_centers=3, base_timepoints=1)
        >>> ep.tl.famd(edata, n_components=10)
        >>> edata.obsm["X_famd"]  # Factor scores for plotting
        >>> edata.uns["famd"]["variance_ratio"]  # Explained variance

    Returns:
        If edata is EHRData and copy=True, returns modified copy. If edata is ndarray, returns (factor_scores, loadings, metadata).
    """
    arr = edata.X if layer is None else edata.layers[layer]
    _raise_array_type_not_implemented(famd, type(arr))
    return None


@function_2D_only()
@famd.register(EHRData)
def _(
    edata: EHRData,
    /,
    *,
    layer: str | None = None,
    n_components: int = 2,
    key_added: str | None = None,
    var_names: Sequence[str] | None = None,
    copy: bool = False,
) -> EHRData | None:
    if key_added is None:
        key_added = "famd"

    edata = edata.copy() if copy else edata

    arr = edata.X if layer is None else edata.layers[layer]
    factor_scores, loadings, metadata = famd(arr, n_components=n_components, var_names=edata.var_names)

    edata.obsm[f"X_{key_added}"] = factor_scores
    edata.varm[f"{key_added}_loadings"] = loadings
    edata.uns[key_added] = {
        "params": {
            "n_components": n_components,
        },
        "variance": metadata["variance"],
        "variance_ratio": metadata["variance_ratio"],
        "quant_mask": metadata["quant_mask"],
        "feature_names": metadata["feature_names"],
        "feature_to_original": metadata["feature_to_original"],
    }

    return edata if copy else None


@famd.register(np.ndarray)
def _(
    arr: np.ndarray, /, *, n_components: int = 2, var_names: Sequence[str] | None = None, **kwargs
) -> tuple[np.ndarray, np.ndarray, dict]:
    if arr.ndim != 3 or arr.shape[2] != 1:
        raise ValueError(f"FAMD requires 3D array with single timepoint (shape[2]=1), got shape {arr.shape}")

    data = arr[:, :, 0]
    n_vars = data.shape[1]

    if var_names is None:
        var_names = [f"var_{i}" for i in range(n_vars)]

    quant_mask = np.zeros(n_vars, dtype=bool)
    for i in range(n_vars):
        col = pd.Series(data[:, i], name=var_names[i])
        feature_type, _ = _detect_feature_type(col)
        quant_mask[i] = feature_type == "numeric"

    qual_mask = ~quant_mask
    n_obs = data.shape[0]
    transformed_cols = []
    feature_names = []
    feature_to_original = []

    if quant_mask.any():
        quant_indices = np.where(quant_mask)[0]
        X_quant = data[:, quant_mask].astype(float)
        X_quant_centered = X_quant - np.nanmean(X_quant, axis=0)
        X_quant_std = np.nanstd(X_quant, axis=0)
        X_quant_std[X_quant_std == 0] = 1
        X_quant_scaled = X_quant_centered / X_quant_std
        transformed_cols.append(X_quant_scaled)

        for idx in quant_indices:
            feature_names.append(f"{var_names[idx]}")
            feature_to_original.append(idx)

    if qual_mask.any():
        qual_indices = np.where(qual_mask)[0]
        for idx in qual_indices:
            col_data = data[:, idx]
            valid_mask = ~pd.isna(col_data)
            categories = pd.Categorical(col_data[valid_mask])

            indicator = np.zeros((n_obs, len(categories.categories)))
            indicator[valid_mask] = pd.get_dummies(categories, drop_first=False).values

            freq = indicator.mean(axis=0)
            freq[freq == 0] = 1
            indicator_scaled = (indicator - freq) / np.sqrt(freq)
            transformed_cols.append(indicator_scaled)

            for cat in categories.categories:
                feature_names.append(f"{var_names[idx]}_{cat}")
                feature_to_original.append(idx)

    X_transformed = np.hstack(transformed_cols)
    X_transformed = np.nan_to_num(X_transformed)

    U, S, Vt = svd(X_transformed, full_matrices=False)
    n_components = min(n_components, min(X_transformed.shape))

    factor_scores = U[:, :n_components] * S[:n_components]
    loadings = Vt[:n_components, :].T

    metadata = {
        "variance": S[:n_components] ** 2 / n_obs,
        "variance_ratio": (S[:n_components] ** 2) / (S**2).sum(),
        "quant_mask": quant_mask,
        "n_components": n_components,
        "feature_names": feature_names,
        "feature_to_original": feature_to_original,
    }

    return factor_scores, loadings, metadata
