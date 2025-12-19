# Plotting

The plotting module `ehrapy.pl.\*` largely parallels the `tl.\*` and a few of the `pp.\*` functions.
For most tools and for some preprocessing functions, you will find a plotting function with the same name.

```{eval-rst}
.. module:: ehrapy
    :no-index:
```

## Generic

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.scatter
    plot.heatmap
    plot.dotplot
    plot.tracksplot
    plot.violin
    plot.stacked_violin
    plot.matrixplot
    plot.clustermap
    plot.ranking
    plot.dendrogram
    plot.catplot
    plot.timeseries
    plot.sankey_diagram
    plot.sankey_diagram_time
```

## Quality Control and missing values

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.missing_values_matrix
    plot.missing_values_barplot
    plot.missing_values_heatmap
    plot.missing_values_dendrogram
```

## Classes

Please refer to [Scanpy's plotting classes documentation](https://scanpy.readthedocs.io/en/stable/api.html#classes).

## Tools

Methods that extract and visualize tool-specific annotation in an AnnData object. For any method in module `tl`, there is a method with the same name in `pl`.

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.pca
    plot.pca_loadings
    plot.pca_variance_ratio
    plot.pca_overview
```

## Embeddings

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.tsne
    plot.umap
    plot.diffmap
    plot.draw_graph
    plot.embedding
    plot.embedding_density
```

## Branching trajectories and pseudotime

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.dpt_groups_pseudotime
    plot.dpt_timeseries
    plot.paga
    plot.paga_path
    plot.paga_compare
```

## Feature Ranking

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.rank_features_groups
    plot.rank_features_groups_violin
    plot.rank_features_groups_stacked_violin
    plot.rank_features_groups_heatmap
    plot.rank_features_groups_dotplot
    plot.rank_features_groups_matrixplot
    plot.rank_features_groups_tracksplot
    plot.rank_features_supervised
```

## Survival Analysis

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.ols
    plot.kaplan_meier
    plot.cox_ph_forestplot
```

## Causal Inference

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.causal_effect
```
