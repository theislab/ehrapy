# Tools

Any transformation of the data matrix that is not preprocessing.
In contrast to a preprocessing function, a tool usually adds an easily interpretable annotation to the data matrix, which can then be visualized with a corresponding plotting function.

```{eval-rst}
.. module:: ehrapy
    :no-index:
```

## Embeddings

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.tsne
    tools.umap
    tools.draw_graph
    tools.diffmap
    tools.embedding_density
```

## Clustering and trajectory inference

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.leiden
    tools.dendrogram
    tools.dpt
    tools.paga
```

## Feature Ranking

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.rank_features_groups
    tools.filter_rank_features_groups
    tools.rank_features_supervised
```

## Dataset integration

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.ingest
```

## Survival Analysis

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.ols
    tools.glm
    tools.kaplan_meier
    tools.test_kmf_logrank
    tools.test_nested_f_statistic
    tools.cox_ph
    tools.cox_ph_adjusted_curves
    tools.weibull_aft
    tools.log_logistic_aft
    tools.nelson_aalen
    tools.weibull

```

## Causal Inference

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.causal_inference
```

## Normalized Complexity Profile

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.ncp
```

## Cohort Tracking & summaries

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.CohortTracker
    tools.stratified_table_one
```
