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
    tools.famd
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

Regression and survival models estimate how covariates relate to outcomes and to the time until an event.

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.ols
    tools.glm
    tools.kaplan_meier
    tools.test_kmf_logrank
    tools.test_nested_f_statistic
    tools.anova_glm
    tools.cox_ph
    tools.cox_ph_adjusted_curves
    tools.weibull_aft
    tools.log_logistic_aft
    tools.nelson_aalen
    tools.weibull

```

## Causal Inference

Causal estimators estimate the average and individual effects of a binary treatment from observational data, and diagnostics check their assumptions.

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.iptw
    tools.g_computation
    tools.aipw
    tools.propensity_score_matching
    tools.t_learner
    tools.s_learner
    tools.x_learner
    tools.covariate_balance
    tools.positivity_check
    tools.CausalEstimate
```

## Patterns across patients, variables and time

Finds groups of patients whose variables follow a similar course over time.

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.ncp
```

## Comorbidity

Scores the comorbidity burden of every patient from ICD-10 codes.

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.comorbidity_index
```

## Cohort Tracking & summaries

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.CohortTracker
    tools.stratified_table_one
```
