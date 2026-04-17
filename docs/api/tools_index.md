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

## Cohort Tracking

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.CohortTracker
```

## Drug Screening

Functions for preparing prescription episodes from therapy data and running the
self-controlled cohort screening workflow ported from the `original/` drug
screening analysis. The current API supports common free-text dosage parsing,
named screening workflows (`actual`, `30days`, `365days`), and the indication
mapping joins used to connect RxNorm, SNOMED, Read codes, CPRD medcodes, and
CPRD product codes. It also includes an optional LLM review layer that mirrors
the original `chatgpt.R` prompt workflow through user-supplied callables rather
than a hard-coded provider integration. For more complex dosage instructions,
prefer precomputed structured fields.

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.compute_ndd_from_text
    tools.prepare_prescriptions_from_therapy
    tools.prepare_prescriptions_with_drugprepr
    tools.extract_rxclass_may_treat
    tools.build_rxcui_medcode_map
    tools.build_bnfcode_prodcode_map
    tools.build_rxcui_prodcode_map
    tools.build_prodcode_medcode_map
    tools.build_indication_map
    tools.review_repurposing_indications
    tools.summarize_drug_indications
    tools.review_repurposing_risk_factors
    tools.review_safety_symptoms
    tools.review_safety_indications
    tools.review_safety_aging
    tools.screen_grouped_therapy
    tools.screen_substance_therapy
    tools.screen_substance_cohort
    tools.screen_drugs
    tools.rate_ratio_test
```
