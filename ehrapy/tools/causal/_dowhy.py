from __future__ import annotations

from typing import Literal

import dowhy
import anndata


def causal_inference(
    adata: anndata.AnnData,
    graph: str,
    treatment: str,
    outcome: str,
    refute_methods: Literal["placebo_treatment_refuter", "random_common_cause", "data_subset_refuter"] = [
        "placebo_treatment_refuter",
        "random_common_cause",
        "data_subset_refuter",
    ],
):

    model = dowhy.CausalModel(
        data=adata.to_df(),  # 0 is not a problem, because impute
        graph=graph,
        treatment=treatment,
        outcome=outcome,
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_stratification")

    refute_results = {}
    for method in refute_methods:
        try:
            refute = model.refute_estimate(identified_estimand, estimate, method_name=method)
            refute_results[str(refute.refutation_type)] = {
                "Estimated effect": refute.estimated_effect,
                "New effect": refute.new_effect,
                "p-value": refute.refutation_result["p_value"],
                "test_significance": refute.estimated_effect,
            }
        except ValueError as e:
            refute_results[method] = str(e)

    return estimate, refute_results
