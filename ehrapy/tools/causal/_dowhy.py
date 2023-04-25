from __future__ import annotations

from typing import Any, Optional, Tuple, Union, Literal, Dict, List

import io
from contextlib import redirect_stdout
import dowhy
import anndata


def causal_inference(
    adata: anndata.AnnData,
    graph: str,
    treatment: str,
    outcome: str,
    refute_methods: List[Literal["placebo_treatment_refuter", "random_common_cause", "data_subset_refuter"]] = [
        "placebo_treatment_refuter",
        "random_common_cause",
        "data_subset_refuter",
    ],
    estimation_method: Literal[
        "backdoor.propensity_score_stratification", 
        "backdoor.propensity_score_weighting",
        "backdoor.linear_regression",
        "backdoor.propensity_score_matching",
        "backdoor.frontdoor_propensity_score_matching",
        "backdoor.instrumental_variable",
        "backdoor.doubly_robust_weighting",
        "backdoor.panel_regression"
    ] = "backdoor.propensity_score_stratification",
    return_as: Literal["summary", "estimate", "refute_results", "all"] = "summary",
    # optional kwargs arguments
    identify_kwargs: Optional[Dict[str, Any]] = None,
    estimate_kwargs: Optional[Dict[str, Any]] = None,
    refute_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[dowhy.CausalEstimate, Dict[str, Union[str, Dict[str, float]]]]:
    """
    Performs causal inference on an AnnData object using the specified causal model and returns a tuple containing the causal estimate and the results of any refutation tests.

    Args:
        adata: An AnnData object containing the input data.
        graph: A str representing the causal graph to use.
        treatment: A str representing the treatment variable in the causal graph.
        outcome: A str representing the outcome variable in the causal graph.
        refute_methods: An optional List of Literal specifying the methods to use for refutation tests. Defaults to ["placebo_treatment_refuter", "random_common_cause", "data_subset_refuter"].
        estimation_method: An optional Literal specifying the estimation method to use. Defaults to "backdoor.propensity_score_stratification".
        return_as: An optional Literal specifying the type of output to return. Defaults to "summary".
        identify_kwargs: Optional keyword arguments for dowhy.CausalModel.identify_effect().
        estimate_kwargs: Optional keyword arguments for dowhy.CausalModel.estimate_effect().
        refute_kwargs: Optional keyword arguments for dowhy.CausalModel.refute_estimate().

    Returns:
        A tuple containing the causal estimate and a dictionary of the results of any refutation tests.

    Raises:
        TypeError: If adata, graph, treatment, outcome, refute_methods, estimation_method, or return_as is not of the expected type.
        ValueError: If refute_methods or estimation_method contains an unknown value, or if return_as is an unknown value.
    """

    if not isinstance(adata, anndata.AnnData):
        raise TypeError("adata must be an instance of anndata.AnnData")

    if not isinstance(graph, str):
        raise TypeError("graph must be a string")

    if not isinstance(treatment, str):
        raise TypeError("treatment must be a string")

    if not isinstance(outcome, str):
        raise TypeError("outcome must be a string")

    if not isinstance(refute_methods, list):
        raise TypeError("refute_methods must be a list")

    for method in refute_methods:
        if method not in ["placebo_treatment_refuter", "random_common_cause", "data_subset_refuter"]:
            raise ValueError(f"Unknown refute method {method}")

    if not isinstance(estimation_method, str):
        raise TypeError("estimation_method must be a string")

    if estimation_method not in [
        "backdoor.propensity_score_stratification", 
        "backdoor.propensity_score_weighting",
        "backdoor.linear_regression",
        "backdoor.propensity_score_matching",
        "backdoor.frontdoor_propensity_score_matching",
        "backdoor.instrumental_variable",
        "backdoor.doubly_robust_weighting",
        "backdoor.panel_regression"
    ]:
        raise ValueError(f"Unknown estimation method {estimation_method}")

    if not isinstance(return_as, str):
        raise TypeError("return_as must be a string")

    if return_as not in ["summary", "estimate", "refute_results", "all"]:
        raise ValueError(f"Unknown value for return_as {return_as}")

        # Define dictionary containing valid kwargs for each function
    valid_kwargs = {
        "identify_effect": ["proceed_when_unidentifiable"],
        "estimate_effect": ["method_name"],
        "refute_estimate": ["method_name", "confounders_frontdoor", "confounders_backdoor"],
    }

    # Create causal model
    model = dowhy.CausalModel(data=adata.to_df(), graph=graph, treatment=treatment, outcome=outcome)

    # Identify effect
    identify_kwargs = identify_kwargs or {}
    assert set(identify_kwargs.keys()).issubset(valid_kwargs["identify_effect"]), "Invalid identify_kwargs"
    identified_estimand = model.identify_effect(**identify_kwargs)

    # Identify effect
    # identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Estimate effect
    estimate_kwargs = estimate_kwargs or {}
    assert set(estimate_kwargs.keys()).issubset(valid_kwargs["estimate_effect"]), "Invalid estimate_kwargs"
    estimate = model.estimate_effect(identified_estimand, method_name=estimation_method, **estimate_kwargs)

    # estimate = model.estimate_effect(identified_estimand, method_name=estimation_method)
    
    # Refute estimate using specified methods
    refute_results = {}
    output_buffer = io.StringIO()
    for method in refute_methods:
        refute_kwargs = refute_kwargs or {}
        assert set(refute_kwargs.keys()).issubset(valid_kwargs["refute_estimate"]), "Invalid refute_kwargs"
        try:
            with redirect_stdout(output_buffer):
                refute = model.refute_estimate(identified_estimand, estimate, method_name=method, verbose=False, **refute_kwargs)
            test_significance = refute.estimated_effect

            refute_results[str(refute.refutation_type)] = {
                "Estimated effect": refute.estimated_effect,
                "New effect": refute.new_effect,
                "p-value": refute.refutation_result["p_value"],
                "test_significance": test_significance,
            }
        except ValueError as e:
            refute_results[method] = str(e)
    output_buffer.close()

    # Create the summary string
    summary = f"Causal inference results for treatment variable '{treatment}' and outcome variable '{outcome}':\n"
    summary += f"Estimated effect: {estimate.value}\n"
    summary += "Refutation results:\n"
    for method, results in refute_results.items():
        if isinstance(results, str):
            summary += f"{method}: {results}\n"
        else:
            summary += f"{method}:\n"
            summary += f"\tEstimated effect: {results['Estimated effect']:.2f}\n"
            summary += f"\tNew effect: {results['New effect']:.2f}\n"
            summary += f"\tp-value: {results['p-value']:.2f}\n"
            summary += f"\tTest significance: {results['test_significance']:.2f}\n"

    if return_as == "summary":
        print(summary)
    elif return_as == "estimate":
        return estimate
    elif return_as == "refute_results":
        return refute_results
    elif return_as == "all":
        return estimate, refute_results
    else:
        raise ValueError(f"Invalid return_as argument: {return_as}")