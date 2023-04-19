from __future__ import annotations

from typing import Tuple, Union, Literal, Dict, List

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
) -> Tuple[dowhy.CausalEstimate, Dict[str, Union[str, Dict[str, float]]]]:
    """
    Perform causal inference analysis using the DoWhy library.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing the data to be analyzed.
    graph : str
        A string specifying the causal graph to be used.
    treatment : str
        A string specifying the column name of the treatment variable in the data.
    outcome : str
        A string specifying the column name of the outcome variable in the data.
    refute_methods : List[Literal["placebo_treatment_refuter", "random_common_cause", "data_subset_refuter"]], optional
        A list of strings specifying the methods to be used to refute the causal effect estimate, by default ["placebo_treatment_refuter", "random_common_cause", "data_subset_refuter"].
    estimation_method : Literal["backdoor.propensity_score_stratification", "backdoor.propensity_score_weighting", "backdoor.linear_regression", "backdoor.propensity_score_matching", "backdoor.frontdoor_propensity_score_matching", "backdoor.instrumental_variable", "backdoor.doubly_robust_weighting", "backdoor.panel_regression"], optional
        A string specifying the method to be used to estimate the causal effect, by default "backdoor.propensity_score_stratification".
    return_as : Literal["summary", "estimate", "refute_results", "all"], optional
        A string specifying the type of result to be returned, by default "summary".

    Returns
    -------
    Tuple[dowhy.CausalEstimate, Dict[str, Union[str, Dict[str, float]]]] or None
        Depending on the value of `return_as`, returns a tuple of a `dowhy.CausalEstimate` object and a dictionary of refutation results, or None.

    Raises
    ------
    ValueError
        If an error occurs during the causal inference analysis.

    Notes
    -----
    This function uses the DoWhy library to perform causal inference analysis. It first creates a `dowhy.CausalModel` object using the specified data and causal graph, and then identifies and estimates the causal effect of the treatment variable on the outcome variable using the specified method. It then refutes the causal effect estimate using the specified methods, and returns a summary of the results or the full results depending on the value of `return_as`.
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

    # Create causal model
    model = dowhy.CausalModel(data=adata.to_df(), graph=graph, treatment=treatment, outcome=outcome)

    # Identify effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Estimate effect
    estimate = model.estimate_effect(identified_estimand, method_name=estimation_method)

    # Refute estimate using specified methods
    refute_results = {}
    for method in refute_methods:
        try:
            refute = model.refute_estimate(identified_estimand, estimate, method_name=method)
            test_significance = refute.estimated_effect

            refute_results[str(refute.refutation_type)] = {
                "Estimated effect": refute.estimated_effect,
                "New effect": refute.new_effect,
                "p-value": refute.refutation_result["p_value"],
                "test_significance": test_significance,
            }
        except ValueError as e:
            refute_results[method] = str(e)

    # Create the summary string
    summary = f"Causal inference results for treatment variable '{treatment}' and outcome variable '{outcome}':\n"
    summary += f"Estimated effect: {estimate.value:.2f}\n"
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