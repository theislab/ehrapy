from __future__ import annotations

import sys
from io import StringIO
import warnings
from typing import Any, Literal

import anndata
import dowhy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

warnings.filterwarnings("ignore")


class capture_output(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def causal_inference(  # type: ignore
    adata: anndata.AnnData,
    graph: nx.DiGraph | str,
    treatment: str,
    outcome: str,
    estimation_method: Literal[
        # dowhy methods from https://github.com/py-why/dowhy/blob/8fb32a7bf617c1a64a2f8b61ed7a4a50ccaf8d8c/dowhy/causal_model.py#L247
        "backdoor.propensity_score_matching",
        "backdoor.propensity_score_stratification",
        "backdoor.propensity_score_weighting",
        "backdoor.linear_regression",
        "backdoor.generalized_linear_model",
        "iv.instrumental_variable",
        "iv.regression_discontinuity",
        # EconML estimators
        "backdoor.econml.linear_model.LinearDML",
        "backdoor.econml.nonparametric_model.NonParamDML",
        "backdoor.econml.causal_forest.CausalForestDML",
        "backdoor.econml.forecast_model.ForestDML",
        "backdoor.econml.dml.DML",
        "backdoor.econml.dml.DMLCate",
        "backdoor.econml.xgboost.XGBTRegressor",
        "backdoor.econml.xgboost.XGBTEstimator",
        "backdoor.econml.metalearners.XLearner",
    ],
    refute_methods: list[
        Literal[
            "placebo_treatment_refuter", "random_common_cause", "data_subset_refuter", "add_unobserved_common_cause"
        ]
    ] = None,
    print_summary: bool = True,
    return_as: Literal["estimate", "refute", "estimate+refute"] = "estimate",
    show_graph: bool = False,
    show_refute_plots: bool | Literal["colormesh", "contour", "line"] | None = None,
    *,
    identify_kwargs: dict[str, Any] | None = None,
    estimate_kwargs: dict[str, Any] | None = None,
    refute_kwargs: dict[str, Any] | None = None,
) -> tuple[dowhy.CausalEstimate, dict[str, str | dict[str, float]]]:
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
        show_graph: Whether to display the graph or not, default is False.
        show_refute_plots: Whether to display the refutation plots or not, default is False.
        identify_kwargs: Optional keyword arguments for dowhy.CausalModel.identify_effect().
        estimate_kwargs: Optional keyword arguments for dowhy.CausalModel.estimate_effect().
        refute_kwargs: Optional keyword arguments for dowhy.CausalModel.refute_estimate().

    Returns:
        A tuple containing the causal estimate and a dictionary of the results of any refutation tests.

    Raises:
        TypeError: If adata, graph, treatment, outcome, refute_methods, estimation_method, or return_as is not of the expected type.
        ValueError: If refute_methods or estimation_method contains an unknown value, or if return_as is an unknown value.

    Examples:
        >>> data = dowhy.datasets.linear_dataset(
        >>>     beta=10,
        >>>     num_common_causes=5,
        >>>     num_instruments=2,
        >>>     num_samples=1000,
        >>>     treatment_is_binary=True,
        >>> )
        >>>
        >>> ep.tl.causal_inference(
        >>>     adata=anndata.AnnData(data["df"]),
        >>>     graph=data["gml_graph"],
        >>>     treatment="v0",
        >>>     outcome="y",
        >>>     estimation_method="backdoor.propensity_score_stratification",
        >>> )
    """
    if not isinstance(adata, anndata.AnnData):
        raise TypeError("Parameter 'adata' must be an instance of anndata.AnnData.")

    if not isinstance(graph, (nx.DiGraph, str)):
        raise TypeError("Input graph must be a networkx DiGraph or string.")

    if not isinstance(treatment, str):
        raise TypeError("treatment must be a string")

    if not isinstance(outcome, str):
        raise TypeError("outcome must be a string")

    valid_refute_methods = [
        "placebo_treatment_refuter",
        "random_common_cause",
        "data_subset_refuter",
        "add_unobserved_common_cause",
    ]

    if refute_methods is None:
        refute_methods = valid_refute_methods

    if not isinstance(refute_methods, (list, str)):
        raise TypeError("Parameter 'refute_methods' must be a list or a string")

    if isinstance(refute_methods, str):
        refute_methods = [refute_methods]

    if isinstance(refute_methods, list):
        if not all([isinstance(rm, str) for rm in refute_methods]):
            raise TypeError("When parameter 'refute_methods' is a list, all of them must be strings.")

    for method in refute_methods:
        if method not in valid_refute_methods:
            raise ValueError(f"Unknown refute method {method}")

    if not isinstance(estimation_method, str):
        raise TypeError("estimation_method must be a string")

    # if estimation_method not in [
    #     "backdoor.propensity_score_stratification",
    #     "backdoor.propensity_score_weighting",
    #     "backdoor.linear_regression",
    #     "backdoor.propensity_score_matching",
    #     "backdoor.frontdoor_propensity_score_matching",
    #     "backdoor.instrumental_variable",
    #     "backdoor.doubly_robust_weighting",
    #     "backdoor.panel_regression",
    # ]:
    #     raise ValueError(f"Unknown estimation method '{estimation_method}': {estimation_method}")

    if not isinstance(return_as, str):
        raise TypeError("return_as must be a string")

    if return_as not in ["estimate", "refute", "estimate+refute"]:
        raise ValueError(f"Unknown value for return_as '{return_as}': {return_as}")

    if not isinstance(show_graph, bool):
        raise TypeError("Parameter 'show_graph' must be a boolean.")

    # Define dictionary containing valid kwargs for each function
    valid_kwargs = {
        "identify_effect": ["proceed_when_unidentifiable"],
        "estimate_effect": ["method_name", "method_params"],
        "refute_estimate": ["method_name", "confounders_frontdoor", "confounders_backdoor", "plotmethod"],
    }

    # Create causal model
    model = dowhy.CausalModel(data=adata.to_df(), graph=graph, treatment=treatment, outcome=outcome)

    if show_graph:
        model.view_model()

    # Identify effect
    identify_kwargs = identify_kwargs or {}
    assert set(identify_kwargs.keys()).issubset(valid_kwargs["identify_effect"]), "Invalid identify_kwargs"
    identified_estimand = model.identify_effect(**identify_kwargs)

    # Estimate effect
    estimate_kwargs = estimate_kwargs or {}
    assert set(estimate_kwargs.keys()).issubset(valid_kwargs["estimate_effect"]), "Invalid estimate_kwargs"

    with capture_output() as _:
        estimate = model.estimate_effect(identified_estimand, method_name=estimation_method, **estimate_kwargs)

    # Refute estimate using specified methods
    refute_results = {}
    for method in refute_methods:
        refute_kwargs = refute_kwargs or {}
        if show_refute_plots is None or show_refute_plots is False:
            refute_kwargs["plotmethod"] = None
        elif isinstance(show_refute_plots, str):
            refute_kwargs["plotmethod"] = show_refute_plots
        elif show_refute_plots is True:
            refute_kwargs["plotmethod"] = "colormesh"

        assert set(refute_kwargs.keys()).issubset(valid_kwargs["refute_estimate"]), "Invalid refute_kwargs"

        # Attempt to refute
        try:
            with capture_output() as _:
                refute = model.refute_estimate(
                    identified_estimand, estimate, method_name=method, verbose=False, **refute_kwargs
                )
                refute_failed = False

        except ValueError as e:
            refute_failed = True
            refute_results[method] = str(e)  # type: ignore

        if not refute_failed:
            test_significance = refute.estimated_effect

            # Try to extract pval, fails for "add_unobserved_common_cause" refuter
            try:
                pval = f"{refute.refutation_result['p_value']:.2f}"
            except:
                pval = "Not applicable"

            # Format effect, can be list when refuter is "add_unobserved_common_cause"
            if isinstance(refute.new_effect, (list, tuple)):
                new_effect = ", ".join([str(np.round(x, 2)) for x in refute.new_effect])
            else:
                new_effect = f"{refute.new_effect:.2f}"

            refute_results[str(refute.refutation_type)] = {
                "Estimated effect": refute.estimated_effect,
                "New effect": new_effect,
                "p-value": pval,
                "test_significance": test_significance,
            }

    # Create the summary string
    summary = f"Causal inference results for treatment variable '{treatment}' and outcome variable '{outcome}':\n"

    with capture_output() as output:
        estimate.interpret(method_name="textual_effect_interpreter")
    if output is not None:
        summary += f"└- {''.join(output)}\n"
    else:
        summary += f"└- Estimated effect: {estimate.value}\n"
    summary += "\nRefutation results\n"
    for idx, (method, results) in enumerate(refute_results.items()):  # type: ignore
        left_char = "|" if (idx + 1) != len(refute_results.keys()) else " "
        branch_char = "├" if (idx + 1) != len(refute_results.keys()) else "└"
        if isinstance(results, str):
            summary += f"├-Refute: {method}\n"
            summary += f"{left_char}    └- {results}\n"
        else:
            summary += f"{branch_char}-{method}\n"
            summary += f"{left_char}    ├- Estimated effect: {results['Estimated effect']:.2f}\n"
            summary += f"{left_char}    ├- New effect: {results['New effect']}\n"
            summary += f"{left_char}    ├- p-value: {results['p-value']}\n"
            summary += f"{left_char}    └- Test significance: {results['test_significance']:.2f}\n"

    if print_summary:
        print(summary)

    if return_as == "estimate":
        return estimate
    elif return_as == "refute":
        return refute_results  # type: ignore
    elif return_as == "estimate+refute":
        return estimate, refute_results  # type: ignore
    else:
        raise ValueError(f"Invalid return_as argument: {return_as}")


def plot_causal_effect(
    estimate: dowhy.causal_estimator.CausalEstimate,
    precision: int = 3,
    **kwargs
):
    """Plot the causal effect estimate."""

    if not isinstance(estimate, dowhy.causal_estimator.CausalEstimate):
        raise TypeError("Parameter 'estimate' must be a dowhy.causal_estimator.CausalEstimate object")

    if "LinearRegressionEstimator" not in str(estimate.params["estimator_class"]):
        raise ValueError(f"Estimation method {estimate.params['estimator_class']} is not supported for this plot type.")

    treatment_name = estimate.estimator._treatment_name
    outcome_name = estimate.estimator._outcome_name
    data = estimate.estimator._data
    treatment = data[treatment_name].values
    outcome = data[outcome_name]

    _, ax = plt.subplots()
    x_min = 0
    x_max = max(treatment)
    y_min = estimate.params["intercept"]
    y_max = y_min + estimate.value * (x_max - x_min)
    ax.scatter(treatment, outcome, c="gray", marker="o", label="Observed data")
    ax.plot([x_min, x_max], [y_min, y_max], c="black", ls="solid", lw=4, label="Causal variation")
    ax.set_ylim(0, max(outcome))
    ax.set_xlim(0, x_max)
    ax.set_title(r"DoWhy estimate $\rho$ (slope) = " + str(round(estimate.value, precision)))
    ax.legend(loc="upper left")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Outcome")
