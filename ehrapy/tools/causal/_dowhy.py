from __future__ import annotations

import importlib
import sys
import warnings
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal

import ehrdata as ed
import numpy as np
from ehrdata._logger import logger

from ehrapy._compat import function_2D_only, use_ehrdata

if TYPE_CHECKING:
    import dowhy
    import networkx as nx
    from anndata import AnnData
    from ehrdata import EHRData

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


@function_2D_only()
@use_ehrdata(deprecated_after="1.0.0")
def causal_inference(
    edata: EHRData | AnnData,
    graph: nx.DiGraph | str,
    treatment: str,
    outcome: str,
    estimation_method: Literal[
        "backdoor.propensity_score_matching",
        "backdoor.propensity_score_stratification",
        "backdoor.propensity_score_weighting",
        "backdoor.linear_regression",
        "backdoor.generalized_linear_model",
        "iv.instrumental_variable",
        "iv.regression_discontinuity",
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
    refute_methods: None
    | list[str]
    | (
        list[
            Literal[
                "placebo_treatment_refuter",
                "random_common_cause",
                "data_subset_refuter",
                "add_unobserved_common_cause",
                "bootstrap_refuter",
                "dummy_outcome_refuter",
            ]
        ]
    ) = None,
    print_causal_estimate: bool = False,
    print_summary: bool = True,
    return_as: Literal["estimate", "refute", "estimate+refute"] = "estimate",
    show_graph: bool = False,
    show_refute_plots: bool | Literal["colormesh", "contour", "line"] | None = None,
    attempts: int = 10,
    *,
    layer: str | None = None,
    identify_kwargs: dict[str, Any] | None = None,
    estimate_kwargs: dict[str, Any] | None = None,
    refute_kwargs: dict[str, Any] | None = None,
) -> tuple[dowhy.CausalEstimate, dict[str, str | dict[str, float]]]:
    """Performs causal inference using the specified causal model and returns a tuple containing the causal estimate and the results of any refutation tests.

    Args:
        edata: Central data object.
        graph: A str representing the causal graph to use.
        treatment: A str representing the treatment variable in the causal graph.
        outcome: A str representing the outcome variable in the causal graph.
        estimation_method: An optional Literal specifying the estimation method to use.
        refute_methods: An optional List of Literal specifying the methods to use for refutation tests.
        print_causal_estimate: Whether to print the causal estimate or not.
        print_summary: Whether to print the causal model summary or not.
        return_as: An optional Literal specifying the type of output to return.
        show_graph: Whether to display the graph or not.
        show_refute_plots: Whether to display the refutation plots or not.
        attempts: Number of attempts to try to generate a valid causal estimate.
        layer: The layer to use.
        identify_kwargs: Optional keyword arguments for dowhy.CausalModel.identify_effect().
        estimate_kwargs: Optional keyword arguments for dowhy.CausalModel.estimate_effect().
        refute_kwargs: Optional keyword arguments for dowhy.CausalModel.refute_estimate().

    Returns:
        A tuple containing the causal estimate and a dictionary of the results of any refutation tests.

    Raises:
        TypeError: If edata, graph, treatment, outcome, refute_methods, estimation_method, or return_as is not of the expected type.
        ValueError: If refute_methods or estimation_method contains an unknown value, or if return_as is an unknown value.

    Examples:
        >>> data = dowhy.datasets.linear_dataset(
        ...     beta=10,
        ...     num_common_causes=5,
        ...     num_instruments=2,
        ...     num_samples=1000,
        ...     treatment_is_binary=True,
        ... )

        >>> ci = ep.tl.causal_inference(
        ...     edata=ehrdata.EHRData(data["df"]),
        ...     graph=data["gml_graph"],
        ...     treatment="v0",
        ...     outcome="y",
        ...     estimation_method="backdoor.propensity_score_stratification",
        ... )

        >>> estimate = ep.tl.causal_inference(
        ...     edata=ci.linear_data,
        ...     graph=ci.linear_graph,
        ...     treatment="treatment",
        ...     outcome="outcome",
        ...     estimation_method="backdoor.linear_regression",
        ...     return_as="estimate",
        ...     show_graph=True,
        ...     show_refute_plots=True,
        ... )
        ... ep.tl.plot_causal_effect(estimate)
    """
    valid_refute_methods = [
        "placebo_treatment_refuter",
        "random_common_cause",
        "data_subset_refuter",
        "add_unobserved_common_cause",
        "bootstrap_refuter",
        "dummy_outcome_refuter",
    ]

    if refute_methods is None:
        refute_methods = valid_refute_methods

    if isinstance(refute_methods, str):
        refute_methods = [refute_methods]

    for method in refute_methods:
        if method not in valid_refute_methods:
            raise ValueError(f"Unknown refute method {method}")

    if return_as not in ["estimate", "refute", "estimate+refute"]:
        raise ValueError(f"Unknown value for return_as '{return_as}': {return_as}")

    identify_kwargs = identify_kwargs or {}
    estimate_kwargs = estimate_kwargs or {}
    refute_kwargs = refute_kwargs or {}

    if show_refute_plots is None or show_refute_plots is False:
        refute_kwargs["plotmethod"] = None
    elif isinstance(show_refute_plots, str):
        refute_kwargs["plotmethod"] = show_refute_plots
    elif show_refute_plots is True:
        refute_kwargs["plotmethod"] = "colormesh"

    user_gave_num_simulations = "num_simulations" in refute_kwargs
    user_gave_random_seed = "random_state" in refute_kwargs
    found_problematic_pvalues = True

    if importlib.util.find_spec("dowhy") is None:
        raise ImportError("Required package 'dowhy' is not installed. Install with: pip install dowhy")

    import dowhy

    model = dowhy.CausalModel(
        data=ed.io.to_pandas(edata, layer=layer), graph=graph, treatment=treatment, outcome=outcome
    )

    if show_graph:
        model.view_model()

    # For some reason, dowhy sometimes fails to calculate a pval
    # and spits out NaN or values greater than 1. In that case we just try again.
    failed_attempts = 0
    while found_problematic_pvalues:
        if not user_gave_num_simulations:
            refute_kwargs["num_simulations"] = np.random.default_rng().integers(70, 90)
        if not user_gave_random_seed:
            refute_kwargs["random_seed"] = np.random.default_rng().integers(0, 100)

        identified_estimand = model.identify_effect(**identify_kwargs)

        # otherwise prints estimation_method
        with capture_output() as _:
            # input validation since `dowhy` does not do it
            if "." not in estimation_method:
                raise ValueError(f"Estimation method '{estimation_method}' not supported.")
            else:
                if len(estimation_method.split(".")) > 2 and not any(
                    ["dowhy" in estimation_method, "_estimator" in estimation_method]
                ):
                    raise ValueError(f"Estimation method '{estimation_method}' not supported.")

            estimate = model.estimate_effect(identified_estimand, method_name=estimation_method, **estimate_kwargs)

        refute_results: dict[str, str | dict[str, str]] = {}

        for method in refute_methods:
            try:
                with capture_output() as _:
                    refute = model.refute_estimate(
                        identified_estimand, estimate, method_name=method, verbose=False, **refute_kwargs
                    )
                    refute_failed = False
            except ValueError as e:
                refute_failed = True
                refute_results[method] = str(e)  # type: ignore

            if refute_failed:
                logger.warning(f"Refutation '{method}' failed.")
            else:
                # the dummy refuter returns a list with 1 element
                if isinstance(refute, list):
                    refute = refute[0]
                # only returns dict when pval should be a number
                if isinstance(refute.refutation_result, dict):
                    if 0 <= refute.refutation_result["p_value"] <= 1:
                        found_problematic_pvalues = False
                    else:
                        failed_attempts += 1
                        if failed_attempts <= attempts:
                            found_problematic_pvalues = True
                            logger.warning(
                                f"Refutation '{method}' returned invalid pval '{str(refute.refutation_result['p_value'])}', retrying ({failed_attempts}/{attempts})"
                            )
                            break
                        else:
                            found_problematic_pvalues = False
                else:
                    found_problematic_pvalues = False

            if not refute_failed:
                test_significance = refute.estimated_effect

                # Try to extract pval, fails for "add_unobserved_common_cause" refuter
                try:
                    pval = f"{refute.refutation_result['p_value']:.3f}"
                except TypeError:
                    pval = "Not applicable"

                # Format effect, can be list when refuter is "add_unobserved_common_cause"
                if isinstance(refute.new_effect, list | tuple):
                    new_effect = ", ".join([str(np.round(x, 2)) for x in refute.new_effect])
                else:
                    new_effect = f"{refute.new_effect:.3f}"

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

    if print_causal_estimate:
        print(estimate)

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
