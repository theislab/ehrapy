from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import dowhy


def causal_effect(estimate: dowhy.causal_estimator.CausalEstimate, precision: int = 3) -> matplotlib.axes:
    """Plot the causal effect estimate.

    This function plots the causal effect of treatment on outcome, assuming a
    linear relationship between the two. It uses the data, treatment name,
    outcome name, and estimate object to determine the data to plot. It then
    creates a plot with the treatment on the x-axis and the outcome on the
    y-axis. The observed data is plotted as gray dots, and the causal variation
    is plotted as a black line. The function then returns the plot.

    Args:
        estimate: The causal effect estimate to plot.
        precision: The number of decimal places to round the estimate to in the plot title.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object containing the plot.

    Raises:
        TypeError: If the `estimate` parameter is not an instance of `dowhy.causal_estimator.CausalEstimate`.
        ValueError: If the estimation method in `estimate` is not supported for this plot type.
    """
    if "LinearRegressionEstimator" not in str(estimate.params["estimator_class"]):
        raise ValueError(f"Estimation method {estimate.params['estimator_class']} is not supported for this plot type.")

    treatment_name = estimate.estimator._target_estimand.treatment_variable[0]
    outcome_name = estimate.estimator._target_estimand.outcome_variable[0]
    data = estimate._data
    treatment = data[treatment_name].values
    outcome = data[outcome_name]

    _, ax = plt.subplots()
    x_min = 0
    x_max = max(treatment)
    if isinstance(x_max, np.ndarray) and len(x_max) == 1:
        x_max = x_max[0]
    y_min = estimate.params["intercept"]
    y_max = y_min + estimate.value * (x_max - x_min)
    if isinstance(y_max, np.ndarray) and len(y_max) == 1:
        y_max = y_max[0]
    ax.scatter(treatment, outcome, c="gray", marker="o", label="Observed data")
    ax.plot([x_min, x_max], [y_min, y_max], c="black", ls="solid", lw=4, label="Causal variation")
    ax.set_ylim(0, max(outcome))
    ax.set_xlim(0, x_max)
    ax.set_title(r"DoWhy estimate $\rho$ (slope) = " + str(round(estimate.value, precision)))
    ax.legend(loc="upper left")
    ax.set_xlabel(treatment_name)
    ax.set_ylabel(outcome_name)
    plt.tight_layout()

    return ax
