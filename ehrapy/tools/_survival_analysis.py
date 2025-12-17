from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import ehrdata as ed
import numpy as np  # noqa: TC002
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ehrdata.core.constants import CATEGORICAL_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from lifelines import (
    CoxPHFitter,
    KaplanMeierFitter,
    LogLogisticAFTFitter,
    NelsonAalenFitter,
    WeibullAFTFitter,
    WeibullFitter,
)
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import StatisticalResult, logrank_test
from scipy import stats
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper  # noqa

from ehrapy._compat import function_2D_only, use_ehrdata

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from ehrdata import EHRData


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
def ols(
    edata: EHRData | AnnData,
    var_names: list[str] | None | None = None,
    formula: str | None = None,
    *,
    missing: Literal["none", "drop", "raise"] | None = "none",
    use_feature_types: bool = False,
    layer: str | None = None,
) -> sm.OLS:
    """Create an Ordinary Least Squares (OLS) Model from a formula and the data object.

    See https://www.statsmodels.org/stable/generated/statsmodels.formula.api.ols.html#statsmodels.formula.api.ols

    Args:
        edata: Central data object.
        var_names: A list of var names indicating which columns are for the OLS model.
        formula: The formula specifying the model.
        use_feature_types: If True, the feature types in the data objects .var are used.
        missing: Available options are 'none', 'drop', and 'raise'.
                 If 'none', no nan checking is done. If 'drop', any observations with nans are dropped.
                 If 'raise', an error is raised.
        layer: The layer to use.

    Returns:
        The OLS model instance.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> formula = "tco2_first ~ pco2_first"
        >>> var_names = ["tco2_first", "pco2_first"]
        >>> ols = ep.tl.ols(edata, var_names, formula, missing="drop")
    """
    if isinstance(var_names, list):
        data = ed.io.to_pandas(edata[:, var_names], layer=layer)
    else:
        data = ed.io.to_pandas(edata, layer=layer)

    if use_feature_types:
        for col in data.columns:
            if col in edata.var.index:
                feature_type = edata.var[FEATURE_TYPE_KEY][col]
                if feature_type == CATEGORICAL_TAG:
                    data[col] = data[col].astype("category")
                elif feature_type == NUMERIC_TAG:
                    data[col] = data[col].astype(float)
    else:
        data = data.astype(float)

    ols = smf.ols(formula, data=data, missing=missing)

    return ols


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
def glm(
    edata: EHRData | AnnData,
    var_names: Iterable[str] | None = None,
    formula: str | None = None,
    *,
    family: Literal["Gaussian", "Binomial", "Gamma", "Gaussian", "InverseGaussian"] = "Gaussian",
    use_feature_types: bool = False,
    missing: Literal["none", "drop", "raise"] = "none",
    as_continuous: Iterable[str] | None | None = None,
    layer: str | None = None,
) -> sm.GLM:
    """Create a Generalized Linear Model (GLM) from a formula, a distribution, and the data object.

    See https://www.statsmodels.org/stable/generated/statsmodels.formula.api.glm.html#statsmodels.formula.api.glm

    Args:
        edata: Central data object.
        var_names: A list of var names indicating which columns are for the GLM model.
        formula: The formula specifying the model.
        family: The distribution families. Available options are 'Gaussian', 'Binomial', 'Gamma', and 'InverseGaussian'.
        use_feature_types: If True, the feature types in the data objects .var are used.
        missing: Available options are 'none', 'drop', and 'raise'. If 'none', no nan checking is done.
                 If 'drop', any observations with nans are dropped. If 'raise', an error is raised.
        as_continuous: A list of var names indicating which columns are continuous rather than categorical.
                    The corresponding columns will be set as type float.
        layer: The layer to use.

    Returns:
        The GLM model instance.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> formula = "day_28_flg ~ age"
        >>> var_names = ["day_28_flg", "age"]
        >>> family = "Binomial"
        >>> glm = ep.tl.glm(edata, var_names, formula, family, missing="drop", as_continuous=["age"])
    """
    family_dict = {
        "Gaussian": sm.families.Gaussian(),
        "Binomial": sm.families.Binomial(),
        "Gamma": sm.families.Gamma(),
        "InverseGaussian": sm.families.InverseGaussian(),
    }
    if family in ["Gaussian", "Binomial", "Gamma", "Gaussian", "InverseGaussian"]:
        family = family_dict[family]
    if isinstance(var_names, list):
        data = ed.io.to_pandas(edata[:, var_names], layer=layer)
    else:
        data = ed.io.to_pandas(edata, layer=layer)
    if as_continuous is not None:
        data[as_continuous] = data[as_continuous].astype(float)
    if use_feature_types:
        for col in data.columns:
            if col in edata.var.index:
                feature_type = edata.var[FEATURE_TYPE_KEY][col]
                if feature_type == CATEGORICAL_TAG:
                    data[col] = data[col].astype("category")
                elif feature_type == NUMERIC_TAG:
                    data[col] = data[col].astype(float)

    glm = smf.glm(formula, data=data, family=family, missing=missing)

    return glm


def kmf(
    durations: Iterable,
    event_observed: Iterable | None = None,
    timeline: Iterable = None,
    entry: Iterable | None = None,
    label: str | None = None,
    alpha: float | None = None,
    ci_labels: tuple[str, str] = None,
    weights: Iterable | None = None,
    censoring: Literal["right", "left"] = None,
) -> KaplanMeierFitter:
    """DEPRECATION WARNING: This function is deprecated and will be removed in the next release -use `kaplan_meier` instead.

    Fit the Kaplan-Meier estimate for the survival function.

    The Kaplan-Meier estimator, also known as the product limit estimator, is a non-parametric statistic used to estimate the survival function from lifetime data.
    In medical research, it is often used to measure the fraction of patients living for a certain amount of time after treatment.

    See https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator
        https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html#module-lifelines.fitters.kaplan_meier_fitter

    Args:
        durations: length n -- duration (relative to subject's birth) the subject was alive for.
        event_observed: True if the death was observed, False if the event was lost (right-censored). Defaults to all True if event_observed is equal to `None`.
        timeline: return the best estimate at the values in timelines (positively increasing)
        entry: Relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations.
               If None, all members of the population entered study when they were "born".
        label: A string to name the column of the estimate.
        alpha: The alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
        ci_labels: Add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>] (default: <label>_lower_<1-alpha/2>).
        weights: If providing a weighted dataset. For example, instead of providing every subject
                 as a single element of `durations` and `event_observed`, one could weigh subject differently.
        censoring: 'right' for fitting the model to a right-censored dataset.
                   'left' for fitting the model to a left-censored dataset (default: fit the model to a right-censored dataset).

    Returns:
        Fitted KaplanMeierFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> kmf = ep.tl.kmf(edata[:, ["mort_day_censored"]].X, edata[:, ["censor_flg"]].X)
    """
    warnings.warn(
        "This function is deprecated and will be removed in the next release. Use `ep.tl.kaplan_meier` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    kmf = KaplanMeierFitter()
    if censoring == "None" or "right":
        kmf.fit(
            durations=durations,
            event_observed=event_observed,
            timeline=timeline,
            entry=entry,
            label=label,
            alpha=alpha,
            ci_labels=ci_labels,
            weights=weights,
        )
    elif censoring == "left":
        kmf.fit_left_censoring(
            durations=durations,
            event_observed=event_observed,
            timeline=timeline,
            entry=entry,
            label=label,
            alpha=alpha,
            ci_labels=ci_labels,
            weights=weights,
        )

    return kmf


@function_2D_only()
@use_ehrdata(deprecated_after="1.0.0")
def kaplan_meier(
    edata: EHRData | AnnData,
    duration_col: str,
    event_col: str | None = None,
    *,
    uns_key: str = "kaplan_meier",
    timeline: list[float] | None = None,
    entry: str | None = None,
    label: str | None = None,
    alpha: float | None = None,
    ci_labels: list[str] | None = None,
    weights: list[float] | None = None,
    fit_options: dict | None = None,
    censoring: Literal["right", "left"] = "right",
    layer: str | None = None,
) -> KaplanMeierFitter:
    """Fit the Kaplan-Meier estimate for the survival function.

    The Kaplan–Meier estimator, also known as the product limit estimator, is a non-parametric statistic used to estimate the survival function from lifetime data.
    In medical research, it is often used to measure the fraction of patients living for a certain amount of time after treatment.
    The results will be stored in the `.uns` slot of the data object under the key 'kaplan_meier' unless specified otherwise in the `uns_key` parameter.

    See `Kaplan Meier on Wikipedia <https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator>`_ and `Kaplan Meier on Lifelines <https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html#module-lifelines.fitters.kaplan_meier_fitter>`_.

    Args:
        edata: Central data object.
        duration_col: The name of the column in the data object that contains the subjects' lifetimes.
        event_col: The name of the column in the data object that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
        uns_key: The key to use for the `.uns` slot in the data object.
        timeline: Return the best estimate at the values in timelines (positively increasing)
        entry: Relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations.
               If None, all members of the population entered study when they were "born".
        label: A string to name the column of the estimate.
        alpha: The alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
        ci_labels: Add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>] (default: <label>_lower_<1-alpha/2>).
        weights: If providing a weighted dataset. For example, instead of providing every subject
                 as a single element of `durations` and `event_observed`, one could weigh subject differently.
        fit_options: Additional keyword arguments to pass into the estimator.
        censoring: 'right' for fitting the model to a right-censored dataset. (default, calls fit).
                   'left' for fitting the model to a left-censored dataset (calls fit_left_censoring).
        layer: The layer to use.

    Returns:
        Fitted KaplanMeierFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> kmf = ep.tl.kaplan_meier(edata, "mort_day_censored", "censor_flg", label="Mortality")
    """
    return _univariate_model(
        edata,
        duration_col,
        event_col,
        KaplanMeierFitter,
        uns_key,
        True,
        timeline,
        entry,
        label,
        alpha,
        ci_labels,
        weights,
        fit_options,
        censoring,
        layer,
    )


def test_kmf_logrank(
    kmf_A: KaplanMeierFitter,
    kmf_B: KaplanMeierFitter,
    t_0: float | None = -1,
    weightings: Literal["wilcoxon", "tarone-ware", "peto", "fleming-harrington"] | None = None,
) -> StatisticalResult:
    """Calculates the p-value for the logrank test comparing the survival functions of two groups.

    Measures and reports on whether two intensity processes are different.
    That is, given two event series, determines whether the data generating processes are statistically different.
    The test-statistic is chi-squared under the null hypothesis.

    See https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html

    Args:
        kmf_A: The first KaplanMeierFitter object containing the durations and events.
        kmf_B: The second KaplanMeierFitter object containing the durations and events.
        t_0: The final time period under observation, and subjects who experience the event after this time are set to be censored.
             Specify -1 to use all time.
        weightings: Apply a weighted logrank test: options are "wilcoxon" for Wilcoxon (also known as Breslow), "tarone-ware"
                    for Tarone-Ware, "peto" for Peto test and "fleming-harrington" for Fleming-Harrington test.
                    These are useful for testing for early or late differences in the survival curve. For the Fleming-Harrington
                    test, keyword arguments p and q must also be provided with non-negative values.

    Returns:
        The p-value for the logrank test comparing the survival functions of the two groups.
    """
    results_pairwise = logrank_test(
        durations_A=kmf_A.durations,
        durations_B=kmf_B.durations,
        event_observed_A=kmf_A.event_observed,
        event_observed_B=kmf_B.event_observed,
        weights_A=kmf_A.weights,
        weights_B=kmf_B.weights,
        t_0=t_0,
        weightings=weightings,
    )

    return results_pairwise


def test_nested_f_statistic(small_model: GLMResultsWrapper, big_model: GLMResultsWrapper) -> float:
    """Calculate the P value indicating if a larger GLM, encompassing a smaller GLM's parameters, adds explanatory power.

    See https://stackoverflow.com/questions/27328623/anova-test-for-glm-in-python/60769343#60769343

    Args:
        small_model: fitted generalized linear models.
        big_model: fitted generalized linear models.

    Returns:
        float: p_value of Anova test.
    """
    addtl_params = big_model.df_model - small_model.df_model
    f_stat = (small_model.deviance - big_model.deviance) / (addtl_params * big_model.scale)
    df_numerator = addtl_params
    df_denom = big_model.fittedvalues.shape[0] - big_model.df_model
    p_value = stats.f.sf(f_stat, df_numerator, df_denom)

    return p_value


def anova_glm(
    result_1: GLMResultsWrapper,
    result_2: GLMResultsWrapper,
    formula_1: str,
    formula_2: str,
) -> pd.DataFrame:
    """Anova table for two fitted generalized linear models.

    Args:
        result_1: fitted generalized linear models.
        result_2: fitted generalized linear models.
        formula_1: The formula specifying the model.
        formula_2: The formula specifying the model.

    Returns:
        pd.DataFrame: Anova table.
    """
    p_value = test_nested_f_statistic(result_1, result_2)

    table = {
        "Model": [1, 2],
        "formula": [formula_1, formula_2],
        "Df Resid.": [result_1.df_resid, result_2.df_resid],
        "Dev.": [result_1.deviance, result_2.deviance],
        "Df_diff": [None, result_2.df_model - result_1.df_model],
        "Pr(>Chi)": [None, p_value],
    }
    dataframe = pd.DataFrame(data=table)
    return dataframe


def _build_model_input_dataframe(
    edata: EHRData | AnnData, duration_col: str, accept_zero_duration=True, layer: str | None = None
):
    """Convenience function for regression models."""
    df = ed.io.to_pandas(edata, layer=layer)
    df = df.dropna()

    if not accept_zero_duration:
        df.loc[df[duration_col] == 0, duration_col] += 1e-5

    return df


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
def cox_ph(
    edata: EHRData | AnnData,
    duration_col: str,
    event_col: str = None,
    *,
    uns_key: str = "cox_ph",
    alpha: float = 0.05,
    label: str | None = None,
    baseline_estimation_method: Literal["breslow", "spline", "piecewise"] = "breslow",
    penalizer: float | np.ndarray = 0.0,
    l1_ratio: float = 0.0,
    strata: list[str] | str | None = None,
    n_baseline_knots: int = 4,
    knots: list[float] | None = None,
    breakpoints: list[float] | None = None,
    weights_col: str | None = None,
    cluster_col: str | None = None,
    entry_col: str = None,
    robust: bool = False,
    formula: str = None,
    batch_mode: bool = None,
    show_progress: bool = False,
    initial_point: np.ndarray | None = None,
    fit_options: dict | None = None,
    layer: str | None = None,
) -> CoxPHFitter:
    """Fit the Cox’s proportional hazard for the survival function.

    The Cox proportional hazards model (CoxPH) examines the relationship between the survival time of subjects and one or more predictor variables.
    It models the hazard rate as a product of a baseline hazard function and an exponential function of the predictors, assuming proportional hazards over time.
    The results will be stored in the `.uns` slot of the data object under the key 'cox_ph' unless specified otherwise in the `uns_key` parameter.

    See https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html

    Args:
        edata: Central data object.
        duration_col: The name of the column in the data objects that contains the subjects’ lifetimes.
        event_col: The name of the column in the data object that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
        uns_key: The key to use for the `.uns` slot in the data object.
        alpha: The alpha value in the confidence intervals.
        label: The name of the column of the estimate.
        baseline_estimation_method: The method used to estimate the baseline hazard. Options are 'breslow', 'spline', and 'piecewise'.
        penalizer: Attach a penalty to the size of the coefficients during regression. This improves stability of the estimates and controls for high correlation between covariates.
        l1_ratio: Specify what ratio to assign to a L1 vs L2 penalty. Same as scikit-learn. See penalizer above.
        strata: specify a list of columns to use in stratification. This is useful if a categorical covariate does not obey the proportional hazard assumption. This is used similar to the strata expression in R. See http://courses.washington.edu/b515/l17.pdf.
        n_baseline_knots: Used when baseline_estimation_method="spline". Set the number of knots (interior & exterior) in the baseline hazard, which will be placed evenly along the time axis.
            Should be at least 2. Royston et. al, the authors of this model, suggest 4 to start, but any values between 2 and 8 are reasonable.
            If you need to customize the timestamps used to calculate the curve, use the knots parameter instead.
        knots: When baseline_estimation_method="spline", this allows customizing the points in the time axis for the baseline hazard curve. To use evenly-spaced points in time, the n_baseline_knots parameter can be employed instead.
        breakpoints: Used when baseline_estimation_method="piecewise". Set the positions of the baseline hazard breakpoints.
        weights_col: The name of the column in DataFrame that contains the weights for each subject.
        cluster_col: The name of the column in DataFrame that contains the cluster variable.
            Using this forces the sandwich estimator (robust variance estimator) to be used.
        entry_col: Column denoting when a subject entered the study, i.e. left-truncation.
        robust: Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate.
            This does not handle ties, so if there are high number of ties, results may significantly differ.
        formula: an Wilkinson formula, like in R and statsmodels, for the right-hand-side.
            If left as None, all columns not assigned as durations, weights, etc. are used.
            Uses the library Formulaic for parsing.
        batch_mode:  Enabling batch_mode can be faster for datasets with a large number of ties.
            If left as `None`, lifelines will choose the best option.
        show_progress: Since the fitter is iterative, show convergence diagnostics. Useful if convergence is failing.
        initial_point: set the starting point for the iterative solver.
        fit_options: Additional keyword arguments to pass into the estimator.
        layer: The layer to use.

    Returns:
        Fitted CoxPHFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> cph = ep.tl.cox_ph(
        ...     edata, "mort_day_censored", "censor_flg", formula="gender_num + afib_flg + day_icu_intime_num"
        ... )
    """
    df = _build_model_input_dataframe(edata, duration_col, layer=layer)
    cox_ph = CoxPHFitter(
        alpha=alpha,
        label=label,
        strata=strata,
        baseline_estimation_method=baseline_estimation_method,
        penalizer=penalizer,
        l1_ratio=l1_ratio,
        n_baseline_knots=n_baseline_knots,
        knots=knots,
        breakpoints=breakpoints,
    )
    try:
        cox_ph.fit(
            df,
            duration_col=duration_col,
            event_col=event_col,
            entry_col=entry_col,
            robust=robust,
            initial_point=initial_point,
            weights_col=weights_col,
            cluster_col=cluster_col,
            batch_mode=batch_mode,
            formula=formula,
            fit_options=fit_options,
            show_progress=show_progress,
        )
    except (ValueError, ConvergenceError) as e:
        special_cols = {duration_col, event_col, entry_col, weights_col, cluster_col} - {None}
        numeric_cols = [c for c in df.columns if c not in special_cols and df[c].dtype.kind in "iufb"]

        if "could not convert string to float" in str(e):
            non_numeric = [c for c in df.columns if c not in special_cols and df[c].dtype.kind not in "iufb"]
            raise ValueError(
                f"Non-numeric columns found: {non_numeric}\n"
                f"Specify numeric covariates with formula=, e.g.:\n"
                f'  ep.tl.cox_ph(..., formula="{" + ".join(numeric_cols[:3])}")'
            ) from e
        elif "singular" in str(e).lower() or "collinearity" in str(e).lower():
            raise ValueError(
                f"Matrix singularity (likely collinear or constant columns).\n"
                f"Specify covariates explicitly with formula=, e.g.:\n"
                f'  ep.tl.cox_ph(..., formula="{" + ".join(numeric_cols[:3])}")'
            ) from e
        raise

    summary = cox_ph.summary
    edata.uns[uns_key] = summary

    return cox_ph


@function_2D_only()
@use_ehrdata(deprecated_after="1.0.0")
def weibull_aft(
    edata: EHRData | AnnData,
    duration_col: str,
    event_col: str,
    *,
    uns_key: str = "weibull_aft",
    alpha: float = 0.05,
    fit_intercept: bool = True,
    penalizer: float | np.ndarray = 0.0,
    l1_ratio: float = 0.0,
    model_ancillary: bool = True,
    ancillary: bool | pd.DataFrame | str | None = None,
    show_progress: bool = False,
    weights_col: str | None = None,
    robust: bool = False,
    initial_point=None,
    entry_col: str | None = None,
    formula: str | None = None,
    fit_options: dict | None = None,
    layer: str | None = None,
) -> WeibullAFTFitter:
    """Fit the Weibull accelerated failure time regression for the survival function.

    The Weibull Accelerated Failure Time (AFT) survival regression model is a statistical method used to analyze time-to-event data,
    where the underlying assumption is that the logarithm of survival time follows a Weibull distribution.
    It models the survival time as an exponential function of the predictors, assuming a specific shape parameter
    for the distribution and allowing for accelerated or decelerated failure times based on the covariates.
    The results will be stored in the `.uns` slot of the data object under the key 'weibull_aft' unless specified otherwise in the `uns_key` parameter.

    See https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html

    Args:
        edata: Central data object.
        duration_col: Name of the column in the data objects that contains the subjects’ lifetimes.
        event_col: The name of the column in the data object that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
        uns_key: The key to use for the `.uns` slot in the data object.
        alpha: The alpha value in the confidence intervals.
        fit_intercept: Whether to fit an intercept term in the model.
        penalizer: Attach a penalty to the size of the coefficients during regression. This improves stability of the estimates and controls for high correlation between covariates.
        l1_ratio: Specify what ratio to assign to a L1 vs L2 penalty. Same as scikit-learn. See penalizer above.
        model_ancillary: set the model instance to always model the ancillary parameter with the supplied Dataframe. This is useful for grid-search optimization.
        ancillary: Choose to model the ancillary parameters.
            If None or False, explicitly do not fit the ancillary parameters using any covariates.
            If True, model the ancillary parameters with the same covariates as ``df``.
            If DataFrame, provide covariates to model the ancillary parameters. Must be the same row count as ``df``.
            If str, should be a formula
        show_progress: since the fitter is iterative, show convergence diagnostics. Useful if convergence is failing.
        weights_col: The name of the column in DataFrame that contains the weights for each subject.
        robust: Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle ties, so if there are high number of ties, results may significantly differ.
        initial_point: set the starting point for the iterative solver.
        entry_col: Column denoting when a subject entered the study, i.e. left-truncation.
        formula: Use an R-style formula for modeling the dataset. See formula syntax: https://matthewwardrop.github.io/formulaic/basic/grammar/
            If a formula is not provided, all variables in the dataframe are used (minus those used for other purposes like event_col, etc.)
        fit_options: Additional keyword arguments to pass into the estimator.
        layer: The layer to use.


    Returns:
        Fitted WeibullAFTFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> edata = edata[:, ["mort_day_censored", "censor_flg"]]
        >>> aft = ep.tl.weibull_aft(edata, duration_col="mort_day_censored", event_col="censor_flg")
        >>> aft.print_summary()
    """
    df = _build_model_input_dataframe(edata, duration_col, accept_zero_duration=False, layer=layer)

    weibull_aft = WeibullAFTFitter(
        alpha=alpha,
        fit_intercept=fit_intercept,
        penalizer=penalizer,
        l1_ratio=l1_ratio,
        model_ancillary=model_ancillary,
    )

    weibull_aft.fit(
        df,
        duration_col=duration_col,
        event_col=event_col,
        entry_col=entry_col,
        ancillary=ancillary,
        show_progress=show_progress,
        weights_col=weights_col,
        robust=robust,
        initial_point=initial_point,
        formula=formula,
        fit_options=fit_options,
    )

    summary = weibull_aft.summary
    edata.uns[uns_key] = summary

    return weibull_aft


@function_2D_only()
@use_ehrdata(deprecated_after="1.0.0")
def log_logistic_aft(
    edata: EHRData | AnnData,
    duration_col: str,
    event_col: str | None = None,
    *,
    uns_key: str = "log_logistic_aft",
    alpha: float = 0.05,
    fit_intercept: bool = True,
    penalizer: float | np.ndarray = 0.0,
    l1_ratio: float = 0.0,
    model_ancillary: bool = False,
    ancillary: bool | pd.DataFrame | str | None = None,
    show_progress: bool = False,
    weights_col: str | None = None,
    robust: bool = False,
    initial_point=None,
    entry_col: str | None = None,
    formula: str | None = None,
    fit_options: dict | None = None,
    layer: str | None = None,
) -> LogLogisticAFTFitter:
    """Fit the log logistic accelerated failure time regression for the survival function.

    The Log-Logistic Accelerated Failure Time (AFT) survival regression model is employed in the analysis of time-to-event data.
    This model operates under the assumption that the logarithm of survival time adheres to a log-logistic distribution.
    By modeling survival time as a function of predictors, the Log-Logistic AFT model enables to explore
    how specific factors influence the acceleration or deceleration of failure times.

    See https://lifelines.readthedocs.io/en/latest/fitters/regression/LogLogisticAFTFitter.html.

    Args:
        edata: Central data object.
        duration_col: Name of the column in the data objects that contains the subjects' lifetimes.
        event_col: The name of the column in the data object that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
        uns_key: The key to use for the `.uns` slot in the data object.
        alpha: The alpha value in the confidence intervals.
        fit_intercept: Whether to fit an intercept term in the model.
        penalizer: Attach a penalty to the size of the coefficients during regression. This improves stability of the estimates and controls for high correlation between covariates.
        l1_ratio: Specify what ratio to assign to a L1 vs L2 penalty. Same as scikit-learn. See penalizer above.
        model_ancillary: Set the model instance to always model the ancillary parameter with the supplied Dataframe. This is useful for grid-search optimization.
        ancillary: Choose to model the ancillary parameters.
            If None or False, explicitly do not fit the ancillary parameters using any covariates.
            If True, model the ancillary parameters with the same covariates as ``df``.
            If DataFrame, provide covariates to model the ancillary parameters. Must be the same row count as ``df``.
            If str, should be a formula
        show_progress: Since the fitter is iterative, show convergence diagnostics. Useful if convergence is failing.
        weights_col: The name of the column in DataFrame that contains the weights for each subject.
        robust: Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle ties, so if there are high number of ties, results may significantly differ.
        initial_point: set the starting point for the iterative solver.
        entry_col: Column denoting when a subject entered the study, i.e. left-truncation.
        formula: Use an R-style formula for modeling the dataset. See formula syntax: https://matthewwardrop.github.io/formulaic/basic/grammar/
            If a formula is not provided, all variables in the dataframe are used (minus those used for other purposes like event_col, etc.)
        fit_options: Additional keyword arguments to pass into the estimator.
        layer: The layer to use.

    Returns:
        Fitted LogLogisticAFTFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> edata = edata[:, ["mort_day_censored", "censor_flg"]]
        >>> llf = ep.tl.log_logistic_aft(edata, duration_col="mort_day_censored", event_col="censor_flg")
    """
    df = _build_model_input_dataframe(edata, duration_col, accept_zero_duration=False, layer=layer)

    log_logistic_aft = LogLogisticAFTFitter(
        alpha=alpha,
        fit_intercept=fit_intercept,
        penalizer=penalizer,
        l1_ratio=l1_ratio,
        model_ancillary=model_ancillary,
    )

    log_logistic_aft.fit(
        df,
        duration_col=duration_col,
        event_col=event_col,
        entry_col=entry_col,
        ancillary=ancillary,
        show_progress=show_progress,
        weights_col=weights_col,
        robust=robust,
        initial_point=initial_point,
        formula=formula,
        fit_options=fit_options,
    )

    summary = log_logistic_aft.summary
    edata.uns[uns_key] = summary

    return log_logistic_aft


def _univariate_model(
    edata: EHRData | AnnData,
    duration_col: str,
    event_col: str,
    model_class,
    uns_key: str,
    accept_zero_duration=True,
    timeline: list[float] | None = None,
    entry: str | None = None,
    label: str | None = None,
    alpha: float | None = None,
    ci_labels: list[str] | None = None,
    weights: list[float] | None = None,
    fit_options: dict | None = None,
    censoring: Literal["right", "left"] = "right",
    layer: str | None = None,
):
    """Convenience function for univariate models."""
    df = _build_model_input_dataframe(edata, duration_col, accept_zero_duration, layer)
    T = df[duration_col]
    E = df[event_col]

    model = model_class()
    function_name = "fit" if censoring == "right" else "fit_left_censoring"
    # get fit function, default to fit if not found
    fit_function = getattr(model, function_name, model.fit)

    fit_function(
        T,
        event_observed=E,
        timeline=timeline,
        entry=entry,
        label=label,
        alpha=alpha,
        ci_labels=ci_labels,
        weights=weights,
        fit_options=fit_options,
    )

    if isinstance(model, NelsonAalenFitter) or isinstance(
        model, KaplanMeierFitter
    ):  # NelsonAalenFitter and KaplanMeierFitter have no summary attribute
        summary = model.event_table
    else:
        summary = model.summary
    edata.uns[uns_key] = summary

    return model


@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
def nelson_aalen(
    edata: EHRData | AnnData,
    duration_col: str,
    event_col: str | None = None,
    *,
    uns_key: str = "nelson_aalen",
    timeline: list[float] | None = None,
    entry: str | None = None,
    label: str | None = None,
    alpha: float | None = None,
    ci_labels: list[str] | None = None,
    weights: list[float] | None = None,
    fit_options: dict | None = None,
    censoring: Literal["right", "left"] = "right",
    layer: str | None = None,
) -> NelsonAalenFitter:
    """Employ the Nelson-Aalen estimator to estimate the cumulative hazard function from censored survival data.

    The Nelson-Aalen estimator is a non-parametric method used in survival analysis to estimate the cumulative hazard function.
    It accounts for the presence of individuals whose event times are unknown due to censoring.
    By estimating the cumulative hazard function, the Nelson-Aalen estimator assessing the risk of an event occurring over time.
    The results will be stored in the `.uns` slot of the data object under the key 'nelson_aalen' unless specified otherwise in the `uns_key` parameter.
    See https://lifelines.readthedocs.io/en/latest/fitters/univariate/NelsonAalenFitter.html

    Args:
        edata: Central data object.
        duration_col: The name of the column in the data objects that contains the subjects' lifetimes.
        event_col: The name of the column in the data object that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
        uns_key: The key to use for the `.uns` slot in the data object.
        timeline: Return the best estimate at the values in timelines (positively increasing)
        entry: Relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations.
               If None, all members of the population entered study when they were "born".
        label: A string to name the column of the estimate.
        alpha: The alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
        ci_labels: Add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>] (default: <label>_lower_<1-alpha/2>).
        weights: If providing a weighted dataset. For example, instead of providing every subject
                 as a single element of `durations` and `event_observed`, one could weigh subject differently.
        fit_options: Additional keyword arguments to pass into the estimator.
        censoring: 'right' for fitting the model to a right-censored dataset. (default, calls fit).
                   'left' for fitting the model to a left-censored dataset (calls fit_left_censoring).
        layer: The layer to use.

    Returns:
        Fitted NelsonAalenFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> naf = ep.tl.nelson_aalen(edata, "mort_day_censored", "censor_flg")
    """
    return _univariate_model(
        edata,
        duration_col,
        event_col,
        NelsonAalenFitter,
        uns_key=uns_key,
        accept_zero_duration=True,
        timeline=timeline,
        entry=entry,
        label=label,
        alpha=alpha,
        ci_labels=ci_labels,
        weights=weights,
        fit_options=fit_options,
        censoring=censoring,
        layer=layer,
    )


@function_2D_only()
@use_ehrdata(deprecated_after="1.0.0")
def weibull(
    edata: EHRData | AnnData,
    duration_col: str,
    event_col: str,
    *,
    uns_key: str = "weibull",
    timeline: list[float] | None = None,
    entry: str | None = None,
    label: str | None = None,
    alpha: float | None = None,
    ci_labels: list[str] | None = None,
    weights: list[float] | None = None,
    fit_options: dict | None = None,
    layer: str | None = None,
) -> WeibullFitter:
    """Employ the Weibull model in univariate survival analysis to understand event occurrence dynamics.

    In contrast to the non-parametric Nelson-Aalen estimator, the Weibull model employs a parametric approach with shape and scale parameters,
    enabling a more structured analysis of survival data.
    This technique is particularly useful when dealing with censored data, as it accounts for the presence of individuals whose event times are unknown due to censoring.
    By fitting the Weibull model to censored survival data, researchers can estimate these parameters and gain insights
    into the hazard rate over time, facilitating comparisons between different groups or treatments.
    This method provides a comprehensive framework for examining survival data and offers valuable insights into the factors influencing event occurrence dynamics.
    The results will be stored in the `.uns` slot of the data object under the key 'weibull' unless specified otherwise in the `uns_key` parameter.
    See https://lifelines.readthedocs.io/en/latest/fitters/univariate/WeibullFitter.html

    Args:
        edata: Central data object.
        duration_col: Name of the column in the data objects that contains the subjects’ lifetimes.
        event_col: The name of the column in the data object that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
        uns_key: The key to use for the `.uns` slot in the data object.
        timeline: Return the best estimate at the values in timelines (positively increasing)
        entry: Relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations.
               If None, all members of the population entered study when they were "born".
        label: A string to name the column of the estimate.
        alpha: The alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
        ci_labels: Add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>] (default: <label>_lower_<1-alpha/2>).
        weights: If providing a weighted dataset. For example, instead of providing every subject
                 as a single element of `durations` and `event_observed`, one could weigh subject differently.
        fit_options: Additional keyword arguments to pass into the estimator.
        layer: The layer to use.

    Returns:
        Fitted WeibullFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> wf = ep.tl.weibull(edata, "mort_day_censored", "censor_flg")
    """
    return _univariate_model(
        edata,
        duration_col,
        event_col,
        WeibullFitter,
        uns_key=uns_key,
        accept_zero_duration=False,
        timeline=timeline,
        entry=entry,
        label=label,
        alpha=alpha,
        ci_labels=ci_labels,
        weights=weights,
        fit_options=fit_options,
        layer=layer,
    )
