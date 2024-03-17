from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np  # This package is implicitly used
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import (
    CoxPHFitter,
    KaplanMeierFitter,
    LogLogisticAFTFitter,
    NelsonAalenFitter,
    WeibullAFTFitter,
    WeibullFitter,
)
from lifelines.statistics import StatisticalResult, logrank_test
from scipy import stats

from ehrapy.anndata import anndata_to_df

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper


def ols(
    adata: AnnData,
    var_names: list[str] | None | None = None,
    formula: str | None = None,
    missing: Literal["none", "drop", "raise"] | None = "none",
) -> sm.OLS:
    """Create a Ordinary Least Squares (OLS) Model from a formula and AnnData.

    See https://www.statsmodels.org/stable/generated/statsmodels.formula.api.ols.html#statsmodels.formula.api.ols

    Args:
        adata: The AnnData object for the OLS model.
        var_names: A list of var names indicating which columns are for the OLS model.
        formula: The formula specifying the model.
        missing: Available options are 'none', 'drop', and 'raise'.
                 If 'none', no nan checking is done. If 'drop', any observations with nans are dropped.
                 If 'raise', an error is raised. Defaults to 'none'.

    Returns:
        The OLS model instance.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> formula = "tco2_first ~ pco2_first"
        >>> var_names = ["tco2_first", "pco2_first"]
        >>> ols = ep.tl.ols(adata, var_names, formula, missing="drop")
    """
    if isinstance(var_names, list):
        data = pd.DataFrame(adata[:, var_names].X, columns=var_names).astype(float)
    else:
        data = pd.DataFrame(adata.X, columns=adata.var_names)
    ols = smf.ols(formula, data=data, missing=missing)

    return ols


def glm(
    adata: AnnData,
    var_names: Iterable[str] | None = None,
    formula: str | None = None,
    family: Literal["Gaussian", "Binomial", "Gamma", "Gaussian", "InverseGaussian"] = "Gaussian",
    missing: Literal["none", "drop", "raise"] = "none",
    as_continuous: Iterable[str] | None | None = None,
) -> sm.GLM:
    """Create a Generalized Linear Model (GLM) from a formula, a distribution, and AnnData.

    See https://www.statsmodels.org/stable/generated/statsmodels.formula.api.glm.html#statsmodels.formula.api.glm

    Args:
        adata: The AnnData object for the GLM model.
        var_names: A list of var names indicating which columns are for the GLM model.
        formula: The formula specifying the model.
        family: The distribution families. Available options are 'Gaussian', 'Binomial', 'Gamma', and 'InverseGaussian'.
                Defaults to 'Gaussian'.
        missing: Available options are 'none', 'drop', and 'raise'. If 'none', no nan checking is done.
                 If 'drop', any observations with nans are dropped. If 'raise', an error is raised (default: 'none').
        as_continuous: A list of var names indicating which columns are continuous rather than categorical.
                    The corresponding columns will be set as type float.

    Returns:
        The GLM model instance.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> formula = "day_28_flg ~ age"
        >>> var_names = ["day_28_flg", "age"]
        >>> family = "Binomial"
        >>> glm = ep.tl.glm(adata, var_names, formula, family, missing="drop", ascontinus=["age"])
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
        data = pd.DataFrame(adata[:, var_names].X, columns=var_names)
    else:
        data = pd.DataFrame(adata.X, columns=adata.var_names)
    if as_continuous is not None:
        data[as_continuous] = data[as_continuous].astype(float)
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
    """Fit the Kaplan-Meier estimate for the survival function.

    The Kaplan–Meier estimator, also known as the product limit estimator, is a non-parametric statistic used to estimate the survival function from lifetime data.
    In medical research, it is often used to measure the fraction of patients living for a certain amount of time after treatment.

    See https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator
        https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html#module-lifelines.fitters.kaplan_meier_fitter

    Args:
        durations: length n -- duration (relative to subject's birth) the subject was alive for.
        event_observed: True if the death was observed, False if the event was lost (right-censored). Defaults to all True if event_observed==None.
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
        Fitted KaplanMeierFitter

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> adata[:, ["censor_flg"]].X = np.where(adata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> kmf = ep.tl.kmf(adata[:, ["mort_day_censored"]].X, adata[:, ["censor_flg"]].X)
    """
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
             Specify -1 to use all time. Defaults to -1.
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
    """ "Calculate the P value indicating if a larger GLM, encompassing a smaller GLM's parameters, adds explanatory power."

    See https://stackoverflow.com/questions/27328623/anova-test-for-glm-in-python/60769343#60769343

    Args:
        small_model: fitted generalized linear models.
        big_model: fitted generalized linear models.

    Returns:
        float: p_value
    """
    addtl_params = big_model.df_model - small_model.df_model
    f_stat = (small_model.deviance - big_model.deviance) / (addtl_params * big_model.scale)
    df_numerator = addtl_params
    df_denom = big_model.fittedvalues.shape[0] - big_model.df_model
    p_value = stats.f.sf(f_stat, df_numerator, df_denom)

    return p_value


def anova_glm(result_1: GLMResultsWrapper, result_2: GLMResultsWrapper, formula_1: str, formula_2: str) -> pd.DataFrame:
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


def _regression_model(
    model_class, adata: AnnData, duration_col: str, event_col: str, entry_col: str = None, accept_zero_duration=True
):
    """Convenience function for regression models."""
    df = anndata_to_df(adata)
    df = df.dropna()

    if not accept_zero_duration:
        df[duration_col][df[duration_col] == 0] += 1e-5

    model = model_class()
    model.fit(df, duration_col, event_col, entry_col=entry_col)

    return model


def cox_ph(adata: AnnData, duration_col: str, event_col: str, entry_col: str = None) -> CoxPHFitter:
    """Fit the Cox’s proportional hazard for the survival function.

    The Cox proportional hazards model (CoxPH) examines the relationship between the survival time of subjects and one or more predictor variables.
    It models the hazard rate as a product of a baseline hazard function and an exponential function of the predictors, assuming proportional hazards over time.

    See https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html

    Args:
        adata: adata: AnnData object with necessary columns `duration_col` and `event_col`.
        duration_col: The name of the column in the AnnData objects that contains the subjects’ lifetimes.
        event_col: The name of the column in anndata that contains the subjects’ death observation.
                   If left as None, assume all individuals are uncensored.
        entry_col: Column denoting when a subject entered the study, i.e. left-truncation.

    Returns:
        Fitted CoxPHFitter.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> adata[:, ["censor_flg"]].X = np.where(adata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> cph = ep.tl.cox_ph(adata, "mort_day_censored", "censor_flg")
    """
    return _regression_model(CoxPHFitter, adata, duration_col, event_col, entry_col)


def weibull_aft(adata: AnnData, duration_col: str, event_col: str, entry_col: str = None) -> WeibullAFTFitter:
    """Fit the Weibull accelerated failure time regression for the survival function.

    The Weibull Accelerated Failure Time (AFT) survival regression model is a statistical method used to analyze time-to-event data,
    where the underlying assumption is that the logarithm of survival time follows a Weibull distribution.
    It models the survival time as an exponential function of the predictors, assuming a specific shape parameter
    for the distribution and allowing for accelerated or decelerated failure times based on the covariates.
    See https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html

    Args:
        adata: adata: AnnData object with necessary columns `duration_col` and `event_col`.
        duration_col: Name of the column in the AnnData objects that contains the subjects’ lifetimes.
        event_col: Name of the column in anndata that contains the subjects’ death observation.
                   If left as None, assume all individuals are uncensored.
        entry_col: Column denoting when a subject entered the study, i.e. left-truncation.

    Returns:
        Fitted WeibullAFTFitter

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> adata[:, ["censor_flg"]].X = np.where(adata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> aft = ep.tl.weibull_aft(adata, "mort_day_censored", "censor_flg")
    """
    return _regression_model(WeibullAFTFitter, adata, duration_col, event_col, entry_col, accept_zero_duration=False)


def log_rogistic_aft(adata: AnnData, duration_col: str, event_col: str, entry_col: str = None) -> LogLogisticAFTFitter:
    """Fit the log logistic accelerated failure time regression for the survival function.
    The Log-Logistic Accelerated Failure Time (AFT) survival regression model is a powerful statistical tool employed in the analysis of time-to-event data.
    This model operates under the assumption that the logarithm of survival time adheres to a log-logistic distribution, offering a flexible framework for understanding the impact of covariates on survival times.
    By modeling survival time as a function of predictors, the Log-Logistic AFT model enables researchers to explore
    how specific factors influence the acceleration or deceleration of failure times, providing valuable insights into the underlying mechanisms driving event occurrence.
    See https://lifelines.readthedocs.io/en/latest/fitters/regression/LogLogisticAFTFitter.html

    Args:
        adata: adata: AnnData object with necessary columns `duration_col` and `event_col`.
        duration_col: Name of the column in the AnnData objects that contains the subjects’ lifetimes.
        event_col: Name of the column in anndata that contains the subjects’ death observation.
                   If left as None, assume all individuals are uncensored.
        entry_col: Column denoting when a subject entered the study, i.e. left-truncation.

    Returns:
        Fitted LogLogisticAFTFitter

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> adata[:, ["censor_flg"]].X = np.where(adata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> llf = ep.tl.log_rogistic_aft(adata, "mort_day_censored", "censor_flg")
    """
    return _regression_model(
        LogLogisticAFTFitter, adata, duration_col, event_col, entry_col, accept_zero_duration=False
    )


def _univariate_model(adata: AnnData, duration_col: str, event_col: str, model_class, accept_zero_duration=True):
    """Convenience function for univariate models."""
    df = anndata_to_df(adata)

    if not accept_zero_duration:
        df[duration_col][df[duration_col] == 0] += 1e-5
    T = df[duration_col]
    E = df[event_col]

    model = model_class()
    model.fit(T, event_observed=E)

    return model


def nelson_alen(adata: AnnData, duration_col: str, event_col: str) -> NelsonAalenFitter:
    """Employ the Nelson-Aalen estimator to estimate the cumulative hazard function from censored survival data

    The Nelson-Aalen estimator is a non-parametric method used in survival analysis to estimate the cumulative hazard function.
    This technique is particularly useful when dealing with censored data, as it accounts for the presence of individuals whose event times are unknown due to censoring.
    By estimating the cumulative hazard function, the Nelson-Aalen estimator allows researchers to assess the risk of an event occurring over time, providing valuable insights into the underlying dynamics of the survival process.
    See https://lifelines.readthedocs.io/en/latest/fitters/univariate/NelsonAalenFitter.html

    Args:
        adata: adata: AnnData object with necessary columns `duration_col` and `event_col`.
        duration_col: The name of the column in the AnnData objects that contains the subjects’ lifetimes.
        event_col: The name of the column in anndata that contains the subjects’ death observation.
                   If left as None, assume all individuals are uncensored.

    Returns:
        Fitted NelsonAalenFitter

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> adata[:, ["censor_flg"]].X = np.where(adata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> naf = ep.tl.nelson_alen(adata, "mort_day_censored", "censor_flg")
    """
    return _univariate_model(adata, duration_col, event_col, NelsonAalenFitter)


def weibull(adata: AnnData, duration_col: str, event_col: str) -> WeibullFitter:
    """Employ the Weibull model in univariate survival analysis to understand event occurrence dynamics.

    In contrast to the non-parametric Nelson-Aalen estimator, the Weibull model employs a parametric approach with shape and scale parameters,
    enabling a more structured analysis of survival data.
    This technique is particularly useful when dealing with censored data, as it accounts for the presence of individuals whose event times are unknown due to censoring.
    By fitting the Weibull model to censored survival data, researchers can estimate these parameters and gain insights
    into the hazard rate over time, facilitating comparisons between different groups or treatments.
    This method provides a comprehensive framework for examining survival data and offers valuable insights into the factors influencing event occurrence dynamics.
    See https://lifelines.readthedocs.io/en/latest/fitters/univariate/WeibullFitter.html

    Args:
        adata: adata: AnnData object with necessary columns `duration_col` and `event_col`.
        duration_col: Name of the column in the AnnData objects that contains the subjects’ lifetimes.
        event_col: Name of the column in the AnnData object that contains the subjects’ death observation.
                   If left as None, assume all individuals are uncensored.

    Returns:
        Fitted WeibullFitter

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> adata[:, ["censor_flg"]].X = np.where(adata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> wf = ep.tl.weibull(adata, "mort_day_censored", "censor_flg")
    """
    return _univariate_model(adata, duration_col, event_col, WeibullFitter, accept_zero_duration=False)
