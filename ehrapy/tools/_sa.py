from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np  # This package is implicitly used
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
from lifelines.statistics import StatisticalResult, logrank_test
from scipy import stats

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
    Internally use the statsmodel to create a OLS Model from a formula and dataframe.

    Args:
        adata: The AnnData object for the OLS model.
        var_names: A list of var names indicating which columns are for the OLS model.
        formula: The formula specifying the model.
        missing: Available options are 'none', 'drop', and 'raise'. If 'none', no nan checking is done. If 'drop', any observations with nans are dropped. If 'raise', an error is raised. Default is 'none'.

    Returns:
        The OLS model instance.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> formula = 'tco2_first ~ pco2_first'
        >>> var_names = ['tco2_first', 'pco2_first']
        >>> ols = ep.tl.ols(adata, var_names, formula, missing = 'drop')
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
    Internally use the statsmodel to create a GLM Model from a formula, a distribution, and dataframe.

    Args:
        adata: The AnnData object for the GLM model.
        var_names: A list of var names indicating which columns are for the GLM model.
        formula: The formula specifying the model.
        family: The distribution families. Available options are 'Gaussian', 'Binomial', 'Gamma', and 'InverseGaussian'.
                Defaults to 'Gaussian'.
        missing: Available options are 'none', 'drop', and 'raise'. If 'none', no nan checking is done.
                 If 'drop', any observations with nans are dropped. If 'raise', an error is raised (default: 'none').
        ascontinus: A list of var names indicating which columns are continuous rather than categorical.
                    The corresponding columns will be set as type float.

    Returns:
        The GLM model instance.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> formula = 'day_28_flg ~ age'
        >>> var_names = ['day_28_flg', 'age']
        >>> family = 'Binomial'
        >>> glm = ep.tl.glmglm(adata, var_names, formula, family, missing = 'drop', ascontinus = ['age'])
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

    See https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html#module-lifelines.fitters.kaplan_meier_fitter
    Class for fitting the Kaplan-Meier estimate for the survival function.

    Args:
        durations: length n -- duration (relative to subject's birth) the subject was alive for.
        event_observed: True if the the death was observed, False if the event was lost (right-censored). Defaults to all True if event_observed==None.
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
        >>> # Because in MIMIC-II database, `censor_fl` is censored or death (binary: 0 = death, 1 = censored).
        >>> # While in KaplanMeierFitter, `event_observed` is True if the the death was observed, False if the event was lost (right-censored).
        >>> # So we need to flip `censor_fl` when pass `censor_fl` to KaplanMeierFitter
        >>> adata[:, ['censor_flg']].X = np.where(adata[:, ['censor_flg']].X == 0, 1, 0)
        >>> kmf = ep.tl.kmf(adata[:, ['mort_day_censored']].X, adata[:, ['censor_flg']].X)
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

    See https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html

    Measures and reports on whether two intensity processes are different.
    That is, given two event series, determines whether the data generating processes are statistically different.
    The test-statistic is chi-squared under the null hypothesis.

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
    """Given two fitted GLMs, the larger of which contains the parameter space of the smaller, return the P value corresponding to the larger model adding explanatory power.

    See https://stackoverflow.com/questions/27328623/anova-test-for-glm-in-python/60769343#60769343

    Args:
        small_model (GLMResultsWrapper): fitted generalized linear models.
        big_model (GLMResultsWrapper): fitted generalized linear models.

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
        result_1 (GLMResultsWrapper): fitted generalized linear models.
        result_2 (GLMResultsWrapper): fitted generalized linear models.
        formula_1 (str): The formula specifying the model.
        formula_2 (str): The formula specifying the model.

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
