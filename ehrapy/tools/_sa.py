from __future__ import annotations

from typing import Literal

import numpy as np  # noqa: F401 # This package is implicitly used
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from anndata import AnnData
from lifelines import KaplanMeierFitter


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

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encoded=False)
            formula = 'tco2_first ~ pco2_first'
            var_names = ['tco2_first', 'pco2_first']
            ols = ep.tl.ols(adata, var_names, formula, missing = 'drop')
    """
    if isinstance(var_names, list):
        data = pd.DataFrame(adata[:, var_names].X, columns=var_names).astype(float)
    else:
        data = pd.DataFrame(adata.X, columns=adata.var_names)
    ols = smf.ols(formula, data=data, missing=missing)
    return ols


def glm(
    adata: AnnData,
    var_names: list[str] | None = None,
    formula: str | None = None,
    family: Literal["Gaussian", "Binomial", "Gamma", "Gaussian", "InverseGaussian"] = "Gaussian",
    missing: Literal["none", "drop", "raise"] = "none",
    ascontinus: list[str] | None | None = None,
) -> sm.GLM:
    """Create a Generalized Linear Model (GLM) from a formula, a distribution, and AnnData.

    See https://www.statsmodels.org/stable/generated/statsmodels.formula.api.glm.html#statsmodels.formula.api.glm
    Internally use the statsmodel to create a GLM Model from a formula, a distribution, and dataframe.

    Args:
        adata: The AnnData object for the GLM model.
        var_names: A list of var names indicating which columns are for the GLM model.
        formula: The formula specifying the model.
        family: The distribution families. Available options are 'Gaussian', 'Binomial', 'Gamma', and 'InverseGaussian'. Default is 'Gaussian'.
        missing: Available options are 'none', 'drop', and 'raise'. If 'none', no nan checking is done. If 'drop', any observations with nans are dropped. If 'raise', an error is raised. Default is 'none'.
        ascontinus: A list of var names indicating which columns are continus rather than categorical. The corresponding columns will be set as type float.

    Returns:
        The GLM model instance.

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encoded=False)
            formula = 'day_28_flg ~ age'
            var_names = ['day_28_flg', 'age']
            family = 'Binomial'
            glm = ep.tl.glmglm(adata, var_names, formula, family, missing = 'drop', ascontinus = ['age'])
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
    if ascontinus is not None:
        data[ascontinus] = data[ascontinus].astype(float)
    glm = smf.glm(formula, data=data, family=family, missing=missing)
    return glm


def kmf(
    durations,
    event_observed=None,
    timeline=None,
    entry=None,
    label=None,
    alpha=None,
    ci_labels=None,
    weights=None,
    censoring=None,
) -> sm.GLM:
    """Fit the Kaplan-Meier estimate for the survival function.

    See https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html#module-lifelines.fitters.kaplan_meier_fitter
    Class for fitting the Kaplan-Meier estimate for the survival function.

    Args:
        durations: an array, list, pd.DataFrame or pd.Series
            length n -- duration (relative to subject's birth) the subject was alive for.
        event_observed: an array, list, pd.DataFrame, or pd.Series, optional
            True if the the death was observed, False if the event was lost (right-censored). Defaults all True if event_observed==None
        timeline: an array, list, pd.DataFrame, or pd.Series, optional
            return the best estimate at the values in timelines (positively increasing)
        entry: an array, list, pd.DataFrame, or pd.Series, optional
            relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations. If None, all members of the population
            entered study when they were "born".
        label: string, optional
            a string to name the column of the estimate.
        alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
        ci_labels: tuple, optional
            add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<1-alpha/2>
        weights: an array, list, pd.DataFrame, or pd.Series, optional
            if providing a weighted dataset. For example, instead
            of providing every subject as a single element of `durations` and `event_observed`, one could
            weigh subject differently.
        censoring: string, optional. One of ('right', 'left)
            'right' for fitting the model to a right-censored dataset. 'left' for fitting the model to a left-censored dataset.
            Default is to fit the model to a right-censored dataset.

    Returns:
        Fitted KaplanMeierFitter

    Example:
        .. code-block:: python

            import ehrapy as ep
            adata = ep.dt.mimic_2(encoded=False)
            # Because in MIMIC-II database, `censor_fl` is censored or death (binary: 0 = death, 1 = censored).
            # While in KaplanMeierFitter, `event_observed` is True if the the death was observed, False if the event was lost (right-censored).
            # So we need to flip `censor_fl` when pass `censor_fl` to KaplanMeierFitter
            adata[:, ['censor_flg']].X = np.where(adata[:, ['censor_flg']].X == 0, 1, 0)
            kmf = ep.tl.kmf(adata[:, ['mort_day_censored']].X, adata[:, ['censor_flg']].X)
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
