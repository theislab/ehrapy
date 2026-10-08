from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ehrdata._logger import logger
from ehrdata.core.constants import CATEGORICAL_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from formulaic import Formula
from lifelines import (
    AalenJohansenFitter,
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

from ehrapy._compat import _materialize, _tem_times
from ehrapy.get import obs_df

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ehrdata import EHRData

    from ehrapy._compat import Array


def _model_frame(edata: EHRData, keys: Iterable[str | None], *, layer: str | None, dropna: bool = True) -> pd.DataFrame:
    """Values of the obs columns and variables a model uses, without the observations that miss any of them."""
    keys = list(dict.fromkeys(key for key in keys if key is not None))
    frame = obs_df(edata, keys=keys, layer=layer).infer_objects()
    return _drop_incomplete(frame) if dropna else frame


def _drop_incomplete(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame.notna().all(axis=1)
    if n_dropped := int((~complete).sum()):
        logger.info(f"Dropped {n_dropped} of {len(frame)} observations with missing values in {list(frame.columns)}.")
    return frame[complete]


def _survival_frame(
    edata: EHRData, duration_col: str | None, event_col: str | None, keys: Iterable[str | None], *, layer: str | None
) -> tuple[pd.DataFrame, str]:
    """Model frame of a survival model and its duration column, which is derived along with the event if `event_col` is a variable of 3D data."""
    X = edata.X if layer is None else edata.layers[layer]
    if event_col is None or getattr(X, "ndim", 2) != 3 or event_col not in edata.var_names:
        if duration_col is None:
            raise ValueError("Pass a `duration_col`, or an `event_col` that is a variable of 3D data.")
        return _model_frame(edata, [duration_col, event_col, *keys], layer=layer), duration_col
    if duration_col is not None:
        raise ValueError(f"The durations are derived from the 3D variable {event_col!r}, pass no `duration_col`.")
    duration_col = f"{event_col}_duration"
    events = pd.DataFrame(
        _time_to_event(X[:, edata.var_names.get_loc(event_col)], _tem_times(edata, "interval_start_offset"), event_col),
        index=edata.obs_names,
        columns=[duration_col, event_col],
    )
    covariates = _model_frame(edata, [key for key in keys if key != event_col], layer=layer, dropna=False)
    return _drop_incomplete(pd.concat([events, covariates], axis=1)), duration_col


def _time_to_event(values: Array, times: np.ndarray, name: str) -> np.ndarray:
    """Time of the first 1 of every row and 1, or the time of its last non-missing value and 0 if it has no 1."""
    (values,) = _materialize(values)
    observed = ~np.isnan(values)
    if not np.isin(values[observed], [0, 1]).all():
        raise ValueError(f"The longitudinal event {name!r} must be 1 at the timepoints with the event and 0 otherwise.")
    occurred = values == 1
    happened = occurred.any(axis=1)
    last = values.shape[1] - 1 - observed[:, ::-1].argmax(axis=1)
    durations = np.where(happened, times[occurred.argmax(axis=1)], times[last])
    return np.where(observed.any(axis=1)[:, None], np.column_stack([durations, happened]), np.nan)


def _formula_variables(formula: str | None) -> list[str]:
    return [] if formula is None else sorted(Formula(formula).required_variables)


def _covariates(edata: EHRData, covariates: Sequence[str] | None, formula: str | None) -> list[str]:
    if covariates is not None:
        return list(covariates)
    if formula is not None:
        return _formula_variables(formula)
    return list(edata.var_names)


def _cast_variables(edata: EHRData, data: pd.DataFrame, use_feature_types: bool) -> pd.DataFrame:
    """Cast variable columns to float, or to their feature type if `use_feature_types`; obs columns keep their dtype."""
    for col in data.columns.intersection(edata.var_names):
        feature_type = edata.var[FEATURE_TYPE_KEY][col] if use_feature_types else NUMERIC_TAG
        if feature_type == CATEGORICAL_TAG:
            data[col] = data[col].astype("category")
        elif feature_type == NUMERIC_TAG:
            data[col] = data[col].astype(float)
    return data


def _shift_zero_durations(data: pd.DataFrame, duration_col: str) -> pd.DataFrame:
    data.loc[data[duration_col] == 0, duration_col] += 1e-5
    return data


def ols(
    edata: EHRData,
    *,
    var_names: Sequence[str] | None = None,
    formula: str | None = None,
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
        layer: The layer to take variables from.

    Returns:
        The OLS model instance.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> formula = "tco2_first ~ pco2_first"
        >>> var_names = ["tco2_first", "pco2_first"]
        >>> ols = ep.tl.ols(edata, var_names=var_names, formula=formula, missing="drop")
    """
    keys = var_names if var_names is not None else _formula_variables(formula)
    data = _cast_variables(edata, _model_frame(edata, keys, layer=layer, dropna=False), use_feature_types)

    ols = smf.ols(formula, data=data, missing=missing)

    return ols


def glm(
    edata: EHRData,
    *,
    var_names: Sequence[str] | None = None,
    formula: str | None = None,
    family: Literal["Gaussian", "Binomial", "Gamma", "InverseGaussian"] = "Gaussian",
    use_feature_types: bool = False,
    missing: Literal["none", "drop", "raise"] = "none",
    as_continuous: Sequence[str] | None = None,
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
        layer: The layer to take variables from.

    Returns:
        The GLM model instance.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> formula = "day_28_flg ~ age"
        >>> var_names = ["day_28_flg", "age"]
        >>> family = "Binomial"
        >>> glm = ep.tl.glm(
        ...     edata, var_names=var_names, formula=formula, family=family, missing="drop", as_continuous=["age"]
        ... )
    """
    family_dict = {
        "Gaussian": sm.families.Gaussian(),
        "Binomial": sm.families.Binomial(),
        "Gamma": sm.families.Gamma(),
        "InverseGaussian": sm.families.InverseGaussian(),
    }
    keys = var_names if var_names is not None else _formula_variables(formula)
    data = _model_frame(edata, keys, layer=layer, dropna=False)
    if use_feature_types:
        data = _cast_variables(edata, data, use_feature_types=True)
    if as_continuous is not None:
        data[list(as_continuous)] = data[list(as_continuous)].astype(float)

    glm = smf.glm(formula, data=data, family=family_dict[family], missing=missing)

    return glm


def kaplan_meier(
    edata: EHRData,
    duration_col: str | None = None,
    *,
    event_col: str | None = None,
    key_added: str = "kaplan_meier",
    timeline: Sequence[float] | None = None,
    entry_col: str | None = None,
    label: str | None = None,
    alpha: float | None = None,
    ci_labels: Sequence[str] | None = None,
    weights_col: str | None = None,
    fit_options: Mapping[str, Any] | None = None,
    censoring: Literal["right", "left"] = "right",
    layer: str | None = None,
    event_of_interest: int | None = None,
    random_state: int = 0,
) -> KaplanMeierFitter | AalenJohansenFitter:
    """Fit the Kaplan-Meier estimate for the survival function.

    The Kaplan–Meier estimator, also known as the product limit estimator, is a non-parametric statistic used to estimate the survival function from lifetime data.
    In medical research, it is often used to measure the fraction of patients living for a certain amount of time after treatment.
    With competing events, such as death before the event of interest, one minus the Kaplan-Meier estimate overestimates the probability of the event.
    Passing `event_of_interest` instead estimates its cumulative incidence with the Aalen-Johansen estimator, which accounts for the competing events.
    The results will be stored in the `.uns` slot of the data object under the key 'kaplan_meier' unless specified otherwise in the `key_added` parameter.

    See `Kaplan Meier on Wikipedia <https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator>`_ and `Kaplan Meier on Lifelines <https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html#module-lifelines.fitters.kaplan_meier_fitter>`_.

    Args:
        edata: Central data object.
        duration_col: Column in `edata.obs` or variable with the subjects' lifetimes.
            `None` if `event_col` is a variable of 3D data, from which the durations are derived.
        event_col: Column in `edata.obs` or variable that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
            If it is a variable of 3D data that is 1 at the timepoints with the event and 0 otherwise, the event is whether it is ever 1 and the duration the time of its first 1, or of its last non-missing value for censored observations.
            Times count from the first timepoint, in `edata.tem['interval_start_offset']` if present, with time differences in seconds, or as positions otherwise.
        key_added: The key to use for the `.uns` slot in the data object.
        timeline: Return the best estimate at the values in timelines (positively increasing)
        entry_col: Column in `edata.obs` or variable with the relative time when a subject entered the study.
            This is useful for left-truncated (not left-censored) observations.
            If None, all members of the population entered study when they were "born".
        label: A string to name the column of the estimate.
        alpha: The alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
        ci_labels: Add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>] (default: <label>_lower_<1-alpha/2>).
        weights_col: Column in `edata.obs` or variable with a weight per subject.
        fit_options: Additional keyword arguments to pass into the estimator.
        censoring: 'right' for fitting the model to a right-censored dataset. (default, calls fit).
                   'left' for fitting the model to a left-censored dataset (calls fit_left_censoring).
        layer: The layer to take variables from.
        event_of_interest: The event type in `event_col` to estimate the cumulative incidence of.
            `event_col` then holds 0 for censored subjects and a positive integer per event type, and all event types other than `event_of_interest` are competing events.
            Supports only `censoring='right'`.
        random_state: Seed for breaking tied event times when `event_of_interest` is set.

    Returns:
        Fitted KaplanMeierFitter, or fitted AalenJohansenFitter if `event_of_interest` is set.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.mimic_2()
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> kmf = ep.tl.kaplan_meier(edata, duration_col="mort_day_censored", event_col="censor_flg", label="Mortality")

        Cumulative incidence of death in hospital, with death after discharge as competing event:

        >>> edata = ed.dt.mimic_2()
        >>> died, died_in_hospital = edata[:, ["censor_flg", "hosp_exp_flg"]].X.T.astype(float)
        >>> edata.obs["death"] = np.select([died_in_hospital == 1, died == 1], [1, 2], 0)
        >>> ajf = ep.tl.kaplan_meier(edata, duration_col="mort_day_censored", event_col="death", event_of_interest=1)
    """
    return _univariate_model(
        edata,
        duration_col,
        event_col,
        KaplanMeierFitter if event_of_interest is None else AalenJohansenFitter,
        key_added,
        True,
        timeline,
        entry_col,
        label,
        alpha,
        ci_labels,
        weights_col,
        fit_options,
        censoring,
        layer,
        event_of_interest,
        random_state,
    )


def test_kmf_logrank(
    kmf_A: KaplanMeierFitter,
    kmf_B: KaplanMeierFitter,
    *,
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


def cox_ph(
    edata: EHRData,
    duration_col: str | None = None,
    *,
    event_col: str | None = None,
    key_added: str = "cox_ph",
    alpha: float = 0.05,
    label: str | None = None,
    baseline_estimation_method: Literal["breslow", "spline", "piecewise"] = "breslow",
    penalizer: float | np.ndarray = 0.0,
    l1_ratio: float = 0.0,
    strata: str | Sequence[str] | None = None,
    n_baseline_knots: int = 4,
    knots: Sequence[float] | None = None,
    breakpoints: Sequence[float] | None = None,
    weights_col: str | None = None,
    cluster_col: str | None = None,
    entry_col: str | None = None,
    robust: bool = False,
    formula: str | None = None,
    covariates: Sequence[str] | None = None,
    batch_mode: bool | None = None,
    show_progress: bool = False,
    initial_point: np.ndarray | None = None,
    fit_options: Mapping[str, Any] | None = None,
    layer: str | None = None,
) -> CoxPHFitter:
    """Fit the Cox’s proportional hazard for the survival function.

    The Cox proportional hazards model (CoxPH) examines the relationship between the survival time of subjects and one or more predictor variables.
    It models the hazard rate as a product of a baseline hazard function and an exponential function of the predictors, assuming proportional hazards over time.
    The results will be stored in the `.uns` slot of the data object under the key 'cox_ph' unless specified otherwise in the `key_added` parameter.

    See https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html

    Args:
        edata: Central data object.
        duration_col: Column in `edata.obs` or variable with the subjects' lifetimes.
            `None` if `event_col` is a variable of 3D data, from which the durations are derived.
        event_col: Column in `edata.obs` or variable that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
            If it is a variable of 3D data that is 1 at the timepoints with the event and 0 otherwise, the event is whether it is ever 1 and the duration the time of its first 1, or of its last non-missing value for censored observations.
            Times count from the first timepoint, in `edata.tem['interval_start_offset']` if present, with time differences in seconds, or as positions otherwise.
        key_added: The key to use for the `.uns` slot in the data object.
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
            If left as None, all covariates are used additively.
            Uses the library Formulaic for parsing.
        covariates: Columns in `edata.obs` or variables used as covariates.
            If None, the variables referenced in `formula`, or all variables if no formula is given.
        batch_mode:  Enabling batch_mode can be faster for datasets with a large number of ties.
            If left as `None`, lifelines will choose the best option.
        show_progress: Since the fitter is iterative, show convergence diagnostics. Useful if convergence is failing.
        initial_point: set the starting point for the iterative solver.
        fit_options: Additional keyword arguments to pass into the estimator.
        layer: The layer to take variables from.

    Returns:
        Fitted CoxPHFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.mimic_2()
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> cph = ep.tl.cox_ph(
        ...     edata,
        ...     duration_col="mort_day_censored",
        ...     event_col="censor_flg",
        ...     formula="gender_num + afib_flg + day_icu_intime_num",
        ... )
    """
    strata_cols = [strata] if isinstance(strata, str) else list(strata or [])
    df, duration_col = _survival_frame(
        edata,
        duration_col,
        event_col,
        [
            entry_col,
            weights_col,
            cluster_col,
            *strata_cols,
            *_covariates(edata, covariates, formula),
        ],
        layer=layer,
    )
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
    edata.uns[key_added] = summary

    return cox_ph


def weibull_aft(
    edata: EHRData,
    duration_col: str | None,
    event_col: str,
    *,
    key_added: str = "weibull_aft",
    alpha: float = 0.05,
    fit_intercept: bool = True,
    penalizer: float | np.ndarray = 0.0,
    l1_ratio: float = 0.0,
    model_ancillary: bool = True,
    ancillary: bool | pd.DataFrame | str | None = None,
    show_progress: bool = False,
    weights_col: str | None = None,
    robust: bool = False,
    initial_point: np.ndarray | None = None,
    entry_col: str | None = None,
    formula: str | None = None,
    covariates: Sequence[str] | None = None,
    fit_options: Mapping[str, Any] | None = None,
    layer: str | None = None,
) -> WeibullAFTFitter:
    """Fit the Weibull accelerated failure time regression for the survival function.

    The Weibull Accelerated Failure Time (AFT) survival regression model is a statistical method used to analyze time-to-event data,
    where the underlying assumption is that the logarithm of survival time follows a Weibull distribution.
    It models the survival time as an exponential function of the predictors, assuming a specific shape parameter
    for the distribution and allowing for accelerated or decelerated failure times based on the covariates.
    The results will be stored in the `.uns` slot of the data object under the key 'weibull_aft' unless specified otherwise in the `key_added` parameter.

    See https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html

    Args:
        edata: Central data object.
        duration_col: Name of the column in the data objects that contains the subjects’ lifetimes.
            `None` if `event_col` is a variable of 3D data, from which the durations are derived.
        event_col: Column in `edata.obs` or variable that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
            If it is a variable of 3D data that is 1 at the timepoints with the event and 0 otherwise, the event is whether it is ever 1 and the duration the time of its first 1, or of its last non-missing value for censored observations.
            Times count from the first timepoint, in `edata.tem['interval_start_offset']` if present, with time differences in seconds, or as positions otherwise.
        key_added: The key to use for the `.uns` slot in the data object.
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
            If a formula is not provided, all covariates are used additively.
        covariates: Columns in `edata.obs` or variables used as covariates.
            If None, the variables referenced in `formula`, or all variables if no formula is given.
        fit_options: Additional keyword arguments to pass into the estimator.
        layer: The layer to take variables from.


    Returns:
        Fitted WeibullAFTFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.mimic_2()
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> edata = edata[:, ["mort_day_censored", "censor_flg"]]
        >>> aft = ep.tl.weibull_aft(edata, duration_col="mort_day_censored", event_col="censor_flg")
        >>> aft.print_summary()
    """
    ancillary_cols = _formula_variables(ancillary) if isinstance(ancillary, str) else []
    df, duration_col = _survival_frame(
        edata,
        duration_col,
        event_col,
        [entry_col, weights_col, *_covariates(edata, covariates, formula), *ancillary_cols],
        layer=layer,
    )
    df = _shift_zero_durations(df, duration_col)

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
    edata.uns[key_added] = summary

    return weibull_aft


def log_logistic_aft(
    edata: EHRData,
    duration_col: str | None = None,
    *,
    event_col: str | None = None,
    key_added: str = "log_logistic_aft",
    alpha: float = 0.05,
    fit_intercept: bool = True,
    penalizer: float | np.ndarray = 0.0,
    l1_ratio: float = 0.0,
    model_ancillary: bool = False,
    ancillary: bool | pd.DataFrame | str | None = None,
    show_progress: bool = False,
    weights_col: str | None = None,
    robust: bool = False,
    initial_point: np.ndarray | None = None,
    entry_col: str | None = None,
    formula: str | None = None,
    covariates: Sequence[str] | None = None,
    fit_options: Mapping[str, Any] | None = None,
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
            `None` if `event_col` is a variable of 3D data, from which the durations are derived.
        event_col: Column in `edata.obs` or variable that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
            If it is a variable of 3D data that is 1 at the timepoints with the event and 0 otherwise, the event is whether it is ever 1 and the duration the time of its first 1, or of its last non-missing value for censored observations.
            Times count from the first timepoint, in `edata.tem['interval_start_offset']` if present, with time differences in seconds, or as positions otherwise.
        key_added: The key to use for the `.uns` slot in the data object.
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
            If a formula is not provided, all covariates are used additively.
        covariates: Columns in `edata.obs` or variables used as covariates.
            If None, the variables referenced in `formula`, or all variables if no formula is given.
        fit_options: Additional keyword arguments to pass into the estimator.
        layer: The layer to take variables from.

    Returns:
        Fitted LogLogisticAFTFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.mimic_2()
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> edata = edata[:, ["mort_day_censored", "censor_flg"]]
        >>> llf = ep.tl.log_logistic_aft(edata, duration_col="mort_day_censored", event_col="censor_flg")
    """
    ancillary_cols = _formula_variables(ancillary) if isinstance(ancillary, str) else []
    df, duration_col = _survival_frame(
        edata,
        duration_col,
        event_col,
        [entry_col, weights_col, *_covariates(edata, covariates, formula), *ancillary_cols],
        layer=layer,
    )
    df = _shift_zero_durations(df, duration_col)

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
    edata.uns[key_added] = summary

    return log_logistic_aft


def _univariate_model(
    edata: EHRData,
    duration_col: str | None,
    event_col: str,
    model_class,
    key_added: str,
    accept_zero_duration=True,
    timeline: Sequence[float] | None = None,
    entry_col: str | None = None,
    label: str | None = None,
    alpha: float | None = None,
    ci_labels: Sequence[str] | None = None,
    weights_col: str | None = None,
    fit_options: Mapping[str, Any] | None = None,
    censoring: Literal["right", "left"] = "right",
    layer: str | None = None,
    event_of_interest: int | None = None,
    random_state: int = 0,
):
    """Convenience function for univariate models."""
    df, duration_col = _survival_frame(edata, duration_col, event_col, [entry_col, weights_col], layer=layer)
    if not accept_zero_duration:
        df = _shift_zero_durations(df, duration_col)

    events = None if event_col is None else df[event_col]
    if event_of_interest is None:
        if events is not None and not events.isin([0, 1]).all():
            raise ValueError(
                f"`{event_col}` has more than one event type, pass the one to estimate as `event_of_interest`."
            )
    elif events is None:
        raise ValueError("`event_of_interest` requires an `event_col`.")
    elif model_class is not AalenJohansenFitter:
        events = events == event_of_interest

    fit_kwargs = {
        "timeline": timeline,
        "entry": None if entry_col is None else df[entry_col],
        "label": label,
        "alpha": alpha,
        "ci_labels": ci_labels,
        "weights": None if weights_col is None else df[weights_col],
    }
    if model_class is AalenJohansenFitter:
        if censoring != "right" or fit_options is not None:
            raise ValueError("The cumulative incidence supports neither `censoring='left'` nor `fit_options`.")
        model = AalenJohansenFitter(seed=random_state)
        model.fit(df[duration_col], events.astype(int), event_of_interest, **fit_kwargs)
    else:
        model = model_class()
        function_name = "fit" if censoring == "right" else "fit_left_censoring"
        # get fit function, default to fit if not found
        fit_function = getattr(model, function_name, model.fit)
        fit_function(df[duration_col], event_observed=events, fit_options=fit_options, **fit_kwargs)

    if isinstance(
        model, NelsonAalenFitter | KaplanMeierFitter | AalenJohansenFitter
    ):  # the non-parametric fitters have no summary attribute
        summary = model.event_table
    else:
        summary = model.summary
    edata.uns[key_added] = summary

    return model


def nelson_aalen(
    edata: EHRData,
    duration_col: str | None = None,
    *,
    event_col: str | None = None,
    key_added: str = "nelson_aalen",
    timeline: Sequence[float] | None = None,
    entry_col: str | None = None,
    label: str | None = None,
    alpha: float | None = None,
    ci_labels: Sequence[str] | None = None,
    weights_col: str | None = None,
    fit_options: Mapping[str, Any] | None = None,
    censoring: Literal["right", "left"] = "right",
    layer: str | None = None,
    event_of_interest: int | None = None,
) -> NelsonAalenFitter:
    """Employ the Nelson-Aalen estimator to estimate the cumulative hazard function from censored survival data.

    The Nelson-Aalen estimator is a non-parametric method used in survival analysis to estimate the cumulative hazard function.
    It accounts for the presence of individuals whose event times are unknown due to censoring.
    By estimating the cumulative hazard function, the Nelson-Aalen estimator assessing the risk of an event occurring over time.
    The results will be stored in the `.uns` slot of the data object under the key 'nelson_aalen' unless specified otherwise in the `key_added` parameter.
    See https://lifelines.readthedocs.io/en/latest/fitters/univariate/NelsonAalenFitter.html

    Args:
        edata: Central data object.
        duration_col: Column in `edata.obs` or variable with the subjects' lifetimes.
            `None` if `event_col` is a variable of 3D data, from which the durations are derived.
        event_col: Column in `edata.obs` or variable that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
            If it is a variable of 3D data that is 1 at the timepoints with the event and 0 otherwise, the event is whether it is ever 1 and the duration the time of its first 1, or of its last non-missing value for censored observations.
            Times count from the first timepoint, in `edata.tem['interval_start_offset']` if present, with time differences in seconds, or as positions otherwise.
        key_added: The key to use for the `.uns` slot in the data object.
        timeline: Return the best estimate at the values in timelines (positively increasing)
        entry_col: Column in `edata.obs` or variable with the relative time when a subject entered the study.
            This is useful for left-truncated (not left-censored) observations.
            If None, all members of the population entered study when they were "born".
        label: A string to name the column of the estimate.
        alpha: The alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
        ci_labels: Add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>] (default: <label>_lower_<1-alpha/2>).
        weights_col: Column in `edata.obs` or variable with a weight per subject.
        fit_options: Additional keyword arguments to pass into the estimator.
        censoring: 'right' for fitting the model to a right-censored dataset. (default, calls fit).
                   'left' for fitting the model to a left-censored dataset (calls fit_left_censoring).
        layer: The layer to take variables from.
        event_of_interest: The event type in `event_col` to estimate the cause-specific cumulative hazard of.
            `event_col` then holds 0 for censored subjects and a positive integer per event type, and subjects with other event types count as censored at their event.

    Returns:
        Fitted NelsonAalenFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.mimic_2()
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> naf = ep.tl.nelson_aalen(edata, duration_col="mort_day_censored", event_col="censor_flg")
    """
    return _univariate_model(
        edata,
        duration_col,
        event_col,
        NelsonAalenFitter,
        key_added=key_added,
        accept_zero_duration=True,
        timeline=timeline,
        entry_col=entry_col,
        label=label,
        alpha=alpha,
        ci_labels=ci_labels,
        weights_col=weights_col,
        fit_options=fit_options,
        censoring=censoring,
        layer=layer,
        event_of_interest=event_of_interest,
    )


def weibull(
    edata: EHRData,
    duration_col: str | None,
    event_col: str,
    *,
    key_added: str = "weibull",
    timeline: Sequence[float] | None = None,
    entry_col: str | None = None,
    label: str | None = None,
    alpha: float | None = None,
    ci_labels: Sequence[str] | None = None,
    weights_col: str | None = None,
    fit_options: Mapping[str, Any] | None = None,
    layer: str | None = None,
) -> WeibullFitter:
    """Employ the Weibull model in univariate survival analysis to understand event occurrence dynamics.

    In contrast to the non-parametric Nelson-Aalen estimator, the Weibull model employs a parametric approach with shape and scale parameters,
    enabling a more structured analysis of survival data.
    This technique is particularly useful when dealing with censored data, as it accounts for the presence of individuals whose event times are unknown due to censoring.
    By fitting the Weibull model to censored survival data, researchers can estimate these parameters and gain insights
    into the hazard rate over time, facilitating comparisons between different groups or treatments.
    This method provides a comprehensive framework for examining survival data and offers valuable insights into the factors influencing event occurrence dynamics.
    The results will be stored in the `.uns` slot of the data object under the key 'weibull' unless specified otherwise in the `key_added` parameter.
    See https://lifelines.readthedocs.io/en/latest/fitters/univariate/WeibullFitter.html

    Args:
        edata: Central data object.
        duration_col: Name of the column in the data objects that contains the subjects’ lifetimes.
            `None` if `event_col` is a variable of 3D data, from which the durations are derived.
        event_col: Column in `edata.obs` or variable that specifies whether the event has been observed, or censored.
            Column values are `True` if the event was observed, `False` if the event was lost (right-censored).
            If left `None`, all individuals are assumed to be uncensored.
            If it is a variable of 3D data that is 1 at the timepoints with the event and 0 otherwise, the event is whether it is ever 1 and the duration the time of its first 1, or of its last non-missing value for censored observations.
            Times count from the first timepoint, in `edata.tem['interval_start_offset']` if present, with time differences in seconds, or as positions otherwise.
        key_added: The key to use for the `.uns` slot in the data object.
        timeline: Return the best estimate at the values in timelines (positively increasing)
        entry_col: Column in `edata.obs` or variable with the relative time when a subject entered the study.
            This is useful for left-truncated (not left-censored) observations.
            If None, all members of the population entered study when they were "born".
        label: A string to name the column of the estimate.
        alpha: The alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
        ci_labels: Add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>] (default: <label>_lower_<1-alpha/2>).
        weights_col: Column in `edata.obs` or variable with a weight per subject.
        fit_options: Additional keyword arguments to pass into the estimator.
        layer: The layer to take variables from.

    Returns:
        Fitted WeibullFitter.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.mimic_2()
        >>> # Flip 'censor_fl' because 0 = death and 1 = censored
        >>> edata[:, ["censor_flg"]].X = np.where(edata[:, ["censor_flg"]].X == 0, 1, 0)
        >>> wf = ep.tl.weibull(edata, duration_col="mort_day_censored", event_col="censor_flg")
    """
    return _univariate_model(
        edata,
        duration_col,
        event_col,
        WeibullFitter,
        key_added=key_added,
        accept_zero_duration=False,
        timeline=timeline,
        entry_col=entry_col,
        label=label,
        alpha=alpha,
        ci_labels=ci_labels,
        weights_col=weights_col,
        fit_options=fit_options,
        layer=layer,
    )


def cox_ph_adjusted_curves(
    edata: EHRData,
    cph: CoxPHFitter,
    strata: str,
    duration_col: str | None,
    event_col: str,
    *,
    method: Literal["average", "conditional"] = "average",
    reference_values: Mapping[str, str | int | float] | None = None,
    n_bootstrap: int = 200,
    ci_alpha: float = 0.05,
    times: np.ndarray | None = None,
    layer: str | None = None,
    key_added: str = "cox_ph_adjusted_curves",
    copy: bool = False,
) -> EHRData | None:
    """Compute CoxPH adjusted survival curves stratified by a grouping variable.

    Adjusted survival curves account for differences in baseline covariates between groups, allowing fairer comparison of survival outcomes in observational cohorts where groups may not be balanced.
    This mirrors the functionality of R's survminer::surv_adjustedcurves().
    The results will be stored in the `.uns` slot of the data object under the key 'cox_ph_adjusted_curves',
    unless specified otherwise in the `key_added` parameter.
    See Therneau, Crowson & Atkinson (2015), 'Adjusted Survival Curves': https://cran.r-project.org/web/packages/survival/vignettes/adjcurve.pdf.

    Args:
        edata: Central data object.
        cph: Fitted CoxPHFitter, as returned by :func:`~ehrapy.tools.cox_ph`.
        strata: Name of the column to stratify by.
            Must be present in the data and should not be included in the Cox model formula.
        duration_col: The name of the column that contains the subjects' lifetimes.
            `None` if `event_col` is a variable of 3D data, from which the durations are derived.
        event_col: The name of the column that specifies whether the event has been observed or censored.
            Column values are True if the event was observed, False if the event was lost (right-censored).
            If it is a variable of 3D data that is 1 at the timepoints with the event and 0 otherwise, the event is whether it is ever 1 and the duration the time of its first 1, or of its last non-missing value for censored observations.
            Times count from the first timepoint, in `edata.tem['interval_start_offset']` if present, with time differences in seconds, or as positions otherwise.
        method: The method used to compute adjusted survival curves. Options are:
            * `'average'` one population-averaged curve per group, no rebalancing.
            * `'conditional'` one curve per group for a synthetic reference patient with cohort-average covariates, varying only the strata variable.
        reference_values: A dict of values to override the default reference patient values for method = 'conditional' (mean for continuous, mode for categorical).
        n_bootstrap: Number of bootstrap resamples used to compute confidence intervals.
            Only used when method is 'average'.
        ci_alpha: Significance level for confidence intervals.
        times: Evaluation time grid.
            Defaults to 100 evenly-spaced points from 0 to the maximum observed time.
        layer: The layer to use when reconstructing the covariate data for prediction.
        key_added: The key to use for the `.uns` slot in the data object.
        copy: Copy `edata` before computation and return a copy. Otherwise, perform computation in place and return `None`.

    Returns:
        Depending on `copy`, returns or updates `edata` with the results in `edata.uns[key_added]`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> cph = ep.tl.cox_ph(
        ...     edata,
        ...     duration_col="mort_day_censored",
        ...     event_col="censor_flg",
        ...     formula="gender_num + afib_flg + day_icu_intime_num",
        ... )
        >>> ep.tl.cox_ph_adjusted_curves(
        ...     edata,
        ...     cph=cph,
        ...     strata="gender_num",
        ...     duration_col="mort_day_censored",
        ...     event_col="censor_flg",
        ... )
    """
    edata = edata.copy() if copy else edata
    strata_cols = [cph.strata] if isinstance(cph.strata, str) else list(cph.strata or [])
    cph_covariates = _formula_variables(cph.formula) if cph.formula else list(cph.params_.index)
    df, duration_col = _survival_frame(
        edata, duration_col, event_col, [strata, *strata_cols, *cph_covariates], layer=layer
    )

    t_max = df[duration_col].max()
    _times = times if times is not None else np.linspace(0, t_max, 100)

    # Columns to pass to predict (everything except duration and event)
    predict_cols = [c for c in df.columns if c not in (duration_col, event_col)]

    groups = df[strata].unique()
    results: dict = {}

    if method == "average":
        full_df = df[predict_cols].copy().reset_index(drop=True)

        for group in groups:
            # Assign every patient in the cohort to this group
            full_df_group = full_df.copy()
            # keep the dtype seen at fit time so that the model encodes the strata variable consistently
            full_df_group[strata] = pd.Series([group] * len(full_df_group), dtype=df[strata].dtype)

            surv_matrix = cph.predict_survival_function(full_df_group, times=_times)
            mean_surv = surv_matrix.values.mean(axis=1)

            boot_means = _bootstrap_average_survival(
                cph=cph,
                group_df=full_df_group,
                times=_times,
                predict_cols=predict_cols,
                n_bootstrap=n_bootstrap,
                rng=np.random.default_rng(seed=42),
            )
            ci_lower = np.percentile(boot_means, 100 * (ci_alpha / 2), axis=0)
            ci_upper = np.percentile(boot_means, 100 * (1 - ci_alpha / 2), axis=0)

            results[str(group)] = {
                "times": _times,
                "survival": mean_surv,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
    elif method == "conditional":
        ref_patient = _build_reference_patient(df, predict_cols, reference_values or {}, cph)
        for group in groups:
            ref = ref_patient.copy()
            ref[strata] = group  # vary only the strata variable

            ref_df = df[predict_cols].iloc[[0]].copy()
            for col, val in ref.items():
                if col in ref_df.columns:
                    if hasattr(ref_df[col], "cat"):
                        # preserve categorical dtype when assigning scalar
                        ref_df[col] = pd.Categorical([val], categories=ref_df[col].cat.categories)
                    else:
                        ref_df[col] = val

            surv = cph.predict_survival_function(ref_df, times=_times)
            survival = surv.values.flatten()

            results[str(group)] = {
                "times": _times,
                "survival": survival,
                "ci_lower": None,
                "ci_upper": None,
            }

    else:
        raise ValueError(f"method must be 'average' or 'conditional', got '{method}'")

    results["_meta"] = {
        "strata": strata,
        "method": method,
        "n_bootstrap": n_bootstrap if method == "average" else None,
        "ci_alpha": ci_alpha,
        "duration_col": duration_col,
        "event_col": event_col,
    }
    edata.uns[key_added] = results

    return edata if copy else None


def _bootstrap_average_survival(
    cph: CoxPHFitter,
    group_df: pd.DataFrame,
    times: np.ndarray,
    predict_cols: Sequence[str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute one mean survival curve per bootstrap resample of the group."""
    boot_means = np.zeros((n_bootstrap, len(times)))
    n = len(group_df)
    for i in range(n_bootstrap):
        sample = group_df.sample(n=n, replace=True, random_state=int(rng.integers(1e6)))
        surv_matrix = cph.predict_survival_function(sample[predict_cols], times=times)
        boot_means[i] = surv_matrix.values.mean(axis=1)
    return boot_means


def _build_reference_patient(
    df: pd.DataFrame,
    predict_cols: Sequence[str],
    overrides: Mapping[str, str | int | float],
    cph: CoxPHFitter,
) -> dict:
    """Build a synthetic reference patient.

    A reference patient represents a patient with average characteristics of the entire cohort.

    - continuous columns -> column mean
    - categorical/object columns -> column mode
    - any key in `overrides` -> use that value.
    """
    # columns that got dummy-encoded during fitting have names like "col[T.x]"
    dummy_encoded_cols = {param.split("[")[0] for param in cph.params_.index.get_level_values(0) if "[" in param}

    ref = {}
    for col in predict_cols:
        if col in overrides:
            ref[col] = overrides[col]
        elif col in dummy_encoded_cols:
            # must pass an original level, not a mean — use mode
            ref[col] = df[col].mode().iloc[0]
        elif pd.api.types.is_numeric_dtype(df[col]):
            ref[col] = df[col].mean()
        else:
            ref[col] = df[col].mode().iloc[0]
    return ref
