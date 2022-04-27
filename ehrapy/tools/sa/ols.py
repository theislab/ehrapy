from __future__ import annotations

from typing import Literal

import numpy as np  # noqa: F401 # This package is implicitly used
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from anndata import AnnData


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
