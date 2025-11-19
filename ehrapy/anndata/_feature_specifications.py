from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Literal

import ehrdata as ed
import numpy as np
import pandas as pd
from anndata import AnnData
from dateutil.parser import isoparse  # type: ignore
from ehrdata._logger import logger
from ehrdata.core.constants import CATEGORICAL_TAG, DATE_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from rich import print
from rich.tree import Tree

from ehrapy._compat import function_2D_only, function_future_warning, use_ehrdata

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ehrdata import EHRData


def _detect_feature_type(col: pd.Series) -> tuple[Literal["date", "categorical", "numeric"], bool]:
    """Detect the feature type of a column in a pandas DataFrame.

    Args:
        col: The column to detect the feature type for.
        verbose: Whether to print warnings for uncertain feature types.

    Returns:
        The detected feature type (one of 'date', 'categorical', or 'numeric') and a boolean, which is True if the feature type is uncertain.
    """
    n_elements = len(col)
    col = col.dropna()
    if len(col) == 0:
        raise ValueError(
            f"Feature '{col.name}' contains only NaN values. Please drop this feature to infer the feature type."
        )
    majority_type = col.apply(type).value_counts().idxmax()

    if majority_type == pd.Timestamp:
        return DATE_TAG, False  # type: ignore

    if majority_type is str:
        try:
            col.apply(isoparse)
            return DATE_TAG, False  # type: ignore
        except ValueError:
            try:
                col = pd.to_numeric(col, errors="raise")  # Could be an encoded categorical or a numeric feature
                majority_type = float
            except ValueError:
                # Features stored as Strings that cannot be converted to float are assumed to be categorical
                return CATEGORICAL_TAG, False  # type: ignore

    if majority_type not in [int, float]:
        return CATEGORICAL_TAG, False  # type: ignore

    # Guess categorical if the feature is an integer and the values are 0/1 to n-1/n with no gaps
    if (
        (majority_type is int or (np.all(i.is_integer() for i in col)))
        and (n_elements != col.nunique())
        and (
            (col.min() == 0 and np.all(np.sort(col.unique()) == np.arange(col.nunique())))
            or (col.min() == 1 and np.all(np.sort(col.unique()) == np.arange(1, col.nunique() + 1)))
        )
    ):
        return CATEGORICAL_TAG, True  # type: ignore

    return NUMERIC_TAG, False  # type: ignore


@use_ehrdata(deprecated_after="1.0.0")
@function_future_warning("ep.ad.infer_feature_types", "ehrdata.infer_feature_types")
@function_2D_only()
def infer_feature_types(
    edata: EHRData | AnnData,
    layer: str | None = None,
    output: Literal["tree", "dataframe"] | None = "tree",
    verbose: bool = True,
) -> pd.Series | None:
    """Infer feature types from EHRData object.

    For each feature in edata.var_names, the method infers one of the following types: 'date', 'categorical', or 'numeric'.
    The inferred types are stored in edata.var['feature_type']. Please check the inferred types and adjust if necessary using
    edata.var['feature_type']['feature1']='corrected_type'.
    Be aware that not all features stored numerically are of 'numeric' type, as categorical features might be stored in a numerically encoded format.
    For example, a feature with values [0, 1, 2] might be a categorical feature with three categories. This is accounted for in the method, but it is
    recommended to check the inferred types.

    Args:
        edata: Central data object.
        layer: The layer to use from the EHRData object. If None, the X layer is used.
        output: The output format. Choose between 'tree', 'dataframe', or None. If 'tree', the feature types will be printed to the console in a tree format.
            If 'dataframe', a pandas DataFrame with the feature types will be returned. If None, nothing will be returned.
        verbose: Whether to print warnings for uncertain feature types.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.ad.infer_feature_types(edata)
    """
    from ehrapy.anndata.anndata_ext import anndata_to_df

    feature_types = {}
    uncertain_features = []

    df = anndata_to_df(edata, layer=layer)
    for feature in edata.var_names:
        if (
            FEATURE_TYPE_KEY in edata.var.keys()
            and edata.var[FEATURE_TYPE_KEY][feature] is not None
            and not pd.isna(edata.var[FEATURE_TYPE_KEY][feature])
        ):
            feature_types[feature] = edata.var[FEATURE_TYPE_KEY][feature]
        else:
            feature_types[feature], raise_warning = _detect_feature_type(df[feature])
            if raise_warning:
                uncertain_features.append(feature)

    edata.var[FEATURE_TYPE_KEY] = pd.Series(feature_types)[edata.var_names]

    if verbose:
        logger.warning(
            f"{'Features' if len(uncertain_features) > 1 else 'Feature'} {str(uncertain_features)[1:-1]} {'were' if len(uncertain_features) > 1 else 'was'} detected as categorical features stored numerically."
            f"Please verify and correct using `ep.ad.replace_feature_types` if necessary."
        )

        logger.info(
            f"Stored feature types in edata.var['{FEATURE_TYPE_KEY}']."
            f" Please verify and adjust if necessary using `ep.ad.replace_feature_types`."
        )

    if output == "tree":
        feature_type_overview(edata)
    elif output == "dataframe":
        return edata.var[FEATURE_TYPE_KEY]
    elif output is not None:
        raise ValueError(f"Output format {output} not recognized. Choose between 'tree', 'dataframe', or None.")

    return None


# TODO: this function is a different flavor of inferring feature types. We should decide on a single implementation in the future.
def _infer_numerical_column_indices(
    edata: EHRData | AnnData, layer: str | None = None, column_indices: Iterable[int] | None = None
) -> list[int]:
    mtx = edata.X if layer is None else edata[layer]
    indices = (
        list(range(mtx.shape[1])) if column_indices is None else [i for i in column_indices if i < mtx.shape[1] - 1]
    )
    non_numerical_indices = []
    for i in indices:
        # The astype("float64") call will throw only if the feature’s data type cannot be cast to float64, meaning in
        # practice it contains non-numeric values. Consequently, it won’t throw if the values are numeric but stored
        # as an "object" dtype, as astype("float64") can successfully convert them to floats.
        try:
            mtx[::, i].astype("float64")
        except ValueError:
            non_numerical_indices.append(i)

    return [idx for idx in indices if idx not in non_numerical_indices]


def _check_feature_types(func):
    @wraps(func)
    def wrapper(edata, *args, **kwargs):
        # Account for class methods that pass self as first argument
        from ehrdata import EHRData

        _self = None
        if (
            not (isinstance(edata, EHRData) or isinstance(edata, AnnData))
            and len(args) > 0
            and (isinstance(args[0], EHRData) or isinstance(args[0], AnnData))
        ):
            _self = edata
            edata = args[0]
            args = args[1:]

        layer = kwargs.get("layer", None)

        if FEATURE_TYPE_KEY not in edata.var.keys():
            ed.infer_feature_types(edata, layer=layer, output=None)
            logger.warning(
                f"Feature types were inferred and stored in edata.var[{FEATURE_TYPE_KEY}]. Please verify using `ehrdata.feature_type_overview` and adjust if necessary using `ehrdata.replace_feature_types`."
            )

        for feature in edata.var_names:
            feature_type = edata.var[FEATURE_TYPE_KEY][feature]
            if (
                feature_type is not None
                and (not pd.isna(feature_type))
                and feature_type not in [CATEGORICAL_TAG, NUMERIC_TAG, DATE_TAG]
            ):
                logger.warning(
                    f"Feature '{feature}' has an invalid feature type '{feature_type}'. Please correct using `ep.ad.replace_feature_types`."
                )

        if _self is not None:
            return func(_self, edata, *args, **kwargs)
        return func(edata, *args, **kwargs)

    return wrapper


@_check_feature_types
@use_ehrdata(deprecated_after="1.0.0")
@function_future_warning("ep.ad.feature_type_overview", "ehrdata.feature_type_overview")
def feature_type_overview(edata: EHRData | AnnData) -> None:
    """Print an overview of the feature types and encoding modes in the EHRData object.

    Args:
        edata: Central data object.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.ad.feature_type_overview(edata)
    """
    from ehrapy.anndata.anndata_ext import anndata_to_df

    tree = Tree(
        f"[b] Detected feature types for AnnData object with {len(edata.obs_names)} obs and {len(edata.var_names)} vars",
        guide_style="underline2",
    )

    branch = tree.add("📅[b] Date features")
    for date in sorted(edata.var_names[edata.var[FEATURE_TYPE_KEY] == DATE_TAG]):
        branch.add(date)

    branch = tree.add("📐[b] Numerical features")
    for numeric in sorted(edata.var_names[edata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG]):
        branch.add(numeric)

    branch = tree.add("🗂️[b] Categorical features")
    cat_features = edata.var_names[edata.var[FEATURE_TYPE_KEY] == CATEGORICAL_TAG]
    df = anndata_to_df(edata[:, cat_features])

    if "encoding_mode" in edata.var.keys():
        unencoded_vars = edata.var.loc[cat_features, "unencoded_var_names"].unique().tolist()

        for unencoded in sorted(unencoded_vars):
            if unencoded in edata.var_names:
                branch.add(f"{unencoded} ({df.loc[:, unencoded].nunique()} categories)")
            else:
                enc_mode = edata.var.loc[edata.var["unencoded_var_names"] == unencoded, "encoding_mode"].values[0]
                branch.add(f"{unencoded} ({edata.obs[unencoded].nunique()} categories); {enc_mode} encoded")

    else:
        for categorical in sorted(cat_features):
            branch.add(f"{categorical} ({df.loc[:, categorical].nunique()} categories)")

    print(tree)


@use_ehrdata(deprecated_after="1.0.0")
@function_future_warning("ep.ad.replace_feature_types", "ehrdata.replace_feature_types")
def replace_feature_types(edata: EHRData | AnnData, features: Iterable[str], corrected_type: str) -> None:
    """Correct the feature types for a list of features inplace.

    Args:
        edata: Central data object.
        features: The features to correct.
        corrected_type: The corrected feature type. One of 'date', 'categorical', or 'numeric'.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.diabetes_130_fairlearn()
        >>> ep.ad.infer_feature_types(edata)
        >>> ep.ad.replace_feature_types(edata, ["time_in_hospital", "number_diagnoses", "num_procedures"], "numeric")
    """
    if corrected_type not in [CATEGORICAL_TAG, NUMERIC_TAG, DATE_TAG]:
        raise ValueError(
            f"Corrected type {corrected_type} not recognized. Choose between '{DATE_TAG}', '{CATEGORICAL_TAG}', or '{NUMERIC_TAG}'."
        )

    if FEATURE_TYPE_KEY not in edata.var.keys():
        raise ValueError(
            "Feature types were not inferred. Please infer feature types using 'infer_feature_types' before correcting."
        )

    if isinstance(features, str):
        features = [features]

    edata.var.loc[features, FEATURE_TYPE_KEY] = corrected_type
