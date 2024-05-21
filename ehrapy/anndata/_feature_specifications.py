from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from dateutil.parser import isoparse  # type: ignore
from lamin_utils import logger
from rich import print
from rich.tree import Tree

from ehrapy.anndata._constants import CATEGORICAL_TAG, DATE_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG

if TYPE_CHECKING:
    from collections.abc import Iterable


def _detect_feature_type(col: pd.Series) -> tuple[Literal["date", "categorical", "numeric"], bool]:
    """Detect the feature type of a column in a pandas DataFrame.

    Args:
        col: The column to detect the feature type for.
        verbose: Whether to print warnings for uncertain feature types. Defaults to True.

    Returns:
        The detected feature type (one of 'date', 'categorical', or 'numeric') and a boolean, which is True if the feature type is uncertain.
    """
    n_elements = len(col)
    col = col.dropna()
    if len(col) == 0:
        raise ValueError(
            f"Feature {col.name} has only NaN values. Please drop the feature if you want to infer the feature type."
        )
    majority_type = col.apply(type).value_counts().idxmax()

    if majority_type == pd.Timestamp:
        return DATE_TAG, False  # type: ignore

    if majority_type == str:
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
        (majority_type == int or (np.all(i.is_integer() for i in col)))
        and (n_elements != col.nunique())
        and (
            (col.min() == 0 and np.all(np.sort(col.unique()) == np.arange(col.nunique())))
            or (col.min() == 1 and np.all(np.sort(col.unique()) == np.arange(1, col.nunique() + 1)))
        )
    ):
        return CATEGORICAL_TAG, True  # type: ignore

    return NUMERIC_TAG, False  # type: ignore


def infer_feature_types(
    adata: AnnData, layer: str | None = None, output: Literal["tree", "dataframe"] | None = "tree", verbose: bool = True
):
    """Infer feature types from AnnData object.

    For each feature in adata.var_names, the method infers one of the following types: 'date', 'categorical', or 'numeric'.
    The inferred types are stored in adata.var['feature_type']. Please check the inferred types and adjust if necessary using
    adata.var['feature_type']['feature1']='corrected_type'.
    Be aware that not all features stored numerically are of 'numeric' type, as categorical features might be stored in a numerically encoded format.
    For example, a feature with values [0, 1, 2] might be a categorical feature with three categories. This is accounted for in the method, but it is
    recommended to check the inferred types.

    Args:
        adata: :class:`~anndata.AnnData` object storing the EHR data.
        layer: The layer to use from the AnnData object. If None, the X layer is used.
        output: The output format. Choose between 'tree', 'dataframe', or None. If 'tree', the feature types will be printed to the console in a tree format.
            If 'dataframe', a pandas DataFrame with the feature types will be returned. If None, nothing will be returned. Defaults to 'tree'.
        verbose: Whether to print warnings for uncertain feature types. Defaults to True.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=False)
        >>> ep.ad.infer_feature_types(adata)
    """
    from ehrapy.anndata.anndata_ext import anndata_to_df

    feature_types = {}
    uncertain_features = []

    df = anndata_to_df(adata, layer=layer)
    for feature in adata.var_names:
        if (
            FEATURE_TYPE_KEY in adata.var.keys()
            and adata.var[FEATURE_TYPE_KEY][feature] is not None
            and not pd.isna(adata.var[FEATURE_TYPE_KEY][feature])
        ):
            feature_types[feature] = adata.var[FEATURE_TYPE_KEY][feature]
        else:
            feature_types[feature], raise_warning = _detect_feature_type(df[feature])
            if raise_warning:
                uncertain_features.append(feature)

    adata.var[FEATURE_TYPE_KEY] = pd.Series(feature_types)[adata.var_names]

    if verbose:
        logger.warning(
            f"{'Features' if len(uncertain_features) >1 else 'Feature'} {str(uncertain_features)[1:-1]} {'were' if len(uncertain_features) >1 else 'was'} detected as categorical features stored numerically."
            f"Please verify and correct using `ep.ad.replace_feature_types` if necessary."
        )

        logger.info(
            f"Stored feature types in adata.var['{FEATURE_TYPE_KEY}']."
            f" Please verify and adjust if necessary using `ep.ad.replace_feature_types`."
        )

    if output == "tree":
        feature_type_overview(adata)
    elif output == "dataframe":
        return adata.var[FEATURE_TYPE_KEY]
    elif output is not None:
        raise ValueError(f"Output format {output} not recognized. Choose between 'tree', 'dataframe', or None.")


def check_feature_types(func):
    @wraps(func)
    def wrapper(adata, *args, **kwargs):
        # Account for class methods that pass self as first argument
        _self = None
        if not isinstance(adata, AnnData) and len(args) > 0 and isinstance(args[0], AnnData):
            _self = adata
            adata = args[0]
            args = args[1:]

        if FEATURE_TYPE_KEY not in adata.var.keys():
            infer_feature_types(adata, output=None)
            logger.warning(
                f"Feature types were inferred and stored in adata.var[{FEATURE_TYPE_KEY}]. Please verify using `ep.ad.feature_type_overview` and adjust if necessary using `ep.ad.replace_feature_types`."
            )

        for feature in adata.var_names:
            feature_type = adata.var[FEATURE_TYPE_KEY][feature]
            if (
                feature_type is not None
                and (not pd.isna(feature_type))
                and feature_type not in [CATEGORICAL_TAG, NUMERIC_TAG, DATE_TAG]
            ):
                logger.warning(
                    f"Feature '{feature}' has an invalid feature type '{feature_type}'. Please correct using `ep.ad.replace_feature_types`."
                )

        if _self is not None:
            return func(_self, adata, *args, **kwargs)
        return func(adata, *args, **kwargs)

    return wrapper


@check_feature_types
def feature_type_overview(adata: AnnData):
    """Print an overview of the feature types and encoding modes in the AnnData object.

    Args:
        adata: The AnnData object storing the EHR data.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.ad.feature_type_overview(adata)
    """
    from ehrapy.anndata.anndata_ext import anndata_to_df

    tree = Tree(
        f"[b] Detected feature types for AnnData object with {len(adata.obs_names)} obs and {len(adata.var_names)} vars",
        guide_style="underline2",
    )

    branch = tree.add("📅[b] Date features")
    for date in sorted(adata.var_names[adata.var[FEATURE_TYPE_KEY] == DATE_TAG]):
        branch.add(date)

    branch = tree.add("📐[b] Numerical features")
    for numeric in sorted(adata.var_names[adata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG]):
        branch.add(numeric)

    branch = tree.add("🗂️[b] Categorical features")
    cat_features = adata.var_names[adata.var[FEATURE_TYPE_KEY] == CATEGORICAL_TAG]
    df = anndata_to_df(adata[:, cat_features])

    if "encoding_mode" in adata.var.keys():
        unencoded_vars = adata.var.loc[cat_features, "unencoded_var_names"].unique().tolist()

        for unencoded in sorted(unencoded_vars):
            if unencoded in adata.var_names:
                branch.add(f"{unencoded} ({df.loc[:, unencoded].nunique()} categories)")
            else:
                enc_mode = adata.var.loc[adata.var["unencoded_var_names"] == unencoded, "encoding_mode"].values[0]
                branch.add(f"{unencoded} ({adata.obs[unencoded].nunique()} categories); {enc_mode} encoded")

    else:
        for categorical in sorted(cat_features):
            branch.add(f"{categorical} ({df.loc[:, categorical].nunique()} categories)")

    print(tree)


def replace_feature_types(adata, features: Iterable[str], corrected_type: str):
    """Correct the feature types for a list of features inplace.

    Args:
        adata: :class:`~anndata.AnnData` object storing the EHR data.
        features: The features to correct.
        corrected_type: The corrected feature type. One of 'date', 'categorical', or 'numeric'.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.diabetes_130_fairlearn()
        >>> ep.ad.infer_feature_types(adata)
        >>> ep.ad.replace_feature_types(adata, ["time_in_hospital", "number_diagnoses", "num_procedures"], "numeric")
    """
    if corrected_type not in [CATEGORICAL_TAG, NUMERIC_TAG, DATE_TAG]:
        raise ValueError(
            f"Corrected type {corrected_type} not recognized. Choose between '{DATE_TAG}', '{CATEGORICAL_TAG}', or '{NUMERIC_TAG}'."
        )

    if FEATURE_TYPE_KEY not in adata.var.keys():
        raise ValueError(
            "Feature types were not inferred. Please infer feature types using 'infer_feature_types' before correcting."
        )

    if isinstance(features, str):
        features = [features]

    adata.var[FEATURE_TYPE_KEY].loc[features] = corrected_type
