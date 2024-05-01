from functools import wraps
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from lamin_utils import logger
from rich import print
from rich.tree import Tree

from ehrapy.anndata._constants import CATEGORICAL_TAG, CONTINUOUS_TAG, DATE_TAG, FEATURE_TYPE_KEY
from ehrapy.anndata.anndata_ext import anndata_to_df


def infer_feature_types(adata: AnnData, layer: str | None = None, output: Literal["tree", "dataframe"] | None = "tree"):
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
    """
    feature_types = {}

    df = anndata_to_df(adata, layer=layer)
    for feature in adata.var_names:
        col = df[feature].dropna()
        majority_type = col.apply(type).value_counts().idxmax()
        if majority_type == pd.Timestamp:
            feature_types[feature] = DATE_TAG
        elif majority_type not in [int, float, complex]:
            feature_types[feature] = CATEGORICAL_TAG
        # Guess categorical if the feature is an integer and the values are 0/1 to n-1 with no gaps
        elif np.all(i.is_integer() for i in col) and (
            (col.min() == 0 and np.all(np.sort(col.unique()) == np.arange(col.nunique())))
            or (col.min() == 1 and np.all(np.sort(col.unique()) == np.arange(1, col.nunique() + 1)))
        ):
            feature_types[feature] = CATEGORICAL_TAG
        else:
            feature_types[feature] = CONTINUOUS_TAG

    adata.var[FEATURE_TYPE_KEY] = pd.Series(feature_types)[adata.var_names]

    logger.info(
        f"Stored feature types in adata.var['{FEATURE_TYPE_KEY}']."
        f" Please verify and adjust if necessary using adata.var['{FEATURE_TYPE_KEY}']['feature1']='corrected_type'."
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
        if FEATURE_TYPE_KEY not in adata.var.keys():
            raise ValueError("Feature types are not specified in adata.var. Please run `infer_feature_types` first.")
        np.all(adata.var[FEATURE_TYPE_KEY].isin([CATEGORICAL_TAG, CONTINUOUS_TAG, DATE_TAG]))
        return func(adata, *args, **kwargs)

    return wrapper


@check_feature_types
def feature_type_overview(adata: AnnData):
    """Print an overview of the feature types in the AnnData object."""
    tree = Tree(
        f"[b] Detected feature types for AnnData object with {len(adata.obs_names)} obs and {len(adata.var_names)} vars",
        guide_style="underline2",
    )

    branch = tree.add("📅[b] Date features")
    for date in sorted(adata.var_names[adata.var[FEATURE_TYPE_KEY] == DATE_TAG]):
        branch.add(date)

    branch = tree.add("📐[b] Numerical features")
    for numeric in sorted(adata.var_names[adata.var[FEATURE_TYPE_KEY] == CONTINUOUS_TAG]):
        branch.add(numeric)

    branch = tree.add("🗂️[b] Categorical features")
    cat_features = adata.var_names[adata.var[FEATURE_TYPE_KEY] == CATEGORICAL_TAG]
    df = anndata_to_df(adata[:, cat_features])
    for categorical in sorted(cat_features):
        branch.add(f"{categorical} ({df.loc[:, categorical].nunique()} categories)")

    print(tree)
