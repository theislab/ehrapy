import numpy as np
import pandas as pd

from ehrapy.anndata._constants import CATEGORICAL_TAG, CONTINUOUS_TAG, DATE_TAG, FEATURE_TYPE_KEY
from ehrapy.anndata.anndata_ext import anndata_to_df


def infer_feature_types(adata, layer: str | None = None):
    """
    Infer feature types from AnnData object.

    Args:
        adata: :class:`~anndata.AnnData` object storing the EHR data.
        layer: The layer to use from the AnnData object. If None, the X layer is used.
    """
    feature_types = {}  # TODO: Add date type

    df = anndata_to_df(adata, layer=layer)
    for feature in adata.var_names:
        majority_type = df[feature].dropna().apply(type).value_counts().idxmax()
        if majority_type == pd.Timestamp:  # TODO: Check
            feature_types[feature] = DATE_TAG
        elif majority_type not in [int, float, complex]:
            feature_types[feature] = CATEGORICAL_TAG
        # Guess categorical if the feature is an integer and the values are 0/1 to n-1 with no gaps
        elif np.all(i.is_integer() for i in df[feature]) and (
            (df[feature].min() == 0 and df[feature].max() == df[feature].nunique() - 1)
            or (df[feature].min() == 1 and df[feature].max() == df[feature].nunique())
        ):
            feature_types[feature] = CATEGORICAL_TAG
        else:
            feature_types[feature] = CONTINUOUS_TAG

    adata.var[FEATURE_TYPE_KEY] = pd.Series(feature_types)[adata.var_names]


def check_feature_types(func):
    def wrapper(adata, *args, **kwargs):
        if FEATURE_TYPE_KEY not in adata.var.columns:
            raise ValueError("Feature types are not specified in adata.var. Please run `infer_feature_types` first.")
        np.all(adata.var[FEATURE_TYPE_KEY].isin([CATEGORICAL_TAG, CONTINUOUS_TAG, DATE_TAG]))
        func(adata, *args, **kwargs)

    return wrapper
