import numpy as np
import pandas as pd

import ehrapy as ep
from ehrapy.anndata._constants import CATEGORICAL_TAG, CONTINUOUS_TAG, FEATURE_TYPE_KEY
from ehrapy.anndata.anndata_ext import df_to_anndata


def test_feature_type_inference():
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 2, 0],
            "feature2": ["a", "b", "c", "d"],
            "feature3": [1.0, 2.0, 3.0, 2.0],
            "feature4": [0.0, 0.3, 0.5, 4.6],
            "feature5": ["a", "b", np.nan, "d"],
            "feature6": [1.4, 0.2, np.nan, np.nan],
        }
    )
    adata = df_to_anndata(df)

    ep.ad.infer_feature_types(adata)
    assert all(
        adata.var[FEATURE_TYPE_KEY]
        == [CATEGORICAL_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG, CONTINUOUS_TAG, CATEGORICAL_TAG, CONTINUOUS_TAG]
    )
