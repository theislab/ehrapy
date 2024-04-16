import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from ehrapy.anndata import check_feature_types, df_to_anndata
from ehrapy.anndata._constants import CATEGORICAL_TAG, CONTINUOUS_TAG, DATE_TAG, FEATURE_TYPE_KEY


@pytest.fixture
def adata():
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 2, 0],
            "feature2": ["a", "b", "c", "d"],
            "feature3": [1.0, 2.0, 3.0, 2.0],
            "feature4": [0.0, 0.3, 0.5, 4.6],
            "feature5": ["a", "b", np.nan, "d"],
            "feature6": [1.4, 0.2, np.nan, np.nan],
            "feature7": pd.to_datetime(["2021-01-01", "2024-04-16", "2021-01-03", "2067-07-02"]),
        }
    )
    adata = df_to_anndata(df)
    return adata


def test_feature_type_inference(adata):
    ep.ad.infer_feature_types(adata)
    assert adata.var[FEATURE_TYPE_KEY]["feature1"] == CATEGORICAL_TAG
    assert adata.var[FEATURE_TYPE_KEY]["feature2"] == CATEGORICAL_TAG
    assert adata.var[FEATURE_TYPE_KEY]["feature3"] == CATEGORICAL_TAG
    assert adata.var[FEATURE_TYPE_KEY]["feature4"] == CONTINUOUS_TAG
    assert adata.var[FEATURE_TYPE_KEY]["feature5"] == CATEGORICAL_TAG
    assert adata.var[FEATURE_TYPE_KEY]["feature6"] == CONTINUOUS_TAG
    assert adata.var[FEATURE_TYPE_KEY]["feature7"] == DATE_TAG


def test_check_feature_types(adata):
    @check_feature_types
    def test_func(adata):
        pass

    with pytest.raises(ValueError) as e:
        test_func(adata)
    assert str(e.value).startswith("Feature types are not specified in adata.var.")

    ep.ad.infer_feature_types(adata)
    test_func(adata)
    assert FEATURE_TYPE_KEY in adata.var.keys()
