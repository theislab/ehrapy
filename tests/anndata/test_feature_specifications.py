from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from ehrapy.anndata import check_feature_types, df_to_anndata
from ehrapy.anndata._constants import CATEGORICAL_TAG, CONTINUOUS_TAG, DATE_TAG, FEATURE_TYPE_KEY
from ehrapy.io._read import read_csv

_TEST_PATH = f"{Path(__file__).parents[1]}/preprocessing/test_data_imputation"


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
    ep.ad.infer_feature_types(adata, output=None)
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

    ep.ad.infer_feature_types(adata, output=None)
    test_func(adata)
    assert FEATURE_TYPE_KEY in adata.var.keys()

    @check_feature_types
    def test_func_with_return(adata):
        return adata

    adata = test_func_with_return(adata)
    assert FEATURE_TYPE_KEY in adata.var.keys()


def test_feature_types_impute_num_adata():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_num.csv")
    ep.ad.infer_feature_types(adata, output=None)
    assert np.all(adata.var[FEATURE_TYPE_KEY] == [CONTINUOUS_TAG, CONTINUOUS_TAG, CONTINUOUS_TAG])
    return adata


def test_feature_types_impute_adata():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute.csv")
    ep.ad.infer_feature_types(adata, output=None)
    assert np.all(adata.var[FEATURE_TYPE_KEY] == [CATEGORICAL_TAG, CONTINUOUS_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG])


def test_feature_types_impute_iris():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_iris.csv")
    ep.ad.infer_feature_types(adata, output=None)
    assert np.all(
        adata.var[FEATURE_TYPE_KEY] == [CONTINUOUS_TAG, CONTINUOUS_TAG, CONTINUOUS_TAG, CONTINUOUS_TAG, CATEGORICAL_TAG]
    )


def test_feature_types_impute_feature_types_titanic():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/test_impute_titanic.csv")
    ep.ad.infer_feature_types(adata, output=None)
    adata.var[FEATURE_TYPE_KEY] = [
        CATEGORICAL_TAG,
        CATEGORICAL_TAG,
        CATEGORICAL_TAG,
        CATEGORICAL_TAG,
        CATEGORICAL_TAG,
        CONTINUOUS_TAG,
        CONTINUOUS_TAG,
        CONTINUOUS_TAG,
        CONTINUOUS_TAG,
        CONTINUOUS_TAG,
        CATEGORICAL_TAG,
        CATEGORICAL_TAG,
    ]
