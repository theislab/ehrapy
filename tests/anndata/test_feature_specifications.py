import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from ehrapy.anndata import check_feature_types, df_to_anndata
from ehrapy.anndata._constants import CATEGORICAL_TAG, DATE_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from tests.conftest import TEST_DATA_PATH

IMPUTATION_DATA_PATH = TEST_DATA_PATH / "imputation"


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
    assert adata.var[FEATURE_TYPE_KEY]["feature4"] == NUMERIC_TAG
    assert adata.var[FEATURE_TYPE_KEY]["feature5"] == CATEGORICAL_TAG
    assert adata.var[FEATURE_TYPE_KEY]["feature6"] == NUMERIC_TAG
    assert adata.var[FEATURE_TYPE_KEY]["feature7"] == DATE_TAG


def test_check_feature_types(adata):
    @check_feature_types
    def test_func(adata):
        pass

    assert FEATURE_TYPE_KEY not in adata.var.keys()
    test_func(adata)
    assert FEATURE_TYPE_KEY in adata.var.keys()

    ep.ad.infer_feature_types(adata, output=None)
    test_func(adata)
    assert FEATURE_TYPE_KEY in adata.var.keys()

    @check_feature_types
    def test_func_with_return(adata):
        return adata

    adata = test_func_with_return(adata)
    assert FEATURE_TYPE_KEY in adata.var.keys()


def test_feature_types_impute_num_adata(impute_num_adata):
    ep.ad.infer_feature_types(impute_num_adata, output=None)
    assert np.all(impute_num_adata.var[FEATURE_TYPE_KEY] == [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG])


def test_feature_types_impute_adata(impute_adata):
    ep.ad.infer_feature_types(impute_adata, output=None)
    assert np.all(impute_adata.var[FEATURE_TYPE_KEY] == [NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG])


def test_feature_types_impute_iris(impute_iris_adata):
    ep.ad.infer_feature_types(impute_iris_adata, output=None)
    assert np.all(
        impute_iris_adata.var[FEATURE_TYPE_KEY] == [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG]
    )


def test_feature_types_impute_feature_types_titanic(impute_titanic_adata):
    ep.ad.infer_feature_types(impute_titanic_adata, output=None)
    impute_titanic_adata.var[FEATURE_TYPE_KEY] = [
        CATEGORICAL_TAG,
        CATEGORICAL_TAG,
        CATEGORICAL_TAG,
        CATEGORICAL_TAG,
        CATEGORICAL_TAG,
        NUMERIC_TAG,
        NUMERIC_TAG,
        NUMERIC_TAG,
        NUMERIC_TAG,
        NUMERIC_TAG,
        CATEGORICAL_TAG,
        CATEGORICAL_TAG,
    ]


def test_date_detection():
    df = pd.DataFrame(
        {
            "date1": pd.to_datetime(["2021-01-01", "2024-04-16", "2021-01-03"]),
            "date2": ["2021-01-01", "2024-04-16", "2021-01-03"],
            "date3": ["2024-04-16 07:45:13", "2024-04-16", "2024-04"],
            "not_date": ["not_a_date", "2024-04-16", "2021-01-03"],
        }
    )
    adata = df_to_anndata(df)
    ep.ad.infer_feature_types(adata, output=None)
    assert np.all(adata.var[FEATURE_TYPE_KEY] == [DATE_TAG, DATE_TAG, DATE_TAG, CATEGORICAL_TAG])


def test_all_possible_types():
    df = pd.DataFrame(
        {
            "f1": [42, 17, 93, 235],
            "f2": ["apple", "banana", "cherry", "date"],
            "f3": [1, 2, 3, 1],
            "f4": [1.0, 2.0, 1.0, 2.0],
            "f5": ["20200101", "20200102", "20200103", "20200104"],
            "f6": [True, False, True, False],
            "f7": [np.nan, 1, np.nan, 2],
            "f8": ["apple", 1, "banana", 2],
            "f9": ["001", "002", "003", "002"],
            "f10": ["5", "5", "5", "5"],
            "f11": ["A1", "A2", "B1", "B2"],
            "f12": [90210, 10001, 60614, 80588],
            "f13": [0.25, 0.5, 0.75, 1.0],
            "f14": ["2125551234", "2125555678", "2125559012", "2125553456"],
            "f15": ["$100", "€150", "£200", "¥250"],
            "f16": [101, 102, 103, 104],
            "f17": [1e3, 5e-2, 3.1e2, 2.7e-1],
            "f18": ["23.5", "324", "4.5", "0.5"],
            "f19": [1, 2, 3, 4],
            "f20": ["001", "002", "003", "004"],
        }
    )

    adata = df_to_anndata(df)
    ep.ad.infer_feature_types(adata, output=None)

    assert np.all(
        adata.var[FEATURE_TYPE_KEY]
        == [
            NUMERIC_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            DATE_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            CATEGORICAL_TAG,
            NUMERIC_TAG,
            CATEGORICAL_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
            CATEGORICAL_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
            NUMERIC_TAG,
        ]
    )


def test_partial_annotation(adata):
    adata.var[FEATURE_TYPE_KEY] = ["dummy", np.nan, np.nan, NUMERIC_TAG, None, np.nan, None]
    ep.ad.infer_feature_types(adata, output=None)
    assert np.all(
        adata.var[FEATURE_TYPE_KEY]
        == ["dummy", CATEGORICAL_TAG, CATEGORICAL_TAG, NUMERIC_TAG, CATEGORICAL_TAG, NUMERIC_TAG, DATE_TAG]
    )
