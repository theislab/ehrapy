import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import CATEGORICAL_TAG, DATE_TAG, DEFAULT_TEM_LAYER_NAME, FEATURE_TYPE_KEY, NUMERIC_TAG

import ehrapy as ep
from ehrapy.anndata import _check_feature_types, df_to_anndata
from tests.conftest import TEST_DATA_PATH

IMPUTATION_DATA_PATH = TEST_DATA_PATH / "imputation"


def test_feature_type_inference(edata_feature_type_specifications):
    edata = edata_feature_type_specifications
    ep.ad.infer_feature_types(edata, output=None)
    assert edata.var[FEATURE_TYPE_KEY]["feature1"] == CATEGORICAL_TAG
    assert edata.var[FEATURE_TYPE_KEY]["feature2"] == CATEGORICAL_TAG
    assert edata.var[FEATURE_TYPE_KEY]["feature3"] == CATEGORICAL_TAG
    assert edata.var[FEATURE_TYPE_KEY]["feature4"] == NUMERIC_TAG
    assert edata.var[FEATURE_TYPE_KEY]["feature5"] == CATEGORICAL_TAG
    assert edata.var[FEATURE_TYPE_KEY]["feature6"] == NUMERIC_TAG
    assert edata.var[FEATURE_TYPE_KEY]["feature7"] == DATE_TAG


def test_check_feature_types(edata_feature_type_specifications):
    edata = edata_feature_type_specifications

    @_check_feature_types
    def test_func(edata):
        pass

    assert FEATURE_TYPE_KEY not in edata.var.keys()
    test_func(edata)
    assert FEATURE_TYPE_KEY in edata.var.keys()

    ep.ad.infer_feature_types(edata, output=None)
    test_func(edata)
    assert FEATURE_TYPE_KEY in edata.var.keys()

    @_check_feature_types
    def test_func_with_return(edata):
        return edata

    edata = test_func_with_return(edata)
    assert FEATURE_TYPE_KEY in edata.var.keys()


def test_feature_types_impute_num_edata(impute_num_edata):
    ep.ad.infer_feature_types(impute_num_edata, output=None)
    assert np.all(impute_num_edata.var[FEATURE_TYPE_KEY] == [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG])


def test_feature_types_impute_edata(impute_edata):
    ep.ad.infer_feature_types(impute_edata, output=None)
    assert np.all(
        impute_edata.var[FEATURE_TYPE_KEY] == [NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, DATE_TAG, CATEGORICAL_TAG]
    )


def test_feature_types_impute_iris(impute_iris_edata):
    ep.ad.infer_feature_types(impute_iris_edata, output=None)
    assert np.all(
        impute_iris_edata.var[FEATURE_TYPE_KEY] == [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG]
    )


def test_feature_types_impute_feature_types_titanic(impute_titanic_edata):
    ep.ad.infer_feature_types(impute_titanic_edata, output=None)
    impute_titanic_edata.var[FEATURE_TYPE_KEY] = [
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


def test_partial_annotation(edata_feature_type_specifications):
    edata_feature_type_specifications.var[FEATURE_TYPE_KEY] = ["dummy", np.nan, np.nan, NUMERIC_TAG, None, np.nan, None]
    ep.ad.infer_feature_types(edata_feature_type_specifications, output=None)
    assert np.all(
        edata_feature_type_specifications.var[FEATURE_TYPE_KEY]
        == ["dummy", CATEGORICAL_TAG, CATEGORICAL_TAG, NUMERIC_TAG, CATEGORICAL_TAG, NUMERIC_TAG, DATE_TAG]
    )


def test_infer_feature_types_3D(edata_blob_small):
    ep.ad.infer_feature_types(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.ad.infer_feature_types(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)
