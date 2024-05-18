from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pandas import DataFrame
from pandas.testing import assert_frame_equal

import ehrapy as ep
from ehrapy.anndata._constants import CATEGORICAL_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from ehrapy.anndata.anndata_ext import (
    NotEncodedError,
    _assert_encoded,
    anndata_to_df,
    assert_numeric_vars,
    delete_from_obs,
    df_to_anndata,
    get_numeric_vars,
    move_to_obs,
    move_to_x,
    set_numeric_vars,
)

CUR_DIR = Path(__file__).parent.resolve()


@pytest.fixture
def adata_move_obs_num() -> AnnData:
    return ep.io.read_csv(CUR_DIR / "../io/test_data_io/dataset_move_obs_num.csv")


@pytest.fixture
def adata_move_obs_mix() -> AnnData:
    return ep.io.read_csv(CUR_DIR / "../io/test_data_io/dataset_move_obs_mix.csv")


@pytest.fixture
def setup_df_to_anndata() -> tuple[DataFrame, list, list, list]:
    col1_val = ["str" + str(idx) for idx in range(100)]
    col2_val = ["another_str" + str(idx) for idx in range(100)]
    col3_val = list(range(100))
    df = DataFrame({"col1": col1_val, "col2": col2_val, "col3": col3_val})

    return df, col1_val, col2_val, col3_val


@pytest.fixture
def setup_binary_df_to_anndata() -> DataFrame:
    col1_val = ["str" + str(idx) for idx in range(100)]
    col2_val = ["another_str" + str(idx) for idx in range(100)]
    col3_val = [0 for _ in range(100)]
    col4_val = [1.0 for _ in range(100)]
    col5_val = [0.0 if idx % 2 == 0 else np.NaN for idx in range(100)]
    col6_val = [idx % 2 for idx in range(100)]
    col7_val = [float(idx % 2) for idx in range(100)]
    col8_val = [idx % 3 if idx % 3 in {0, 1} else np.NaN for idx in range(100)]
    df = DataFrame(
        {
            "col1": col1_val,
            "col2": col2_val,
            "col3": col3_val,
            "col4": col4_val,
            "col5": col5_val,
            "col6_binary_int": col6_val,
            "col7_binary_float": col7_val,
            "col8_binary_missing_values": col8_val,
        }
    )

    return df


@pytest.fixture
def setup_anndata_to_df() -> tuple[list, list, list]:
    col1_val = ["patient" + str(idx) for idx in range(100)]
    col2_val = ["feature" + str(idx) for idx in range(100)]
    col3_val = list(range(100))

    return col1_val, col2_val, col3_val


def test_move_to_obs_only_num(adata_move_obs_num: AnnData):
    move_to_obs(adata_move_obs_num, ["los_days", "b12_values"])
    assert list(adata_move_obs_num.obs.columns) == ["los_days", "b12_values"]
    assert {str(col) for col in adata_move_obs_num.obs.dtypes} == {"float32"}
    assert_frame_equal(
        adata_move_obs_num.obs,
        DataFrame(
            {"los_days": [14.0, 7.0, 10.0, 11.0, 3.0], "b12_values": [500.0, 330.0, 800.0, 765.0, 800.0]},
            index=[str(idx) for idx in range(5)],
        ).astype({"b12_values": "float32", "los_days": "float32"}),
    )


def test_move_to_obs_mixed(adata_move_obs_mix: AnnData):
    move_to_obs(adata_move_obs_mix, ["name", "clinic_id"])
    assert set(adata_move_obs_mix.obs.columns) == {"name", "clinic_id"}
    assert {str(col) for col in adata_move_obs_mix.obs.dtypes} == {"float32", "category"}
    assert_frame_equal(
        adata_move_obs_mix.obs,
        DataFrame(
            {"clinic_id": list(range(1, 6)), "name": ["foo", "bar", "baz", "buz", "ber"]},
            index=[str(idx) for idx in range(5)],
        ).astype({"clinic_id": "float32", "name": "category"}),
    )


def test_move_to_obs_copy_obs(adata_move_obs_mix: AnnData):
    adata_dim_old = adata_move_obs_mix.X.shape
    move_to_obs(adata_move_obs_mix, ["name", "clinic_id"], copy_obs=True)
    assert set(adata_move_obs_mix.obs.columns) == {"name", "clinic_id"}
    assert adata_move_obs_mix.X.shape == adata_dim_old
    assert {str(col) for col in adata_move_obs_mix.obs.dtypes} == {"float32", "category"}
    assert_frame_equal(
        adata_move_obs_mix.obs,
        DataFrame(
            {"clinic_id": list(range(1, 6)), "name": ["foo", "bar", "baz", "buz", "ber"]},
            index=[str(idx) for idx in range(5)],
        ).astype({"clinic_id": "float32", "name": "category"}),
    )


def test_move_to_obs_invalid_column_name(adata_move_obs_mix: AnnData):
    with pytest.raises(ValueError):
        _ = move_to_obs(adata_move_obs_mix, "nam")
        _ = move_to_obs(adata_move_obs_mix, "clic_id")
        _ = move_to_obs(adata_move_obs_mix, ["nam", "clic_id"])


def test_move_to_x(adata_move_obs_mix):
    move_to_obs(adata_move_obs_mix, ["name"], copy_obs=True)
    move_to_obs(adata_move_obs_mix, ["clinic_id"], copy_obs=False)
    new_adata_non_num = move_to_x(adata_move_obs_mix, ["name"])
    new_adata_num = move_to_x(adata_move_obs_mix, ["clinic_id"])
    assert set(new_adata_non_num.obs.columns) == {"name", "clinic_id"}
    assert set(new_adata_num.obs.columns) == {"name"}
    assert {str(col) for col in new_adata_num.obs.dtypes} == {"category"}
    assert {str(col) for col in new_adata_non_num.obs.dtypes} == {"float32", "category"}

    assert_frame_equal(
        new_adata_non_num.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG]},
            index=["los_days", "b12_values", "name"],
        ),
    )

    assert_frame_equal(
        new_adata_num.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, np.nan]},
            index=["los_days", "b12_values", "name", "clinic_id"],
        ),
    )
    ep.ad.infer_feature_types(new_adata_num, output=None)
    assert np.all(new_adata_num.var[FEATURE_TYPE_KEY] == [NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, NUMERIC_TAG])

    assert_frame_equal(
        new_adata_num.obs,
        DataFrame(
            {"name": ["foo", "bar", "baz", "buz", "ber"]},
            index=[str(idx) for idx in range(5)],
        ).astype({"name": "category"}),
    )

    assert_frame_equal(
        new_adata_non_num.obs,
        DataFrame(
            {"name": ["foo", "bar", "baz", "buz", "ber"], "clinic_id": list(range(1, 6))},
            index=[str(idx) for idx in range(5)],
        ).astype({"clinic_id": "float32", "name": "category"}),
    )


def test_move_to_x_invalid_column_names(adata_move_obs_mix):
    move_to_obs(adata_move_obs_mix, ["name"], copy_obs=True)
    move_to_obs(adata_move_obs_mix, ["clinic_id"], copy_obs=False)
    with pytest.raises(ValueError):
        _ = move_to_x(adata_move_obs_mix, ["blabla1"])
        _ = move_to_x(adata_move_obs_mix, ["blabla1", "blabla2"])


def test_move_to_x_move_to_obs(adata_move_obs_mix):
    adata_dim_old = adata_move_obs_mix.X.shape
    # moving columns from X to obs and back
    # case 1:  move some column from obs to X and this col was copied previously from X to obs
    move_to_obs(adata_move_obs_mix, ["name"], copy_obs=True)
    adata = move_to_x(adata_move_obs_mix, ["name"])
    assert {"name"}.issubset(set(adata.var_names))
    assert adata.X.shape == adata_dim_old
    delete_from_obs(adata, ["name"])

    # case 2: move some column from obs to X and this col was previously moved inplace from X to obs
    move_to_obs(adata, ["clinic_id"], copy_obs=False)
    adata = move_to_x(adata, ["clinic_id"])
    assert not {"clinic_id"}.issubset(set(adata.obs.columns))
    assert {"clinic_id"}.issubset(set(adata.var_names))
    assert adata.X.shape == adata_dim_old

    # case 3: move multiple columns from obs to X and some of them were copied or moved inplace previously from X to obs
    move_to_obs(adata, ["los_days"], copy_obs=True)
    move_to_obs(adata, ["b12_values"], copy_obs=False)
    adata = move_to_x(adata, ["los_days", "b12_values"])
    delete_from_obs(adata, ["los_days"])
    assert not {"los_days"}.issubset(
        set(adata.obs.columns)
    )  # check if the copied column was removed from obs by delete_from_obs()
    assert not {"b12_values"}.issubset(set(adata.obs.columns))
    assert {"los_days", "b12_values"}.issubset(set(adata.var_names))
    assert adata.X.shape == adata_dim_old


def test_delete_from_obs(adata_move_obs_mix):
    adata = move_to_obs(adata_move_obs_mix, ["los_days"], copy_obs=True)
    adata = delete_from_obs(adata, ["los_days"])
    assert not {"los_days"}.issubset(set(adata.obs.columns))
    assert {"los_days"}.issubset(set(adata.var_names))


def test_df_to_anndata_simple(setup_df_to_anndata):
    df, col1_val, col2_val, col3_val = setup_df_to_anndata
    expected_x = np.array([col1_val, col2_val, col3_val], dtype="object").transpose()
    adata = df_to_anndata(df)

    assert adata.X.dtype == "object"
    assert adata.X.shape == (100, 3)
    np.testing.assert_array_equal(adata.X, expected_x)


def test_df_to_anndata_index_column(setup_df_to_anndata):
    df, col1_val, col2_val, col3_val = setup_df_to_anndata
    expected_x = np.array([col2_val, col3_val], dtype="object").transpose()
    adata = df_to_anndata(df, index_column="col1")

    assert adata.X.dtype == "object"
    assert adata.X.shape == (100, 2)
    np.testing.assert_array_equal(adata.X, expected_x)
    assert list(adata.obs.index) == col1_val
    assert adata.obs.index.name == "col1"


def test_df_to_anndata_index_column_num(setup_df_to_anndata):
    df, col1_val, col2_val, col3_val = setup_df_to_anndata
    expected_x = np.array([col2_val, col3_val], dtype="object").transpose()
    adata = df_to_anndata(df, index_column=0)

    assert adata.X.dtype == "object"
    assert adata.X.shape == (100, 2)
    np.testing.assert_array_equal(adata.X, expected_x)
    assert list(adata.obs.index) == col1_val
    assert adata.obs.index.name == "col1"


def test_df_to_anndata_index_column_index():
    d = {"col1": [0, 1, 2, 3], "col2": pd.Series([2, 3])}
    df = pd.DataFrame(data=d, index=[0, 1, 2, 3])
    df.index.set_names("quarter", inplace=True)
    adata = ep.ad.df_to_anndata(df, index_column="quarter")
    assert adata.obs.index.name == "quarter"
    assert list(adata.obs.index) == ["0", "1", "2", "3"]


def test_df_to_anndata_invalid_index_throws_error(setup_df_to_anndata):
    df, col1_val, col2_val, col3_val = setup_df_to_anndata
    with pytest.raises(ValueError):
        _ = df_to_anndata(df, index_column="UnknownCol")


def test_df_to_anndata_cols_obs_only(setup_df_to_anndata):
    df, col1_val, col2_val, col3_val = setup_df_to_anndata
    adata = df_to_anndata(df, columns_obs_only=["col1", "col2"])
    assert adata.X.dtype == "float64"
    assert adata.X.shape == (100, 1)
    assert_frame_equal(
        adata.obs,
        DataFrame({"col1": col1_val, "col2": col2_val}, index=[str(idx) for idx in range(100)]).astype("category"),
    )


def test_df_to_anndata_all_num():
    test_array = np.random.randint(0, 100, (4, 5))
    df = DataFrame(test_array, columns=["col" + str(idx) for idx in range(5)])
    adata = df_to_anndata(df)

    assert adata.X.dtype == "float64"
    np.testing.assert_array_equal(test_array, adata.X)


def test_anndata_to_df_simple(setup_anndata_to_df):
    col1_val, col2_val, col3_val = setup_anndata_to_df
    expected_df = DataFrame({"col1": col1_val, "col2": col2_val, "col3": col3_val}, dtype="object")
    adata_x = np.array([col1_val, col2_val, col3_val], dtype="object").transpose()
    adata = AnnData(
        X=adata_x,
        obs=DataFrame(index=list(range(100))),
        var=DataFrame(index=["col" + str(idx) for idx in range(1, 4)]),
    )
    anndata_df = anndata_to_df(adata)

    assert_frame_equal(anndata_df, expected_df)


def test_anndata_to_df_all_from_obs(setup_anndata_to_df):
    col1_val, col2_val, col3_val = setup_anndata_to_df
    expected_df = DataFrame({"col1": col1_val, "col2": col2_val, "col3": col3_val})
    obs = DataFrame({"col2": col2_val, "col3": col3_val})
    adata_x = np.array([col1_val], dtype="object").transpose()
    adata = AnnData(X=adata_x, obs=obs, var=DataFrame(index=["col1"]))
    anndata_df = anndata_to_df(adata, obs_cols=list(adata.obs.columns))

    assert_frame_equal(anndata_df, expected_df)


def test_anndata_to_df_some_from_obs(setup_anndata_to_df):
    col1_val, col2_val, col3_val = setup_anndata_to_df
    expected_df = DataFrame({"col1": col1_val, "col3": col3_val})
    obs = DataFrame({"col2": col2_val, "col3": col3_val})
    adata_x = np.array([col1_val], dtype="object").transpose()
    adata = AnnData(X=adata_x, obs=obs, var=DataFrame(index=["col1"]))
    anndata_df = anndata_to_df(adata, obs_cols=["col3"])

    assert_frame_equal(anndata_df, expected_df)


def test_anndata_to_df_throws_error_with_empty_obs():
    col1_val = ["patient" + str(idx) for idx in range(100)]
    adata_x = np.array([col1_val], dtype="object").transpose()
    adata = AnnData(X=adata_x, obs=DataFrame(index=list(range(100))), var=DataFrame(index=["col1"]))

    with pytest.raises(ValueError):
        _ = anndata_to_df(adata, obs_cols=["some_missing_column"])


def test_anndata_to_df_all_columns(setup_anndata_to_df):
    col1_val, col2_val, col3_val = setup_anndata_to_df
    expected_df = DataFrame({"col1": col1_val})
    var = DataFrame(index=["col1"])
    adata_x = np.array([col1_val], dtype="object").transpose()
    adata = AnnData(X=adata_x, obs=DataFrame({"col2": col2_val, "col3": col3_val}), var=var)
    anndata_df = anndata_to_df(adata, obs_cols=list(adata.var.columns))

    assert_frame_equal(anndata_df, expected_df)


def test_anndata_to_df_layers(setup_anndata_to_df):
    col1_val, col2_val, col3_val = setup_anndata_to_df
    expected_df = DataFrame({"col1": col1_val, "col2": col2_val, "col3": col3_val})
    obs = DataFrame({"col2": col2_val, "col3": col3_val})
    adata_x = np.array([col1_val], dtype="object").transpose()
    adata = AnnData(X=adata_x, obs=obs, var=DataFrame(index=["col1"]), layers={"raw": adata_x.copy()})
    anndata_df = anndata_to_df(adata, obs_cols=list(adata.obs.columns), layer="raw")

    assert_frame_equal(anndata_df, expected_df)


def test_detect_binary_columns(setup_binary_df_to_anndata):
    adata = df_to_anndata(setup_binary_df_to_anndata)
    ep.ad.infer_feature_types(adata, output=None)

    assert_frame_equal(
        adata.var,
        DataFrame(
            {
                FEATURE_TYPE_KEY: [
                    CATEGORICAL_TAG,
                    CATEGORICAL_TAG,
                    CATEGORICAL_TAG,
                    CATEGORICAL_TAG,
                    CATEGORICAL_TAG,
                    CATEGORICAL_TAG,
                    CATEGORICAL_TAG,
                    CATEGORICAL_TAG,
                ]
            },
            index=[
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col6_binary_int",
                "col7_binary_float",
                "col8_binary_missing_values",
            ],
        ),
    )


def test_detect_mixed_binary_columns():
    df = pd.DataFrame(
        {"Col1": list(range(4)), "Col2": ["str" + str(i) for i in range(4)], "Col3": [1.0, 0.0, np.nan, 1.0]}
    )
    adata = ep.ad.df_to_anndata(df)
    ep.ad.infer_feature_types(adata, output=None)

    assert_frame_equal(
        adata.var,
        DataFrame(
            {FEATURE_TYPE_KEY: [NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG]},
            index=["Col1", "Col2", "Col3"],
        ),
    )


@pytest.fixture
def adata_strings_encoded():
    obs_data = {"ID": ["Patient1", "Patient2", "Patient3"], "Age": [31, 94, 62]}
    X_strings = np.array(
        [
            [1, 3.4, "A string", "A different string"],
            [2, 5.4, "Silly string", "A different string"],
            [2, 5.7, "A string", "What string?"],
        ],
        dtype=pd.StringDtype,
    )
    var_strings = {
        "Feature": ["Numeric1", "Numeric2", "String1", "String2"],
        "Type": ["Numeric", "Numeric", "String", "String"],
    }

    adata_strings = AnnData(
        X=X_strings,
        obs=pd.DataFrame(data=obs_data),
        var=pd.DataFrame(data=var_strings, index=var_strings["Feature"]),
    )
    adata_strings.var[FEATURE_TYPE_KEY] = [NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG]

    adata_encoded = ep.pp.encode(adata_strings.copy(), autodetect=True, encodings="label")

    return adata_strings, adata_encoded


@pytest.fixture
def adata_encoded(adata_strings):
    return ep.pp.encode(adata_strings.copy(), autodetect=True, encodings="label")


def test_assert_encoded(adata_strings_encoded):
    adata_strings, adata_encoded = adata_strings_encoded
    _assert_encoded(adata_encoded)
    with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
        _assert_encoded(adata_strings)


def test_get_numeric_vars(adata_strings_encoded):
    adata_strings, adata_encoded = adata_strings_encoded
    vars = get_numeric_vars(adata_encoded)
    assert vars == ["Numeric1", "Numeric2"]
    with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
        get_numeric_vars(adata_strings)


def test_get_numeric_vars_numeric_only():
    adata = AnnData(X=np.array([[1, 2, 3], [4, 0, 6]], dtype=np.float32))
    vars = get_numeric_vars(adata)
    assert vars == ["0", "1", "2"]


def test_assert_numeric_vars(adata_strings_encoded):
    adata_strings, adata_encoded = adata_strings_encoded
    assert_numeric_vars(adata_encoded, ["Numeric1", "Numeric2"])
    with pytest.raises(ValueError, match=r"Some selected vars are not numeric"):
        assert_numeric_vars(adata_encoded, ["Numeric2", "String1"])


def test_set_numeric_vars(adata_strings_encoded):
    """Test for the numeric vars setter."""
    adata_strings, adata_encoded = adata_strings_encoded
    values = np.array(
        [[1.2, 2.2], [3.2, 4.2], [5.2, 6.2]],
        dtype=np.dtype(np.float32),
    )
    adata_set = set_numeric_vars(adata_encoded, values, copy=True)
    np.testing.assert_array_equal(adata_set.X[:, 2], values[:, 0]) and np.testing.assert_array_equal(
        adata_set.X[:, 3], values[:, 1]
    )

    with pytest.raises(ValueError, match=r"Some selected vars are not numeric"):
        set_numeric_vars(adata_encoded, values, vars=["ehrapycat_String1"])

    string_values = np.array(
        [
            ["A"],
            ["B"],
            ["A"],
        ]
    )

    with pytest.raises(TypeError, match=r"Values must be numeric"):
        set_numeric_vars(adata_encoded, string_values)

    extra_values = np.array(
        [
            [1.2, 1.3, 1.4],
            [2.2, 2.3, 2.4],
            [2.2, 2.3, 2.4],
        ],
        dtype=np.dtype(np.float32),
    )

    with pytest.raises(ValueError, match=r"does not match number of vars"):
        set_numeric_vars(adata_encoded, extra_values)

    with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
        set_numeric_vars(adata_strings, values)
