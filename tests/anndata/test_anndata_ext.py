from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pandas import DataFrame
from pandas.testing import assert_frame_equal

import ehrapy as ep
from ehrapy.anndata.anndata_ext import (
    IndexNotFoundError,
    NotEncodedError,
    ObsEmptyError,
    _assert_encoded,
    anndata_to_df,
    assert_numeric_vars,
    delete_from_obs,
    df_to_anndata,
    generate_anndata,
    get_numeric_vars,
    move_to_obs,
    move_to_x,
    set_numeric_vars,
)

CUR_DIR = Path(__file__).parent.resolve()


class TestAnndataExt:
    def test_move_to_obs_only_num(self):
        adata = ep.io.read_csv(CUR_DIR / "../io/test_data_io/dataset_move_obs_num.csv")
        move_to_obs(adata, ["los_days", "b12_values"])
        assert list(adata.obs.columns) == ["los_days", "b12_values"]
        assert {str(col) for col in adata.obs.dtypes} == {"float32"}
        assert_frame_equal(
            adata.obs,
            DataFrame(
                {"los_days": [14.0, 7.0, 10.0, 11.0, 3.0], "b12_values": [500.0, 330.0, 800.0, 765.0, 800.0]},
                index=[str(idx) for idx in range(5)],
            ).astype({"b12_values": "float32", "los_days": "float32"}),
        )

    def test_move_to_obs_mixed(self):
        adata = ep.io.read_csv(CUR_DIR / "../io/test_data_io/dataset_move_obs_mix.csv")
        move_to_obs(adata, ["name", "clinic_id"])
        assert set(adata.obs.columns) == {"name", "clinic_id"}
        assert {str(col) for col in adata.obs.dtypes} == {"float32", "category"}
        assert_frame_equal(
            adata.obs,
            DataFrame(
                {"clinic_id": [i for i in range(1, 6)], "name": ["foo", "bar", "baz", "buz", "ber"]},
                index=[str(idx) for idx in range(5)],
            ).astype({"clinic_id": "float32", "name": "category"}),
        )

    def test_move_to_obs_copy_obs(self):
        adata = ep.io.read_csv(CUR_DIR / "../io/test_data_io/dataset_move_obs_mix.csv")
        adata_dim_old = adata.X.shape
        move_to_obs(adata, ["name", "clinic_id"], copy_obs=True)
        assert set(adata.obs.columns) == {"name", "clinic_id"}
        assert adata.X.shape == adata_dim_old
        assert {str(col) for col in adata.obs.dtypes} == {"float32", "category"}
        assert_frame_equal(
            adata.obs,
            DataFrame(
                {"clinic_id": [i for i in range(1, 6)], "name": ["foo", "bar", "baz", "buz", "ber"]},
                index=[str(idx) for idx in range(5)],
            ).astype({"clinic_id": "float32", "name": "category"}),
        )

    def test_move_to_obs_invalid_column_name(self):
        adata = ep.io.read_csv(CUR_DIR / "../io/test_data_io/dataset_move_obs_mix.csv")
        with pytest.raises(ValueError):
            _ = move_to_obs(adata, "nam")
            _ = move_to_obs(adata, "clic_id")
            _ = move_to_obs(adata, ["nam", "clic_id"])

    def test_move_to_x(self):
        adata = ep.io.read_csv(CUR_DIR / "../io/test_data_io/dataset_move_obs_mix.csv")
        move_to_obs(adata, ["name"], copy_obs=True)
        move_to_obs(adata, ["clinic_id"], copy_obs=False)
        new_adata_non_num = move_to_x(adata, ["name"])
        new_adata_num = move_to_x(adata, ["clinic_id"])
        assert set(new_adata_non_num.obs.columns) == {"name", "clinic_id"}
        assert set(new_adata_num.obs.columns) == {"name"}
        assert {str(col) for col in new_adata_num.obs.dtypes} == {"category"}
        assert {str(col) for col in new_adata_non_num.obs.dtypes} == {"float32", "category"}
        assert len(sum(list(new_adata_num.uns.values()), [])) == len(list(new_adata_num.var_names))
        assert len(sum(list(new_adata_non_num.uns.values()), [])) == len(list(new_adata_non_num.var_names))
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
                {"name": ["foo", "bar", "baz", "buz", "ber"], "clinic_id": [i for i in range(1, 6)]},
                index=[str(idx) for idx in range(5)],
            ).astype({"clinic_id": "float32", "name": "category"}),
        )

    def test_move_to_x_invalid_column_names(self):
        adata = ep.io.read_csv(CUR_DIR / "../io/test_data_io/dataset_move_obs_mix.csv")
        move_to_obs(adata, ["name"], copy_obs=True)
        move_to_obs(adata, ["clinic_id"], copy_obs=False)
        with pytest.raises(ValueError):
            _ = move_to_x(adata, ["blabla1"])
            _ = move_to_x(adata, ["blabla1", "blabla2"])

    def test_move_to_x_move_to_obs(self):
        adata = ep.io.read_csv(CUR_DIR / "../io/test_data_io/dataset_move_obs_mix.csv")
        adata_dim_old = adata.X.shape
        # moving columns from X to obs and back
        # case 1:  move some column from obs to X and this col was copied previously from X to obs
        move_to_obs(adata, ["name"], copy_obs=True)
        adata = move_to_x(adata, ["name"], copy=False)
        assert {"name"}.issubset(set(adata.var_names))  # check if the copied column is still in X
        assert adata.X.shape == adata_dim_old  # the shape of X should be the same as previously
        assert "name" in [item for sublist in adata.uns.values() for item in sublist]  # check if the column in in uns
        delete_from_obs(adata, ["name"])  # delete the column from obs to restore the original adata state

        # case 2: move some column from obs to X and this col was previously moved inplace from X to obs
        move_to_obs(adata, ["clinic_id"], copy_obs=False)
        adata = move_to_x(adata, ["clinic_id"], copy=False)
        assert not {"clinic_id"}.issubset(set(adata.obs.columns))  # check if the copied column was removed from obs
        assert {"clinic_id"}.issubset(set(adata.var_names))  # check if the copied column is now in X
        assert adata.X.shape == adata_dim_old  # the shape of X should be the same as previously
        assert "clinic_id" in [
            item for sublist in adata.uns.values() for item in sublist
        ]  # check if the column in in uns

        # case 3: move multiple columns from obs to X and some of them were copied or moved inplace previously from X to obs
        move_to_obs(adata, ["los_days"], copy_obs=True)
        move_to_obs(adata, ["b12_values"], copy_obs=False)
        adata = move_to_x(adata, ["los_days", "b12_values"], copy=False)
        delete_from_obs(adata, ["los_days"])
        assert not {"los_days"}.issubset(
            set(adata.obs.columns)
        )  # check if the copied column was removed from obs by delete_from_obs()
        assert not {"b12_values"}.issubset(set(adata.obs.columns))  # check if the moved column was removed from obs
        assert {"los_days", "b12_values"}.issubset(set(adata.var_names))  # check if the copied column is now in X
        assert adata.X.shape == adata_dim_old  # the shape of X should be the same as previously
        assert {"los_days", "b12_values"}.issubset(
            {item for sublist in adata.uns.values() for item in sublist}
        )  # check if the column in in uns

    def test_delete_from_obs(self):
        adata = ep.io.read_csv(CUR_DIR / "../io/test_data_io/dataset_move_obs_mix.csv")
        adata = move_to_obs(adata, ["los_days"], copy_obs=True)
        adata = delete_from_obs(adata, ["los_days"])
        assert not {"los_days"}.issubset(set(adata.obs.columns))
        assert {"los_days"}.issubset(set(adata.var_names))
        assert {"los_days"}.issubset({item for sublist in adata.uns.values() for item in sublist})

    def test_df_to_anndata_simple(self):
        df, col1_val, col2_val, col3_val = TestAnndataExt._setup_df_to_anndata()
        expected_x = np.array([col1_val, col2_val, col3_val], dtype="object").transpose()
        adata = df_to_anndata(df)

        assert adata.X.dtype == "object"
        assert adata.X.shape == (100, 3)
        np.testing.assert_array_equal(adata.X, expected_x)

    def test_df_to_anndata_index_column(self):
        df, col1_val, col2_val, col3_val = TestAnndataExt._setup_df_to_anndata()
        expected_x = np.array([col2_val, col3_val], dtype="object").transpose()
        adata = df_to_anndata(df, index_column="col1")

        assert adata.X.dtype == "object"
        assert adata.X.shape == (100, 2)
        np.testing.assert_array_equal(adata.X, expected_x)
        assert list(adata.obs.index) == col1_val
        assert adata.obs.index.name == "col1"

    def test_df_to_anndata_index_column_num(self):
        df, col1_val, col2_val, col3_val = TestAnndataExt._setup_df_to_anndata()
        expected_x = np.array([col2_val, col3_val], dtype="object").transpose()
        adata = df_to_anndata(df, index_column=0)

        assert adata.X.dtype == "object"
        assert adata.X.shape == (100, 2)
        np.testing.assert_array_equal(adata.X, expected_x)
        assert list(adata.obs.index) == col1_val
        assert adata.obs.index.name == "col1"

    def test_df_to_anndata_index_column_index(self):
        d = {"col1": [0, 1, 2, 3], "col2": pd.Series([2, 3])}
        df = pd.DataFrame(data=d, index=[0, 1, 2, 3])
        df.index.set_names("quarter", inplace=True)
        adata = ep.ad.df_to_anndata(df, index_column="quarter")
        assert adata.obs.index.name == "quarter"
        assert list(adata.obs.index) == ["0", "1", "2", "3"]

    def test_df_to_anndata_invalid_index_throws_error(self):
        df, col1_val, col2_val, col3_val = TestAnndataExt._setup_df_to_anndata()
        with pytest.raises(IndexNotFoundError):
            _ = df_to_anndata(df, index_column="UnknownCol")

    def test_df_to_anndata_cols_obs_only(self):
        df, col1_val, col2_val, col3_val = TestAnndataExt._setup_df_to_anndata()
        adata = df_to_anndata(df, columns_obs_only=["col1", "col2"])
        assert adata.X.dtype == "float32"
        assert adata.X.shape == (100, 1)
        assert_frame_equal(
            adata.obs,
            DataFrame({"col1": col1_val, "col2": col2_val}, index=[str(idx) for idx in range(100)]).astype("category"),
        )

    def test_df_to_anndata_all_num(self):
        test_array = np.random.randint(0, 100, (4, 5))
        df = DataFrame(test_array, columns=["col" + str(idx) for idx in range(5)])
        adata = df_to_anndata(df)

        assert adata.X.dtype == "float32"
        np.testing.assert_array_equal(test_array, adata.X)

    def test_anndata_to_df_simple(self):
        col1_val, col2_val, col3_val = TestAnndataExt._setup_anndata_to_df()
        expected_df = DataFrame({"col1": col1_val, "col2": col2_val, "col3": col3_val}, dtype="object")
        adata_x = np.array([col1_val, col2_val, col3_val], dtype="object").transpose()
        adata = AnnData(
            X=adata_x,
            obs=DataFrame(index=[idx for idx in range(100)]),
            var=DataFrame(index=["col" + str(idx) for idx in range(1, 4)]),
            dtype="object",
        )
        anndata_df = anndata_to_df(adata)

        assert_frame_equal(anndata_df, expected_df)

    def test_anndata_to_df_all_from_obs(self):
        col1_val, col2_val, col3_val = TestAnndataExt._setup_anndata_to_df()
        expected_df = DataFrame({"col1": col1_val, "col2": col2_val, "col3": col3_val})
        obs = DataFrame({"col2": col2_val, "col3": col3_val})
        adata_x = np.array([col1_val], dtype="object").transpose()
        adata = AnnData(X=adata_x, obs=obs, var=DataFrame(index=["col1"]), dtype="object")
        anndata_df = anndata_to_df(adata, obs_cols=list(adata.obs.columns))

        assert_frame_equal(anndata_df, expected_df)

    def test_anndata_to_df_some_from_obs(self):
        col1_val, col2_val, col3_val = TestAnndataExt._setup_anndata_to_df()
        expected_df = DataFrame({"col1": col1_val, "col3": col3_val})
        obs = DataFrame({"col2": col2_val, "col3": col3_val})
        adata_x = np.array([col1_val], dtype="object").transpose()
        adata = AnnData(X=adata_x, obs=obs, var=DataFrame(index=["col1"]), dtype="object")
        anndata_df = anndata_to_df(adata, obs_cols=["col3"])

        assert_frame_equal(anndata_df, expected_df)

    def test_anndata_to_df_throws_error_with_empty_obs(self):
        col1_val = ["patient" + str(idx) for idx in range(100)]
        adata_x = np.array([col1_val], dtype="object").transpose()
        adata = AnnData(
            X=adata_x, obs=DataFrame(index=[idx for idx in range(100)]), var=DataFrame(index=["col1"]), dtype="object"
        )

        with pytest.raises(ObsEmptyError):
            _ = anndata_to_df(adata, obs_cols=["some_missing_column"])

    def test_anndata_to_df_all_columns(self):
        col1_val, col2_val, col3_val = TestAnndataExt._setup_anndata_to_df()
        expected_df = DataFrame({"col1": col1_val})
        var = DataFrame(index=["col1"])
        adata_x = np.array([col1_val], dtype="object").transpose()
        adata = AnnData(X=adata_x, obs=DataFrame({"col2": col2_val, "col3": col3_val}), var=var, dtype="object")
        anndata_df = anndata_to_df(adata, obs_cols=list(adata.var.columns))

        assert_frame_equal(anndata_df, expected_df)

    def test_anndata_to_df_layers(self):
        col1_val, col2_val, col3_val = TestAnndataExt._setup_anndata_to_df()
        expected_df = DataFrame({"col1": col1_val, "col2": col2_val, "col3": col3_val})
        obs = DataFrame({"col2": col2_val, "col3": col3_val})
        adata_x = np.array([col1_val], dtype="object").transpose()
        adata = AnnData(
            X=adata_x, obs=obs, var=DataFrame(index=["col1"]), dtype="object", layers={"raw": adata_x.copy()}
        )
        anndata_df = anndata_to_df(adata, obs_cols=list(adata.obs.columns), layer="raw")

        assert_frame_equal(anndata_df, expected_df)

    def test_detect_binary_columns(self):
        binary_df = TestAnndataExt._setup_binary_df_to_anndata()
        adata = df_to_anndata(binary_df)
        assert set(adata.uns["non_numerical_columns"]) == {
            "col1",
            "col2",
        }
        assert set(adata.uns["numerical_columns"]) == {
            "col3",
            "col4",
            "col5",
            "col6",
            "col7_binary_int",
            "col8_binary_float",
            "col9_binary_missing_values",
        }

    def test_detect_mixed_binary_columns(self):
        df = pd.DataFrame(
            {"Col1": [i for i in range(4)], "Col2": ["str" + str(i) for i in range(4)], "Col3": [1.0, 0.0, np.nan, 1.0]}
        )
        adata = ep.ad.df_to_anndata(df)
        assert set(adata.uns["non_numerical_columns"]) == {"Col2"}
        assert set(adata.uns["numerical_columns"]) == {"Col1", "Col3"}

    @staticmethod
    def _setup_df_to_anndata() -> Tuple[DataFrame, list, list, list]:
        col1_val = ["str" + str(idx) for idx in range(100)]
        col2_val = ["another_str" + str(idx) for idx in range(100)]
        col3_val = [idx for idx in range(100)]
        df = DataFrame({"col1": col1_val, "col2": col2_val, "col3": col3_val})

        return df, col1_val, col2_val, col3_val

    @staticmethod
    def _setup_binary_df_to_anndata() -> DataFrame:
        col1_val = ["str" + str(idx) for idx in range(100)]
        col2_val = ["another_str" + str(idx) for idx in range(100)]
        col3_val = [0 for _ in range(100)]
        col4_val = [1.0 for _ in range(100)]
        col5_val = [np.NaN for _ in range(100)]
        col6_val = [0.0 if idx % 2 == 0 else np.NaN for idx in range(100)]
        col7_val = [idx % 2 for idx in range(100)]
        col8_val = [float(idx % 2) for idx in range(100)]
        col9_val = [idx % 3 if idx % 3 in {0, 1} else np.NaN for idx in range(100)]
        df = DataFrame(
            {
                "col1": col1_val,
                "col2": col2_val,
                "col3": col3_val,
                "col4": col4_val,
                "col5": col5_val,
                "col6": col6_val,
                "col7_binary_int": col7_val,
                "col8_binary_float": col8_val,
                "col9_binary_missing_values": col9_val,
            }
        )

        return df

    @staticmethod
    def _setup_anndata_to_df() -> Tuple[list, list, list]:
        col1_val = ["patient" + str(idx) for idx in range(100)]
        col2_val = ["feature" + str(idx) for idx in range(100)]
        col3_val = [idx for idx in range(100)]

        return col1_val, col2_val, col3_val

    def test_generate_anndata(self):
        adata = generate_anndata((3, 3), include_nlp=False)
        assert adata.X.shape == (3, 3)

        adata = generate_anndata((2, 2), include_nlp=True)
        assert adata.X.shape == (2, 2)
        assert "nlp" in adata.obs.columns


class TestAnnDataUtil:
    def setup_method(self):
        obs_data = {"ID": ["Patient1", "Patient2", "Patient3"], "Age": [31, 94, 62]}

        X_numeric = np.array([[1, 3.4, 2.1, 4], [2, 6.9, 7.6, 2], [1, 4.5, 1.3, 7]], dtype=np.dtype(object))
        var_numeric = {
            "Feature": ["Numeric1", "Numeric2", "Numeric3", "Numeric4"],
            "Type": ["Numeric", "Numeric", "Numeric", "Numeric"],
        }

        X_strings = np.array(
            [
                [1, 3.4, "A string", "A different string"],
                [2, 5.4, "Silly string", "A different string"],
                [2, 5.7, "A string", "What string?"],
            ],
            dtype=np.dtype(object),
        )
        var_strings = {
            "Feature": ["Numeric1", "Numeric2", "String1", "String2"],
            "Type": ["Numeric", "Numeric", "String", "String"],
        }

        self.adata_numeric = AnnData(
            X=X_numeric,
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_numeric, index=var_numeric["Feature"]),
            dtype=np.dtype(object),
            uns=OrderedDict(),
        )

        self.adata_numeric.uns["numerical_columns"] = ["Numeric1", "Numeric2"]
        self.adata_numeric.uns["non_numerical_columns"] = ["String1", "String2"]
        self.adata_strings = AnnData(
            X=X_strings,
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_strings, index=var_strings["Feature"]),
            dtype=np.dtype(object),
        )
        self.adata_strings.uns["numerical_columns"] = ["Numeric1", "Numeric2"]
        self.adata_strings.uns["non_numerical_columns"] = ["String1", "String2"]
        self.adata_encoded = ep.pp.encode(self.adata_strings.copy(), autodetect=True)

    def test_assert_encoded(self):
        """Test for the encoding assertion."""
        _assert_encoded(self.adata_encoded)

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            _assert_encoded(self.adata_numeric)

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            _assert_encoded(self.adata_strings)

    def test_get_numeric_vars(self):
        """Test for the numeric vars getter."""
        vars = get_numeric_vars(self.adata_encoded)
        assert vars == ["Numeric1", "Numeric2"]

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            get_numeric_vars(self.adata_numeric)

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            get_numeric_vars(self.adata_strings)

    def test_assert_numeric_vars(self):
        "Test for the numeric vars assertion."
        assert_numeric_vars(self.adata_encoded, ["Numeric1", "Numeric2"])

        with pytest.raises(ValueError, match=r"Some selected vars are not numeric"):
            assert_numeric_vars(self.adata_encoded, ["Numeric2", "String1"])

    def test_set_numeric_vars(self):
        """Test for the numeric vars setter."""
        values = np.array(
            [[1.2, 2.2], [3.2, 4.2], [5.2, 6.2]],
            dtype=np.dtype(np.float32),
        )
        adata_set = set_numeric_vars(self.adata_encoded, values, copy=True)
        np.testing.assert_array_equal(adata_set.X[:, 2], values[:, 0]) and np.testing.assert_array_equal(
            adata_set.X[:, 3], values[:, 1]
        )

        with pytest.raises(ValueError, match=r"Some selected vars are not numeric"):
            set_numeric_vars(self.adata_encoded, values, vars=["ehrapycat_String1"])

        string_values = np.array(
            [
                ["A"],
                ["B"],
                ["A"],
            ]
        )

        with pytest.raises(TypeError, match=r"values must be numeric"):
            set_numeric_vars(self.adata_encoded, string_values)

        extra_values = np.array(
            [
                [1.2, 1.3, 1.4],
                [2.2, 2.3, 2.4],
                [2.2, 2.3, 2.4],
            ],
            dtype=np.dtype(np.float32),
        )

        with pytest.raises(ValueError, match=r"does not match number of vars"):
            set_numeric_vars(self.adata_encoded, extra_values)

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            set_numeric_vars(self.adata_numeric, values)

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            set_numeric_vars(self.adata_strings, values)
