import numpy as np
import pytest
from anndata import AnnData
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from ehrapy.api.anndata_ext import ObsEmptyError, anndata_to_df, df_to_anndata


class TestAnndataExt:
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

    def test_df_to_anndata_cols_obs_only(self):
        df, col1_val, col2_val, col3_val = TestAnndataExt._setup_df_to_anndata()
        adata = df_to_anndata(df, columns_obs_only=["col1", "col2"])
        assert adata.X.dtype == "float32"
        assert adata.X.shape == (100, 1)
        assert_frame_equal(
            adata.obs, DataFrame({"col1": col1_val, "col2": col2_val}, index=[str(idx) for idx in range(100)])
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
        anndata_df = anndata_to_df(adata, add_from_obs="all")

        assert_frame_equal(anndata_df, expected_df)

    def test_anndata_to_df_some_from_obs(self):
        col1_val, col2_val, col3_val = TestAnndataExt._setup_anndata_to_df()
        expected_df = DataFrame({"col1": col1_val, "col3": col3_val})
        obs = DataFrame({"col2": col2_val, "col3": col3_val})
        adata_x = np.array([col1_val], dtype="object").transpose()
        adata = AnnData(X=adata_x, obs=obs, var=DataFrame(index=["col1"]), dtype="object")
        anndata_df = anndata_to_df(adata, add_from_obs=["col3"])

        assert_frame_equal(anndata_df, expected_df)

    def test_anndata_to_df_throws_error_with_empty_obs(self):
        col1_val = ["patient" + str(idx) for idx in range(100)]
        adata_x = np.array([col1_val], dtype="object").transpose()
        adata = AnnData(
            X=adata_x, obs=DataFrame(index=[idx for idx in range(100)]), var=DataFrame(index=["col1"]), dtype="object"
        )

        with pytest.raises(ObsEmptyError):
            _ = anndata_to_df(adata, add_from_obs=["some_missing_column"])

    @staticmethod
    def _setup_df_to_anndata() -> (DataFrame, list, list, list):
        col1_val = ["str" + str(idx) for idx in range(100)]
        col2_val = ["another_str" + str(idx) for idx in range(100)]
        col3_val = [idx for idx in range(100)]
        df = DataFrame({"col1": col1_val, "col2": col2_val, "col3": col3_val})

        return df, col1_val, col2_val, col3_val

    @staticmethod
    def _setup_anndata_to_df() -> (list, list, list):
        col1_val = ["patient" + str(idx) for idx in range(100)]
        col2_val = ["feature" + str(idx) for idx in range(100)]
        col3_val = [idx for idx in range(100)]

        return col1_val, col2_val, col3_val
