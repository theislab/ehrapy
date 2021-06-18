import pandas as pd
import pytest

from ehrapy.api.data.dataloader import Dataloader

_TEST_PATH = "tests/api/data/test_data"


class TestDataloader:
    @pytest.mark.parametrize(
        "path,sep,on,shape",
        [
            (f"{_TEST_PATH}/dataset_1.csv", ",", "patient_id", (5, 4)),
            ([f"{_TEST_PATH}/dataset_2.tsv", f"{_TEST_PATH}/dataset_3.tsv"], "\t", "patient_id", (5, 6)),
            (_TEST_PATH, "\t", "patient_id", (5, 6)),
        ],
    )
    class TestReadCSVs:
        def test_read_single_csv_(self, path, sep, on, shape):
            ds_1_df = Dataloader.read_csvs(path, sep=sep, on=on)
            assert isinstance(ds_1_df, pd.DataFrame)
            assert ds_1_df.shape == shape

        def test_read_several_csvs(self, path, sep, on, shape):
            ds_2_3_df = Dataloader.read_csvs(path, sep=sep, on=on)
            assert isinstance(ds_2_3_df, pd.DataFrame)
            assert ds_2_3_df.shape == shape

        def test_read_folder_csvs(self, path, sep, on, shape):
            ds_2_3_df = Dataloader.read_csvs(_TEST_PATH, sep=sep, on=on)
            assert isinstance(ds_2_3_df, pd.DataFrame)
            assert ds_2_3_df.shape == shape

    class TestDataFrameToAnnData:
        def test_df_to_anndata(self):
            pass
