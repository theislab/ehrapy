import numpy as np
import pandas as pd
import pytest
from pandas import CategoricalDtype

from ehrapy.io._read import read_csv, read_fhir, read_h5ad
from tests.conftest import TEST_DATA_PATH

_TEST_PATH = f"{TEST_DATA_PATH}/io"
_TEST_PATH_H5AD = f"{_TEST_PATH}/h5ad"
_TEST_PATH_FHIR = f"{_TEST_PATH}/fhir/json"


def test_read_csv():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv")
    matrix = np.array(
        [[12, 14, 500, False], [13, 7, 330, False], [14, 10, 800, True], [15, 11, 765, True], [16, 3, 800, True]]
    )
    assert adata.X.shape == (5, 4)
    assert (adata.X == matrix).all()
    assert adata.var_names.to_list() == ["patient_id", "los_days", "b12_values", "survival"]
    assert (adata.layers["original"] == matrix).all()
    assert id(adata.layers["original"]) != id(adata.X)


def test_read_tsv():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_tsv.tsv", sep="\t")
    matrix = np.array(
        [
            [12, 54, 185.34, False],
            [13, 25, 175.39, True],
            [14, 36, 183.29, False],
            [15, 44, 173.93, True],
            [16, 27, 190.32, True],
        ]
    )
    assert adata.X.shape == (5, 4)
    assert (adata.X == matrix).all()
    assert adata.var_names.to_list() == ["patient_id", "age", "height", "gamer"]
    assert (adata.layers["original"] == matrix).all()
    assert id(adata.layers["original"]) != id(adata.X)


def test_read_multiple_csv_to_anndatas():
    adatas = read_csv(dataset_path=f"{_TEST_PATH}")
    adata_ids = set(adatas.keys())
    assert all(adata_id in adata_ids for adata_id in {"dataset_non_num_with_missing", "dataset_num_with_missing"})
    assert set(adatas["dataset_non_num_with_missing"].var_names) == {
        "indexcol",
        "intcol",
        "strcol",
        "boolcol",
        "binary_col",
    }
    assert set(adatas["dataset_num_with_missing"].var_names) == {"col" + str(i) for i in range(1, 4)}


def test_read_multiple_csvs_to_dfs():
    dfs = read_csv(dataset_path=f"{_TEST_PATH}", return_dfs=True)
    dfs_ids = set(dfs.keys())
    assert all(id in dfs_ids for id in {"dataset_non_num_with_missing", "dataset_num_with_missing"})
    assert set(dfs["dataset_non_num_with_missing"].columns) == {
        "indexcol",
        "intcol",
        "strcol",
        "boolcol",
        "binary_col",
        "datetime",
    }


def test_read_multiple_csv_with_obs_only():
    adatas = read_csv(
        dataset_path=f"{_TEST_PATH}",
        columns_obs_only={"dataset_non_num_with_missing": ["strcol"], "dataset_num_with_missing": ["col1"]},
    )
    adata_ids = set(adatas.keys())
    assert all(adata_id in adata_ids for adata_id in {"dataset_non_num_with_missing", "dataset_num_with_missing"})
    assert set(adatas["dataset_non_num_with_missing"].var_names) == {"indexcol", "intcol", "boolcol", "binary_col"}
    assert set(adatas["dataset_num_with_missing"].var_names) == {"col" + str(i) for i in range(2, 4)}
    assert all(
        obs_name in set(adatas["dataset_non_num_with_missing"].obs.columns) for obs_name in {"datetime", "strcol"}
    )
    assert "col1" in set(adatas["dataset_num_with_missing"].obs.columns)


def test_read_h5ad():
    adata = read_h5ad(dataset_path=f"{_TEST_PATH_H5AD}/dataset9.h5ad")

    assert adata.X.shape == (4, 3)
    assert set(adata.var_names) == {"col" + str(i) for i in range(1, 4)}
    assert set(adata.obs.columns) == set()


def test_read_multiple_h5ad():
    adatas = read_h5ad(dataset_path=f"{_TEST_PATH_H5AD}")
    adata_ids = set(adatas.keys())

    assert all(adata_id in adata_ids for adata_id in {"dataset8", "dataset9"})
    assert set(adatas["dataset8"].var_names) == {"indexcol", "intcol", "boolcol", "binary_col", "strcol"}
    assert set(adatas["dataset9"].var_names) == {"col" + str(i) for i in range(1, 4)}
    assert all(obs_name in set(adatas["dataset8"].obs.columns) for obs_name in {"datetime"})


def test_read_csv_without_index_column():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_index.csv")
    matrix = np.array(
        [[1, 14, 500, False], [2, 7, 330, False], [3, 10, 800, True], [4, 11, 765, True], [5, 3, 800, True]]
    )
    assert adata.X.shape == (5, 4)
    assert (adata.X == matrix).all()
    assert adata.var_names.to_list() == ["clinic_id", "los_days", "b12_values", "survival"]
    assert (adata.layers["original"] == matrix).all()
    assert id(adata.layers["original"]) != id(adata.X)
    assert list(adata.obs.index) == ["0", "1", "2", "3", "4"]


def test_read_csv_with_bools_obs_only():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv", columns_obs_only=["survival", "b12_values"])
    matrix = np.array([[12, 14], [13, 7], [14, 10], [15, 11], [16, 3]])
    assert adata.X.shape == (5, 2)
    assert (adata.X == matrix).all()
    assert adata.var_names.to_list() == ["patient_id", "los_days"]
    assert (adata.layers["original"] == matrix).all()
    assert id(adata.layers["original"]) != id(adata.X)
    assert set(adata.obs.columns) == {"b12_values", "survival"}
    assert pd.api.types.is_bool_dtype(adata.obs["survival"].dtype)
    assert pd.api.types.is_numeric_dtype(adata.obs["b12_values"].dtype)


def test_read_csv_with_bools_and_cats_obs_only():
    adata = read_csv(
        dataset_path=f"{_TEST_PATH}/dataset_bools_and_str.csv", columns_obs_only=["b12_values", "name", "survival"]
    )
    matrix = np.array([[1, 14], [2, 7], [3, 10], [4, 11], [5, 3]])
    assert adata.X.shape == (5, 2)
    assert (adata.X == matrix).all()
    assert adata.var_names.to_list() == ["clinic_id", "los_days"]
    assert (adata.layers["original"] == matrix).all()
    assert id(adata.layers["original"]) != id(adata.X)
    assert set(adata.obs.columns) == {"b12_values", "survival", "name"}
    assert pd.api.types.is_bool_dtype(adata.obs["survival"].dtype)
    assert pd.api.types.is_numeric_dtype(adata.obs["b12_values"].dtype)
    assert isinstance(adata.obs["name"].dtype, CategoricalDtype)


def test_set_default_index():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_index.csv")
    assert adata.X.shape == (5, 4)
    assert not adata.obs_names.name
    assert adata.obs.index.values.tolist() == [f"{i}" for i in range(5)]


def test_set_given_str_index():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv", index_column="los_days")
    assert adata.X.shape == (5, 3)
    assert adata.obs_names.name == "los_days"
    assert adata.obs.index.values.tolist() == ["14", "7", "10", "11", "3"]


def test_set_given_int_index():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv", index_column=1)
    assert adata.X.shape == (5, 3)
    assert adata.obs_names.name == "los_days"
    assert adata.obs.index.values.tolist() == ["14", "7", "10", "11", "3"]


def test_move_single_column_misspelled():
    with pytest.raises(ValueError):
        _ = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv", columns_obs_only=["b11_values"])


def test_move_single_column_to_obs():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv", columns_obs_only=["b12_values"])
    assert adata.X.shape == (5, 3)
    assert list(adata.obs.columns) == ["b12_values"]
    assert "b12_values" not in list(adata.var_names.values)


def test_move_multiple_columns_to_obs():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv", columns_obs_only=["b12_values", "survival"])
    assert adata.X.shape == (5, 2)
    assert list(adata.obs.columns) == ["b12_values", "survival"]
    assert "b12_values" not in list(adata.var_names.values) and "survival" not in list(adata.var_names.values)


def test_read_raises_error_with_duplicates_columns_only_single_1():
    with pytest.raises(ValueError):
        _ = read_csv(
            dataset_path=f"{_TEST_PATH}/dataset_basic.csv",
            columns_obs_only=["survival", "b12_values"],
            columns_x_only=["survival", "b12_values"],
        )


def test_read_raises_error_with_duplicates_columns_only_single_2():
    with pytest.raises(ValueError):
        _ = read_csv(
            dataset_path=f"{_TEST_PATH}/dataset_basic.csv",
            columns_obs_only=["survival"],
            columns_x_only=["survival", "b12_values"],
        )


def test_read_raises_error_with_duplicates_columns_only_multiple_1():
    with pytest.raises(ValueError):
        _ = read_csv(
            dataset_path=f"{_TEST_PATH}",
            columns_obs_only={
                "dataset_non_num_with_missing": ["intcol"],
                "dataset_num_with_missing": ["col1", "col2"],
            },
            columns_x_only={"dataset_non_num_with_missing": ["intcol"]},
        )


def test_read_raises_error_with_duplicates_columns_only_multiple_2():
    with pytest.raises(ValueError):
        _ = read_csv(
            dataset_path=f"{_TEST_PATH}",
            columns_obs_only={
                "dataset_non_num_with_missing": ["intcol"],
                "dataset_num_with_missing": ["col1", "col2"],
            },
            columns_x_only={"dataset_non_num_with_missing": ["indexcol"], "dataset_num_with_missing": ["col3"]},
        )


def test_move_single_column_to_x():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv", columns_x_only=["b12_values"])
    assert adata.X.shape == (5, 1)
    assert list(adata.var_names) == ["b12_values"]
    assert "b12_values" not in list(adata.obs.columns)
    assert all(obs_names in list(adata.obs.columns) for obs_names in ["los_days", "patient_id", "survival"])


def test_move_multiple_columns_to_x():
    adata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv", columns_x_only=["b12_values", "survival"])
    assert adata.X.shape == (5, 2)
    assert all(var_names in list(adata.var_names) for var_names in ["b12_values", "survival"])
    assert all(obs_names in list(adata.obs.columns) for obs_names in ["los_days", "patient_id"])
    assert all(var_names not in list(adata.obs.columns) for var_names in ["b12_values", "survival"])


def test_read_multiple_csv_with_x_only():
    adatas = read_csv(
        dataset_path=f"{_TEST_PATH}",
        columns_x_only={"dataset_non_num_with_missing": ["strcol"], "dataset_num_with_missing": ["col1"]},
    )
    adata_ids = set(adatas.keys())
    assert all(adata_id in adata_ids for adata_id in {"dataset_non_num_with_missing", "dataset_num_with_missing"})
    assert set(adatas["dataset_non_num_with_missing"].obs.columns) == {
        "indexcol",
        "intcol",
        "boolcol",
        "binary_col",
        "datetime",
    }
    assert set(adatas["dataset_num_with_missing"].obs.columns) == {"col" + str(i) for i in range(2, 4)}
    assert set(adatas["dataset_non_num_with_missing"].var_names) == {"strcol"}
    assert set(adatas["dataset_num_with_missing"].var_names) == {"col1"}


def test_read_multiple_csv_with_x_only_2():
    adatas = read_csv(
        dataset_path=f"{_TEST_PATH}",
        columns_x_only={
            "dataset_non_num_with_missing": ["strcol", "intcol", "boolcol"],
            "dataset_num_with_missing": ["col1", "col3"],
        },
    )
    adata_ids = set(adatas.keys())
    assert all(adata_id in adata_ids for adata_id in {"dataset_non_num_with_missing", "dataset_num_with_missing"})
    assert set(adatas["dataset_non_num_with_missing"].obs.columns) == {"indexcol", "binary_col", "datetime"}
    assert set(adatas["dataset_num_with_missing"].obs.columns) == {"col2"}
    assert set(adatas["dataset_non_num_with_missing"].var_names) == {"strcol", "intcol", "boolcol"}
    assert set(adatas["dataset_num_with_missing"].var_names) == {"col1", "col3"}


def test_read_fhir_json():
    adata = read_fhir(_TEST_PATH_FHIR)

    assert adata.shape == (4928, 75)
    assert "birthDate" in adata.obs.columns


def test_read_fhir_json_obs_only():
    adata = read_fhir(_TEST_PATH_FHIR, columns_obs_only=["fullUrl"])

    assert adata.shape == (4928, 74)
    assert "fullUrl" in adata.obs.columns
