from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME, FEATURE_TYPE_KEY, NUMERIC_TAG

import ehrapy as ep
import ehrapy.tools.feature_ranking._rank_features_groups as _utils
from ehrapy.io._read import read_csv
from tests.conftest import TEST_DATA_PATH

CURRENT_DIR = Path(__file__).parent


class TestHelperFunctions:
    def test_adjust_pvalues(self):
        groups = ("group1", "group2")

        pvals: np.recarray = pd.DataFrame(
            {"group1": (0.01, 0.02, 0.03, 1.00), "group2": (0.04, 0.05, 0.06, 0.99)}
        ).to_records(index=False)
        expected_result_bh = pd.DataFrame(
            {"group1": (0.04, 0.04, 0.04, 1.00), "group2": (0.08, 0.08, 0.08, 0.99)}
        ).to_records(index=False)
        expected_result_bf = pd.DataFrame(
            {"group1": (0.04, 0.08, 0.12, 1.00), "group2": (0.16, 0.20, 0.24, 1.00)}
        ).to_records(index=False)

        result_bh = _utils._adjust_pvalues(pvals, corr_method="benjamini-hochberg")

        for group in groups:
            assert (pvals[group] <= result_bh[group]).all()
            assert np.allclose(result_bh[group], expected_result_bh[group])

        result_bf = _utils._adjust_pvalues(pvals, corr_method="bonferroni")

        for group in groups:
            assert (pvals[group] <= result_bf[group]).all()
            assert np.allclose(result_bf[group], expected_result_bf[group])

    def test_sort_features(self):
        def _is_sorted(arr: np.array):
            return np.all(arr[:-1] <= arr[1:])

        adata = ep.dt.mimic_2(encoded=False)
        adata.uns["rank_features_groups"] = {
            "params": {"whatever": "here is", "it does": "not matter", "but this key should be": "present for testing"},
            "names": pd.DataFrame(
                {
                    "group1": ("feature2", "feature1", "feature4", "feature3"),
                    "group2": ("feature5", "feature6", "feature7", "feature8"),
                }
            ).to_records(index=False),
            # Let's mix features in group1, and leave them sorted in group2
            "pvals": pd.DataFrame({"group1": (0.02, 0.01, 1.00, 0.03), "group2": (0.04, 0.05, 0.06, 0.99)}).to_records(
                index=False
            ),
            "scores": pd.DataFrame({"group1": (2, 1, 100, 3), "group2": (4, 5, 6, 7)}).to_records(index=False),
            "log2foldchanges": pd.DataFrame({"group1": (2, 1, 10, 3), "group2": (4, 5, 6, 7)}).to_records(index=False),
        }
        # Doesn't really matter that they are the same here but order should be preserved
        adata.uns["rank_features_groups"]["pvals_adj"] = adata.uns["rank_features_groups"]["pvals"].copy()

        _utils._sort_features(adata)

        # Check that every feature is sorted
        # Actually, some of them would be sorted in the opposite direction (e.g. scores) for real data
        # But what matters here is that the permutation is the same for every key and is based on adjusted p-values
        for key in adata.uns["rank_features_groups"]:
            if key == "params":
                continue
            assert _is_sorted(adata.uns["rank_features_groups"][key]["group1"])
            assert _is_sorted(adata.uns["rank_features_groups"][key]["group2"])

    def test_save_rank_features_result(self):
        groups = ("group1", "group2")

        adata = ep.dt.mimic_2(encoded=False)
        adata.uns["rank_features_groups"] = {
            "params": {"whatever": "here is", "it does": "not matter", "but this key should be": "present for testing"}
        }

        names = pd.DataFrame(
            {
                "group1": ("feature2", "feature1", "feature4", "feature3"),
                "group2": ("feature5", "feature6", "feature7", "feature8"),
            }
        ).to_records(index=False)
        pvals = pd.DataFrame({"group1": (0.02, 0.01, 1.00, 0.03), "group2": (0.04, 0.05, 0.06, 0.99)}).to_records(
            index=False
        )
        scores = pd.DataFrame({"group1": (2, 1, 100, 3), "group2": (4, 5, 6, 7)}).to_records(index=False)
        logfoldchanges = pd.DataFrame({"group1": (2, 1, 10, 3), "group2": (4, 5, 6, 7)}).to_records(index=False)

        # Check that adding only required keys works
        adata_only_required = adata.copy()
        _utils._save_rank_features_result(
            adata_only_required,
            key_added="rank_features_groups",
            groups_order=groups,
            names=names,
            scores=scores,
            pvals=pvals,
        )

        assert "names" in adata_only_required.uns["rank_features_groups"]
        assert "pvals" in adata_only_required.uns["rank_features_groups"]
        assert "scores" in adata_only_required.uns["rank_features_groups"]
        assert "log2foldchanges" not in adata_only_required.uns["rank_features_groups"]
        assert "pvals_adj" not in adata_only_required.uns["rank_features_groups"]
        assert "pts" not in adata_only_required.uns["rank_features_groups"]
        assert adata_only_required.uns["rank_features_groups"]["names"].dtype.names == groups
        assert (
            len(adata_only_required.uns["rank_features_groups"]["names"]) == 4
        )  # It only captures the length of each group
        assert len(adata_only_required.uns["rank_features_groups"]["pvals"]) == 4
        assert len(adata_only_required.uns["rank_features_groups"]["scores"]) == 4

        # Check that running the function on adata with existing keys merges the arrays correctly
        _utils._save_rank_features_result(
            adata_only_required,
            key_added="rank_features_groups",
            groups_order=groups,
            names=names,
            scores=scores,
            pvals=pvals,
        )
        assert len(adata_only_required.uns["rank_features_groups"]["names"]) == 8
        assert len(adata_only_required.uns["rank_features_groups"]["pvals"]) == 8
        assert len(adata_only_required.uns["rank_features_groups"]["scores"]) == 8

        # Check that adding other keys works
        adata_all_keys = adata.copy()
        _utils._save_rank_features_result(
            adata_all_keys,
            key_added="rank_features_groups",
            groups_order=groups,
            names=names,
            scores=scores,
            pvals=pvals,
            pvals_adj=pvals.copy(),
            pts=pvals.copy(),
            logfoldchanges=logfoldchanges,
        )

        assert "names" in adata_all_keys.uns["rank_features_groups"]
        assert "pvals" in adata_all_keys.uns["rank_features_groups"]
        assert "scores" in adata_all_keys.uns["rank_features_groups"]
        assert "logfoldchanges" in adata_all_keys.uns["rank_features_groups"]
        assert "pvals_adj" in adata_all_keys.uns["rank_features_groups"]
        assert "pts" in adata_all_keys.uns["rank_features_groups"]
        assert adata_all_keys.uns["rank_features_groups"]["names"].dtype.names == groups
        assert len(adata_all_keys.uns["rank_features_groups"]["names"]) == 4
        assert len(adata_all_keys.uns["rank_features_groups"]["pvals"]) == 4
        assert len(adata_all_keys.uns["rank_features_groups"]["pvals_adj"]) == 4
        assert len(adata_all_keys.uns["rank_features_groups"]["logfoldchanges"]) == 4
        assert len(adata_all_keys.uns["rank_features_groups"]["scores"]) == 4
        assert len(adata_all_keys.uns["rank_features_groups"]["pts"]) == 4

        # Check that passing empty objects doesn't add keys
        _utils._save_rank_features_result(
            adata,
            key_added="rank_features_groups",
            groups_order=groups,
            names=names,
            scores=scores,
            pvals=pvals,
            pvals_adj=[],
            pts=np.array([]),
            logfoldchanges=pd.DataFrame([]),
        )
        assert "names" in adata.uns["rank_features_groups"]
        assert "pvals" in adata.uns["rank_features_groups"]
        assert "scores" in adata.uns["rank_features_groups"]
        assert "logfoldchanges" not in adata.uns["rank_features_groups"]
        assert "pvals_adj" not in adata.uns["rank_features_groups"]
        assert "pts" not in adata.uns["rank_features_groups"]

    def test_get_groups_order(self):
        assert _utils._get_groups_order(groups_subset="all", group_names=("A", "B", "C"), reference="B") == (
            "A",
            "B",
            "C",
        )
        assert _utils._get_groups_order(groups_subset=("A", "B"), group_names=("A", "B", "C"), reference="B") == (
            "A",
            "B",
        )
        assert _utils._get_groups_order(groups_subset=("A", "B"), group_names=("A", "B", "C"), reference="rest") == (
            "A",
            "B",
        )
        assert _utils._get_groups_order(groups_subset=("A", "B"), group_names=("A", "B", "C"), reference="C") == (
            "A",
            "B",
            "C",
        )

        # Check that array with ints (e.g. leiden clusters) works correctly
        assert _utils._get_groups_order(groups_subset=(1, 2), group_names=np.arange(3), reference="rest") == ("1", "2")

        with pytest.raises(ValueError):
            # Reference not in group_names
            _utils._get_groups_order(groups_subset=("A", "B"), group_names=("A", "B", "C"), reference="D")

    def test_evaluate_categorical_features(self):
        adata = ep.dt.mimic_2(encoded=False)
        ep.ad.infer_feature_types(adata, output=None)
        adata.var.loc["hour_icu_intime", FEATURE_TYPE_KEY] = (
            NUMERIC_TAG  # This is detected as categorical, so we need to correct that
        )
        adata = ep.pp.encode(adata, autodetect=True, encodings="label")

        group_names = pd.Categorical(adata.obs["service_unit"].astype(str)).categories.tolist()

        for method in ("chi-square", "g-test", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read"):
            names, scores, pvals, logfc, pts = _utils._evaluate_categorical_features(
                adata, groupby="service_unit", group_names=group_names, categorical_method=method
            )

            # Check that important fields are not empty
            assert len(names)
            assert len(scores)
            assert len(pvals)
            assert len(logfc)
            assert not len(pts)  # Because pts == False by default

            assert (pvals >= 0).all()
            assert (pvals <= 1).all()

            # Check that grouping feature is not in the results
            assert "service_unit" not in names
            assert "ehrapycat_service_unit" not in names

            # Check that the only other categorical feature is in the results
            assert "ehrapycat_day_icu_intime" in names


def test_real_dataset(mimic_2_encoded):
    adata = mimic_2_encoded
    ep.tl.rank_features_groups(adata, groupby="service_unit")

    assert "rank_features_groups" in adata.uns
    assert "names" in adata.uns["rank_features_groups"]
    assert "pvals" in adata.uns["rank_features_groups"]
    assert "scores" in adata.uns["rank_features_groups"]
    assert "logfoldchanges" in adata.uns["rank_features_groups"]
    assert "pvals_adj" in adata.uns["rank_features_groups"]

    assert "params" in adata.uns["rank_features_groups"]
    assert "method" in adata.uns["rank_features_groups"]["params"]
    assert "categorical_method" in adata.uns["rank_features_groups"]["params"]
    assert "reference" in adata.uns["rank_features_groups"]["params"]
    assert "groupby" in adata.uns["rank_features_groups"]["params"]
    assert "layer" in adata.uns["rank_features_groups"]["params"]
    assert "corr_method" in adata.uns["rank_features_groups"]["params"]


def test_only_continous_features(mimic_2_encoded):
    adata = mimic_2_encoded
    ep.tl.rank_features_groups(adata, groupby="service_unit")

    assert "rank_features_groups" in adata.uns
    assert "names" in adata.uns["rank_features_groups"]
    assert "pvals" in adata.uns["rank_features_groups"]
    assert "scores" in adata.uns["rank_features_groups"]
    assert "logfoldchanges" in adata.uns["rank_features_groups"]
    assert "pvals_adj" in adata.uns["rank_features_groups"]


def test_only_cat_features(mimic_2_encoded):
    adata = mimic_2_encoded
    ep.tl.rank_features_groups(adata, groupby="service_unit")
    assert "rank_features_groups" in adata.uns
    assert "names" in adata.uns["rank_features_groups"]
    assert "pvals" in adata.uns["rank_features_groups"]
    assert "scores" in adata.uns["rank_features_groups"]
    assert "logfoldchanges" in adata.uns["rank_features_groups"]
    assert "pvals_adj" in adata.uns["rank_features_groups"]


@pytest.mark.parametrize("field_to_rank", ["layer", "obs", "layer_and_obs"])
def test_rank_adata_immutability_property(field_to_rank):
    """
    Test that rank_features_group does not modify the adata object passed to it, except for the desired .uns field.
    This test is important because to save memory, copies are made conservatively in rank_features_groups
    """
    adata = read_csv(
        dataset_path=f"{TEST_DATA_PATH}/dataset1.csv", columns_x_only=["station", "sys_bp_entry", "dia_bp_entry"]
    )
    adata = ep.pp.encode(adata, encodings={"label": ["station"]})
    adata_orig = adata.copy()

    ep.tl.rank_features_groups(adata, groupby="disease", field_to_rank=field_to_rank)

    assert adata_orig.shape == adata.shape
    assert adata_orig.X.shape == adata.X.shape
    assert adata_orig.obs.shape == adata.obs.shape
    assert adata_orig.var.shape == adata.var.shape

    assert np.allclose(adata_orig.X, adata.X)
    assert np.array_equal(adata_orig.obs, adata.obs)

    assert "rank_features_groups" in adata.uns


@pytest.mark.parametrize("field_to_rank", ["layer", "obs", "layer_and_obs"])
def test_rank_features_groups_generates_outputs(field_to_rank):
    """Test that the desired output is generated."""
    adata = read_csv(
        dataset_path=f"{TEST_DATA_PATH}/dataset1.csv",
        columns_obs_only=["disease", "station", "sys_bp_entry", "dia_bp_entry"],
    )
    ep.tl.rank_features_groups(adata, groupby="disease", field_to_rank=field_to_rank)

    # check standard rank_features_groups entries
    assert "names" in adata.uns["rank_features_groups"]
    assert "pvals" in adata.uns["rank_features_groups"]
    assert "scores" in adata.uns["rank_features_groups"]
    assert "pvals_adj" in adata.uns["rank_features_groups"]
    assert "logfoldchanges" in adata.uns["rank_features_groups"]
    assert "log2foldchanges" not in adata.uns["rank_features_groups"]
    assert "pts" not in adata.uns["rank_features_groups"]

    if field_to_rank == "layer":
        assert len(adata.uns["rank_features_groups"]["names"]) == 4
        assert len(adata.uns["rank_features_groups"]["pvals"]) == 4
        assert len(adata.uns["rank_features_groups"]["scores"]) == 4

    elif field_to_rank == "obs":
        assert len(adata.uns["rank_features_groups"]["names"]) == 3  # It only captures the length of each group
        assert len(adata.uns["rank_features_groups"]["pvals"]) == 3
        assert len(adata.uns["rank_features_groups"]["scores"]) == 3

    elif field_to_rank == "layer_and_obs":
        assert len(adata.uns["rank_features_groups"]["names"]) == 7  # It only captures the length of each group
        assert len(adata.uns["rank_features_groups"]["pvals"]) == 7
        assert len(adata.uns["rank_features_groups"]["scores"]) == 7


def test_rank_features_groups_consistent_results():
    adata_features_in_x = read_csv(
        dataset_path=f"{TEST_DATA_PATH}/dataset1.csv",
        columns_x_only=["station", "sys_bp_entry", "dia_bp_entry", "glucose"],
    )
    adata_features_in_x = ep.pp.encode(adata_features_in_x, encodings={"label": ["station"]})

    adata_features_in_obs = read_csv(
        dataset_path=f"{TEST_DATA_PATH}/dataset1.csv",
        columns_obs_only=["disease", "station", "sys_bp_entry", "dia_bp_entry", "glucose"],
    )

    adata_features_in_x_and_obs = read_csv(
        dataset_path=f"{TEST_DATA_PATH}/dataset1.csv",
        columns_obs_only=["disease", "station"],
    )
    # to keep the same variables as in the datasets above, in order to make the comparison of consistency
    adata_features_in_x_and_obs = adata_features_in_x_and_obs[:, ["sys_bp_entry", "dia_bp_entry", "glucose"]]

    ep.tl.rank_features_groups(adata_features_in_x, groupby="disease")
    ep.tl.rank_features_groups(adata_features_in_obs, groupby="disease", field_to_rank="obs")
    ep.tl.rank_features_groups(adata_features_in_x_and_obs, groupby="disease", field_to_rank="layer_and_obs")

    for record in adata_features_in_x.uns["rank_features_groups"]["names"].dtype.names:
        assert np.allclose(
            adata_features_in_x.uns["rank_features_groups"]["scores"][record],
            adata_features_in_obs.uns["rank_features_groups"]["scores"][record],
        )
        assert np.allclose(
            np.array(adata_features_in_x.uns["rank_features_groups"]["pvals"][record]),
            np.array(adata_features_in_obs.uns["rank_features_groups"]["pvals"][record]),
        )
        assert np.array_equal(
            np.array(adata_features_in_x.uns["rank_features_groups"]["names"][record]),
            np.array(adata_features_in_obs.uns["rank_features_groups"]["names"][record]),
        )
    for record in adata_features_in_x.uns["rank_features_groups"]["names"].dtype.names:
        assert np.allclose(
            adata_features_in_x.uns["rank_features_groups"]["scores"][record],
            adata_features_in_x_and_obs.uns["rank_features_groups"]["scores"][record],
        )
        assert np.allclose(
            np.array(adata_features_in_x.uns["rank_features_groups"]["pvals"][record]),
            np.array(adata_features_in_x_and_obs.uns["rank_features_groups"]["pvals"][record]),
        )
        assert np.array_equal(
            np.array(adata_features_in_x.uns["rank_features_groups"]["names"][record]),
            np.array(adata_features_in_x_and_obs.uns["rank_features_groups"]["names"][record]),
        )


def test_rank_features_group_column_to_rank():
    adata = read_csv(
        dataset_path=f"{TEST_DATA_PATH}/dataset1.csv",
        columns_obs_only=["disease", "station", "sys_bp_entry", "dia_bp_entry"],
        index_column="idx",
    )

    # get a fresh adata for every test to not have any side effects
    adata_copy = adata.copy()

    ep.tl.rank_features_groups(adata, groupby="disease", columns_to_rank="all")
    assert len(adata.uns["rank_features_groups"]["names"]) == 3

    # want to check a "complete selection" works
    adata = adata_copy.copy()
    ep.tl.rank_features_groups(adata, groupby="disease", columns_to_rank={"var_names": ["glucose", "weight"]})
    assert len(adata.uns["rank_features_groups"]["names"]) == 2

    # want to check a "sub-selection" works
    adata = adata_copy.copy()
    ep.tl.rank_features_groups(adata, groupby="disease", columns_to_rank={"var_names": ["glucose"]})
    assert len(adata.uns["rank_features_groups"]["names"]) == 1

    # want to check a "complete" selection works
    adata = adata_copy.copy()
    ep.tl.rank_features_groups(
        adata,
        groupby="disease",
        field_to_rank="obs",
        columns_to_rank={"obs_names": ["station", "sys_bp_entry", "dia_bp_entry"]},
    )
    assert len(adata.uns["rank_features_groups"]["names"]) == 3

    # want to check a "sub-selection" selection works
    adata = adata_copy.copy()
    ep.tl.rank_features_groups(
        adata,
        groupby="disease",
        field_to_rank="obs",
        columns_to_rank={"obs_names": ["sys_bp_entry", "dia_bp_entry"]},
    )
    assert len(adata.uns["rank_features_groups"]["names"]) == 2


def test_rank_features_groups_3D_edata(edata_blob_small):
    ep.tl.rank_features_groups(edata_blob_small, groupby="cluster", layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.tl.rank_features_groups(edata_blob_small, groupby="cluster", layer=DEFAULT_TEM_LAYER_NAME)


def test_filter_rank_features_groups_edata(mimic_2):
    mimic_2 = ep.ad.move_to_obs(mimic_2, to_obs=["service_unit"])
    ep.tl.rank_features_groups(mimic_2, "service_unit")
    ep.tl.rank_features_groups(mimic_2, "service_unit")
