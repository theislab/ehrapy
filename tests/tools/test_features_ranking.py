import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from ehrapy.tools import _utils


class TestHelperFunctions:
    def test_adjust_pvalues(self):
        groups = ("group1", "group2")

        pvals: np.recarray = pd.DataFrame({"group1": (0.01, 0.02, 0.03, 1.00), "group2": (0.04, 0.05, 0.06, 0.99)}).to_records()
        expected_result_bh = pd.DataFrame({"group1": (0.04, 0.04, 0.04, 1.00), "group2": (0.08, 0.08, 0.08, 0.99)}).to_records()
        expected_result_bf = pd.DataFrame({"group1": (0.04, 0.08, 0.12, 1.00), "group2": (0.16, 0.20, 0.24, 1.00)}).to_records()

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
            "names": pd.DataFrame({"group1": ("gene2", "gene1", "gene4", "gene3"), "group2": ("gene5", "gene6", "gene7", "gene8")}).to_records(),
            # Let's mix genes in group1, and leave them sorted in group2
            "pvals": pd.DataFrame({"group1": (0.02, 0.01, 1.00, 0.03), "group2": (0.04, 0.05, 0.06, 0.99)}).to_records(),
            "scores": pd.DataFrame({"group1": (2, 1, 100, 3), "group2": (4, 5, 6, 7)}).to_records(),
            "log2foldchanges": pd.DataFrame({"group1": (2, 1, 10, 3), "group2": (4, 5, 6, 7)}).to_records(),
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
        
        names =  pd.DataFrame({"group1": ("gene2", "gene1", "gene4", "gene3"), "group2": ("gene5", "gene6", "gene7", "gene8")}).to_records()
        pvals =  pd.DataFrame({"group1": (0.02, 0.01, 1.00, 0.03), "group2": (0.04, 0.05, 0.06, 0.99)}).to_records()
        scores =  pd.DataFrame({"group1": (2, 1, 100, 3), "group2": (4, 5, 6, 7)}).to_records()
        logfoldchanges = pd.DataFrame({"group1": (2, 1, 10, 3), "group2": (4, 5, 6, 7)}).to_records()

        # Chack that adding onle required keys works        
        adata_only_required = adata.copy()
        _utils._save_rank_features_result(adata_only_required, key_added="rank_features_groups", groups_order=groups, names=names, scores=scores, pvals=pvals)

        assert "names" in adata_only_required.uns["rank_features_groups"]
        assert "pvals" in adata_only_required.uns["rank_features_groups"]
        assert "scores" in adata_only_required.uns["rank_features_groups"]
        assert "log2foldchanges" not in adata_only_required.uns["rank_features_groups"]
        assert "pvals_adj" not in adata_only_required.uns["rank_features_groups"]
        assert "pts" not in adata_only_required.uns["rank_features_groups"]
        assert adata_only_required.uns["rank_features_groups"]["names"].dtype.names[1: ] == groups
        assert len(adata_only_required.uns["rank_features_groups"]["names"]) == 4  # It only captures the length of each group
        assert len(adata_only_required.uns["rank_features_groups"]["pvals"]) == 4
        assert len(adata_only_required.uns["rank_features_groups"]["scores"]) == 4

        # Check that running the function on adata with existing keys merges the arrays correctly
        _utils._save_rank_features_result(adata_only_required, key_added="rank_features_groups", groups_order=groups, names=names, scores=scores, pvals=pvals)
        assert len(adata_only_required.uns["rank_features_groups"]["names"]) == 8
        assert len(adata_only_required.uns["rank_features_groups"]["pvals"]) == 8
        assert len(adata_only_required.uns["rank_features_groups"]["scores"]) == 8

        # Check that adding other keys works
        adata_all_keys = adata.copy()
        _utils._save_rank_features_result(adata_all_keys, key_added="rank_features_groups", groups_order=groups, names=names, scores=scores, 
                                          pvals=pvals, pvals_adj=pvals.copy(), pts=pvals.copy(), logfoldchanges=logfoldchanges)

        assert "names" in adata_all_keys.uns["rank_features_groups"]
        assert "pvals" in adata_all_keys.uns["rank_features_groups"]
        assert "scores" in adata_all_keys.uns["rank_features_groups"]
        assert "logfoldchanges" in adata_all_keys.uns["rank_features_groups"]
        assert "pvals_adj" in adata_all_keys.uns["rank_features_groups"]
        assert "pts" in adata_all_keys.uns["rank_features_groups"]
        assert adata_all_keys.uns["rank_features_groups"]["names"].dtype.names[1: ] == groups
        assert len(adata_all_keys.uns["rank_features_groups"]["names"]) == 4
        assert len(adata_all_keys.uns["rank_features_groups"]["pvals"]) == 4
        assert len(adata_all_keys.uns["rank_features_groups"]["pvals_adj"]) == 4
        assert len(adata_all_keys.uns["rank_features_groups"]["logfoldchanges"]) == 4
        assert len(adata_all_keys.uns["rank_features_groups"]["scores"]) == 4
        assert len(adata_all_keys.uns["rank_features_groups"]["pts"]) == 4

        # Check that passing empty objects doesn't add keys
        _utils._save_rank_features_result(adata, key_added="rank_features_groups", groups_order=groups, names=names, scores=scores, 
                                          pvals=pvals, pvals_adj=[], pts=np.array([]), logfoldchanges=pd.DataFrame([]))
        assert "names" in adata.uns["rank_features_groups"]
        assert "pvals" in adata.uns["rank_features_groups"]
        assert "scores" in adata.uns["rank_features_groups"]
        assert "logfoldchanges" not in adata.uns["rank_features_groups"]
        assert "pvals_adj" not in adata.uns["rank_features_groups"]
        assert "pts" not in adata.uns["rank_features_groups"]


    def test_get_groups_order(self):
        assert _utils._get_groups_order(groups_subset="all", group_names=("A", "B", "C"), reference="B") == ("A", "B", "C") 
        assert _utils._get_groups_order(groups_subset=("A", "B"), group_names=("A", "B", "C"), reference="B") == ("A", "B")
        assert _utils._get_groups_order(groups_subset=("A", "B"), group_names=("A", "B", "C"), reference="rest") == ("A", "B")
        assert _utils._get_groups_order(groups_subset=("A", "B"), group_names=("A", "B", "C"), reference="C") == ("A", "B", "C")

        # Check that array with ints (e.g. leiden clusters) works correctly
        assert _utils._get_groups_order(groups_subset=(1, 2), group_names=np.arange(3), reference="rest") == ("1", "2")

        with pytest.raises(ValueError):
            # Reference not in group_names
            _utils._get_groups_order(groups_subset=("A", "B"), group_names=("A", "B", "C"), reference="D")


    def test_evaluate_categorical_features(self):
        adata = ep.dt.mimic_2(encoded=True)

        if not adata.uns["non_numerical_columns"]:
            # Manually set categorical features because of the issue #543
            adata.uns["non_numerical_columns"] = ['ehrapycat_day_icu_intime', 'ehrapycat_service_unit']

        group_names = pd.Categorical(adata.obs["service_unit"].astype(str)).categories.tolist()

        for method in ("chi-square", "g-test", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read"):
            names, scores, pvals, logfc, pts = _utils._evaluate_categorical_features(
                adata, groupby="service_unit", group_names=group_names, categorical_method=method)
            
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


class TestRankFeaturesGroups():
    def test_real_dataset(self):
        adata = ep.dt.mimic_2(encoded=True)

        if not adata.uns["non_numerical_columns"]:
            # Manually set categorical features because of the issue #543
            adata.uns["non_numerical_columns"] = ['ehrapycat_day_icu_intime', 'ehrapycat_service_unit']

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

    def test_only_continous_features(self):
        adata = ep.dt.mimic_2(encoded=True)
        adata.uns["non_numerical_columns"] = []

        ep.tl.rank_features_groups(adata, groupby="service_unit")
        assert "rank_features_groups" in adata.uns
        assert "names" in adata.uns["rank_features_groups"]
        assert "pvals" in adata.uns["rank_features_groups"]
        assert "scores" in adata.uns["rank_features_groups"]
        assert "logfoldchanges" in adata.uns["rank_features_groups"]
        assert "pvals_adj" in adata.uns["rank_features_groups"]

    def test_only_cat_features(self):
        adata = ep.dt.mimic_2(encoded=True)
        adata.uns["numerical_columns"] = []

        if not adata.uns["non_numerical_columns"]:
            # Manually set categorical features because of the issue #543
            adata.uns["non_numerical_columns"] = ['ehrapycat_day_icu_intime', 'ehrapycat_service_unit']

        ep.tl.rank_features_groups(adata, groupby="service_unit")
        assert "rank_features_groups" in adata.uns
        assert "names" in adata.uns["rank_features_groups"]
        assert "pvals" in adata.uns["rank_features_groups"]
        assert "scores" in adata.uns["rank_features_groups"]
        assert "logfoldchanges" in adata.uns["rank_features_groups"]
        assert "pvals_adj" in adata.uns["rank_features_groups"]