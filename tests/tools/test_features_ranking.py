import numpy as np
import pandas as pd

import ehrapy as ep
from ehrapy.tools import _utils

def _save_rank_features_result(adata, key_added, names, scores, pvals, pvals_adj=None, logfoldchanges=None, pts=None, groups_order=None) -> None:
    """Write keys with statistical test results to adata.uns
    
    Args:
        adata: Annotated data matrix after running :func:`~ehrapy.tl.rank_features_groups`
        key_added: The key in `adata.uns` information is saved to.
        names: Structured array storing the feature names
        scores: Array with the statistics
        logfoldchanges: logarithm of fold changes or other info to store under logfoldchanges key
        pvals: p-values of a statistical test 
        pts: Percentages of cells containing features
        groups_order: order of groups in structured arrays

    Returns:
        Nothing. The operation is performed in place
    """
    fields = (names, scores, pvals, pvals_adj, logfoldchanges, pts)
    field_names = ("names", "scores", "pvals", "pvals_adj", "logfoldchanges", "pts")

    for values, key in zip(fields, field_names):
        if values is None or not len(values):
            continue

        if key not in adata.uns[key_added]:
            adata.uns[key_added][key] = values
        else:
            adata.uns[key_added][key] = _merge_arrays(
                recarray=adata.uns[key_added][key],
                array=np.array(values),
                groups_order=groups_order
            )


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

    # Write a test for _save_rank_features_result
    # Test several cases
    # 1. All keys are present
    # 2. Some keys are missing
    # 3. Some keys are present but empty (e.g. lists, empty arrays)
    def test_save_rank_features_result(self):
        groups = ("group1", "group2")

        adata = ep.dt.mimic_2(encoded=False)
        adata.uns["rank_features_groups"] = {
            "params": {"whatever": "here is", "it does": "not matter", "but this key should be": "present for testing"}
        }
        
        names =  pd.DataFrame({"group1": ("gene2", "gene1", "gene4", "gene3"), "group2": ("gene5", "gene6", "gene7", "gene8")}).to_records()
        pvals =  pd.DataFrame({"group1": (0.02, 0.01, 1.00, 0.03), "group2": (0.04, 0.05, 0.06, 0.99)}).to_records()
        scores =  pd.DataFrame({"group1": (2, 1, 100, 3), "group2": (4, 5, 6, 7)}).to_records()
        log2foldchanges = pd.DataFrame({"group1": (2, 1, 10, 3), "group2": (4, 5, 6, 7)}).to_records()
        
        adata_only_required = adata.copy()
        _utils._save_rank_features_result(adata_only_required, key_added="rank_features_groups", groups_order=groups, names=names, scores=scores, pvals=pvals)

        assert "names" in adata_only_required.uns["rank_features_groups"]
        assert "pvals" in adata_only_required.uns["rank_features_groups"]
        assert "scores" in adata_only_required.uns["rank_features_groups"]
        assert "log2foldchanges" not in adata_only_required.uns["rank_features_groups"]
        assert "pvals_adj" not in adata_only_required.uns["rank_features_groups"]
        assert "pts" not in adata_only_required.uns["rank_features_groups"]
        assert adata_only_required.uns["rank_features_groups"]["names"].dtype.names[1: ] == ("group1", "group2")
        assert len(adata_only_required.uns["rank_features_groups"]["names"]) == 4  # It only captures the length of each group
        assert len(adata_only_required.uns["rank_features_groups"]["pvals"]) == 4
        assert len(adata_only_required.uns["rank_features_groups"]["scores"]) == 4

        # Check that running the function on adata with existing keys merges the arrays correctly
        _utils._save_rank_features_result(adata_only_required, key_added="rank_features_groups", groups_order=groups, names=names, scores=scores, pvals=pvals)
        assert len(adata_only_required.uns["rank_features_groups"]["names"]) == 8
        assert len(adata_only_required.uns["rank_features_groups"]["pvals"]) == 8
        assert len(adata_only_required.uns["rank_features_groups"]["scores"]) == 8