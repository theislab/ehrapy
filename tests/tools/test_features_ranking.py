import numpy as np
import pandas as pd

import ehrapy as ep
from ehrapy.tools import _utils


class TestHelperFunctions:
    def test_adjust_pvalues(self):
        pvals: np.recarray = pd.DataFrame({"group1": (0.01, 0.02, 0.03, 1.00), "group2": (0.04, 0.05, 0.06, 0.99)}).to_records()
        expected_result_bh = pd.DataFrame({"group1": (0.04, 0.04, 0.04, 1.00), "group2": (0.08, 0.08, 0.08, 0.99)}).to_records()
        expected_result_bf = pd.DataFrame({"group1": (0.04, 0.08, 0.12, 1.00), "group2": (0.16, 0.20, 0.24, 1.00)}).to_records()

        result_bh = _utils._adjust_pvalues(pvals, method="benjamini-hochberg")
        assert pvals["group1"] <= result_bh["group1"]
        assert pvals["group2"] <= result_bh["group2"]
        assert np.allclose(result_bh, expected_result_bh)
        
        result_bf = _utils._adjust_pvalues(pvals, method="bonferroni")
        assert pvals["group1"] <= result_bf["group1"]
        assert pvals["group2"] <= result_bf["group2"]
        assert np.allclose(result_bf, expected_result_bf)


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
        adata.uns["rank_features_groups"]["pvals_adj"] = adata.uns["rank_features_groups"]["pvals"]

        _utils._sort_features(adata)

        # Check that every feature is sorted
        # Actually, some of them would be sorted in the opposite direction (e.g. scores) for real data
        # But what matters here is that the permutation is the same for every key and is based on adjusted p-values
        for key in adata.uns["rank_features_groups"]:
            if key == "params":
                continue
            assert _is_sorted(adata.uns["rank_features_groups"][key]["group1"])
            assert _is_sorted(adata.uns["rank_features_groups"][key]["group2"])
