import numpy as np
import pandas as pd

import ehrapy as ep


class TestHelperFunctions:
    def test_adjust_pvalues(self):
        pvals: np.recarray = pd.DataFrame({"group1": (0.01, 0.02, 0.03, 1.00), "group2": (0.04, 0.05, 0.06, 0.99)}).to_records()
        expected_result_bh = pd.DataFrame({"group1": (0.04, 0.04, 0.04, 1.00), "group2": (0.08, 0.08, 0.08, 0.99)}).to_records()
        expected_result_bf = pd.DataFrame({"group1": (0.04, 0.08, 0.12, 1.00), "group2": (0.16, 0.20, 0.24, 1.00)}).to_records()

        result_bh = ep.tl._adjust_pvalues(pvals, method="benjamini-hochberg")
        assert pvals["group1"] <= result_bh["group1"]
        assert pvals["group2"] <= result_bh["group2"]
        assert np.allclose(result_bh, expected_result_bh)
        
        result_bf = ep.tl._adjust_pvalues(pvals, method="bonferroni")
        assert pvals["group1"] <= result_bf["group1"]
        assert pvals["group2"] <= result_bf["group2"]
        assert np.allclose(result_bf, expected_result_bf)
