from pathlib import Path

import numpy as np
from anndata import AnnData

from ehrapy.api.preprocessing import Imputation

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_preprocessing"


class TestImputation:
    def test_explicit_replace_single_value(self):
        """Tests for scenario one of explicit_replace.

        We want to ensure that all np.NaNs as well as empty strings are being imputed.
        """
        # NaN value and empty string replacement with single value
        nan_empty_str_array = np.array([["column 1", "column 2", "column 3", "column 4"], [5, np.NaN, "", "not empty"]])
        adata = AnnData(X=nan_empty_str_array, dtype=np.dtype(object))
        imputed_adata = Imputation.explicit(adata, replacement=0, impute_empty_strings=True, copy=True)

        # Run costly, but explicit checks
        for col in imputed_adata.X:
            for val in col:
                assert val != "" and val != np.NaN

    def test_explicit_replace_subset_columns(self):
        pass

    def test_explicit_replace_multiple_subset_columns(self):
        pass
