from pathlib import Path

import numpy as np
from anndata import AnnData

from ehrapy.api.preprocessing import Normalization

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_preprocessing"


class TestNormalization:
    def test_identity(self):
        """Test for the identity normalization.

        Created as a template during development. Should be removed before merging.
        """
        # NaN value and empty string replacement with single value
        nan_empty_str_array = np.array([["column 1", "column 2", "column 3", "column 4"], [5, np.NaN, "", "not empty"]])
        adata = AnnData(X=nan_empty_str_array, dtype=np.dtype(object))
        adata_norm = Normalization.identity(adata, copy=True)

        assert (adata.X == adata_norm.X).all()
