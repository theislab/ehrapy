from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from ehrapy.api.preprocessing import Normalization
from ehrapy.api.preprocessing.encoding._encode import Encoder

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_preprocessing"


class TestNormalization:
    def setup_method(self):
        obs_data = {"ID": ["Patient1", "Patient2", "Patient3"], "Age": [31, 94, 62]}

        X_numeric = np.array([[1, 3.4, 2.1, 4], [2, 6.9, 7.6, 2], [1, 4.5, 1.3, 7]], dtype=np.dtype(object))
        var_numeric = {
            "Feature": ["Numeric1", "Numeric2", "Numeric3", "Numeric4"],
            "Type": ["Numeric", "Numeric", "Numeric", "Numeric"],
        }
        X_strings = np.array(
            [
                [1, 3.4, "A string", "A different string"],
                [2, 5.4, "Silly string", "A different string"],
                [2, 5.7, "A string", "What string?"],
            ],
            dtype=np.dtype(object),
        )
        var_strings = {
            "Feature": ["Numeric1", "Numeric2", "String1", "String2"],
            "Type": ["Numeric", "Numeric", "String", "String"],
        }

        self.adata_numeric = AnnData(
            X=X_numeric,
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_numeric, index=var_numeric["Feature"]),
            dtype=np.dtype(object),
        )

        self.adata_strings = AnnData(
            X=X_strings,
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_strings, index=var_strings["Feature"]),
            dtype=np.dtype(object),
        )

        adata = self.adata_strings
        adata.layers["original"] = adata.X.copy()  # Can be removed once Issue 117 is fixed
        self.adata_encoded = Encoder.encode(adata, autodetect=True)

    def test_identity(self):
        """Test for the identity normalization.

        Created as a template during development. Should be removed before merging.
        """
        # NaN value and empty string replacement with single value
        nan_empty_str_array = np.array([["column 1", "column 2", "column 3", "column 4"], [5, np.NaN, "", "not empty"]])
        adata = AnnData(X=nan_empty_str_array, dtype=np.dtype(object))
        adata_norm = Normalization.identity(adata, copy=True)

        assert (adata.X == adata_norm.X).all()
