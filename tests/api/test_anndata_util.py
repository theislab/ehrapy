from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import ehrapy.api as ep
from ehrapy.api._anndata_util import (
    NotEncodedError,
    assert_encoded,
    assert_numeric_vars,
    get_numeric_vars,
    set_numeric_vars,
)

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}"


class TestAnnDataUtil:
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

        self.adata_encoded = ep.pp.encode(self.adata_strings.copy(), autodetect=True, encodings={})

    def test_assert_encoded(self):
        """Test for the encoding assertion."""

        assert_encoded(self.adata_encoded)

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            assert_encoded(self.adata_numeric)

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            assert_encoded(self.adata_strings)

    def test_get_numeric_vars(self):
        """Test for the numeric vars getter."""

        vars = get_numeric_vars(self.adata_encoded)
        assert vars == ["Numeric2"]

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            get_numeric_vars(self.adata_numeric)

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            get_numeric_vars(self.adata_strings)

    def test_assert_numeric_vars(self):
        "Test for the numeric vars assertion."

        assert_numeric_vars(self.adata_encoded, ["Numeric2"])

        with pytest.raises(ValueError, match=r"Some selected vars are not numeric"):
            assert_numeric_vars(self.adata_encoded, ["Numeric2", "String1"])

    def test_set_numeric_vars(self):
        """Test for the numeric vars setter."""

        values = np.array(
            [
                [1.2],
                [2.2],
                [2.2],
            ],
            dtype=np.dtype(np.float32),
        )
        adata_set = set_numeric_vars(self.adata_encoded, values, copy=True)
        assert (adata_set.X[:, 3] == values[:, 0]).all()

        with pytest.raises(ValueError, match=r"Some selected vars are not numeric"):
            set_numeric_vars(self.adata_encoded, values, vars=["ehrapycat_String1"])

        string_values = np.array(
            [
                ["A"],
                ["B"],
                ["A"],
            ]
        )

        with pytest.raises(TypeError, match=r"values must be numeric"):
            set_numeric_vars(self.adata_encoded, string_values)

        extra_values = np.array(
            [
                [1.2, 1.3],
                [2.2, 2.3],
                [2.2, 2.3],
            ],
            dtype=np.dtype(np.float32),
        )

        with pytest.raises(ValueError, match=r"does not match number of vars"):
            set_numeric_vars(self.adata_encoded, extra_values)

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            set_numeric_vars(self.adata_numeric, values)

        with pytest.raises(NotEncodedError, match=r"not yet been encoded"):
            set_numeric_vars(self.adata_strings, values)
