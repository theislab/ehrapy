from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import ehrapy.api as ep

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_preprocessing"


class TestNormalization:
    def setup_method(self):
        obs_data = {"ID": ["Patient1", "Patient2", "Patient3"], "Age": [31, 94, 62]}

        X_data = np.array(
            [
                [1, 3.4, 2.0, 1.0, "A string", "A different string"],
                [2, 5.4, 5.0, 2.0, "Silly string", "A different string"],
                [2, 5.7, 3.0, np.nan, "A string", "What string?"],
            ],
            dtype=np.dtype(object),
        )
        var_data = {
            "Feature": ["Integer1", "Numeric1", "Numeric2", "Numeric3", "String1", "String2"],
            "Type": ["Integer", "Numeric", "Numeric", "Numeric", "String", "String"],
        }

        self.adata = AnnData(
            X=X_data,
            obs=pd.DataFrame(data=obs_data),
            var=pd.DataFrame(data=var_data, index=var_data["Feature"]),
            dtype=np.dtype(object),
        )

        self.adata = ep.pp.encode(self.adata, autodetect=True, encodings={})

    def test_methods_checks(self):
        """Test for checks that methods argument is valid."""

        with pytest.raises(ValueError, match=r"Some keys of methods are not available normalization methods"):
            ep.pp.normalize(self.adata, methods={"fail_method": ["Numeric2"]})

        with pytest.raises(ValueError, match=r"Some values of methods contain items which are not numeric variables"):
            ep.pp.normalize(self.adata, methods={"identity": ["String1"]})

    def test_norm_scale(self):
        """Test for the scaling normalization method."""

        adata_norm = ep.pp.normalize(self.adata, methods="scale", copy=True)

        num1_norm = np.array([-1.4039999, 0.55506986, 0.84893], dtype=np.float32)
        num2_norm = np.array([-1.069045, 1.3363061, -0.2672612], dtype=np.float32)

        assert np.array_equal(adata_norm.X[:, 0], self.adata.X[:, 0])
        assert np.array_equal(adata_norm.X[:, 1], self.adata.X[:, 1])
        assert np.array_equal(adata_norm.X[:, 2], self.adata.X[:, 2])
        assert np.allclose(adata_norm.X[:, 3], num1_norm)
        assert np.allclose(adata_norm.X[:, 4], num2_norm)
        assert np.allclose(adata_norm.X[:, 5], self.adata.X[:, 5], equal_nan=True)

    def test_norm_minmax(self):
        """Test for the minmax normalization method."""

        adata_norm = ep.pp.normalize(self.adata, methods="minmax", copy=True)

        num1_norm = np.array([0.0, 0.86956537, 0.9999999], dtype=np.dtype(np.float32))
        num2_norm = np.array([0.0, 1.0, 0.3333333], dtype=np.float32)

        assert np.array_equal(adata_norm.X[:, 0], self.adata.X[:, 0])
        assert np.array_equal(adata_norm.X[:, 1], self.adata.X[:, 1])
        assert np.array_equal(adata_norm.X[:, 2], self.adata.X[:, 2])
        assert np.allclose(adata_norm.X[:, 3], num1_norm)
        assert np.allclose(adata_norm.X[:, 4], num2_norm)
        assert np.allclose(adata_norm.X[:, 5], self.adata.X[:, 5], equal_nan=True)

    def test_norm_maxabs(self):
        """Test for the maxabs normalization method."""

        adata_norm = ep.pp.normalize(self.adata, methods="maxabs", copy=True)

        num1_norm = np.array([0.5964913, 0.94736844, 1.0], dtype=np.float32)
        num2_norm = np.array([0.4, 1.0, 0.6], dtype=np.float32)

        assert np.array_equal(adata_norm.X[:, 0], self.adata.X[:, 0])
        assert np.array_equal(adata_norm.X[:, 1], self.adata.X[:, 1])
        assert np.array_equal(adata_norm.X[:, 2], self.adata.X[:, 2])
        assert np.allclose(adata_norm.X[:, 3], num1_norm)
        assert np.allclose(adata_norm.X[:, 4], num2_norm)
        assert np.allclose(adata_norm.X[:, 5], self.adata.X[:, 5], equal_nan=True)

    def test_norm_robust_scale(self):
        """Test for the robust_scale normalization method."""

        adata_norm = ep.pp.normalize(self.adata, methods="robust_scale", copy=True)

        num1_norm = np.array([-1.73913043, 0.0, 0.26086957], dtype=np.float32)
        num2_norm = np.array([-0.66666667, 1.33333333, 0.0], dtype=np.float32)

        assert np.array_equal(adata_norm.X[:, 0], self.adata.X[:, 0])
        assert np.array_equal(adata_norm.X[:, 1], self.adata.X[:, 1])
        assert np.array_equal(adata_norm.X[:, 2], self.adata.X[:, 2])
        assert np.allclose(adata_norm.X[:, 3], num1_norm)
        assert np.allclose(adata_norm.X[:, 4], num2_norm)
        assert np.allclose(adata_norm.X[:, 5], self.adata.X[:, 5], equal_nan=True)

    def test_norm_quantile_uniform(self):
        """Test for the uniform quantile normalization method."""

        adata_norm = ep.pp.normalize(self.adata, methods="quantile_uniform", copy=True)

        num1_norm = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        num2_norm = np.array([0.0, 1.0, 0.5], dtype=np.float32)

        assert np.array_equal(adata_norm.X[:, 0], self.adata.X[:, 0])
        assert np.array_equal(adata_norm.X[:, 1], self.adata.X[:, 1])
        assert np.array_equal(adata_norm.X[:, 2], self.adata.X[:, 2])
        assert np.allclose(adata_norm.X[:, 3], num1_norm)
        assert np.allclose(adata_norm.X[:, 4], num2_norm)
        assert np.allclose(adata_norm.X[:, 5], self.adata.X[:, 5], equal_nan=True)

        # Single column requires extra manipulation so check this works
        adata_norm = ep.pp.normalize(self.adata, methods={"quantile_uniform": ["Numeric1"]}, copy=True)

        assert np.allclose(adata_norm.X[:, 3], num1_norm)

    def test_norm_quantile_normal(self):
        """Test for the normal quantile normalization method."""

        adata_norm = ep.pp.normalize(self.adata, methods="quantile_normal", copy=True)

        num1_norm = np.array([-5.19933758, 0.0, 5.19933758], dtype=np.float32)
        num2_norm = np.array([-5.19933758, 5.19933758, 0.0], dtype=np.float32)

        assert np.array_equal(adata_norm.X[:, 0], self.adata.X[:, 0])
        assert np.array_equal(adata_norm.X[:, 1], self.adata.X[:, 1])
        assert np.array_equal(adata_norm.X[:, 2], self.adata.X[:, 2])
        assert np.allclose(adata_norm.X[:, 3], num1_norm)
        assert np.allclose(adata_norm.X[:, 4], num2_norm)
        assert np.allclose(adata_norm.X[:, 5], self.adata.X[:, 5], equal_nan=True)

        # Single column requires extra manipulation so check this works
        adata_norm = ep.pp.normalize(self.adata, methods={"quantile_normal": ["Numeric1"]}, copy=True)

        assert np.allclose(adata_norm.X[:, 3], num1_norm)

    def test_norm_power_yeo_johnson(self):
        """Test for the Yeo-Johnson power normalization method."""

        adata_norm = ep.pp.normalize(self.adata, methods="power_yeo_johnson", copy=True)

        num1_norm = np.array([-1.3821232, 0.43163615, 0.950487], dtype=np.float32)
        num2_norm = np.array([-1.1953654, 1.252149, -0.05678357], dtype=np.float32)

        assert np.array_equal(adata_norm.X[:, 0], self.adata.X[:, 0])
        assert np.array_equal(adata_norm.X[:, 1], self.adata.X[:, 1])
        assert np.array_equal(adata_norm.X[:, 2], self.adata.X[:, 2])
        assert np.allclose(adata_norm.X[:, 3], num1_norm)
        assert np.allclose(adata_norm.X[:, 4], num2_norm)
        assert np.allclose(adata_norm.X[:, 5], self.adata.X[:, 5], equal_nan=True)

        # Single column requires extra manipulation so check this works
        adata_norm = ep.pp.normalize(self.adata, methods={"power_yeo_johnson": ["Numeric1"]}, copy=True)

        assert np.allclose(adata_norm.X[:, 3], num1_norm)

    def test_norm_power_box_cox(self):
        """Test for the Box-Cox power normalization method."""

        adata_norm = ep.pp.normalize(self.adata, methods="power_box_cox", copy=True)

        num1_norm = np.array([-1.3841851, 0.44104755, 0.9431376], dtype=np.float32)
        num2_norm = np.array([-1.205321, 1.2432859, -0.037965], dtype=np.float32)

        assert np.array_equal(adata_norm.X[:, 0], self.adata.X[:, 0])
        assert np.array_equal(adata_norm.X[:, 1], self.adata.X[:, 1])
        assert np.array_equal(adata_norm.X[:, 2], self.adata.X[:, 2])
        assert np.allclose(adata_norm.X[:, 3], num1_norm)
        assert np.allclose(adata_norm.X[:, 4], num2_norm)
        assert np.allclose(adata_norm.X[:, 5], self.adata.X[:, 5], equal_nan=True)

        # Single column requires extra manipulation so check this works
        adata_norm = ep.pp.normalize(self.adata, methods={"power_box_cox": ["Numeric1"]}, copy=True)

        assert np.allclose(adata_norm.X[:, 3], num1_norm)

    def test_norm_identity(self):
        """Test for the identity normalization method."""

        adata_norm = ep.pp.normalize(self.adata, methods="identity", copy=True)

        assert np.allclose(adata_norm.X, self.adata.X, equal_nan=True)
        assert np.allclose(adata_norm.layers["raw"], self.adata.X, equal_nan=True)

    def test_norm_mixed(self):
        """Test for normalization with mixed methods."""

        adata_norm = ep.pp.normalize(self.adata, methods={"minmax": ["Numeric1"], "scale": ["Numeric2"]}, copy=True)

        num1_norm = np.array([0.0, 0.86956537, 0.9999999], dtype=np.dtype(np.float32))
        num2_norm = np.array([-1.069045, 1.3363061, -0.2672612], dtype=np.float32)

        assert np.array_equal(adata_norm.X[:, 0], self.adata.X[:, 0])
        assert np.array_equal(adata_norm.X[:, 1], self.adata.X[:, 1])
        assert np.array_equal(adata_norm.X[:, 2], self.adata.X[:, 2])
        assert np.allclose(adata_norm.X[:, 3], num1_norm)
        assert np.allclose(adata_norm.X[:, 4], num2_norm)
        assert np.allclose(adata_norm.X[:, 5], self.adata.X[:, 5], equal_nan=True)
