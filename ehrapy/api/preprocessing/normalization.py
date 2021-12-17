from typing import Dict, Optional, Union

from anndata import AnnData
from numpy import array
from sklearn.preprocessing import minmax_scale, scale

from ehrapy.api._anndata_util import assert_encoded, get_column_indices, get_column_values, get_numeric_vars


class Normalization:
    """Provides functions to normalize continuous features"""

    available_methods = {"scale", "minmax", "identity"}

    @staticmethod
    def _normalize(adata: AnnData, methods: Union[Dict[str, str], str], copy: bool = False) -> Optional[AnnData]:
        """Normalize numeric variable.

        This function normalizes the numeric variables in an AnnData object.

        Available normalization methods are:

        1. scale (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale)
        2. minmax (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)
        3. identity (return the un-normalized values)

        Args:
            adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encode using ~ehrapy.preprocessing.encode.encode.
            methods: Methods to use for normalization. Either:

                str: Name of the method to use for all numeric variable

                Dict: A dictionary specifying the method for each numeric variable where keys are variable and values are methods
            copy: Whether to return a copy or act in place

        Returns:
            :class:`~anndata.AnnData` object with normalized X
        """

        assert_encoded(adata)

        num_vars = get_numeric_vars(adata)
        if isinstance(methods, str):
            methods = dict.fromkeys(num_vars, methods)
        else:
            if not set(methods.keys()) <= set(num_vars):
                raise ValueError("Some keys of methods are not numeric variables")
            if not set(methods.values()) <= Normalization.available_methods:
                raise ValueError(
                    "Some values of methods are not available normalization methods. Available methods are:"
                    f"{Normalization.available_methods}"
                )

        if copy:
            adata = adata.copy()

        adata.layers["raw"] = adata.X.copy()

        for var, method in methods.items():
            var_idx = get_column_indices(adata, var)
            var_values = get_column_values(adata, var_idx)

            if method == "scale":
                adata.X[:, var_idx] = Normalization._norm_scale(var_values)
            elif method == "minmax":
                adata.X[:, var_idx] = Normalization._norm_minmax(var_values)
            elif method == "identity":
                adata.X[:, var_idx] = Normalization._norm_identity(var_values)

        return adata

    @staticmethod
    def _norm_scale(values: array) -> Optional[AnnData]:
        """Apply standard scaling normalization.

        Args:
            values: A single column numpy array

        Returns:
            Single column numpy array with scaled values
        """

        return scale(values)

    @staticmethod
    def _norm_minmax(values: array) -> Optional[AnnData]:
        """Apply minmax normalization.

        Args:
            values: A single column numpy array

        Returns:
            Single column numpy array with minmax scaled values
        """

        return minmax_scale(values)

    @staticmethod
    def _norm_identity(values: array) -> Optional[AnnData]:
        """Apply identity normalization.

        Args:
            values: A single column numpy array

        Returns:
            Single column numpy array with normalized values
        """

        return values
