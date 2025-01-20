from functools import singledispatch

import ehrdata as ed
import numpy as np
import xarray as xr

from ehrapy._compat import _raise_array_type_not_implemented


class StandardScaler3D:
    def __init__(self):
        """Standardize features by removing the mean and scaling to unit variance, across all samples and timesteps.

        This class is in concept similar to :class:`sklearn.preprocessing.StandardScaler`, but for a 3D array of shape (n_samples, n_features, n_timesteps).
        """
        self.mean_ = None
        self.std_ = None

    def _fit(self, data: np.ndarray) -> None:
        # Compute mean and std along the (0, 2) axes (samples and timesteps)
        self.mean_ = np.nanmean(data, axis=(0, 2), keepdims=True)
        self.scale_ = np.nanstd(data, axis=(0, 2), keepdims=True)

        self.scale_[self.scale_ == 0] = 1.0

    def fit(self, edata: ed.EHRData) -> None:
        """Fit the StandardScaler3D object to the input data.

        Computes the mean and standard deviation for each feature across all samples and timesteps from the input EHRData's `.r` field.
        If missing values are present, they are ignored during this computation.

        Args:
            edata: Input EHRData, from which's `.r` field, the statistics for the normalization are computed on.
        """

        if not isinstance(edata, ed.EHRData):
            raise ValueError("Input must be an EHRData object.")

        if edata.r.ndim != 3:
            raise ValueError("Input EHRData's .r field must be a 3D array.")

        self._fit(edata.r)

    def _transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_) / self.scale_

    def transform(self, edata: ed.EHRData, copy: bool = False) -> ed.EHRData | None:
        """
        Standardize the input data.

        If missing values are present, they are ignored and remain missing values.

        Args:
            edata: Input EHRData, which's `.r` field, the statistics for the normalization are computed on.

        Returns:
            The standardized input data.
        """
        if edata.r.ndim != 3:
            raise ValueError("Input EHRData's .r field must be a 3D array.")

        if copy:
            edata = edata.copy()
            edata.r = self._transform(edata.r)
            return edata
        else:
            edata.r = self._transform(edata.r)
            return None

    def fit_transform(self, edata: ed.EHRData, copy: bool = False) -> ed.EHRData | None:
        """
        Fit the StandardScaler3D to the input data, and apply the normalization to it.

        Args:
            edata: EHRData, or array of shape (n_samples, n_features, n_timesteps).

        Returns:
            The standardized input data.
        """
        self.fit(edata)
        return self.transform(edata, copy=copy)


def scale_norm_3d(edata: ed.EHRData, copy: bool = False) -> ed.EHRData | None:
    """
    Normalize the input data by scaling each feature across all samples and timesteps.

    Args:
        edata: Anndata object with shape (n_samples, n_features, n_timesteps).

    Returns:
        The normalized input data.
    """
    scaler = StandardScaler3D()
    return scaler.fit_transform(edata, copy=copy)


class LOCFImputer:
    def __init__(self, fallback_method: str = "mean"):
        """Impute missing values by carrying forward the last observed value.

        Args:
            fallback_method: The method to use for imputing missing values of timesteps before the first observation for a subject.
            Currently, only 'mean' is supported.
        """

        if fallback_method != "mean":
            raise ValueError("Only 'mean' is supported as a fallback method.")

        self.fallback_method = fallback_method
        self.mean_ = None

    def fit(self, edata: ed.EHRData) -> None:
        self.feature_means = np.nanmean(np.nanmean(edata.r, axis=0), axis=1)

    def transform(self, edata: ed.EHRData, copy: bool = False) -> ed.EHRData:
        X = xr.DataArray(edata.r)
        X = X.ffill("dim_2")

        mask = X.isnull().values

        feature_means_broadcasted = np.repeat(
            np.repeat(self.feature_means.reshape(1, -1, 1), X.shape[0], axis=0), X.shape[2], axis=2
        )

        X.values[mask] = feature_means_broadcasted[mask]

        if copy:
            edata = edata.copy()
            edata.r = X.values
            return edata
        else:
            edata.r = X.values

            return None

    def fit_transform(self, edata: ed.EHRData, copy: bool = False) -> ed.EHRData | None:
        self.fit(edata)
        return self.transform(edata)


def locf_impute(edata: ed.EHRData, copy: bool = False) -> ed.EHRData | None:
    """
    Impute missing values by carrying forward the last observed value.

    Args:
        edata: Anndata object with shape (n_samples, n_features, n_timesteps).

    Returns:
        The input data with missing values imputed.
    """
    imputer = LOCFImputer()
    return imputer.fit_transform(edata, copy=copy)
