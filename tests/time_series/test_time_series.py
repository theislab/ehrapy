import ehrdata as ed
import numpy as np
import pytest

import ehrapy as ep


@pytest.fixture
def edata_mini_timeseries():
    edata = ed.EHRData(
        X=np.zeros((2, 2)),
        r=np.array(
            [
                [[1, 2], [np.nan, 5]],
                [[3, np.nan], [5, 8]],
            ]
        ),
    )
    return edata


def test_StandardScaler3D(edata_mini_timeseries):
    scaler3d = ep.ts.StandardScaler3D()
    scaler3d.fit(edata_mini_timeseries)
    assert scaler3d.mean_ is not None
    assert scaler3d.scale_ is not None

    assert np.allclose(scaler3d.mean_, np.array([[2], [6]]))
    assert np.allclose(scaler3d.scale_, np.array([[0.81649658], [1.41421356]]))

    scaler3d.transform(edata_mini_timeseries)

    assert np.allclose(
        edata_mini_timeseries.r,
        [[[-1.22474487, 0.0], [np.nan, -0.70710678]], [[1.22474487, np.nan], [-0.70710678, 1.41421356]]],
        equal_nan=True,
    )


def test_scale_norm_3d(edata_mini_timeseries):
    ep.ts.scale_norm_3d(edata_mini_timeseries)

    assert np.allclose(
        edata_mini_timeseries.r,
        [[[-1.22474487, 0.0], [np.nan, -0.70710678]], [[1.22474487, np.nan], [-0.70710678, 1.41421356]]],
        equal_nan=True,
    )


def test_LOCFImputer(edata_mini_timeseries):
    locf_imputer = ep.ts.LOCFImputer()
    locf_imputer.fit(edata_mini_timeseries)

    locf_imputer.transform(edata_mini_timeseries)

    assert np.allclose(
        edata_mini_timeseries.r,
        np.array(
            [
                [[1, 2], [5, 4]],
                [[3, 3], [5, 6]],
            ]
        ),
    )


def test_locf_impute(edata_mini_timeseries):
    ep.ts.locf_impute(edata_mini_timeseries)

    assert np.allclose(
        edata_mini_timeseries.r,
        np.array(
            [
                [[1, 2], [5, 4]],
                [[3, 3], [5, 6]],
            ]
        ),
    )
