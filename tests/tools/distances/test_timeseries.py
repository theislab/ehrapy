from functools import partial

import ehrdata as ed
import numpy as np
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

from ehrapy.tools.distances.timeseries import timeseries_distance


@pytest.mark.parametrize("metric", ["dtw", "soft_dtw", "gak"])
def test_patient_timeseries_distance_numpy(metric):
    """Test timeseries distance function with raw numpy arrays."""
    time_series_data = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
            [[1.1, 2.1, 3.1, 4.1, 5.1, 6.1], [2.1, 3.1, 4.1, 5.1, 6.1, 7.1]],
            [[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [7.0, 6.0, 5.0, 4.0, 3.0, 2.0]],
            [[1.5, np.nan, 3.2, 4.1, 5.3, 6.1], [2.2, 3.1, 4.3, np.nan, 6.1, 7.2]],
        ]
    )

    patient_0 = np.array([0])
    distance_same = timeseries_distance(patient_0, patient_0, time_series_data, metric=metric)
    assert distance_same == 0.0

    patient_1 = np.array([1])
    distance_similar = timeseries_distance(patient_0, patient_1, time_series_data, metric=metric)
    assert distance_similar > 0
    assert distance_similar < 1.0

    patient_2 = np.array([2])
    distance_different = timeseries_distance(patient_0, patient_2, time_series_data, metric=metric)
    assert distance_different > distance_similar

    patient_3 = np.array([3])
    distance_with_nan = timeseries_distance(patient_0, patient_3, time_series_data, metric=metric)
    assert distance_with_nan > 0


@pytest.mark.parametrize("metric", ["dtw", "soft_dtw", "gak"])
def test_patient_timeseries_distance_insufficient_overlap(metric):
    """Test behavior when patients have insufficient valid timepoint overlap."""
    time_series_data = np.array(
        [
            [[1.0, 2.0, 3.0, np.nan, np.nan], [2.0, 3.0, 4.0, np.nan, np.nan]],
            [[np.nan, np.nan, np.nan, 4.0, 5.0], [np.nan, np.nan, np.nan, 5.0, 6.0]],
        ]
    )

    patient_0 = np.array([0])
    patient_1 = np.array([1])
    distance = timeseries_distance(patient_0, patient_1, time_series_data, metric=metric)
    assert distance == 0.0


@pytest.mark.parametrize("metric", ["dtw", "soft_dtw", "gak"])
def test_patient_timeseries_distance_with_ehrdata(metric):
    """Test timeseries distance function with EHRData object."""
    edata = ed.dt.ehrdata_blobs(
        n_observations=10,
        base_timepoints=20,
        cluster_std=0.7,
        n_centers=3,
        seasonality=True,
        time_shifts=True,
        variable_length=False,
        layer=DEFAULT_TEM_LAYER_NAME,
    )
    ts_metric = partial(timeseries_distance, arr=edata.layers[DEFAULT_TEM_LAYER_NAME], metric=metric)

    patient_0 = np.array([0])
    distance_same = ts_metric(patient_0, patient_0)
    assert distance_same == 0.0

    patient_1 = np.array([1])
    patient_5 = np.array([5])

    distance_01 = ts_metric(patient_0, patient_1)
    distance_05 = ts_metric(patient_0, patient_5)

    assert distance_01 > 0
    assert distance_05 > 0

    distance_10 = ts_metric(patient_1, patient_0)
    assert abs(distance_01 - distance_10) < 1e-10


@pytest.mark.parametrize("metric", ["dtw", "soft_dtw", "gak"])
def test_patient_timeseries_distance_edge_cases(metric):
    """Test edge cases for timeseries distance function."""
    time_series_data = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0]],
            [[2.0, 3.0, 4.0, 5.0, 6.0]],
        ]
    )

    patient_0 = np.array([0])
    patient_1 = np.array([1])

    distance = timeseries_distance(patient_0, patient_1, time_series_data, metric=metric)
    assert distance > 0
    assert np.isfinite(distance)

    time_series_data_nan = np.array(
        [[[np.nan, np.nan, np.nan, np.nan, np.nan]], [[np.nan, np.nan, np.nan, np.nan, np.nan]]]
    )

    distance_nan = timeseries_distance(patient_0, patient_1, time_series_data_nan, metric=metric)
    assert distance_nan == 0.0
