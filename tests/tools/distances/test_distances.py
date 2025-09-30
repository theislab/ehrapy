from functools import partial

import ehrdata as ed
import numpy as np

from ehrapy.tools.distances.dtw import dtw_distance


def test_patient_dtw_distance_numpy():
    """Test DTW distance function with raw numpy arrays."""
    # Create simple 3D test data: 4 patients, 2 variables, 5 timepoints
    time_series_data = np.array(
        [
            # Patient 0
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
            # Patient 1 (similar to patient 0)
            [[1.1, 2.1, 3.1, 4.1, 5.1, 6.1], [2.1, 3.1, 4.1, 5.1, 6.1, 7.1]],
            # Patient 2 (different pattern)
            [[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [7.0, 6.0, 5.0, 4.0, 3.0, 2.0]],
            # Patient 3 (with NaN values)
            [[1.5, np.nan, 3.2, 4.1, 5.3, 6.1], [2.2, 3.1, 4.3, np.nan, 6.1, 7.2]],
        ]
    )

    # Test same patient (should be 0)
    patient_0 = np.array([0])
    distance_same = dtw_distance(patient_0, patient_0, time_series_data)
    assert distance_same == 0.0

    # Test similar patients (should be small)
    patient_1 = np.array([1])
    distance_similar = dtw_distance(patient_0, patient_1, time_series_data)
    assert distance_similar > 0
    assert distance_similar < 1.0  # Should be small for similar patterns

    # Test different patients (should be larger)
    patient_2 = np.array([2])
    distance_different = dtw_distance(patient_0, patient_2, time_series_data)
    assert distance_different > distance_similar

    # Test patient with NaN values
    patient_3 = np.array([3])
    distance_with_nan = dtw_distance(patient_0, patient_3, time_series_data)
    assert distance_with_nan > 0  # Should still compute distance where valid overlap exists


def test_patient_dtw_distance_insufficient_overlap():
    """Test behavior when patients have insufficient valid timepoint overlap."""
    time_series_data = np.array(
        [
            # Patient 0: valid at timepoints 0,1,2
            [[1.0, 2.0, 3.0, np.nan, np.nan], [2.0, 3.0, 4.0, np.nan, np.nan]],
            # Patient 1: valid at timepoints 3,4 (no overlap)
            [[np.nan, np.nan, np.nan, 4.0, 5.0], [np.nan, np.nan, np.nan, 5.0, 6.0]],
        ]
    )

    patient_0 = np.array([0])
    patient_1 = np.array([1])
    distance = dtw_distance(patient_0, patient_1, time_series_data)
    # Should return 0 when no valid variable pairs exist
    assert distance == 0.0


def test_patient_dtw_distance_with_ehrdata():
    """Test DTW distance function with EHRData object."""
    edata = ed.dt.ehrdata_blobs(
        n_observations=10,
        base_timepoints=20,
        cluster_std=0.7,
        n_centers=3,
        seasonality=True,
        time_shifts=True,
        variable_length=False,
    )
    dtw_metric = partial(dtw_distance, R=edata.R)

    # Test same patient
    patient_0 = np.array([0])
    distance_same = dtw_metric(patient_0, patient_0)
    assert distance_same == 0.0

    # Test different patients
    patient_1 = np.array([1])
    patient_5 = np.array([5])

    distance_01 = dtw_metric(patient_0, patient_1)
    distance_05 = dtw_metric(patient_0, patient_5)

    assert distance_01 > 0
    assert distance_05 > 0

    # Distances should be symmetric
    distance_10 = dtw_metric(patient_1, patient_0)
    assert abs(distance_01 - distance_10) < 1e-10


def test_patient_dtw_distance_edge_cases():
    """Test edge cases for DTW distance function."""
    # Single variable, minimal timepoints
    time_series_data = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0]],  # Patient 0, Variable 0
            [[2.0, 3.0, 4.0, 5.0, 6.0]],  # Patient 1, Variable 0
        ]
    )

    patient_0 = np.array([0])
    patient_1 = np.array([1])

    distance = dtw_distance(patient_0, patient_1, time_series_data)
    assert distance > 0
    assert np.isfinite(distance)

    # Test with all NaN variable
    time_series_data_nan = np.array(
        [[[np.nan, np.nan, np.nan, np.nan, np.nan]], [[np.nan, np.nan, np.nan, np.nan, np.nan]]]
    )

    distance_nan = dtw_distance(patient_0, patient_1, time_series_data_nan)
    assert distance_nan == 0.0  # No valid comparisons
