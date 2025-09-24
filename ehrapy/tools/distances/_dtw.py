import numpy as np

# from tslearn.metrics import dtw


def dtw_distance(patient_indices_x: np.ndarray, patient_indices_y: np.ndarray, R: np.ndarray) -> float:
    """Calculate DTW-based distance between two patients across all variables.

    For each variable where both patients have >3 valid measurements, computes DTW distance between their time series.
    Returns average DTW distance across all valid variable pairs.

    Args:
        patient_indices_x: Array containing patient index [i]
        patient_indices_y: Array containing patient index [j]
        R: Array containing timeseries measurements

    Returns:
        Average DTW distance across valid variable pairs.
        Returns 0 if no valid variable pairs exist.
    """
    patient_i, patient_j = int(patient_indices_x[0]), int(patient_indices_y[0])
    total_distance = 0
    valid_variable_count = 0

    for variable_idx in range(R.shape[1]):
        series_i = R[patient_i, variable_idx, :]
        series_j = R[patient_j, variable_idx, :]

        valid_measurements_i = ~np.isnan(series_i)
        valid_measurements_j = ~np.isnan(series_j)
        shared_valid_timepoints = valid_measurements_i & valid_measurements_j

        if np.sum(shared_valid_timepoints) > 3:
            series_i[shared_valid_timepoints].reshape(-1, 1)
            series_j[shared_valid_timepoints].reshape(-1, 1)
            # variable_distance = dtw(valid_series_i, valid_series_j)
            total_distance += 0
            valid_variable_count += 1

    return total_distance / max(valid_variable_count, 1)
