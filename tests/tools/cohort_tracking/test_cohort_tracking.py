from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from ehrapy.io._read import read_csv

CURRENT_DIR = Path(__file__).parent
_TEST_DATA_PATH = f"{CURRENT_DIR.parent}/test_data_features_ranking"


def _compare_dict_equal(dict1, dict2, tolerance=1e-9):
    if isinstance(dict1, dict) and isinstance(dict2, dict):
        if set(dict1.keys()) != set(dict2.keys()):
            return False
        for key in dict1.keys():
            if not _compare_dict_equal(dict1[key], dict2[key], tolerance):
                return False
        return True
    elif isinstance(dict1, list) and isinstance(dict2, list):
        if len(dict1) != len(dict2):
            return False
        for val1, val2 in zip(dict1, dict2):
            if not _compare_dict_equal(val1, val2, tolerance):
                return False
        return True
    elif isinstance(dict1, float) and isinstance(dict2, float):
        return abs(dict1 - dict2) < tolerance
    elif isinstance(dict1, str) and isinstance(dict2, str):
        return dict1 == dict2
    else:
        return dict1 == dict2


@pytest.fixture
def mini_adata():
    return read_csv(f"{_TEST_DATA_PATH}/dataset1.csv", columns_obs_only=["glucose", "weight", "disease", "station"])


@pytest.mark.parametrize("columns", [None, ["glucose", "weight", "disease", "station"]])
def test_CohortTracker_init_vanilla(columns, mini_adata):
    ct = ep.tl.CohortTracker(mini_adata, columns)
    assert ct._tracked_steps == 0
    assert ct.tracked_steps == 0
    assert ct._tracked_text == []
    assert ct._tracked_operations == []

    target_track = {
        "glucose": [],
        "weight": [],
        "disease": {"A": [], "B": [], "C": []},
        "station": {"ICU": [], "MICU": []},
    }
    assert _compare_dict_equal(ct.track, target_track)


def test_CohortTracker_init_set_columns(mini_adata):
    # limit columns
    ct = ep.tl.CohortTracker(mini_adata, columns=["glucose", "disease"])
    target_track = {
        "glucose": [],
        "disease": {"A": [], "B": [], "C": []},
    }
    assert _compare_dict_equal(ct.track, target_track)

    # invalid column
    with pytest.raises(ValueError):
        ep.tl.CohortTracker(
            mini_adata,
            columns=["glucose", "disease", "non_existing_column"],
        )

    # force categoricalization
    ct = ep.tl.CohortTracker(mini_adata, columns=["glucose", "disease"], categorical=["glucose", "disease"])
    target_track = {
        "glucose": {70: [], 80: [], 85: [], 90: [], 95: [], 120: [], 125: [], 130: [], 135: []},
        "disease": {"A": [], "B": [], "C": []},
    }
    assert _compare_dict_equal(ct.track, target_track)

    # invalid category
    with pytest.raises(ValueError):
        ep.tl.CohortTracker(
            mini_adata,
            columns=["glucose", "disease"],
            categorical=["station"],
        )


def test_CohortTracker_call(mini_adata):
    ct = ep.tl.CohortTracker(mini_adata)

    ct(mini_adata)
    assert ct.tracked_steps == 1
    assert ct._tracked_text == ["Cohort 0\n (n=12)"]
    target_track_1 = {
        "glucose": ["105.0 (23.6)"],
        "weight": ["76.0 (14.9)"],
        "disease": {"A": [33.3], "B": [33.3], "C": [33.3]},
        "station": {"ICU": [50.0], "MICU": [50.0]},
    }
    assert _compare_dict_equal(ct.track, target_track_1)

    ct(mini_adata)
    assert ct.tracked_steps == 2
    assert ct._tracked_text == ["Cohort 0\n (n=12)", "Cohort 1\n (n=12)"]
    target_track_2 = {
        "glucose": ["105.0 (23.6)", "105.0 (23.6)"],
        "weight": ["76.0 (14.9)", "76.0 (14.9)"],
        "disease": {"A": [33.3, 33.3], "B": [33.3, 33.3], "C": [33.3, 33.3]},
        "station": {"ICU": [50.0, 50.0], "MICU": [50.0, 50.0]},
    }
    assert _compare_dict_equal(ct.track, target_track_2)


def test_CohortTracker_reset(mini_adata):
    ct = ep.tl.CohortTracker(mini_adata)

    ct(mini_adata)
    ct(mini_adata)

    ct.reset()
    assert ct.tracked_steps == 0
    assert ct._tracked_text == []
    assert ct._tracked_operations == []

    target_track = {
        "glucose": [],
        "weight": [],
        "disease": {"A": [], "B": [], "C": []},
        "station": {"ICU": [], "MICU": []},
    }
    assert _compare_dict_equal(ct.track, target_track)
    assert _compare_dict_equal(ct._track_backup, target_track)


def test_CohortTracker_flowchart(mini_adata):
    ct = ep.tl.CohortTracker(mini_adata)

    ct(mini_adata, label="First step", operations_done="Some operations")
    ct(mini_adata, label="Second step", operations_done="Some other operations")

    ct.plot_flowchart()


def test_CohortTracker_plot_cohort_change(mini_adata):
    ct = ep.tl.CohortTracker(mini_adata)

    ct(mini_adata)
    ct(mini_adata)

    ct.plot_cohort_change(return_figure=True)
