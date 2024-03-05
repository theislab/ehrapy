from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from ehrapy.io._read import read_csv

CURRENT_DIR = Path(__file__).parent
_TEST_DATA_PATH = f"{CURRENT_DIR.parent}/test_data_features_ranking"


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


def test_CohortTracker_type_detection(mini_adata):
    ct = ep.tl.CohortTracker(mini_adata, ["glucose", "weight", "disease", "station"])
    assert set(ct.categorical) == {"disease", "station"}


def test_CohortTracker_init_set_columns(mini_adata):
    # limit columns
    ep.tl.CohortTracker(mini_adata, columns=["glucose", "disease"])

    # TODO: check plot?

    # invalid column
    with pytest.raises(ValueError):
        ep.tl.CohortTracker(
            mini_adata,
            columns=["glucose", "disease", "non_existing_column"],
        )

    # force categoricalization
    ep.tl.CohortTracker(mini_adata, columns=["glucose", "disease"], categorical=["glucose", "disease"])

    # TODO: check plot?

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

    # TODO: check plot?

    ct(mini_adata)
    assert ct.tracked_steps == 2
    assert ct._tracked_text == ["Cohort 0\n (n=12)", "Cohort 1\n (n=12)"]


# TODO: check plot?


def test_CohortTracker_reset(mini_adata):
    ct = ep.tl.CohortTracker(mini_adata)

    ct(mini_adata)
    ct(mini_adata)

    ct.reset()
    assert ct.tracked_steps == 0
    assert ct._tracked_text == []
    assert ct._tracked_operations == []


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
