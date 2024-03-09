from pathlib import Path

import pytest

import ehrapy as ep
from ehrapy.io._read import read_csv

CURRENT_DIR = Path(__file__).parent
_TEST_DATA_PATH = f"{CURRENT_DIR.parent}/test_data_features_ranking"
_TEST_IMAGE_PATH = f"{CURRENT_DIR.parent}/_images"


@pytest.fixture
def adata_mini():
    return read_csv(f"{_TEST_DATA_PATH}/dataset1.csv", columns_obs_only=["glucose", "weight", "disease", "station"])


@pytest.mark.parametrize("columns", [None, ["glucose", "weight", "disease", "station"]])
def test_CohortTracker_init_vanilla(columns, adata_mini):
    ct = ep.tl.CohortTracker(adata_mini, columns)
    assert ct._tracked_steps == 0
    assert ct.tracked_steps == 0
    assert ct._tracked_text == []
    assert ct._tracked_operations == []


def test_CohortTracker_type_detection(adata_mini):
    ct = ep.tl.CohortTracker(adata_mini, ["glucose", "weight", "disease", "station"])
    assert set(ct.categorical) == {"disease", "station"}


def test_CohortTracker_init_set_columns(adata_mini):
    # limit columns
    ep.tl.CohortTracker(adata_mini, columns=["glucose", "disease"])

    # invalid column
    with pytest.raises(ValueError):
        ep.tl.CohortTracker(
            adata_mini,
            columns=["glucose", "disease", "non_existing_column"],
        )

    # force categoricalization
    ep.tl.CohortTracker(adata_mini, columns=["glucose", "disease"], categorical=["glucose", "disease"])

    # invalid category
    with pytest.raises(ValueError):
        ep.tl.CohortTracker(
            adata_mini,
            columns=["glucose", "disease"],
            categorical=["station"],
        )


def test_CohortTracker_call(adata_mini):
    ct = ep.tl.CohortTracker(adata_mini)

    ct(adata_mini)
    assert ct.tracked_steps == 1
    assert ct._tracked_text == ["Cohort 0\n (n=12)"]

    ct(adata_mini)
    assert ct.tracked_steps == 2
    assert ct._tracked_text == ["Cohort 0\n (n=12)", "Cohort 1\n (n=12)"]


def test_CohortTracker_plot_cohort_change_test_sensitivity(adata_mini, check_same_image):
    ct = ep.tl.CohortTracker(adata_mini)

    # check that e.g. different color triggers error
    ct(adata_mini, label="First step", operations_done="Some operations")
    fig1, _ = ct.plot_cohort_change(show=False, color_palette="husl")

    with pytest.raises(AssertionError):
        check_same_image(
            fig=fig1,
            base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_adata_mini_step1",
            tol=1e-1,
        )


def test_CohortTracker_plot_cohort_change(adata_mini, check_same_image):
    ct = ep.tl.CohortTracker(adata_mini)

    ct(adata_mini, label="First step", operations_done="Some operations")
    fig1, _ = ct.plot_cohort_change(show=False)

    check_same_image(
        fig=fig1,
        base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_adata_mini_step1",
        tol=1e-1,
    )

    ct(adata_mini, label="Second step", operations_done="Some other operations")
    fig2, _ = ct.plot_cohort_change(show=False)

    check_same_image(
        fig=fig2,
        base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_adata_mini_step2",
        tol=1e-1,
    )


def test_CohortTracker_flowchart_sensitivity(adata_mini, check_same_image):
    ct = ep.tl.CohortTracker(adata_mini)

    ct(adata_mini, label="Base Cohort")
    ct(adata_mini, operations_done="Some processing")

    # check that e.g. different arrow size triggers error
    fig, _ = ct.plot_flowchart(show=False, arrow_size=0.5)

    with pytest.raises(AssertionError):
        check_same_image(
            fig=fig,
            base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_adata_mini_flowchart",
            tol=1e-1,
        )


def test_CohortTracker_flowchart(adata_mini, check_same_image):
    ct = ep.tl.CohortTracker(adata_mini)

    ct(adata_mini, label="Base Cohort")
    ct(adata_mini, operations_done="Some processing")

    fig, _ = ct.plot_flowchart(show=False)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_adata_mini_flowchart",
        tol=1e-1,
    )
