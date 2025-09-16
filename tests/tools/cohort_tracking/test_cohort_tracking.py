from pathlib import Path

import pytest

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR.parent}/_images"


@pytest.mark.parametrize("columns", [None, ["glucose", "weight", "disease", "station"]])
def test_CohortTracker_init_vanilla(columns, edata_mini):
    ct = ep.tl.CohortTracker(edata_mini, columns)
    assert ct._tracked_steps == 0
    assert ct.tracked_steps == 0
    assert ct._tracked_text == []
    assert ct._tracked_operations == []


def test_CohortTracker_type_detection(edata_mini):
    ct = ep.tl.CohortTracker(edata_mini, ["glucose", "weight", "disease", "station"])
    assert set(ct.categorical) == {"disease", "station"}


def test_CohortTracker_init_set_columns(edata_mini):
    ep.tl.CohortTracker(edata_mini, columns=["glucose", "disease"])

    # invalid column
    with pytest.raises(ValueError):
        ep.tl.CohortTracker(
            edata_mini,
            columns=["glucose", "disease", "non_existing_column"],
        )

    # force categoricalization
    ep.tl.CohortTracker(edata_mini, columns=["glucose", "disease"], categorical=["glucose", "disease"])

    # invalid category
    with pytest.raises(ValueError):
        ep.tl.CohortTracker(
            edata_mini,
            columns=["glucose", "disease"],
            categorical=["station"],
        )


def test_CohortTracker_call(edata_mini):
    ct = ep.tl.CohortTracker(edata_mini)

    ct(edata_mini)
    assert ct.tracked_steps == 1
    assert ct._tracked_text == ["Cohort 0\n (n=12)"]

    ct(edata_mini)
    assert ct.tracked_steps == 2
    assert ct._tracked_text == ["Cohort 0\n (n=12)", "Cohort 1\n (n=12)"]

    edata_mini_col_name_gone = edata_mini.copy()
    edata_mini_col_name_gone.obs.rename(columns={"disease": "new_disease"}, inplace=True)
    with pytest.raises(ValueError):
        ct(edata_mini_col_name_gone)

    edata_mini_new_category = edata_mini.copy()
    edata_mini_new_category.obs["disease"] = edata_mini_new_category.obs["disease"].astype(str)
    edata_mini_new_category.obs.loc[edata_mini_new_category.obs["disease"] == "A", "disease"] = "new_disease"
    with pytest.raises(ValueError):
        ct(edata_mini_new_category)


def test_CohortTracker_plot_cohort_barplot_test_sensitivity(edata_mini, check_same_image):
    ct = ep.tl.CohortTracker(edata_mini)

    # e.g. different color should trigger error
    ct(edata_mini, label="First step", operations_done="Some operations")
    fig1, _ = ct.plot_cohort_barplot(show=False, color_palette="husl")

    with pytest.raises(AssertionError):
        check_same_image(
            fig=fig1,
            base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_edata_mini_step1_vanilla",
            tol=1e-1,
        )


def test_CohortTracker_plot_cohort_barplot_vanilla(edata_mini, check_same_image):
    ct = ep.tl.CohortTracker(edata_mini)

    ct(edata_mini, label="First step", operations_done="Some operations")
    fig1, _ = ct.plot_cohort_barplot(legend_labels={"weight": "weight(kg)", "glucose": "glucose(mg/dL)"}, show=False)

    check_same_image(
        fig=fig1,
        base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_edata_mini_step1_vanilla",
        tol=1e-1,
    )

    ct(edata_mini, label="Second step", operations_done="Some other operations")
    fig2, _ = ct.plot_cohort_barplot(legend_labels={"weight": "weight(kg)", "glucose": "glucose(mg/dL)"}, show=False)

    check_same_image(
        fig=fig2,
        base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_edata_mini_step2_vanilla",
        tol=1e-1,
    )


def test_CohortTracker_plot_cohort_barplot_use_settings(edata_mini, check_same_image):
    ct = ep.tl.CohortTracker(edata_mini)

    ct(edata_mini, label="First step", operations_done="Some operations")
    fig, _ = ct.plot_cohort_barplot(
        show=False,
        yticks_labels={"weight": "wgt"},
        legend_labels={"A": "Dis. A", "weight": "(kg)", "glucose": "glucose(mg/dL)"},
    )

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_edata_mini_step1_use_settings",
        tol=1e-1,
    )


def test_CohortTracker_plot_cohort_barplot_use_settings_big(edata_mini, check_same_image):
    ct = ep.tl.CohortTracker(edata_mini)

    ct(edata_mini, label="First step", operations_done="Some operations")
    fig, _ = ct.plot_cohort_barplot(
        show=False,
        yticks_labels={"weight": "wgt"},
        legend_labels={"A": "Dis. A", "weight": "(kg)"},
        legend_subtitles=True,
        legend_subtitles_names={"station": "", "disease": "dis", "weight": "wgt", "glucose": "glc"},
    )

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_edata_mini_step1_use_settings_big",
        tol=1e-1,
    )


def test_CohortTracker_plot_cohort_barplot_loosing_category(edata_mini, check_same_image):
    ct = ep.tl.CohortTracker(edata_mini)
    ct(edata_mini, label="First step", operations_done="Some operations")

    edata_mini = edata_mini[edata_mini.obs.disease == "A", :]
    ct(edata_mini)

    fig, _ = ct.plot_cohort_barplot(
        color_palette="colorblind", legend_labels={"weight": "weight(kg)", "glucose": "glucose(mg/dL)"}, show=False
    )

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_edata_mini_step2_loose_category",
        tol=1e-1,
    )


def test_CohortTracker_flowchart_sensitivity(edata_mini, check_same_image):
    ct = ep.tl.CohortTracker(edata_mini)

    ct(edata_mini, label="Base Cohort")
    ct(edata_mini, operations_done="Some processing")

    # check that e.g. different arrow size triggers error
    fig, _ = ct.plot_flowchart(show=False, arrow_size=0.5)

    with pytest.raises(AssertionError):
        check_same_image(
            fig=fig,
            base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_edata_mini_flowchart",
            tol=1e-1,
        )


def test_CohortTracker_flowchart(edata_mini, check_same_image):
    ct = ep.tl.CohortTracker(edata_mini)

    ct(edata_mini, label="Base Cohort")
    ct(edata_mini, operations_done="Some processing")

    fig, _ = ct.plot_flowchart(show=False)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_edata_mini_flowchart",
        tol=1e-1,
    )
