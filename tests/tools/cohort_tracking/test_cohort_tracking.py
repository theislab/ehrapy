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
        tol=35,
    )

    ct(edata_mini, label="Second step", operations_done="Some other operations")
    fig2, _ = ct.plot_cohort_barplot(legend_labels={"weight": "weight(kg)", "glucose": "glucose(mg/dL)"}, show=False)

    check_same_image(
        fig=fig2,
        base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_edata_mini_step2_vanilla",
        tol=35,
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
        tol=35,
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
        tol=35,
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
        tol=35,
    )


def test_CohortTracker_branching_parent_resolution(edata_mini):
    ct = ep.tl.CohortTracker(edata_mini)

    with pytest.raises(ValueError, match="first tracked step"):
        ct(edata_mini, label="Root", parent=0)

    ct(edata_mini, label="Screened")
    ct(edata_mini[:8], label="Enrolled", operations_done="eligibility")
    ct(edata_mini[:4], label="Arm A", operations_done="randomized", parent="Enrolled")
    ct(edata_mini[4:8], label="Arm B", operations_done="randomized", parent=1)

    assert ct._tracked_parents == [None, 0, 1, 1]
    assert not ct._is_linear()

    with pytest.raises(ValueError, match="not found"):
        ct(edata_mini, label="Bad", parent="does-not-exist")
    with pytest.raises(ValueError, match="out of range"):
        ct(edata_mini, label="Bad", parent=99)


def test_CohortTracker_branching_ambiguous_parent(edata_mini):
    ct = ep.tl.CohortTracker(edata_mini)
    ct(edata_mini, label="Dup")
    ct(edata_mini, label="Dup")
    with pytest.raises(ValueError, match="ambiguous"):
        ct(edata_mini, label="Child", parent="Dup")


def test_CohortTracker_branching_linear_backcompat(edata_mini):
    ct = ep.tl.CohortTracker(edata_mini)
    ct(edata_mini, label="A")
    ct(edata_mini, label="B")
    ct(edata_mini, label="C")
    assert ct._tracked_parents == [None, 0, 1]
    assert ct._is_linear()


def test_CohortTracker_flowchart_linear(edata_mini):
    import holoviews as hv

    ct = ep.tl.CohortTracker(edata_mini)
    ct(edata_mini, label="Base Cohort")
    ct(edata_mini, operations_done="Some processing")

    plot = ct.plot_flowchart(title="linear")
    assert isinstance(plot, hv.core.Overlay)


def test_CohortTracker_flowchart_branched(edata_mini):
    import holoviews as hv

    ct = ep.tl.CohortTracker(edata_mini)
    ct(edata_mini, label="Screened")
    ct(edata_mini[:8], label="Enrolled", operations_done="eligibility")
    ct(edata_mini[:4], label="Arm A", operations_done="randomized", parent="Enrolled")
    ct(edata_mini[4:8], label="Arm B", operations_done="randomized", parent="Enrolled")

    plot = ct.plot_flowchart(title="CONSORT")
    assert isinstance(plot, hv.core.Overlay)


def test_CohortTracker_flowchart_empty(edata_mini):
    ct = ep.tl.CohortTracker(edata_mini)
    with pytest.raises(ValueError, match="No tracked steps"):
        ct.plot_flowchart()


def test_CohortTracker_flowchart_image(edata_mini, check_same_image):
    ct = ep.tl.CohortTracker(edata_mini)
    ct(edata_mini, label="Base Cohort")
    ct(edata_mini, operations_done="Some processing")

    plot = ct.plot_flowchart()
    check_same_image(
        fig=plot,
        base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_edata_mini_flowchart",
    )


def test_CohortTracker_flowchart_image_sensitivity(edata_mini, check_same_image):
    ct = ep.tl.CohortTracker(edata_mini)
    ct(edata_mini, label="Base Cohort")
    ct(edata_mini, operations_done="Some processing")

    # different node colour should diverge from the reference
    plot = ct.plot_flowchart(node_color="#ff0000")
    with pytest.raises(AssertionError):
        check_same_image(
            fig=plot,
            base_path=f"{_TEST_IMAGE_PATH}/cohorttracker_edata_mini_flowchart",
            tol=1e-1,
        )
