from pathlib import Path

import ehrdata as ed
import holoviews as hv
import numpy as np
import pandas as pd
import pytest

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


@pytest.fixture
def ehr_3d_mini():
    layer = np.array(
        [
            [[0, 1, 2, 1, 2], [1, 2, 1, 2, 0]],
            [[1, 2, 0, 2, 1], [2, 1, 2, 1, 2]],
            [[2, 0, 1, 2, 0], [2, 0, 1, 1, 2]],
            [[1, 2, 1, 0, 1], [0, 2, 1, 2, 0]],
            [[0, 2, 1, 2, 2], [1, 2, 1, 0, 2]],
            [[0, 1, 2, 1, 2], [1, 2, 1, 2, 0]],
            [[1, 2, 0, 2, 1], [2, 1, 2, 1, 2]],
            [[2, 0, 1, 2, 0], [2, 0, 1, 1, 2]],
            [[1, 2, 1, 0, 1], [0, 2, 1, 2, 0]],
            [[0, 2, 1, 2, 2], [1, 2, 1, 0, 2]],
        ]
    )

    edata = ed.EHRData(
        layers={"layer_1": layer},
        obs=pd.DataFrame(index=["patient 1", "patient 2", "patient 3", "patient 4", "patient 5"]),
        # obs_names=["patient 1", "patient 2", "patient 3"],
        var=pd.DataFrame(index=["treatment", "disease_flare"]),
        # var_names=["treatment", "disease_flare"],
        tem=pd.DataFrame(index=["visit_0", "visit_1", "visit_2", "visit_3", "visit_4"]),
    )
    return edata


@pytest.fixture
def diabetes_130_fairlearn_sample_100():
    edata = ed.dt.diabetes_130_fairlearn(
        columns_obs_only=[
            "race",
            "gender",
        ]
    )[:100]

    return edata


def test_sankey_plot(diabetes_130_fairlearn_sample_100, check_same_image):
    hv.extension("matplotlib")
    edata = diabetes_130_fairlearn_sample_100.copy()

    sankey = ep.pl.plot_sankey(edata, columns=["gender", "race"])
    fig = hv.render(sankey, backend="matplotlib")

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/sankey",
        tol=2e-1,
    )


def test_sankey_time_plot(ehr_3d_mini, check_same_image):
    hv.extension("matplotlib")
    edata = ehr_3d_mini
    sankey_time = ep.pl.plot_sankey_time(
        edata,
        columns=["disease_flare"],
        layer="layer_1",
        state_labels={0: "no flare", 1: "mid flare", 2: "severe flare"},
    )

    fig = hv.render(sankey_time, backend="matplotlib")

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/sankey_time",
        tol=2e-1,
    )


def test_sankey_bokeh_plot(diabetes_130_fairlearn_sample_100):
    hv.extension("bokeh")
    edata = diabetes_130_fairlearn_sample_100.copy()

    sankey = ep.pl.plot_sankey(edata, columns=["gender", "race"])

    assert isinstance(sankey, hv.Sankey)

    data = sankey.data
    required_columns = ["source", "target", "value"]
    for col in required_columns:
        assert col in data.columns

    assert len(data) > 0  # at least one flow
    assert (data["value"] > 0).all()  # flow values positive
    assert data["value"].sum() == len(edata.obs)  # total flow must match total obs

    # each flow matches the expected count
    for _, row in data.iterrows():
        gender_value = row["source"].split(": ")[1]
        race_value = row["target"].split(": ")[1]
        flow_value = row["value"]

        expected_count = len(edata.obs[(edata.obs["gender"] == gender_value) & (edata.obs["race"] == race_value)])

        assert flow_value == expected_count

    for source in data["source"].unique():
        assert source.startswith("gender:")  # sources have the correct prefix
    for target in data["target"].unique():
        assert target.startswith("race:")  # targets have the correct prefix


def test_sankey_time_plot(ehr_3d_mini, check_same_image):
    edata = ehr_3d_mini
    sankey_time = ep.pl.plot_sankey_time(
        edata,
        columns=["disease_flare"],
        layer="layer_1",
        state_labels={0: "no flare", 1: "mid flare", 2: "severe flare"},
        backend="matplotlib",
    )

    fig = hv.render(sankey_time, backend="matplotlib")

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/sankey_time",
        tol=2e-1,
    )


def test_sankey_time_bokeh_plot(ehr_3d_mini):
    hv.extension("bokeh")
    edata = ehr_3d_mini
    sankey = ep.pl.plot_sankey_time(
        edata,
        columns=["disease_flare"],
        layer="layer_1",
        state_labels={0: "no flare", 1: " mid flare", 2: "severe flare"},
        backend="bokeh",
        edata,
        columns=["disease_flare"],
        layer="layer_1",
        state_labels={0: "no flare", 1: " mid flare", 2: "severe flare"},
    )
    assert isinstance(sankey, hv.Sankey)

    data = sankey.data
    required_columns = ["source", "target", "value"]
    for col in required_columns:
        assert col in data.columns

    assert len(data) > 0
    assert (data["value"] > 0).all()

    # check that sources and targets contain state labels
    state_labels = ["no flare", "mid flare", "severe flare"]
    for source in data["source"].unique():
        assert any(label in source for label in state_labels)
        assert "(" in source and ")" in source
    for target in data["target"].unique():
        assert any(label in target for label in state_labels)
        assert "(" in target and ")" in target

    # check conservation of patients across time points
    time_steps = edata.tem.index.tolist()
    first_time = time_steps[0]
    last_time = time_steps[-1]

    outflow_first = data[data["source"].str.contains(f"\\({first_time}\\)", regex=True)]["value"].sum()
    inflow_last = data[data["target"].str.contains(f"\\({last_time}\\)", regex=True)]["value"].sum()

    assert outflow_first == inflow_last

    # total flow equals number of transitions
    n_patients = edata.n_obs
    n_transitions = len(time_steps) - 1
    expected_total_flow = n_patients * n_transitions

    assert data["value"].sum() == expected_total_flow
