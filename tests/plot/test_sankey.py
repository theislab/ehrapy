from pathlib import Path

import ehrdata as ed
import holoviews as hv
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


def test_sankey_plot(diabetes_130_fairlearn_sample_100, check_same_image, hv_backend):
    hv_backend("matplotlib")
    edata = diabetes_130_fairlearn_sample_100.copy()

    sankey = ep.pl.sankey_diagram(edata, columns=["gender", "race"])
    fig = hv.render(sankey, backend="matplotlib")

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/sankey",
        tol=2e-1,
    )


def test_sankey_time_plot(check_same_image, hv_backend):
    hv_backend("matplotlib")
    edata = ed.dt.ehrdata_blobs(base_timepoints=5, n_variables=1, n_observations=5, random_state=59)
    edata.layers[DEFAULT_TEM_LAYER_NAME] = edata.layers[DEFAULT_TEM_LAYER_NAME].astype(int)
    sankey_time = ep.pl.sankey_diagram_time(
        edata,
        var_name="feature_0",
        layer=DEFAULT_TEM_LAYER_NAME,
        state_labels={-2: "no", -3: "mild", -4: "moderate", -5: "severe", -6: "critical"},
    )

    fig = hv.render(sankey_time, backend="matplotlib")

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/sankey_time",
        tol=2e-1,
    )


def test_sankey_bokeh_plot(diabetes_130_fairlearn_sample_100, hv_backend):
    hv_backend("bokeh")
    edata = diabetes_130_fairlearn_sample_100.copy()

    sankey = ep.pl.sankey_diagram(edata, columns=["gender", "race"])

    assert isinstance(sankey, hv.Sankey)

    data = sankey.data
    required_columns = ["source", "target", "value"]
    for col in required_columns:
        assert col in data.columns

    assert len(data) > 0  # at least one flow
    assert (data["value"] > 0).all()  # flow values positive

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


def test_sankey_time_bokeh_plot(hv_backend):
    hv_backend("bokeh")
    edata = ed.dt.ehrdata_blobs(base_timepoints=5, n_variables=1, n_observations=5, random_state=59)
    edata.layers[DEFAULT_TEM_LAYER_NAME] = edata.layers[DEFAULT_TEM_LAYER_NAME].astype(int)
    sankey = ep.pl.sankey_diagram_time(
        edata,
        var_name="feature_0",
        layer=DEFAULT_TEM_LAYER_NAME,
        state_labels={-2: "no", -3: "mild", -4: "moderate", -5: "severe", -6: "critical"},
    )
    assert isinstance(sankey, hv.Sankey)

    data = sankey.data
    required_columns = ["source", "target", "value"]
    for col in required_columns:
        assert col in data.columns

    assert len(data) > 0
    assert (data["value"] > 0).all()

    # check that sources and targets contain state labels
    state_labels = ["no", "mild", "moderate", "severe", "critical"]
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


def test_error_cases():
    edata_time = ed.dt.ehrdata_blobs(base_timepoints=5, n_variables=1, n_observations=5, random_state=59)

    with pytest.raises(ValueError, match="Sankey requires discrete, binned states"):
        ep.pl.sankey_diagram_time(
            edata_time,
            var_name="feature_0",
            layer=DEFAULT_TEM_LAYER_NAME,
        )

    edata_time.layers[DEFAULT_TEM_LAYER_NAME] = edata_time.layers[DEFAULT_TEM_LAYER_NAME].astype(int)

    with pytest.raises(KeyError, match="unknown_feature not found in edata.var_names"):
        ep.pl.sankey_diagram_time(
            edata_time,
            var_name="unknown_feature",
            layer=DEFAULT_TEM_LAYER_NAME,
        )

    with pytest.raises(KeyError, match="unknown_layer not found in edata.layers"):
        ep.pl.sankey_diagram_time(
            edata_time,
            var_name="feature_0",
            layer="unknown_layer",
        )
