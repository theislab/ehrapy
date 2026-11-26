from pathlib import Path

import ehrdata as ed
import holoviews as hv

hv.extension("bokeh")
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import curve_values, forbid_dask_compute

CURRENT_DIR = Path(__file__).parent


def test_timeseries(edata_blob_small):
    edata = edata_blob_small

    plot = ep.pl.timeseries(edata, obs_names="1", layer=DEFAULT_TEM_LAYER_NAME)
    assert plot is not None
    assert isinstance(plot, hv.Layout)


def test_timeseries_multiple_obs(edata_blob_small):
    edata = edata_blob_small

    plot = ep.pl.timeseries(
        edata,
        obs_names=["3", "4"],
        var_names=["feature_1", "feature_2", "feature_3"],
        layer=DEFAULT_TEM_LAYER_NAME,
    )

    assert plot is not None
    assert isinstance(plot, hv.Layout)


def test_timeseries_overlay(edata_blob_small):
    edata = edata_blob_small

    plot = ep.pl.timeseries(
        edata,
        obs_names=["3", "4", "5"],
        var_names="feature_1",
        layer=DEFAULT_TEM_LAYER_NAME,
        overlay=True,
    )
    assert plot is not None
    assert isinstance(plot, hv.Overlay)


def test_timeseries_subset_time(edata_blob_small):
    edata = edata_blob_small

    plot_1 = ep.pl.timeseries(
        edata,
        obs_names=["3", "4"],
        var_names=["feature_1", "feature_2", "feature_3"],
        tem_names=slice(0, 5),
        layer=DEFAULT_TEM_LAYER_NAME,
    )

    assert plot_1 is not None
    assert isinstance(plot_1, hv.Layout)


def test_timeseries_list(edata_blob_small):
    edata = edata_blob_small

    plot = ep.pl.timeseries(
        edata,
        obs_names=["3", "4"],
        var_names=["feature_1", "feature_2", "feature_3"],
        tem_names=["0", "1", "2"],
        layer=DEFAULT_TEM_LAYER_NAME,
    )

    assert plot is not None
    assert isinstance(plot, hv.Layout)


def test_timeseries_error_cases(mar_edata, edata_blob_small):
    edata_2d_layer = mar_edata.X
    edata_2d = ed.EHRData(shape=(100, 10), layers={"X": edata_2d_layer})

    with pytest.raises(ValueError, match="Layer 'X' must be 3D"):
        ep.pl.timeseries(
            edata_2d,
            obs_names="0",
            var_names="feature_1",
            layer="X",
        )

    with pytest.raises(KeyError, match="Layer 'unknown_layer' not found in edata.layers"):
        ep.pl.timeseries(
            edata_blob_small,
            obs_names="0",
            var_names="feature_0",
            layer="unknown_layer",
        )

    with pytest.raises(KeyError, match="unknown_feature not found in edata.var_names"):
        ep.pl.timeseries(
            edata_blob_small,
            obs_names="0",
            var_names="unknown_feature",
            layer=DEFAULT_TEM_LAYER_NAME,
        )
    with pytest.raises(ValueError, match="When overlay=True, only a single var_name can be plotted at a time"):
        ep.pl.timeseries(
            edata_blob_small,
            obs_names=["0", "1"],
            var_names=["feature_1", "feature_2"],
            layer=DEFAULT_TEM_LAYER_NAME,
            overlay=True,
        )


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("layer", [None, DEFAULT_TEM_LAYER_NAME])
@pytest.mark.parametrize("overlay", [False, True])
def test_timeseries_array_types(array_type, layer, overlay, edata_blob_small):
    tensor = edata_blob_small.layers[DEFAULT_TEM_LAYER_NAME]
    var_names = ["feature_1"] if overlay else ["feature_1", "feature_2"]
    kwargs = {
        "obs_names": ["3", "4"],
        "var_names": var_names,
        "tem_names": slice(2, 6),
        "layer": layer,
        "overlay": overlay,
    }

    def make_edata(X):
        return ed.EHRData(X=X, layers={DEFAULT_TEM_LAYER_NAME: X}, obs=edata_blob_small.obs, var=edata_blob_small.var)

    if array_type.flags & Flags.Sparse:
        with pytest.raises(ValueError, match="must be 3D"):
            ep.pl.timeseries(make_edata(array_type(tensor[:, :, 0])), **kwargs)
        return

    expected = ep.pl.timeseries(make_edata(tensor), **kwargs)
    with forbid_dask_compute(allowed=1):
        result = ep.pl.timeseries(make_edata(array_type(tensor)), **kwargs)

    for values, expected_values in zip(curve_values(result), curve_values(expected), strict=True):
        np.testing.assert_array_equal(values, expected_values)


def _panel_curves(plot: hv.Layout) -> dict[str, hv.Curve]:
    """The curves of the first panel of a layout, by label."""
    return {curve.label: curve for curve in plot.values()[0].traverse(lambda curve: curve, [hv.Curve])}


def test_timeseries_obsm_keys_and_events(edata_blob_small):
    """A value per timepoint from obsm is plotted as steps, an event from obs as a line at its position."""
    edata = edata_blob_small
    n_time = len(edata.tem)
    edata.obsm["score"] = np.tile(np.arange(n_time, dtype=float), (edata.n_obs, 1))
    edata.obs["onset"] = 2.0
    edata.obs["later"] = float(n_time + 5)
    edata.obs["missing"] = np.nan

    plot = ep.pl.timeseries(
        edata,
        obs_names="1",
        var_names=["feature_1"],
        layer=DEFAULT_TEM_LAYER_NAME,
        obsm_keys="score",
        events=["onset", "later", "missing"],
    )

    curves = _panel_curves(plot)
    assert list(curves) == ["feature_1", "score", "onset"]
    np.testing.assert_array_equal(curves["score"].dimension_values("value"), np.arange(n_time))
    assert curves["score"].opts.get(backend="bokeh").kwargs["interpolation"] == "steps-post"
    assert set(curves["onset"].dimension_values("time")) == {2.0}
    assert curves["onset"].opts.get(backend="bokeh").kwargs["line_dash"] == "dashed"


def test_timeseries_only_obsm_keys(edata_blob_small):
    """With no variables, a panel shows the obsm values alone."""
    edata = edata_blob_small
    edata.obsm["score"] = np.ones((edata.n_obs, len(edata.tem)))

    plot = ep.pl.timeseries(edata, obs_names=["1", "2"], var_names=[], layer=DEFAULT_TEM_LAYER_NAME, obsm_keys="score")

    assert len(plot) == 2
    assert list(_panel_curves(plot)) == ["score"]


def test_timeseries_time_key(edata_blob_small):
    """time_key places the timepoints and the events by a column of tem."""
    edata = edata_blob_small
    edata.tem["hour"] = 2 * np.arange(len(edata.tem)) - 4
    edata.obs["onset"] = 3.5

    plot = ep.pl.timeseries(
        edata,
        obs_names="1",
        var_names=["feature_1"],
        tem_names=slice(1, 6),
        layer=DEFAULT_TEM_LAYER_NAME,
        events="onset",
        time_key="hour",
    )

    curves = _panel_curves(plot)
    np.testing.assert_array_equal(curves["feature_1"].dimension_values("time"), edata.tem["hour"].to_numpy()[1:6])
    np.testing.assert_array_equal(curves["onset"].dimension_values("time"), [3.0, 3.0])


def test_timeseries_markers(edata_blob_small):
    """A marker gets a row of ticks below the panel: a variable where it has a value, an obsm key where it is not 0."""
    edata = edata_blob_small
    n_time = len(edata.tem)
    edata.layers[DEFAULT_TEM_LAYER_NAME][:, edata.var_names.get_loc("feature_2"), 3] = np.nan
    edata.obsm["flag"] = np.zeros((edata.n_obs, n_time))
    edata.obsm["flag"][:, [1, 4]] = 1
    edata.obsm["flag"][:, 5] = np.nan
    edata.obs["onset"] = 2.0

    plot = ep.pl.timeseries(
        edata,
        obs_names=["1", "2"],
        var_names=["feature_1"],
        layer=DEFAULT_TEM_LAYER_NAME,
        events="onset",
        markers=["feature_2", "flag"],
    )

    assert len(plot) == 4
    strip = plot.values()[1]
    ticks = {spikes.label: spikes for spikes in strip.traverse(lambda element: element, [hv.Spikes])}
    np.testing.assert_array_equal(ticks["feature_2"].dimension_values("time"), np.delete(np.arange(n_time), 3))
    np.testing.assert_array_equal(ticks["flag"].dimension_values("time"), [1, 4])
    assert (
        ticks["feature_2"].opts.get(backend="bokeh").kwargs["position"]
        > ticks["flag"].opts.get(backend="bokeh").kwargs["position"]
    )
    (line,) = strip.traverse(lambda element: element, [hv.Curve])
    np.testing.assert_array_equal(line.dimension_values("time"), [2.0, 2.0])
    hv.render(plot, backend="bokeh")
    hv.render(plot, backend="matplotlib")


def test_timeseries_only_markers(edata_blob_small):
    """With no variables and no obsm values, every observation gets only its strip of markers."""
    edata = edata_blob_small
    edata.obsm["flag"] = np.ones((edata.n_obs, len(edata.tem)))

    plot = ep.pl.timeseries(edata, obs_names=["1", "2"], var_names=[], layer=DEFAULT_TEM_LAYER_NAME, markers="flag")

    assert len(plot) == 2
    assert all(panel.traverse(lambda element: element, [hv.Spikes]) for panel in plot.values())


def test_timeseries_obsm_keys_and_events_errors(edata_blob_small):
    edata = edata_blob_small
    edata.obsm["score"] = np.ones((edata.n_obs, len(edata.tem)))
    edata.obsm["short"] = np.ones((edata.n_obs, 2))
    kwargs = {"obs_names": "1", "layer": DEFAULT_TEM_LAYER_NAME}

    with pytest.raises(KeyError, match="'unknown' not found in edata.obsm"):
        ep.pl.timeseries(edata, obsm_keys="unknown", **kwargs)
    with pytest.raises(ValueError, match="must hold a value per observation and timepoint"):
        ep.pl.timeseries(edata, obsm_keys="short", **kwargs)
    with pytest.raises(KeyError, match="'unknown' not found in edata.obs"):
        ep.pl.timeseries(edata, events="unknown", **kwargs)
    with pytest.raises(KeyError, match="'unknown' not found in edata.tem"):
        ep.pl.timeseries(edata, time_key="unknown", **kwargs)
    with pytest.raises(ValueError, match="need `overlay=False`"):
        ep.pl.timeseries(edata, var_names="feature_1", obsm_keys="score", overlay=True, **kwargs)
    with pytest.raises(ValueError, match="No variables selected"):
        ep.pl.timeseries(edata, var_names=[], **kwargs)
    with pytest.raises(KeyError, match="'unknown' not found in edata.var_names or edata.obsm"):
        ep.pl.timeseries(edata, markers="unknown", **kwargs)
    edata.obsm["feature_1"] = np.ones((edata.n_obs, len(edata.tem)))
    with pytest.raises(ValueError, match="both a variable and a key of edata.obsm"):
        ep.pl.timeseries(edata, markers="feature_1", **kwargs)


@pytest.fixture
def trajectories_edata() -> ed.EHRData:
    X = np.array(
        [
            [[1.0, 2.0, np.nan], [0.0, 0.0, 0.0]],
            [[3.0, np.nan, np.nan], [1.0, 1.0, 1.0]],
            [[5.0, 6.0, 7.0], [2.0, 2.0, 2.0]],
            [[7.0, 8.0, np.nan], [3.0, 3.0, 3.0]],
        ]
    )
    obs = pd.DataFrame({"group": pd.Categorical(["a", "a", "b", "b"])}, index=["0", "1", "2", "3"])
    return ed.EHRData(X=X, obs=obs, var=pd.DataFrame(index=["v0", "v1"]))


def _means(plot: hv.Overlay | hv.Layout) -> list[np.ndarray]:
    return [
        curve.dimension_values(1)
        for curve in plot.traverse(lambda curve: curve, [hv.Curve])
        if not isinstance(curve, hv.Area)
    ]


def test_trajectories_mean_and_ci(trajectories_edata):
    plot = ep.pl.trajectories(trajectories_edata, var_names="v0", groupby="group", ci=0.95)

    assert isinstance(plot, hv.Overlay)
    means = _means(plot)
    np.testing.assert_array_equal(means[0], [2.0, 2.0, np.nan])
    np.testing.assert_array_equal(means[1], [6.0, 7.0, 7.0])
    bands = plot.traverse(lambda area: (area.dimension_values(1), area.dimension_values(2)), [hv.Area])
    half_width = 1.959963984540054 * np.sqrt(2.0) / np.sqrt(2)
    np.testing.assert_allclose(bands[0][0][:1], [2.0 - half_width])
    np.testing.assert_allclose(bands[1][1][:2], [6.0 + half_width, 7.0 + half_width])
    assert np.isnan(bands[0][0][1:]).all()
    assert np.isnan(bands[1][0][2])


def test_trajectories_panels(trajectories_edata):
    layout = ep.pl.trajectories(trajectories_edata)
    without_band = ep.pl.trajectories(trajectories_edata, var_names="v1", ci=None)

    assert isinstance(layout, hv.Layout)
    assert len(layout) == 2
    np.testing.assert_array_equal(_means(without_band)[0], [1.5, 1.5, 1.5])
    assert not without_band.traverse(lambda area: area, [hv.Area])


def test_trajectories_error_cases(trajectories_edata, mar_edata):
    with pytest.raises(KeyError, match="'unknown' not found in edata.obs"):
        ep.pl.trajectories(trajectories_edata, groupby="unknown")
    with pytest.raises(ValueError, match="must be 3D"):
        ep.pl.trajectories(ed.EHRData(X=mar_edata.X))


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu | Flags.Sparse)
def test_trajectories_array_types(array_type, trajectories_edata):
    expected = ep.pl.trajectories(trajectories_edata, groupby="group")
    edata = ed.EHRData(X=array_type(trajectories_edata.X), obs=trajectories_edata.obs, var=trajectories_edata.var)

    with forbid_dask_compute(allowed=1):
        result = ep.pl.trajectories(edata, groupby="group")

    for values, expected_values in zip(curve_values(result), curve_values(expected), strict=True):
        np.testing.assert_array_equal(values, expected_values)
