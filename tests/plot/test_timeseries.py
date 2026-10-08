from pathlib import Path

import ehrdata as ed
import holoviews as hv

hv.extension("bokeh")
import numpy as np
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
