from pathlib import Path

import ehrdata as ed
import holoviews as hv

hv.extension("bokeh")
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent


def test_plot_timeseries(edata_blob_small):
    edata = edata_blob_small

    plot = ep.pl.plot_timeseries(edata, obs_names=1, layer=DEFAULT_TEM_LAYER_NAME)
    assert plot is not None
    assert isinstance(plot, hv.Layout)


def test_plot_timeseries_obs_id(edata_blob_small):
    edata = edata_blob_small

    plot = ep.pl.plot_timeseries(
        edata,
        obs_names="1",
        layer=DEFAULT_TEM_LAYER_NAME,
    )
    assert plot is not None
    assert isinstance(plot, hv.Layout)


def test_plot_timeseries_multiple_obs(edata_blob_small):
    edata = edata_blob_small

    plot = ep.pl.plot_timeseries(
        edata,
        obs_names=[3, 4],
        var_names=["feature_1", "feature_2", "feature_3"],
        layer=DEFAULT_TEM_LAYER_NAME,
    )

    assert plot is not None
    assert isinstance(plot, hv.Layout)


def test_plot_timeseries_overlay(edata_blob_small):
    edata = edata_blob_small

    plot = ep.pl.plot_timeseries(
        edata,
        obs_names=[3, 4, 5],
        var_names="feature_1",
        layer=DEFAULT_TEM_LAYER_NAME,
        overlay=True,
    )
    assert plot is not None
    assert isinstance(plot, hv.Overlay)


def test_plot_timeseries_error_cases(mar_edata, edata_blob_small):
    edata_2d_layer = mar_edata.X
    edata_2d = ed.EHRData(shape=(100, 10), layers={"X": edata_2d_layer})

    with pytest.raises(ValueError, match="Layer 'X' must be 3D"):
        ep.pl.plot_timeseries(
            edata_2d,
            obs_names=0,
            var_names="feature_1",
            layer="X",
        )
    with pytest.raises(KeyError, match="Column 'unknown_time' not found in edata.tem"):
        ep.pl.plot_timeseries(
            edata_blob_small,
            obs_names=0,
            var_names="feature_1",
            tem_time_key="unknown_time",
            layer=DEFAULT_TEM_LAYER_NAME,
        )
    with pytest.raises(KeyError, match="Variable 'unknown_feature' not found in edata.var_names"):
        ep.pl.plot_timeseries(
            edata_blob_small,
            obs_names=0,
            var_names="unknown_feature",
            tem_time_key="timepoint",
            layer=DEFAULT_TEM_LAYER_NAME,
        )
    with pytest.raises(ValueError, match="When overlay=True, only a single var_name can be plotted at a time"):
        ep.pl.plot_timeseries(
            edata_blob_small,
            obs_names=[0, 1],
            var_names=["feature_1", "feature_2"],
            layer=DEFAULT_TEM_LAYER_NAME,
            tem_time_key="timepoint",
            overlay=True,
        )
