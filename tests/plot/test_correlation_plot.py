import ehrdata as ed
import holoviews as hv
import numpy as np
import pytest
from ehrdata import EHRData
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep


def test_correlation_heatmap(edata_blobs_timeseries_small):
    heatmap = ep.pl.variable_correlations(edata_blobs_timeseries_small, layer=DEFAULT_TEM_LAYER_NAME)
    assert heatmap is not None
    assert isinstance(heatmap, hv.Overlay)


def test_correlation_chord(edata_blobs_timeseries_small):
    chord = ep.pl.variable_dependencies(edata_blobs_timeseries_small, layer=DEFAULT_TEM_LAYER_NAME)
    assert chord is not None
    assert isinstance(chord, hv.Chord)


def test_correlation_heatmap_x():
    edata = ed.dt.ehrdata_blobs(n_variables=5, n_centers=2, n_observations=20, base_timepoints=3)

    heatmap = ep.pl.variable_correlations(edata)
    assert heatmap is not None
    assert isinstance(heatmap, hv.Overlay)


def test_correlation_chord_x():
    edata = ed.dt.ehrdata_blobs(n_variables=5, n_centers=2, n_observations=20, base_timepoints=3)

    chord = ep.pl.variable_dependencies(edata)
    assert chord is not None
    assert isinstance(chord, hv.Chord)
