import holoviews as hv
import numpy as np
import pytest
from ehrdata import EHRData
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep


def test_correlation_heatmap(edata_blobs_timeseries_small):
    heatmap = ep.pl.plot_variable_correlations(edata_blobs_timeseries_small, layer=DEFAULT_TEM_LAYER_NAME)
    assert heatmap is not None
    assert isinstance(heatmap, hv.Overlay)


def test_correlation_chord(edata_blobs_timeseries_small):
    chord = ep.pl.plot_variable_dependencies(edata_blobs_timeseries_small, layer=DEFAULT_TEM_LAYER_NAME)
    assert chord is not None
    assert isinstance(chord, hv.Chord)
