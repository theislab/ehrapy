import ehrdata as ed
import holoviews as hv
import numpy as np
import pytest
from ehrdata import EHRData
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep


@pytest.mark.parametrize("layer", [DEFAULT_TEM_LAYER_NAME, None])
def test_correlation_heatmap(edata_blobs_timeseries_small, layer):
    if layer is None:
        edata = ed.dt.ehrdata_blobs(n_variables=5, n_centers=2, n_observations=20, base_timepoints=3)
    else:
        edata = edata_blobs_timeseries_small

    heatmap = ep.pl.variable_correlations(edata, layer=layer)
    assert heatmap is not None
    assert isinstance(heatmap, hv.Overlay)


@pytest.mark.parametrize("layer", [DEFAULT_TEM_LAYER_NAME, None])
def test_correlation_chord(edata_blobs_timeseries_small, layer):
    if layer is None:
        edata = ed.dt.ehrdata_blobs(n_variables=5, n_centers=2, n_observations=20, base_timepoints=3)
    else:
        edata = edata_blobs_timeseries_small

    chord = ep.pl.variable_dependencies(edata, layer=layer)
    assert chord is not None
    assert isinstance(chord, hv.Chord)
