import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from ehrdata import EHRData
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute


def test_correlation_heatmap(edata_blobs_timeseries_small):
    heatmap = ep.pl.variable_correlations(edata_blobs_timeseries_small, layer=DEFAULT_TEM_LAYER_NAME)
    assert heatmap is not None
    assert isinstance(heatmap, hv.Overlay)


def test_correlation_chord(edata_blobs_timeseries_small):
    chord = ep.pl.variable_dependencies(edata_blobs_timeseries_small, layer=DEFAULT_TEM_LAYER_NAME)
    assert chord is not None
    assert isinstance(chord, hv.Chord)


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("plot", [ep.pl.variable_correlations, ep.pl.variable_dependencies])
def test_correlation_plots_array_types(array_type, ndim, plot, edata_blobs_timeseries_small):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    tensor = edata_blobs_timeseries_small.layers[DEFAULT_TEM_LAYER_NAME]
    X = tensor if ndim == 3 else tensor[:, :, 0]
    kwargs = {"only_significant": False, "abs_correlation_threshold": 0} if plot is ep.pl.variable_dependencies else {}
    expected = plot(EHRData(X=X, var=edata_blobs_timeseries_small.var), **kwargs)
    edata = EHRData(X=array_type(X), var=edata_blobs_timeseries_small.var)

    if array_type.flags & Flags.Sparse:
        with pytest.raises(NotImplementedError):
            plot(edata, **kwargs)
        return

    with forbid_dask_compute(allowed=1):
        result = plot(edata, **kwargs)

    frames, expected_frames = (
        drawn.traverse(lambda element: element.dframe(), [hv.HeatMap, hv.Chord]) for drawn in (result, expected)
    )
    for frame, expected_frame in zip(frames, expected_frames, strict=True):
        pd.testing.assert_frame_equal(frame, expected_frame)
