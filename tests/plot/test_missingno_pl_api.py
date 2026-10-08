from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import ehrdata as ed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_missing_values_barplot(mimic_2, check_same_image, layer, clean_up_plots):
    if layer is not None:
        mimic_2.X = None
    plot = ep.pl.missing_values_barplot(mimic_2, filter="bottom", max_cols=15, max_percentage=0.999, layer=layer)
    fig = plot.figure
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/missing_values_barplot",
        tol=25,
    )


def test_missing_values_barplot_3D(edata_blob_small, clean_up_plots):
    ep.pl.missing_values_barplot(edata_blob_small)
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pl.missing_values_barplot(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_missing_values_matrixplot(mimic_2, check_same_image, layer, clean_up_plots):
    if layer is not None:
        mimic_2.X = None
    plot = ep.pl.missing_values_matrix(mimic_2, filter="bottom", max_cols=15, max_percentage=0.999, layer=layer)
    fig = plot.figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/missing_values_matrix",
        tol=25,
    )


def test_missing_values_matrixplot_3D(edata_blob_small, clean_up_plots):
    ep.pl.missing_values_matrix(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pl.missing_values_matrix(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_missing_values_heatmap(mimic_2, check_same_image, layer, clean_up_plots):
    if layer is not None:
        mimic_2.X = None
    plot = ep.pl.missing_values_heatmap(mimic_2, filter="bottom", max_cols=15, max_percentage=0.999, layer=layer)
    fig = plot.figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/missing_values_heatmap",
        tol=25,
    )


def test_missing_values_heatmap_3D(edata_blob_small, clean_up_plots):
    ep.pl.missing_values_heatmap(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pl.missing_values_heatmap(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


@pytest.mark.parametrize("layer", [None, "layer_2"])
def test_missing_values_dendogram(mimic_2, check_same_image, layer):
    if layer is not None:
        mimic_2.X = None
    plot = ep.pl.missing_values_dendrogram(mimic_2, filter="bottom", max_cols=15, max_percentage=0.999, layer=layer)
    fig = plot.figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/missing_values_dendogram",
        tol=25,
    )


def test_missing_values_dendogram_3D(edata_blob_small, clean_up_plots):
    ep.pl.missing_values_dendrogram(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pl.missing_values_dendrogram(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)


def _drawn_data(ax) -> list[np.ndarray]:
    images = [np.asarray(image.get_array()) for image in ax.images]
    patches = [patch.get_bbox().get_points() for patch in ax.patches]
    paths = [path.vertices for collection in ax.collections for path in collection.get_paths()]
    colors = [np.asarray(collection.get_array()) for collection in ax.collections if collection.get_array() is not None]
    return images + patches + paths + colors


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize(
    "plot",
    [
        ep.pl.missing_values_matrix,
        ep.pl.missing_values_barplot,
        ep.pl.missing_values_heatmap,
        ep.pl.missing_values_dendrogram,
    ],
)
def test_missing_values_array_types(array_type, plot, rng, clean_up_plots):
    X = np.where(rng.random((30, 5)) < 0.3, 0, rng.standard_normal((30, 5)))
    X[rng.random((30, 5)) < 0.2] = np.nan
    var = pd.DataFrame(index=["a", "b", "c", "ehrapycat_d", "e"])
    expected = _drawn_data(plot(ed.EHRData(X=X, var=var)))
    edata = ed.EHRData(X=array_type(X), var=var)

    if array_type.flags & Flags.Sparse and array_type.flags & Flags.Dask:
        with pytest.raises(NotImplementedError):
            plot(edata)
        return

    with forbid_dask_compute(allowed=1):
        result = _drawn_data(plot(edata))

    for drawn, expected_drawn in zip(result, expected, strict=True):
        np.testing.assert_allclose(drawn, expected_drawn)
