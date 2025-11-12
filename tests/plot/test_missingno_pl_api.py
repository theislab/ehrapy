from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep

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
        tol=5,
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
        tol=2e-1,
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
        tol=2e-1,
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
        tol=2e-1,
    )


def test_missing_values_dendogram_3D(edata_blob_small, clean_up_plots):
    ep.pl.missing_values_dendrogram(edata_blob_small, layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pl.missing_values_dendrogram(edata_blob_small, layer=DEFAULT_TEM_LAYER_NAME)
