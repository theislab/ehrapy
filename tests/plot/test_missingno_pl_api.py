from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


def test_missing_values_barplot(mimic_2, check_same_image):
    plot = ep.pl.missing_values_barplot(mimic_2, filter="bottom", max_cols=15, max_percentage=0.999)
    fig = plot.figure
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/missing_values_barplot",
        tol=2e-1,
    )


def test_missing_values_matrixplot(mimic_2, check_same_image):
    plot = ep.pl.missing_values_matrix(mimic_2, filter="bottom", max_cols=15, max_percentage=0.999)
    fig = plot.figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/missing_values_matrix",
        tol=2e-1,
    )


def test_missing_values_heatmap(mimic_2, check_same_image):
    plot = ep.pl.missing_values_heatmap(mimic_2, filter="bottom", max_cols=15, max_percentage=0.999)
    fig = plot.figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/missing_values_heatmap",
        tol=2e-1,
    )


def test_missing_values_dendogram(mimic_2, check_same_image):
    plot = ep.pl.missing_values_dendrogram(mimic_2, filter="bottom", max_cols=15, max_percentage=0.999)
    fig = plot.figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/missing_values_dendogram",
        tol=2e-1,
    )
