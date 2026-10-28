from pathlib import Path

import numpy as np

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


def test_catplot_vanilla(edata_mini, check_same_image):
    fig = ep.pl.catplot(edata_mini, jitter=False)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/catplot_vanilla",
        tol=2e-1,
    )


def test_catplot_variables_3D(edata_blobs_3d):
    grid = ep.pl.catplot(edata_blobs_3d, x="cluster", y="feature_0", kind="box")

    np.testing.assert_array_equal(grid.data["feature_0"], edata_blobs_3d.X[:, 0, 0])
    np.testing.assert_array_equal(grid.data["cluster"], edata_blobs_3d.obs["cluster"])
