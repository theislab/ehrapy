from pathlib import Path

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
