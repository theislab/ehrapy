from pathlib import Path

from ehrapy.plot import catplot

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


def test_catplot_vanilla(adata_mini, check_same_image):
    fig = catplot(adata_mini, jitter=False)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/catplot_vanilla",;
        tol=2e-1,
    )
