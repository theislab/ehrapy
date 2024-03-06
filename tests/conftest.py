import os
from pathlib import Path

import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.testing.compare import compare_images


@pytest.fixture
def root_dir():
    return Path(__file__).resolve().parent


# simplified from https://github.com/scverse/scanpy/blob/main/scanpy/tests/conftest.py
@pytest.fixture
def check_same_image(tmp_path):
    def check_same_image(
        fig: Figure,
        base_path: Path | os.PathLike,
        *,
        tol: float,
    ) -> None:
        expected = Path(base_path).parent / (Path(base_path).name + "_expected.png")
        if not Path(expected).is_file():
            raise OSError(f"No expected output found at {expected}.")
        actual = tmp_path / "actual.png"

        fig.tight_layout()
        fig.savefig(actual, dpi=80)

        result = compare_images(expected, actual, tol=tol, in_decorator=True)

        if result is None:
            return None

        raise AssertionError(result)

    return check_same_image
