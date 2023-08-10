from pathlib import Path

import numpy as np
import pytest

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_preprocessing"


@pytest.fixture
def mimic_2_10():
    mimic_2_10 = ep.dt.mimic_2()[:10]

    return mimic_2_10


def test_winsorize_var(mimic_2_10):
    winsorized_adata = ep.pp.winsorize(mimic_2_10, vars=["age"], limits=[0.2, 0.2], copy=True)
    expected = np.array(
        [71.43198, 64.92076, 36.5, 44.49191, 25.41667, 36.54657, 25.41667, 71.43198, 71.43198, 25.41667]
    ).reshape((10, 1))

    np.testing.assert_allclose(np.array(winsorized_adata[:, "age"].X, dtype=np.float32), expected)


def test_winsorized_obs(mimic_2_10):
    to_winsorize_obs = ep.ad.move_to_obs(mimic_2_10, "age")
    winsorized_adata = ep.pp.winsorize(to_winsorize_obs, obs_cols=["age"], limits=[0.2, 0.2], copy=True)
    expected = np.array(
        [71.43198, 64.92076, 36.5, 44.49191, 25.41667, 36.54657, 25.41667, 71.43198, 71.43198, 25.41667]
    )

    np.testing.assert_allclose(np.array(winsorized_adata.obs["age"]), expected)


def test_clip_var(mimic_2_10):
    clipped_adata = ep.pp.clip_quantile(mimic_2_10, vars=["age"], limits=[25, 50], copy=True)
    expected = np.array([50, 50, 36.5, 44.49191, 25, 36.54657, 25, 50, 50, 25.41667]).reshape((10, 1))

    np.testing.assert_allclose(np.array(clipped_adata[:, "age"].X, dtype=np.float32), expected)


def test_clip_obs(mimic_2_10):
    to_clip_obs = ep.ad.move_to_obs(mimic_2_10, "age")
    clipped_adata = ep.pp.clip_quantile(to_clip_obs, obs_cols=["age"], limits=[25, 50], copy=True)
    expected = np.array([50, 50, 36.5, 44.49191, 25, 36.54657, 25, 50, 50, 25.41667])

    np.testing.assert_allclose(np.array(clipped_adata.obs["age"]), expected)
