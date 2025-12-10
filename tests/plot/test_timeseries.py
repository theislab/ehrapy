from pathlib import Path

import ehrdata as ed
import matplotlib
import matplotlib.pyplot as plt
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


@pytest.fixture
def edata_blob_timeseries():
    edata = ed.dt.ehrdata_blobs(
        n_variables=4,
        n_observations=10,
        base_timepoints=100,
        layer="tem_data",
    )
    edata.var.index = ["feature1", "feature2", "feature3", "feature4"]
    edata.obs["subject_id"] = [f"patient{i}" for i in range(1, edata.n_obs + 1)]
    return edata


def test_plot_timeseries(edata_blob_timeseries, check_same_image):
    edata = edata_blob_timeseries

    ax = ep.pl.plot_timeseries(
        edata,
        obs_id=2,
        keys=["feature1", "feature2", "feature3"],
        layer=DEFAULT_TEM_LAYER_NAME,
        tem_time_key="timepoint",
        show=False,
    )

    fig = ax.figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/plot_timeseries_one_obs_row_index",
        tol=2e-1,
    )
    plt.close("all")


def test_plot_timeseries_obs_id(edata_blob_timeseries, check_same_image):
    edata = edata_blob_timeseries

    ax = ep.pl.plot_timeseries(
        edata,
        obs_id="patient3",
        keys=["feature1", "feature2", "feature3"],
        obs_id_key="subject_id",
        layer=DEFAULT_TEM_LAYER_NAME,
        tem_time_key="timepoint",
        show=False,
    )
    fig = ax.figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/plot_timeseries_one_obs_id",
        tol=2e-1,
    )
    plt.close("all")


def test_plot_timeseries_multiple_obs(edata_blob_timeseries, check_same_image):
    edata = edata_blob_timeseries

    axes = ep.pl.plot_timeseries(
        edata,
        obs_id=["patient3", "patient4"],
        keys=["feature1", "feature2", "feature3"],
        obs_id_key="subject_id",
        layer=DEFAULT_TEM_LAYER_NAME,
        tem_time_key="timepoint",
        show=False,
    )
    fig = axes[0].figure
    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/plot_timeseries_multiple_obs_subplots",
        tol=2e-1,
    )
    plt.close("all")


def test_plot_timeseries_overlay(edata_blob_timeseries, check_same_image):
    edata = edata_blob_timeseries
    ax = ep.pl.plot_timeseries(
        edata,
        obs_id=["patient3", "patient4", "patient5"],
        keys="feature1",
        obs_id_key="subject_id",
        layer=DEFAULT_TEM_LAYER_NAME,
        tem_time_key="timepoint",
        overlay=True,
        show=False,
    )
    fig = ax.figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/plot_timeseries_overlay",
        tol=2e-1,
    )

    plt.close("all")
