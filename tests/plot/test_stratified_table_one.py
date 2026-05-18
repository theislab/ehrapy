from __future__ import annotations

from pathlib import Path

import holoviews as hv
import pytest

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


def test_stratified_table_one_plot(edata_mini, check_same_image, hv_backend):
    ep.tl.stratified_table_one(
        edata_mini,
        groupby="station",
        columns=["glucose", "weight", "disease"],
        nonnormal=["glucose"],
    )
    layout = ep.pl.stratified_table_one(edata_mini)
    fig = hv.render(layout, backend="matplotlib")

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/stratified_table_one_station",
        tol=2e-1,
    )


def test_stratified_table_one_plot_sensitivity(edata_mini, check_same_image, hv_backend):
    """Different `n_cols` should give a different image."""
    ep.tl.stratified_table_one(
        edata_mini,
        groupby="station",
        columns=["glucose", "weight", "disease"],
        nonnormal=["glucose"],
    )
    layout = ep.pl.stratified_table_one(edata_mini, n_cols=1)
    fig = hv.render(layout, backend="matplotlib")

    with pytest.raises(AssertionError):
        check_same_image(
            fig=fig,
            base_path=f"{_TEST_IMAGE_PATH}/stratified_table_one_station",
            tol=2e-1,
        )


def test_stratified_table_one_plot_returns_layout(edata_mini, hv_backend):
    ep.tl.stratified_table_one(
        edata_mini, groupby="station", columns=["glucose", "disease"]
    )
    layout = ep.pl.stratified_table_one(edata_mini)
    assert isinstance(layout, hv.Layout)
    assert len(layout) == 2


def test_stratified_table_one_plot_missing_results(edata_mini, hv_backend):
    with pytest.raises(KeyError, match="ep.tl.stratified_table_one"):
        ep.pl.stratified_table_one(edata_mini)
