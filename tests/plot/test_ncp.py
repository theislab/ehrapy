from __future__ import annotations

from pathlib import Path

import ehrdata as ed
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

hv.extension("bokeh")

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


@pytest.fixture
def edata_with_ncp() -> ed.EHRData:
    """EHRData with a 3D layer and NCP results pre-computed.

    Uses the same deterministic data as the reference-image notebooks
    (tests/_scripts/ncp_expected.ipynb, ncp_cluster_trajectories_expected.ipynb).
    """
    rng = np.random.default_rng(0)
    n_obs, n_vars, n_time = 40, 12, 8
    layer = np.abs(rng.standard_normal((n_obs, n_vars, n_time)))
    var_names = [f"disease_{i}" for i in range(n_vars)]
    edata = ed.EHRData(
        shape=(n_obs, n_vars),
        layers={DEFAULT_TEM_LAYER_NAME: layer},
        var=pd.DataFrame(index=var_names),
    )
    ep.tl.ncp(edata, layer=DEFAULT_TEM_LAYER_NAME, rank=3, n_iter_max=50, random_state=0)
    edata.obs["cluster"] = ["A"] * 20 + ["B"] * 20
    return edata


def test_pl_ncp_returns_layout(edata_with_ncp: ed.EHRData) -> None:
    plot = ep.pl.ncp(edata_with_ncp)
    assert plot is not None
    assert isinstance(plot, hv.Layout)


def test_pl_ncp_panel_count(edata_with_ncp: ed.EHRData) -> None:
    rank = edata_with_ncp.obsm["X_ncp"].shape[1]
    plot = ep.pl.ncp(edata_with_ncp)
    assert len(plot) == rank * 3


def test_pl_ncp_custom_key(edata_with_ncp: ed.EHRData) -> None:
    ep.tl.ncp(edata_with_ncp, layer=DEFAULT_TEM_LAYER_NAME, rank=2, key_added="ncp2", n_iter_max=10)
    plot = ep.pl.ncp(edata_with_ncp, key="ncp2")
    assert isinstance(plot, hv.Layout)
    assert len(plot) == 2 * 3


def test_pl_ncp_missing_key_raises(edata_with_ncp: ed.EHRData) -> None:
    with pytest.raises(KeyError, match="Run `ep.tl.ncp"):
        ep.pl.ncp(edata_with_ncp, key="does_not_exist")


def test_pl_ncp_n_top(edata_with_ncp: ed.EHRData) -> None:
    plot = ep.pl.ncp(edata_with_ncp, n_top=5)
    assert isinstance(plot, hv.Layout)


def test_pl_ncp_image(edata_with_ncp: ed.EHRData, check_same_image, hv_backend) -> None:
    plot = ep.pl.ncp(edata_with_ncp, n_top=5)
    fig = hv.render(plot, backend="matplotlib")
    fig.set_size_inches(12, 10.5)
    fig.set_dpi(80)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/ncp",
        tol=2e-1,
    )


def test_pl_ncp_cluster_trajectories_returns_layout(edata_with_ncp: ed.EHRData) -> None:
    plot = ep.pl.ncp_cluster_trajectories(
        edata_with_ncp,
        layer=DEFAULT_TEM_LAYER_NAME,
        cluster_key="cluster",
    )
    assert plot is not None
    assert isinstance(plot, hv.Layout)


def test_pl_ncp_cluster_trajectories_panel_per_cluster(edata_with_ncp: ed.EHRData) -> None:
    n_clusters = edata_with_ncp.obs["cluster"].nunique()
    plot = ep.pl.ncp_cluster_trajectories(
        edata_with_ncp,
        layer=DEFAULT_TEM_LAYER_NAME,
        cluster_key="cluster",
    )
    assert len(plot) == n_clusters


def test_pl_ncp_cluster_trajectories_sigmoid(edata_with_ncp: ed.EHRData) -> None:
    plot = ep.pl.ncp_cluster_trajectories(
        edata_with_ncp,
        layer=DEFAULT_TEM_LAYER_NAME,
        cluster_key="cluster",
        sigmoid_transform=True,
    )
    assert isinstance(plot, hv.Layout)


def test_pl_ncp_cluster_trajectories_missing_cluster_key_raises(edata_with_ncp: ed.EHRData) -> None:
    with pytest.raises(KeyError, match="not found in edata.obs"):
        ep.pl.ncp_cluster_trajectories(
            edata_with_ncp,
            layer=DEFAULT_TEM_LAYER_NAME,
            cluster_key="no_such_column",
        )


def test_pl_ncp_cluster_trajectories_missing_layer_raises(edata_with_ncp: ed.EHRData) -> None:
    with pytest.raises(KeyError, match="not found in edata.layers"):
        ep.pl.ncp_cluster_trajectories(
            edata_with_ncp,
            layer="no_such_layer",
            cluster_key="cluster",
        )


def test_pl_ncp_cluster_trajectories_missing_ncp_raises(edata_with_ncp: ed.EHRData) -> None:
    with pytest.raises(KeyError, match="Run `ep.tl.ncp"):
        ep.pl.ncp_cluster_trajectories(
            edata_with_ncp,
            layer=DEFAULT_TEM_LAYER_NAME,
            cluster_key="cluster",
            key="ghost_key",
        )


def test_pl_ncp_cluster_trajectories_image(edata_with_ncp: ed.EHRData, check_same_image, hv_backend) -> None:
    plot = ep.pl.ncp_cluster_trajectories(
        edata_with_ncp,
        layer=DEFAULT_TEM_LAYER_NAME,
        cluster_key="cluster",
        n_top_diseases=5,
    )
    fig = hv.render(plot, backend="matplotlib")
    fig.set_size_inches(8, 2.75)
    fig.set_dpi(80)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/ncp_cluster_trajectories",
        tol=2e-1,
    )
