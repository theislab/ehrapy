import ehrdata as ed
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from scipy.cluster import hierarchy
from testing.fast_array_utils import Flags

import ehrapy as ep
from ehrapy.plot import _missing_values
from tests.conftest import forbid_dask_compute

VAR = pd.DataFrame(index=["a", "b", "c", "d"])
PLOTS = [
    ep.pl.missing_values_matrix,
    ep.pl.missing_values_barplot,
    ep.pl.missing_values_heatmap,
    ep.pl.missing_values_dendrogram,
]


@pytest.fixture
def X(rng) -> np.ndarray:
    X = rng.standard_normal((30, 4))
    X[rng.random(X.shape) < 0.3] = np.nan
    return X


def _drawn(plot: hv.Element) -> list[np.ndarray]:
    if isinstance(plot, hv.Path):
        return [path.array() for path in plot.split()]
    return [plot.dimension_values(dimension) for dimension in plot.dimensions()]


def _observed(plot: hv.QuadMesh) -> np.ndarray:
    return plot.dimension_values("observed (%)", flat=False)


def _assert_same_plot(result: hv.Element, expected: hv.Element) -> None:
    for drawn, expected_drawn in zip(_drawn(result), _drawn(expected), strict=True):
        if drawn.dtype.kind in "fi":
            np.testing.assert_allclose(drawn, expected_drawn)
        else:
            np.testing.assert_array_equal(drawn, expected_drawn)


def test_missing_values_matrix(X):
    plot = ep.pl.missing_values_matrix(ed.EHRData(X=X, var=VAR))

    np.testing.assert_array_equal(_observed(plot), 100 * ~np.isnan(X))
    assert [label for _, label in plot.opts.get().kwargs["xticks"]] == list(VAR.index)


def test_missing_values_matrix_groups_observations(X, monkeypatch):
    monkeypatch.setattr(_missing_values, "_MAX_ROWS", 4)

    plot = ep.pl.missing_values_matrix(ed.EHRData(X=X, var=VAR))

    edges = np.linspace(0, 30, 5).astype(int)
    expected = [100 * (~np.isnan(X[start:end])).mean(axis=0) for start, end in zip(edges[:-1], edges[1:], strict=True)]
    np.testing.assert_allclose(_observed(plot), expected)


def test_missing_values_matrix_3D(rng):
    X = rng.standard_normal((10, 4, 3))
    X[rng.random(X.shape) < 0.3] = np.nan

    plot = ep.pl.missing_values_matrix(ed.EHRData(X=X, var=VAR))

    np.testing.assert_allclose(_observed(plot), 100 * (~np.isnan(X)).mean(axis=0).T)


def test_missing_values_barplot(X):
    plot = ep.pl.missing_values_barplot(ed.EHRData(X=X, var=VAR))

    assert plot.dimension_values("variable").tolist() == list(VAR.index)
    np.testing.assert_allclose(plot.dimension_values("observed (%)"), 100 * (~np.isnan(X)).mean(axis=0))


def test_missing_values_heatmap(X):
    X[:, 3] = np.nan

    plot = ep.pl.missing_values_heatmap(ed.EHRData(X=X, var=VAR))

    correlation = plot.data.pivot(index="other variable", columns="variable", values="correlation")
    expected = pd.DataFrame(np.isnan(X[:, :3]), columns=["a", "b", "c"]).corr()
    pd.testing.assert_frame_equal(
        correlation.loc[expected.index, expected.columns], expected, check_names=False, check_index_type=False
    )


def test_missing_values_dendrogram(X):
    plot = ep.pl.missing_values_dendrogram(ed.EHRData(X=X, var=VAR))

    expected = hierarchy.dendrogram(
        hierarchy.linkage(np.isnan(X).T.astype(float), "average"), no_plot=True, labels=list(VAR.index)
    )
    assert [label for _, label in plot.opts.get().kwargs["xticks"]] == expected["ivl"]


@pytest.mark.parametrize("plot", PLOTS[1:])
def test_missing_values_3D_counts_every_timepoint(plot, rng):
    X = rng.standard_normal((10, 4, 3))
    X[rng.random(X.shape) < 0.3] = np.nan
    expected = plot(ed.EHRData(X=np.moveaxis(X, 1, 2).reshape(-1, 4), var=VAR))

    result = plot(ed.EHRData(X=X, var=VAR))

    _assert_same_plot(result, expected)


@pytest.mark.parametrize("plot", PLOTS)
def test_missing_values_selects_variables(plot, X):
    var = pd.DataFrame(index=["a", "b", "c", "ehrapycat_d"])
    edata = ed.EHRData(X=X, var=var)

    _assert_same_plot(plot(edata), plot(edata, var_names=["a", "b", "c"]))
    with pytest.raises(KeyError, match="unknown not found"):
        plot(edata, var_names=["unknown"])


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("plot", PLOTS)
def test_missing_values_array_types(array_type, plot, X):
    expected = plot(ed.EHRData(X=X, var=VAR))
    edata = ed.EHRData(X=array_type(X), var=VAR)

    with forbid_dask_compute(allowed=1):
        result = plot(edata)

    _assert_same_plot(result, expected)
