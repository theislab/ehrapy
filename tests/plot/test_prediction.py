import ehrdata as ed
import holoviews as hv
import numpy as np
import pandas as pd
import pytest

import ehrapy as ep


@pytest.fixture
def edata():
    rng = np.random.default_rng(0)
    probability = rng.uniform(size=200)
    y = rng.binomial(1, probability)
    obs = pd.DataFrame(
        {
            "y": y,
            "level": y + rng.integers(0, 2, 200) * y,
            "prediction": probability,
            "group": rng.choice(["a", "b"], 200),
        },
        index=[str(i) for i in range(200)],
    )
    probabilities = np.column_stack([1 - probability, probability / 2, probability / 2])
    return ed.EHRData(obs=obs, obsm={"levels": pd.DataFrame(probabilities, index=obs.index, columns=["0", "1", "2"])})


def test_prediction_performance(edata):
    plot = ep.pl.prediction_performance(edata, ep.ml.Task("y"), split=None)

    assert isinstance(plot, hv.Layout)
    assert len(plot) == 3
    with pytest.raises(ValueError, match="Only predicted probabilities"):
        ep.pl.prediction_performance(edata, ep.ml.Task("y", kind="regression"), split=None)


def test_prediction_performance_per_class(edata):
    roc, pr, calibration = ep.pl.prediction_performance(
        edata, ep.ml.Task("level", kind="multiclass"), key="levels", split=None
    )

    assert (len(roc), len(pr), len(calibration)) == (4, 3, 7)


def test_subgroup_performance(edata):
    performance = ep.ml.evaluate(edata, ep.ml.Task("y"), split=None, groupby="group", n_bootstrap=20)

    plot = ep.pl.subgroup_performance(performance, metrics=["auroc", "brier"])

    assert len(plot) == 2
    assert all(isinstance(panel, hv.Overlay) for panel in plot)


def test_subgroup_performance_keeps_subgroup_order(edata):
    edata.obs["group"] = pd.Categorical(edata.obs["group"], categories=["b", "a"])
    performance = ep.ml.evaluate(edata, ep.ml.Task("y"), split=None, groupby="group", n_bootstrap=0)

    (panel,) = ep.pl.subgroup_performance(performance, metrics=["auroc"])

    rendered = hv.render(panel, backend="bokeh")
    assert list(rendered.x_range.factors) == ["b", "a"]


def test_subgroup_performance_titles_fit_panels(edata):
    performance = ep.ml.evaluate(edata, ep.ml.Task("y"), split=None, groupby="group", n_bootstrap=0)

    (panel,) = ep.pl.subgroup_performance(performance, metrics=["calibration_intercept"])

    difference = performance.loc[("difference", "calibration_intercept"), "value"]
    assert panel.opts.get("plot").kwargs["title"] == f"calibration_intercept\ndifference {difference:.2f}"
