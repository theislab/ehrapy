import ehrdata as ed
import holoviews as hv
import numpy as np
import pandas as pd
import pytest

import ehrapy as ep


def test_prediction_performance():
    rng = np.random.default_rng(0)
    probability = rng.uniform(size=200)
    obs = pd.DataFrame(
        {"y": rng.binomial(1, probability), "prediction": probability}, index=[str(i) for i in range(200)]
    )
    edata = ed.EHRData(obs=obs)

    plot = ep.pl.prediction_performance(edata, ep.ml.Task("y"), split=None)

    assert isinstance(plot, hv.Layout)
    assert len(plot) == 3
    with pytest.raises(ValueError, match="Only binary"):
        ep.pl.prediction_performance(edata, ep.ml.Task("y", kind="regression"), split=None)
