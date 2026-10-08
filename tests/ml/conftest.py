import ehrdata as ed
import numpy as np
import pandas as pd

import ehrapy as ep

TASKS = {
    "binary": ep.ml.Task("label", prediction_time=6, observation_window=4),
    "multiclass": ep.ml.Task("level", kind="multiclass", prediction_time=6, observation_window=4),
    "multilabel": ep.ml.Task(["above", "below"], kind="multilabel", prediction_time=6, observation_window=4),
    "regression": ep.ml.Task("value", kind="regression", prediction_time=6, observation_window=4),
    "survival": ep.ml.Task("time", kind="survival", event="event", prediction_time=6, observation_window=4),
}


def longitudinal(n_obs: int = 300, *, n_timepoints: int = 10, seed: int = 0) -> ed.EHRData:
    """Targets of every kind given by the mean of `signal` at timepoints 2 to 5, with the binary label revealed by `leak` from timepoint 6 on."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs, 3, n_timepoints))
    score = X[:, 0, 2:6].mean(axis=1) + 0.2 * rng.normal(size=n_obs)
    X[:, 1, 6:] = (score > 0)[:, None]
    X[rng.random(X.shape) < 0.1] = np.nan
    obs = pd.DataFrame(
        {
            "label": (score > 0).astype(int),
            "level": np.array(["low", "middle", "high"])[np.digitize(score, [-0.3, 0.3])],
            "above": (score > -0.3).astype(int),
            "below": (score < 0.3).astype(int),
            "value": 2 * score + 0.1 * rng.normal(size=n_obs),
            "time": rng.exponential(np.exp(-2 * score)),
            "event": rng.random(n_obs) < 0.8,
            "patient": rng.integers(0, n_obs // 3, n_obs).astype(str),
            "sex": rng.choice(["f", "m"], n_obs),
            "admission": rng.permutation(n_obs),
        },
        index=[str(i) for i in range(n_obs)],
    )
    return ed.EHRData(X=X, obs=obs, var=pd.DataFrame(index=["signal", "leak", "noise"]))


def static(n_obs: int = 300, *, seed: int = 0) -> ed.EHRData:
    """A label `y` that is a variable as well, and the numeric `outcome` that `x` determines."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_obs)
    y = (x + 0.5 * rng.normal(size=n_obs) > 0).astype(int)
    X = np.column_stack([x, rng.normal(size=n_obs), y]).astype(float)
    X[rng.random(X.shape) < 0.1] = np.nan
    obs = pd.DataFrame({"y": y, "outcome": 2 * x + 0.1 * rng.normal(size=n_obs)}, index=[str(i) for i in range(n_obs)])
    edata = ed.EHRData(X=X, obs=obs, var=pd.DataFrame(index=["x", "noise", "y"]))
    ep.ml.split(edata)
    return edata
