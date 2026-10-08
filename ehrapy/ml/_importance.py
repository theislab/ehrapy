from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ehrapy._compat import _materialize
from ehrapy.ml._evaluate import _metrics, _values
from ehrapy.ml._predictor import _calibrated, _features_of, _held_out, _outputs_of

if TYPE_CHECKING:
    from ehrdata import EHRData

    from ehrapy.ml._predictor import Predictor

_MAIN_METRICS = {
    "binary": "auroc",
    "multiclass": "auroc_macro",
    "multilabel": "auroc_macro",
    "regression": "r2",
    "survival": "c_index",
}


def permutation_importance(
    edata: EHRData,
    predictor: Predictor,
    *,
    split_key: str = "split",
    split: str = "held_out",
    n_repeats: int = 5,
    random_state: int = 0,
    key_added: str = "permutation_importance",
    copy: bool = False,
) -> EHRData | None:
    """Compute how much the predictions of a fitted model rely on every variable.

    The importance of a variable is how much the main metric of the task decreases when the variable is shuffled between observations, which keeps all other variables as they are.
    The main metric is the area under the ROC curve for binary tasks, its macro average for multiclass and multilabel tasks, the coefficient of determination for regression tasks and the concordance index for survival tasks.

    Args:
        edata: Central data object.
        predictor: The model fitted by :func:`~ehrapy.ml.fit`.
        split_key: Column of `obs` with the sets from :func:`~ehrapy.ml.split`.
        split: The set to compute the importances on.
        n_repeats: Number of times every variable is shuffled.
        random_state: Seed for shuffling.
        key_added: Key to store the importances under.
        copy: Whether to return a copy of `edata` instead of modifying it in place.

    Returns:
        ``None`` if ``copy=False``, otherwise the updated data object.
        The mean importance of every variable is stored in `edata.var[key_added]` and the importance of every repeat in `edata.varm[key_added]`, both missing for variables the model does not use.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_observations=200, n_centers=2, base_timepoints=10)
        >>> ep.ml.split(edata, stratify="cluster")
        >>> predictor = ep.ml.fit(edata, ep.ml.Task("cluster"))
        >>> ep.ml.permutation_importance(edata, predictor)
        >>> ep.pl.rank_features_supervised(edata, key="permutation_importance")
    """
    if copy:
        edata = edata.copy()
    rows, y = _held_out(edata, predictor, split_key=split_key, split=split)
    kind = predictor.task.kind
    if kind in {"binary", "multiclass", "multilabel"}:
        y = y.astype(np.int64)
    metric = {_MAIN_METRICS[kind]: _metrics(kind, 0.5)[_MAIN_METRICS[kind]]}

    def score(features: np.ndarray) -> float:
        outputs = _calibrated(predictor, _outputs_of(edata, predictor, features)[0])
        return _values(y, outputs[:, 0] if kind in {"binary", "regression", "survival"} else outputs, metric, kind)[0]

    features = _materialize(_features_of(edata[rows], predictor))[0]
    baseline = score(features)
    rng = np.random.default_rng(random_state)
    names = pd.Index(predictor.feature_names)
    importances = np.full((edata.n_vars, n_repeats), np.nan)
    for var in predictor.var_names:
        candidates = [var] if predictor.preprocessing is None else [var, *(f"{var}_{s}" for s in predictor.statistics)]
        columns = names.get_indexer(candidates)
        columns = columns[columns >= 0]
        for repeat in range(n_repeats):
            permuted = features.copy()
            permuted[:, columns] = features[rng.permutation(len(features))][:, columns]
            importances[edata.var_names.get_loc(var), repeat] = baseline - score(permuted)
    edata.var[key_added] = importances.mean(axis=1)
    edata.varm[key_added] = importances
    return edata if copy else None
