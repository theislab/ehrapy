from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from ehrapy.ml._evaluate import _evaluation_data
from ehrapy.plot._holoviews import load_hv_extensions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData

    from ehrapy.ml import Task

_GUIDE = {"color": "grey", "line_dash": "dashed"}


@load_hv_extensions()
def prediction_performance(
    edata: EHRData,
    task: Task,
    *,
    key: str = "prediction",
    split_key: str = "split",
    split: str | None = "held_out",
    n_bins: int = 10,
    width: int = 300,
    height: int = 300,
) -> hv.Layout:
    """Plot the ROC curve, the precision-recall curve and the calibration curve of predicted probabilities.

    Multiclass and multilabel tasks get a curve for every class or label against all others, labeled with its AUROC or AUPRC.
    The calibration curve compares the mean predicted probability with the observed frequency in `n_bins` bins with equally many observations.
    Dashed lines show a random model for the ROC curve, the frequency of the label for the precision-recall curve of binary tasks and perfect calibration for the calibration curve.

    Args:
        edata: Central data object.
        task: The binary, multiclass or multilabel prediction task.
        key: Key of the predictions from :func:`~ehrapy.ml.predict`.
        split_key: Column of `obs` with the sets from :func:`~ehrapy.ml.split`.
        split: The set to plot.
            If `None`, all observations are plotted.
        n_bins: Number of bins of the calibration curve.
        width: Width of each panel in pixels.
        height: Height of each panel in pixels.

    Returns:
        A HoloViews Layout with the three panels.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_observations=200, n_centers=2, base_timepoints=10)
        >>> ep.ml.split(edata, stratify="cluster")
        >>> task = ep.ml.Task("cluster")
        >>> ep.ml.predict(edata, ep.ml.fit(edata, task))
        >>> ep.pl.prediction_performance(edata, task)

    Preview:
        .. image:: /_static/docstring_previews/prediction_performance.png
    """
    if task.kind not in {"binary", "multiclass", "multilabel"}:
        raise ValueError(f"Only predicted probabilities can be plotted, got a {task.kind} task.")
    y, prediction, _ = _evaluation_data(edata, task, key=key, split_key=split_key, split=split)
    if task.kind == "binary":
        curves = [(y, prediction, "")]
    else:
        predictions = edata.obsm[key]
        names = predictions.columns if isinstance(predictions, pd.DataFrame) else range(prediction.shape[1])
        indicators = y if task.kind == "multilabel" else y[:, None] == np.arange(prediction.shape[1])
        curves = [(indicators[:, i], prediction[:, i], str(name)) for i, name in enumerate(names)]

    roc, pr, calibration = [], [], []
    for observed, probability, name in curves:
        if len(np.unique(observed)) < 2:
            continue
        auroc, auprc = roc_auc_score(observed, probability), average_precision_score(observed, probability)
        false_positive_rate, true_positive_rate, _ = roc_curve(observed, probability)
        precision, recall, _ = precision_recall_curve(observed, probability)
        frequency, mean_probability = calibration_curve(observed, probability, n_bins=n_bins, strategy="quantile")
        roc_label, pr_label = (f"{name} ({score:.2f})" if name else "" for score in (auroc, auprc))
        roc.append(
            hv.Curve(
                (false_positive_rate, true_positive_rate), "False positive rate", "True positive rate", label=roc_label
            )
        )
        pr.append(hv.Curve((recall, precision), "Recall", "Precision", label=pr_label))
        calibration.append(
            hv.Curve((mean_probability, frequency), "Mean predicted probability", "Observed frequency", label=name)
        )
        calibration.append(hv.Scatter((mean_probability, frequency), label=name))

    titles = ["ROC", "Precision-recall", "Calibration"]
    if task.kind == "binary":
        titles[:2] = f"ROC (AUROC {auroc:.2f})", f"Precision-recall (AUPRC {auprc:.2f})"
        pr.append(hv.HLine(y.mean()).opts(**_GUIDE))
    diagonal = hv.Curve([(0, 0), (1, 1)]).opts(**_GUIDE)
    panels = (hv.Overlay([*roc, diagonal]), hv.Overlay(pr), hv.Overlay([*calibration, diagonal]))
    return hv.Layout(
        [panel.opts(title=title, width=width, height=height) for panel, title in zip(panels, titles, strict=True)]
    ).cols(3)


@load_hv_extensions()
def subgroup_performance(
    performance: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    width: int = 300,
    height: int = 250,
) -> hv.Layout:
    """Plot the metrics of every subgroup with their confidence intervals.

    Every panel shows one metric, with its largest difference between subgroups in the title.

    Args:
        performance: Metrics per subgroup from :func:`~ehrapy.ml.evaluate` with `groupby`.
        metrics: The metrics to plot.
            If `None`, all metrics are plotted.
        width: Width of each panel in pixels.
        height: Height of each panel in pixels.

    Returns:
        A HoloViews Layout with a panel for every metric.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.ehrdata_blobs(n_observations=200, n_centers=2, base_timepoints=10)
        >>> edata.obs["sex"] = np.random.default_rng(0).choice(["female", "male"], edata.n_obs)
        >>> ep.ml.split(edata, stratify="cluster")
        >>> task = ep.ml.Task("cluster")
        >>> ep.ml.predict(edata, ep.ml.fit(edata, task))
        >>> performance = ep.ml.evaluate(edata, task, split=None, groupby="sex", n_bootstrap=100)
        >>> ep.pl.subgroup_performance(performance, metrics=["auroc", "sensitivity", "specificity"])

    Preview:
        .. image:: /_static/docstring_previews/subgroup_performance.png
    """
    groupby = performance.index.names[0]
    differences = performance.xs("difference", level=groupby)["value"]
    groups = performance.drop("difference", level=groupby)
    panels = []
    for metric in groups.index.unique("metric") if metrics is None else metrics:
        values = groups.xs(metric, level="metric")
        names = values.index.astype(str)
        lower, upper = values["value"] - values["ci_lower"], values["ci_upper"] - values["value"]
        error_bars = hv.ErrorBars(
            (names, values["value"], lower, upper), kdims=[str(groupby)], vdims=[metric, "lower", "upper"]
        )
        points = hv.Scatter((names, values["value"]), kdims=[str(groupby)], vdims=[metric]).opts(
            size=8, tools=["hover"]
        )
        title = f"{metric} (difference {differences[metric]:.2f})"
        panels.append((error_bars * points).opts(title=title, width=width, height=height))
    return hv.Layout(panels).cols(3)
