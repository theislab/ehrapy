from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from ehrapy.ml._evaluate import _evaluation_data
from ehrapy.plot._holoviews import load_hv_extensions

if TYPE_CHECKING:
    from ehrdata import EHRData

    from ehrapy.ml import Task


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
    """Plot the ROC curve, the precision-recall curve and the calibration curve of binary predictions.

    The calibration curve compares the mean predicted probability with the observed frequency of the label in `n_bins` bins with equally many observations.
    Dashed lines show a random model for the ROC curve, the frequency of the label for the precision-recall curve and perfect calibration for the calibration curve.

    Args:
        edata: Central data object.
        task: The binary prediction task.
        key: Column of `obs` with the predictions from :func:`~ehrapy.ml.predict`.
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
        >>> edata = ed.dt.physionet2012()
        >>> ep.ml.split(edata, stratify="In-hospital_death")
        >>> task = ep.ml.Task("In-hospital_death")
        >>> ep.ml.predict(edata, ep.ml.fit(edata, task, obs_keys=["Age", "Gender", "ICUType"]))
        >>> ep.pl.prediction_performance(edata, task)

    Preview:
        .. image:: /_static/docstring_previews/prediction_performance.png
    """
    if task.kind != "binary":
        raise ValueError(f"Only binary predictions can be plotted, got a {task.kind} task.")
    y, prediction, _ = _evaluation_data(edata, task, key=key, split_key=split_key, split=split)
    false_positive_rate, true_positive_rate, _ = roc_curve(y, prediction)
    precision, recall, _ = precision_recall_curve(y, prediction)
    observed, predicted = calibration_curve(y, prediction, n_bins=n_bins, strategy="quantile")
    guide = {"color": "grey", "line_dash": "dashed"}
    diagonal = hv.Curve([(0, 0), (1, 1)]).opts(**guide)

    roc = hv.Curve((false_positive_rate, true_positive_rate), "False positive rate", "True positive rate") * diagonal
    pr = hv.Curve((recall, precision), "Recall", "Precision") * hv.HLine(y.mean()).opts(**guide)
    calibration = (
        hv.Curve((predicted, observed), "Mean predicted probability", "Observed frequency")
        * hv.Scatter((predicted, observed))
        * diagonal
    )
    return (
        roc.opts(title=f"ROC (AUROC {roc_auc_score(y, prediction):.2f})", width=width, height=height)
        + pr.opts(
            title=f"Precision-recall (AUPRC {average_precision_score(y, prediction):.2f})", width=width, height=height
        )
        + calibration.opts(title="Calibration", width=width, height=height)
    ).cols(3)
