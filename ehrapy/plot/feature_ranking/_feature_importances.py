from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ehrapy._compat import use_ehrdata

if TYPE_CHECKING:
    from anndata import AnnData
    from ehrdata import EHRData
    from matplotlib.axes import Axes


@use_ehrdata(deprecated_after="1.0.0")
def rank_features_supervised(
    edata: EHRData | AnnData,
    key: str = "feature_importances",
    n_features: int = 10,
    ax: Axes | None = None,
    show: bool = True,
    save: str | None = None,
    **kwargs,
) -> Axes | None:
    """Plot features with greatest absolute importances as a barplot.

    Args:
        edata: Central data object. A key in edata.var should contain the feature
            importances, calculated beforehand.
        key: The key in edata.var to use for feature importances.
        n_features: The number of features to plot.
        ax: A matplotlib axes object to plot on. If `None`, a new figure will be created.
        show: If `True`, show the figure. If `False`, return the axes object.
        save: Path to save the figure. If `None`, the figure will not be saved.
        **kwargs: Additional arguments passed to `seaborn.barplot`.

    Returns:
        If `show == False` a `matplotlib.axes.Axes` object, else `None`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.knn_impute(edata, n_neighbors=5)
        >>> input_features = [
        ...     feat for feat in edata.var_names if feat not in {"service_unit", "day_icu_intime", "tco2_first"}
        ... ]
        >>> ep.tl.rank_features_supervised(edata, "tco2_first", "rf", input_features=input_features)
        >>> ep.pl.rank_features_supervised(edata)

        .. image:: /_static/docstring_previews/feature_importances.png
    """
    if key not in edata.var.keys():
        raise ValueError(
            f"Key {key} not found in edata.var. Make sure to calculate feature importances first with ep.tl.feature_importances."
        )

    df = pd.DataFrame({"importance": edata.var[key]}, index=edata.var_names)
    df["absolute_importance"] = df["importance"].abs()
    df = df.sort_values("absolute_importance", ascending=False)

    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.barplot(x=df["importance"][:n_features], y=df.index[:n_features], orient="h", ax=ax, **kwargs)
    plt.ylabel("Feature")
    plt.xlabel("Importance")
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
        return None
    else:
        return ax
