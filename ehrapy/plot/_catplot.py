from __future__ import annotations

from typing import TYPE_CHECKING

import seaborn as sns

from ehrapy._compat import use_ehrdata

if TYPE_CHECKING:
    from anndata import AnnData
    from ehrdata import EHRData
    from seaborn.axisgrid import FacetGrid


@use_ehrdata(deprecated_after="1.0.0")
def catplot(
    edata: EHRData | AnnData,
    x: str = None,
    y: str = None,
    hue: str = None,
    kind: str = "strip",
    **kwargs,
) -> FacetGrid:
    """Plot categorical data.

    Wrapper around `seaborn.catplot <https://seaborn.pydata.org/generated/seaborn.catplot.html>`_. Typically used to show
    the behaviour of one numerical variable with respect to one or several categorical variables.

    Considers edata.obs only.

    Args:
        edata: Central data object.
        x: Variable to plot on the x-axis.
        y: Variable to plot on the y-axis.
        hue: Variable to plot as different colors.
        kind: Kind of plot to make. Options are: "point", "bar", "strip", "swarm", "box", "violin", "boxen", or "count".
        **kwargs: Keyword arguments for seaborn.catplot.

    Returns:
        A Seaborn FacetGrid object for further modifications.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.diabetes_130_fairlearn()
        >>> ep.ad.move_to_obs(edata, ["A1Cresult", "admission_source_id"], copy_obs=True)
        >>> edata.obs["A1Cresult_measured"] = ~edata.obs["A1Cresult"].isna()
        >>> ep.pl.catplot(
        ...     edata=edata,
        ...     y="A1Cresult_measured",
        ...     x="admission_source_id",
        ...     kind="point",
        ...     ci=95,
        ...     join=False,
        ... )

        .. image:: /_static/docstring_previews/catplot.png
    """
    return sns.catplot(data=edata.obs, x=x, y=y, hue=hue, kind=kind, **kwargs)
