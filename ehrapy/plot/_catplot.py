from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from scanpy import AnnData
    from seaborn.axisgrid import FacetGrid


def catplot(adata: AnnData, x: str = None, y: str = None, hue: str = None, kind: str = "strip", **kwargs) -> FacetGrid:
    """Plot categorical data.

    Wrapper around `seaborn.catplot <https://seaborn.pydata.org/generated/seaborn.catplot.html>`_. Typically used to show
    the behaviour of one numerical with respect to one or several categorical variables.

    Considers adata.obs only.

    Args:
        adata: AnnData object.
        x: Variable to plot on the x-axis.
        y: Variable to plot on the y-axis.
        hue: Variable to plot as different colors.
        kind: Kind of plot to make. Options are: "point", "bar", "strip", "swarm", "box", "violin", "boxen", or "count".
        **kwargs: Keyword arguments for seaborn.catplot.

    Returns:
        A Seaborn FacetGrid object for further modifications.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.diabetes_130_fairlearn()
        >>> ep.ad.move_to_obs(adata, ["A1Cresult", "admission_source_id"], copy_obs=True)
        >>> adata.obs["A1Cresult_measured"] = ~adata.obs["A1Cresult"].isna()
        >>> ep.pl.catplot(
        ...     adata=adata,
        ...     y="A1Cresult_measured",
        ...     x="admission_source_id",
        ...     kind="point",
        ...     ci=95,
        ...     join=False,
        ... )

        .. image:: /_static/docstring_previews/catplot.png
    """

    return sns.catplot(data=adata.obs, x=x, y=y, hue=hue, kind=kind, **kwargs)
