from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import seaborn as sns

from ehrapy.get import obs_df

if TYPE_CHECKING:
    from ehrdata import EHRData
    from seaborn.axisgrid import FacetGrid


def catplot(
    edata: EHRData,
    *,
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    kind: Literal["strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"] = "strip",
    **kwargs,
) -> FacetGrid:
    """Plot categorical data.

    Wrapper around `seaborn.catplot <https://seaborn.pydata.org/generated/seaborn.catplot.html>`_. Typically used to show
    the behaviour of one numerical variable with respect to one or several categorical variables.

    Columns of `edata.obs` and variables can be plotted, with variables of 3D data at their first non-missing value, see :func:`~ehrapy.get.obs_df`.

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
        >>> ed.move_to_obs(edata, ["A1Cresult", "admission_source_id"], copy_columns=True)
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
    var_names = [key for key in (x, y, hue) if key in edata.var_names and key not in edata.obs]
    data = edata.obs.join(obs_df(edata, keys=var_names)) if var_names else edata.obs
    return sns.catplot(data=data, x=x, y=y, hue=hue, kind=kind, **kwargs)
