from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import holoviews as hv
import pandas as pd

from ehrapy._compat import choose_hv_backend

_LEADING_NUMBER = re.compile(r"-?\d+\.?\d*")


def _extract_central_value(summary: str) -> float:
    """Extract the leading numeric value from a tableone summary like ``'50.1 (10.2)'`` or ``'1.0 [0.5,2.0]'``."""
    if not summary:
        return 0.0
    match = _LEADING_NUMBER.search(summary)
    if match is None:
        return 0.0
    try:
        return float(match.group(0))
    except ValueError:
        return 0.0


if TYPE_CHECKING:
    from ehrdata import EHRData


def _require_results(edata: EHRData, key: str) -> dict:
    if key not in edata.uns:
        raise KeyError(f"edata.uns[{key!r}] not found. Run `ep.tl.stratified_table_one(edata, groupby=...)` first.")
    return edata.uns[key]


@choose_hv_backend()
def stratified_table_one(
    edata: EHRData,
    *,
    key: str = "stratified_table_one",
    n_cols: int = 2,
    width: int = 380,
    height: int = 260,
    cmap: str | list[str] | None = "Category10",
    show_pvalues: bool = True,
    **kwargs,
) -> hv.Layout:
    """Plot the stratified "Table 1" baseline comparison stored by :func:`~ehrapy.tools.stratified_table_one`.

    Produces one panel per variable laid out in an ``n_cols``-column :class:`holoviews.Layout`:

    - **Categorical** variables — stacked horizontal bars per group (percentage within group).
    - **Continuous** variables — one horizontal bar per group annotated with the summary (e.g. ``mean (SD)`` or ``median [Q1, Q3]`` if listed in ``nonnormal``).

    Each panel title includes the variable name and, when ``show_pvalues=True``, the per-variable p-value as reported by ``tableone``.

    Args:
        edata: Central data object containing results stored by :func:`~ehrapy.tools.stratified_table_one`.
        key: Key under which results are stored in ``edata.uns`` (matches ``key_added``).
        n_cols: Number of columns in the panel layout.
        width: Width of each panel in pixels.
        height: Height of each panel in pixels.
        cmap: Colormap (name or explicit color list) used for categories.
        show_pvalues: Whether to append the p-value to each panel title.
        **kwargs: Additional ``.opts(...)`` styling forwarded to every panel.

    Returns:
        HoloViews Layout of per-variable panels.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.diabetes_130_fairlearn(
        ...     columns_obs_only=["gender", "race", "age", "readmit_binary", "num_procedures"]
        ... )
        >>> ep.tl.stratified_table_one(
        ...     edata,
        ...     groupby="readmit_binary",
        ...     columns=["gender", "race", "age", "num_procedures"],
        ...     nonnormal=["num_procedures"],
        ... )
        >>> ep.pl.stratified_table_one(edata)

        .. image:: /_static/docstring_previews/stratified_table_one.png
    """
    res = _require_results(edata, key)

    columns = res["columns"]
    categorical = set(res["categorical"])
    groups: list[str] = list(res["groups"])
    group_counts = res["group_counts"]
    cat_categories = res["categorical_categories"]
    cat_pct = res["cat_pct"]
    num_summary = res["num_summary"]
    pvalues = res["pvalues"] if show_pvalues else {}
    groupby = res["groupby"]

    max_cats = max((len(cat_categories[c]) for c in categorical), default=1)
    n_colors = max(max_cats, len(groups), 3)
    if isinstance(cmap, str):
        palette = hv.plotting.util.process_cmap(cmap, ncolors=n_colors)
    elif cmap is None:
        palette = hv.plotting.util.process_cmap("Category10", ncolors=n_colors)
    else:
        palette = list(cmap)
        if len(palette) < n_colors:
            raise ValueError(f"cmap has {len(palette)} colors but {n_colors} are needed.")

    group_labels = [f"{g} (n={group_counts[g]})" for g in groups]
    group_str_to_label = dict(zip(groups, group_labels, strict=True))

    is_bokeh = hv.Store.current_backend == "bokeh"

    panels = []
    for col in columns:
        title = col
        if col in pvalues:
            title = f"{col}  (p = {pvalues[col]})"

        common_opts: dict[str, Any] = {
            "title": title,
            "invert_axes": True,
        }
        if is_bokeh:
            common_opts["width"] = width
            common_opts["height"] = height
            common_opts["tools"] = ["hover"]
        common_opts.update(kwargs)

        if col in categorical:
            records = []
            for group in groups:
                for cat in cat_categories[col]:
                    records.append(
                        {
                            "group_label": group_str_to_label[group],
                            "category": str(cat),
                            "pct": float(cat_pct[col][group][str(cat)]),
                        }
                    )
            df = pd.DataFrame.from_records(records)
            bar_opts: dict[str, Any] = {
                "ylabel": "Percentage (%)",
                "xlabel": groupby,
                "show_legend": True,
                "color": hv.Cycle(palette[: len(cat_categories[col])]),
            }
            bar_opts.update(common_opts)
            bars = hv.Bars(
                df,
                kdims=["group_label", "category"],
                vdims=["pct"],
            ).opts(**bar_opts)
            panels.append(bars)
        else:
            records = [
                {
                    "group_label": group_str_to_label[group],
                    "value": _extract_central_value(num_summary[col][group]),
                    "summary": num_summary[col][group],
                }
                for group in groups
            ]
            df = pd.DataFrame.from_records(records)
            bar_opts = {
                "color": palette[0],
                "ylabel": col,
                "xlabel": groupby,
                "show_legend": False,
            }
            bar_opts.update(common_opts)
            bars = hv.Bars(
                df,
                kdims=["group_label"],
                vdims=["value", "summary"],
            ).opts(**bar_opts)
            panels.append(bars)

    if not panels:
        raise ValueError("No variables to plot.")

    return hv.Layout(panels).cols(n_cols)
