from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from ehrdata import EHRData
from ehrdata._feature_types import _detect_feature_type
from ehrdata.core.constants import CATEGORICAL_TAG
from tableone import TableOne

if TYPE_CHECKING:
    from collections.abc import Sequence


_RESERVED_COLS = {"Missing", "P-Value", "Test", "Overall", "P-Value (adjusted)"}


def _parse_count_pct(cell: str | None) -> tuple[float, float]:
    """Parse a ``"count (pct)"`` tableone cell into ``(count, pct)``. Empty cells become ``(0.0, 0.0)``."""
    if cell is None:
        return 0.0, 0.0
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return 0.0, 0.0
    try:
        count_str, rest = s.split("(", 1)
        pct_str = rest.split(")", 1)[0]
        return float(count_str.strip()), float(pct_str.strip())
    except (ValueError, IndexError):
        return 0.0, 0.0


def stratified_table_one(
    edata: EHRData,
    *,
    groupby: str,
    columns: Sequence | None = None,
    categorical: Sequence | None = None,
    nonnormal: Sequence | None = None,
    pval_adjust: str | None = None,
    htest: dict | None = None,
    missing: bool = True,
    key_added: str = "stratified_table_one",
    copy: bool = False,
    **tableone_kwargs,
) -> EHRData | None:
    """Build a stratified "Table 1" comparing baseline characteristics across groups.

    Produces a publication-ready table stratified by ``groupby`` with appropriate per-variable hypothesis tests (chi-square / Fisher's exact for categorical variables, t-test / ANOVA for normally distributed continuous variables, Mann-Whitney U / Kruskal-Wallis for variables listed in ``nonnormal``).
    Wraps the ``tableone`` package [1].

    The rendered table and the intermediate data needed for plotting are stored in ``edata.uns[key_added]``.
    Access the table via ``edata.uns[key_added]["table"]``.
    Use :func:`ehrapy.plot.stratified_table_one` to visualize.

    Args:
        edata: Central data object.
        groupby: Column in ``edata.obs`` to stratify by.
        columns: Columns to include in the table. If `None`, all of ``edata.obs`` except ``groupby`` is used.
        categorical: Columns that contain categorical variables. If `None`, types are inferred.
        nonnormal: Continuous columns that should use a non-parametric test (Mann-Whitney U / Kruskal-Wallis) and report median [Q1, Q3] instead of mean (SD).
        pval_adjust: Multiple-testing correction (e.g. ``"bonferroni"``, ``"holm"``, ``"fdr_bh"``).
        htest: Mapping of column name to a custom hypothesis-test function (advanced).
        missing: If `True`, include a `Missing` column in the rendered table.
        key_added: Key under which results are stored in ``edata.uns``.
        copy: If `True`, return a modified copy of ``edata``; otherwise modify in place and return `None`.
        **tableone_kwargs: Extra keyword arguments forwarded to :class:`tableone.TableOne`.

    Returns:
        ``None`` (default) or a copy of ``edata`` with results stored in ``.uns[key_added]`` when ``copy=True``.

    References:
        [1] Tom Pollard, Alistair E.W. Johnson, Jesse D. Raffa, Roger G. Mark; tableone: An open source Python package for producing summary statistics for research papers, Journal of the American Medical Informatics Association, Volume 24, Issue 2, 1 March 2017, Pages 267-271, https://doi.org/10.1093/jamia/ocw117

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
        >>> edata.uns["stratified_table_one"]["table"]  # the rendered Table 1
        >>> ep.pl.stratified_table_one(edata)
    """
    if not isinstance(edata, EHRData):
        raise ValueError("edata must be an EHRData.")

    if groupby not in edata.obs.columns:
        raise ValueError(f"groupby column {groupby!r} not found in edata.obs.")

    if edata.obs[groupby].nunique(dropna=True) < 2:
        raise ValueError(
            f"groupby column {groupby!r} has fewer than 2 unique non-null values; cannot stratify."
        )

    target = edata.copy() if copy else edata

    if columns is None:
        cols = [c for c in target.obs.columns if c != groupby]
    else:
        missing_cols = set(columns) - set(target.obs.columns)
        if missing_cols:
            raise ValueError(f"Columns {list(missing_cols)} not found in edata.obs.")
        cols = [c for c in columns if c != groupby]

    if categorical is None:
        cats = [
            col for col in cols if _detect_feature_type(target.obs[col])[0] == CATEGORICAL_TAG
        ]
    else:
        missing_cats = set(categorical) - set(target.obs.columns)
        if missing_cats:
            raise ValueError(f"Categorical columns {list(missing_cats)} not found in edata.obs.")
        if set(categorical).difference(set(cols)):
            raise ValueError("categorical columns must be a subset of `columns`.")
        cats = list(categorical)

    categorical_categories = {
        col: list(target.obs[col].astype("category").cat.categories) for col in cats
    }

    # tableone chokes on pandas Categorical dtype — cast the groupby + categorical columns to plain strings
    obs_for_tableone = target.obs.copy()
    obs_for_tableone[groupby] = obs_for_tableone[groupby].astype(str)
    for col in cats:
        obs_for_tableone[col] = obs_for_tableone[col].astype(str)

    t1 = TableOne(
        obs_for_tableone,
        columns=cols,
        categorical=cats,
        groupby=groupby,
        nonnormal=list(nonnormal) if nonnormal else None,
        pval=True,
        pval_adjust=pval_adjust,
        htest=htest,
        missing=missing,
        **tableone_kwargs,
    )

    groups = [g for g in t1.cont_table.columns if g not in _RESERVED_COLS]
    group_counts = {g: int((target.obs[groupby].astype(str) == str(g)).sum()) for g in groups}

    def _format_pval(val) -> str | None:
        s = str(val).strip()
        if s == "" or s.lower() == "nan":
            return None
        try:
            f = float(s)
        except ValueError:
            return s  # already preformatted, e.g. "<0.001"
        if f < 0.001:
            return "<0.001"
        return f"{f:.3f}"

    # parse p-values into a plain dict so the plot fn doesn't need to re-parse the rendered table
    pvalues: dict = {}
    for src in (t1.cont_table, t1.cat_table):
        if src is None or "P-Value" not in src.columns:
            continue
        for idx, val in src["P-Value"].items():
            var = idx[0] if isinstance(idx, tuple) else idx
            formatted = _format_pval(val)
            if formatted is None:
                continue
            pvalues.setdefault(var, formatted)

    # per-(group, category) percentages for plotting categoricals
    cat_pct: dict = {}
    for col in cats:
        per_group: dict = {}
        for group in groups:
            try:
                col_block = t1.cat_table[group].loc[col]
            except KeyError:
                col_block = None
            row: dict = {}
            for cat in categorical_categories[col]:
                pct = 0.0
                if col_block is not None and str(cat) in col_block.index:
                    _, pct = _parse_count_pct(col_block.loc[str(cat)])
                row[str(cat)] = pct
            per_group[str(group)] = row
        cat_pct[col] = per_group

    # per-group summary string for continuous variables
    num_summary: dict = {}
    for col in cols:
        if col in cats:
            continue
        per_group = {}
        for group in groups:
            try:
                per_group[str(group)] = str(t1.cont_table[group].loc[(col, "")]).strip()
            except KeyError:
                per_group[str(group)] = ""
        num_summary[col] = per_group

    target.uns[key_added] = {
        "table": t1.tableone,
        "groupby": groupby,
        "groups": [str(g) for g in groups],
        "group_counts": {str(g): group_counts[g] for g in groups},
        "columns": list(cols),
        "categorical": list(cats),
        "categorical_categories": {k: [str(c) for c in v] for k, v in categorical_categories.items()},
        "pvalues": pvalues,
        "cat_pct": cat_pct,
        "num_summary": num_summary,
    }

    return target if copy else None
