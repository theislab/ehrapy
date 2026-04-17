from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def normalize_screening_result(
    result: pd.DataFrame,
    level: str,
    *,
    drug_col: str | None = None,
) -> pd.DataFrame:
    """Normalize one screening result table to the shared postprocessing schema.

    The original R workflow stores the drug identifier in the first column and then
    annotates the table with the grouping level (`chapter`, `section`, `paragraph`,
    or `substance`). This helper keeps the same behavior while making it explicit.

    Args:
        result: Screening result table for one grouping level.
        level: Drug grouping level represented by the table.
        drug_col: Optional source column to rename to ``drug``. If omitted and the
            table has no ``drug`` column, the first column is renamed.

    Returns:
        A copy of ``result`` with a normalized ``drug`` column and a ``drug.level``
        annotation column.
    """
    if result.empty and len(result.columns) == 0:
        raise ValueError("result must contain at least one column")

    normalized = result.copy()

    if drug_col is not None:
        if drug_col not in normalized.columns:
            raise KeyError(f"{drug_col} is not present in the result table")
        normalized = normalized.rename(columns={drug_col: "drug"})
    elif "drug" not in normalized.columns:
        normalized = normalized.rename(columns={normalized.columns[0]: "drug"})

    normalized["drug.level"] = level
    return normalized


def combine_screening_results(
    results_by_level: Mapping[str, pd.DataFrame | None],
    *,
    levels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Combine screening result tables across grouping levels.

    Args:
        results_by_level: Mapping from grouping level to result dataframe.
        levels: Optional explicit concatenation order. If omitted, insertion order
            from ``results_by_level`` is used.

    Returns:
        One concatenated dataframe with a ``drug.level`` column.
    """
    ordered_levels = list(levels) if levels is not None else list(results_by_level)
    combined: list[pd.DataFrame] = []

    for level in ordered_levels:
        result = results_by_level.get(level)
        if result is None:
            continue
        combined.append(normalize_screening_result(result, level))

    if not combined:
        return pd.DataFrame()

    return pd.concat(combined, ignore_index=True)


def rank_repurposing_hits(
    results: pd.DataFrame,
    *,
    min_unexposed_events: int = 30,
    min_exposed_events: int = 30,
    unexposed_col: str = "N.disease.B.during.unexposed",
    exposed_col: str = "N.disease.C.during.exposed",
    upper_ci_col: str = "IRR.higher.95",
) -> pd.DataFrame:
    """Rank potential repurposing signals from screening results.

    This ports the first repurposing filter from ``original/results.R``.
    Candidates must have enough events in both windows and an upper 95% IRR bound
    below 1. Lower values rank first.
    """
    prioritized = _prioritize_columns(results, ["drug", "disease", upper_ci_col, "age.group", "drug.level"])
    filtered = prioritized[
        (prioritized[unexposed_col] > min_unexposed_events)
        & (prioritized[exposed_col] > min_exposed_events)
        & (prioritized[upper_ci_col] < 1)
    ]
    return filtered.sort_values(by=upper_ci_col, ascending=True).reset_index(drop=True)


def rank_safety_hits(
    results: pd.DataFrame,
    *,
    min_unexposed_events: int = 30,
    min_exposed_events: int = 30,
    unexposed_col: str = "N.disease.B.during.unexposed",
    exposed_col: str = "N.disease.C.during.exposed",
    lower_ci_col: str = "IRR.lower.95",
) -> pd.DataFrame:
    """Rank potential safety signals from screening results.

    This ports the first safety filter from ``original/results.R``.
    Candidates must have enough events in both windows and a lower 95% IRR bound
    above 1. Higher values rank first.
    """
    prioritized = _prioritize_columns(results, ["drug", "disease", lower_ci_col, "age.group", "drug.level"])
    filtered = prioritized[
        (prioritized[unexposed_col] > min_unexposed_events)
        & (prioritized[exposed_col] > min_exposed_events)
        & (prioritized[lower_ci_col] > 1)
    ]
    return filtered.sort_values(by=lower_ci_col, ascending=False).reset_index(drop=True)


def _prioritize_columns(result: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    available = [column for column in columns if column in result.columns]
    remainder = [column for column in result.columns if column not in available]
    return result.loc[:, [*available, *remainder]].copy()
