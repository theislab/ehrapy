from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, get_args

import pandas as pd

GroupingLevel = Literal["substance", "paragraph", "section", "chapter"]

VALID_GROUPING_LEVELS = get_args(GroupingLevel)

DEFAULT_GROUPING_COLUMNS: dict[GroupingLevel, str] = {
    "substance": "drugsubstance",
    "paragraph": "bnf.paragraph",
    "section": "bnf.section",
    "chapter": "bnf.chapter",
}


def validate_grouping_level(level: str) -> GroupingLevel:
    """Validate a drug-screening grouping level."""
    if level not in VALID_GROUPING_LEVELS:
        valid_levels = ", ".join(VALID_GROUPING_LEVELS)
        raise ValueError(f"Unsupported grouping level {level!r}. Expected one of: {valid_levels}.")
    return level  # type: ignore[return-value]


def resolve_grouping_column(level: str, *, grouping_col: str | None = None) -> str:
    """Resolve the source column used for a grouping level."""
    validated = validate_grouping_level(level)
    return grouping_col or DEFAULT_GROUPING_COLUMNS[validated]


def assign_grouping_labels(
    data: pd.DataFrame,
    *,
    level: str,
    mapping: pd.DataFrame | None = None,
    prodcode_col: str = "prodcode",
    grouping_col: str | None = None,
    source_drug_col: str = "drug",
    output_col: str = "drug",
) -> pd.DataFrame:
    """Assign chapter/section/paragraph labels to prepared prescription rows."""
    resolved_col = resolve_grouping_column(level, grouping_col=grouping_col)
    validated = validate_grouping_level(level)

    frame = data.copy()
    if validated == "substance":
        source_col = source_drug_col if source_drug_col in frame.columns else resolved_col
        if source_col not in frame.columns:
            raise KeyError(f"Missing required substance grouping column: {source_col}")
        frame[output_col] = frame[source_col]
        return frame.dropna(subset=[output_col]).reset_index(drop=True)

    if mapping is None:
        raise KeyError("A grouping mapping table is required for chapter, section, and paragraph workflows")
    required_columns = {prodcode_col, resolved_col}
    missing = required_columns.difference(mapping.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required grouping columns: {missing_str}")
    if prodcode_col not in frame.columns:
        raise KeyError(f"Missing required prescription grouping column: {prodcode_col}")

    lookup = mapping.loc[:, [prodcode_col, resolved_col]].dropna().drop_duplicates()
    frame = frame.merge(lookup, how="left", on=prodcode_col)
    frame[output_col] = frame[resolved_col]
    return frame.dropna(subset=[output_col]).reset_index(drop=True)


def group_known_pairs(
    known_pairs: pd.DataFrame | None,
    *,
    level: str,
    mapping: pd.DataFrame | None = None,
    prodcode_col: str = "prodcode",
    disease_col: str = "disease",
    grouping_col: str | None = None,
    output_col: str = "drug",
) -> pd.DataFrame | None:
    """Map known drug-disease pairs to the requested grouping level."""
    if known_pairs is None:
        return None

    resolved_col = resolve_grouping_column(level, grouping_col=grouping_col)
    validated = validate_grouping_level(level)

    if validated == "substance":
        if output_col not in known_pairs.columns or disease_col not in known_pairs.columns:
            raise KeyError("known_pairs must contain the configured drug and disease columns")
        return known_pairs.loc[:, [output_col, disease_col]].dropna().drop_duplicates().reset_index(drop=True)

    if prodcode_col in known_pairs.columns:
        if mapping is None:
            raise KeyError("A grouping mapping table is required to aggregate prodcode-level known pairs")
        required_mapping_columns = {prodcode_col, resolved_col}
        missing = required_mapping_columns.difference(mapping.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise KeyError(f"Missing required grouping columns: {missing_str}")
        if disease_col not in known_pairs.columns:
            raise KeyError(f"Missing required known-pair column: {disease_col}")
        mapped = known_pairs.merge(
            mapping.loc[:, [prodcode_col, resolved_col]].dropna().drop_duplicates(),
            how="left",
            on=prodcode_col,
        )
        mapped[output_col] = mapped[resolved_col]
        return mapped.loc[:, [output_col, disease_col]].dropna().drop_duplicates().reset_index(drop=True)

    if output_col not in known_pairs.columns or disease_col not in known_pairs.columns:
        raise KeyError("known_pairs must contain either prodcode+disease or grouped drug+disease columns")
    return known_pairs.loc[:, [output_col, disease_col]].dropna().drop_duplicates().reset_index(drop=True)


def count_ever_users_by_group(
    prescriptions: pd.DataFrame,
    *,
    patient_col: str = "patid",
    drug_col: str = "drug",
    output_count_col: str = "N.everuser",
) -> pd.DataFrame:
    """Count distinct ever-users for each grouped drug label."""
    required_columns = {patient_col, drug_col}
    missing = required_columns.difference(prescriptions.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns for ever-user counts: {missing_str}")
    return (
        prescriptions.loc[:, [patient_col, drug_col]]
        .dropna()
        .drop_duplicates()
        .groupby(drug_col, as_index=False)
        .agg(**{output_count_col: (patient_col, "nunique")})
    )
