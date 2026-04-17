from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

ACTIVE_INGREDIENT_RELATIONSHIPS: tuple[str, ...] = (
    "Has specific active ingredient (attribute)",
    "Has precise active ingredient (attribute)",
)


def extract_rxclass_may_treat(
    payload: Mapping[str, Any],
    *,
    rxcui: str | int | None = None,
    rxcui_col: str = "rxcui",
    drug_col: str = "drug",
    mesh_col: str = "mesh",
    disease_col: str = "disease",
) -> pd.DataFrame:
    """Extract RxClass ``may_treat`` rows from one RxNav response payload."""
    entries = payload.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
    rows: list[dict[str, object]] = []

    for entry in _as_sequence(entries):
        if not isinstance(entry, Mapping):
            continue
        min_concept = entry.get("minConcept", {})
        rxclass_concept = entry.get("rxclassMinConceptItem", {})
        rows.append(
            {
                rxcui_col: rxcui,
                drug_col: _mapping_get(min_concept, "name"),
                mesh_col: _mapping_get(rxclass_concept, "classId"),
                disease_col: _mapping_get(rxclass_concept, "className"),
            }
        )

    return (
        pd.DataFrame(rows, columns=[rxcui_col, drug_col, mesh_col, disease_col])
        .dropna(subset=[mesh_col])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def extract_readcodev3_from_snomedbrowser(
    text_nodes: Sequence[str],
    *,
    snomed_disease: str | int | None = None,
    text_index: int = 7,
    snomed_col: str = "snomed.disease",
    readcode_col: str = "readcodev3",
) -> pd.DataFrame:
    """Extract Read code v3 rows from the SNOMED browser text-node output.

    The original R workflow reads the eighth text node and splits it on spaces.
    This helper keeps the same assumption while making the data dependency explicit.
    """
    if len(text_nodes) <= text_index:
        return pd.DataFrame(columns=[snomed_col, readcode_col])

    readcode_text = str(text_nodes[text_index]).strip()
    if not readcode_text:
        return pd.DataFrame(columns=[snomed_col, readcode_col])

    readcodes = [token for token in re.split(r"\s+", readcode_text) if token]
    return (
        pd.DataFrame({snomed_col: snomed_disease, readcode_col: readcodes})
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )


def extract_snomed_ingredient_links(
    attr_concepts: pd.DataFrame,
    *,
    snomed_drug: str | int,
    snomed_drug_col: str = "snomed.drug.uk",
    snomed_ingredient_col: str = "snomed.ingredient",
    source_id_col: str = "sourceId",
    source_desc_col: str = "sourceDesc",
    destination_id_col: str = "destinationId",
    type_desc_col: str = "typeDesc",
) -> pd.DataFrame:
    """Extract SNOMED UK drug-to-ingredient links from one ``attrConcept`` result."""
    _require_columns(
        attr_concepts,
        {source_id_col, source_desc_col, destination_id_col, type_desc_col},
        context="SNOMED attribute concepts",
    )

    source_matches = attr_concepts.loc[attr_concepts[source_id_col].astype("string") == str(snomed_drug)].copy()
    if source_matches.empty:
        return pd.DataFrame(columns=[snomed_drug_col, snomed_ingredient_col])

    if source_matches[source_desc_col].astype("string").str.contains(r"\(substance\)", case=False, regex=True).any():
        return pd.DataFrame({snomed_drug_col: [snomed_drug], snomed_ingredient_col: [snomed_drug]})

    ingredient_ids = (
        source_matches.loc[source_matches[type_desc_col].isin(ACTIVE_INGREDIENT_RELATIONSHIPS), destination_id_col]
        .dropna()
        .drop_duplicates()
    )
    if ingredient_ids.empty:
        return pd.DataFrame(columns=[snomed_drug_col, snomed_ingredient_col])
    return (
        pd.DataFrame({snomed_drug_col: snomed_drug, snomed_ingredient_col: ingredient_ids})
        .dropna()
        .drop_duplicates(ignore_index=True)
    )


def extract_rxnav_ingredient_links(
    payload: Mapping[str, Any],
    *,
    rxcui: str | int,
    include_self: bool = True,
    rxcui_col: str = "rxcui",
    ingredient_col: str = "rxcui.ingredient",
    tty_col: str = "tty",
) -> pd.DataFrame:
    """Extract ingredient RxCUIs from one RxNav ``allrelated`` payload."""
    groups = payload.get("allRelatedGroup", {}).get("conceptGroup", [])
    rows: list[dict[str, object]] = []

    for group in _as_sequence(groups):
        if not isinstance(group, Mapping):
            continue
        for concept in _as_sequence(group.get("conceptProperties", [])):
            if not isinstance(concept, Mapping) or concept.get(tty_col) != "IN":
                continue
            rows.append({rxcui_col: rxcui, ingredient_col: concept.get("rxcui")})

    if include_self:
        rows.append({rxcui_col: rxcui, ingredient_col: rxcui})

    return pd.DataFrame(rows, columns=[rxcui_col, ingredient_col]).dropna().drop_duplicates().reset_index(drop=True)


def normalize_readcodev3_to_v2_map(
    mapping: pd.DataFrame,
    *,
    readcodev3_col: str = "readcodev3",
    readcodev2_col: str = "readcodev2",
    excluded_readcodev2: Sequence[str] = ("_NONE", "_DRUG"),
) -> pd.DataFrame:
    """Normalize the NHS Read code v3 to v2 mapping table used by the R workflow."""
    _require_columns(mapping, {readcodev3_col, readcodev2_col}, context="Read code v3 to v2 mapping")

    normalized = mapping.loc[:, [readcodev3_col, readcodev2_col]].dropna().drop_duplicates()
    if excluded_readcodev2:
        normalized = normalized.loc[~normalized[readcodev2_col].isin(excluded_readcodev2)]
    return normalized.reset_index(drop=True)


def normalize_readcodev2_medcode_map(
    medical: pd.DataFrame,
    *,
    medcode_col: str = "medcode",
    readcode_col: str = "readcode",
    output_readcode_col: str = "readcodev2",
    prefix_len: int = 5,
) -> pd.DataFrame:
    """Normalize CPRD ``medical.txt`` rows to the Read code v2 mapping used in R."""
    _require_columns(medical, {medcode_col, readcode_col}, context="Read code v2 to medcode mapping")

    normalized = medical.loc[:, [medcode_col, readcode_col]].dropna().copy()
    normalized[output_readcode_col] = normalized[readcode_col].astype("string").str.slice(0, prefix_len)
    return normalized.loc[:, [output_readcode_col, medcode_col]].dropna().drop_duplicates().reset_index(drop=True)


def normalize_snomed_bnf_map(
    mapping: pd.DataFrame,
    *,
    bnfcode_col: str = "bnfcode",
    snomed_drug_col: str = "snomed.drug.uk",
    prefix_len: int = 6,
) -> pd.DataFrame:
    """Normalize the BNF to SNOMED drug mapping used on the drug side."""
    _require_columns(mapping, {bnfcode_col, snomed_drug_col}, context="BNF to SNOMED mapping")

    normalized = mapping.loc[:, [bnfcode_col, snomed_drug_col]].dropna().copy()
    normalized[bnfcode_col] = normalized[bnfcode_col].astype("string").str.slice(0, prefix_len)
    return normalized.dropna().drop_duplicates().reset_index(drop=True)


def build_rxcui_snomed_disease_map(
    rxcui_mesh: pd.DataFrame,
    mesh_snomed_disease: pd.DataFrame,
    *,
    rxcui_col: str = "rxcui",
    mesh_col: str = "mesh",
    snomed_disease_col: str = "snomed.disease",
) -> pd.DataFrame:
    """Build the RxCUI to SNOMED disease map from MeSH links."""
    _require_columns(rxcui_mesh, {rxcui_col, mesh_col}, context="RxCUI to MeSH mapping")
    _require_columns(mesh_snomed_disease, {mesh_col, snomed_disease_col}, context="MeSH to SNOMED disease mapping")

    return (
        rxcui_mesh.loc[:, [rxcui_col, mesh_col]]
        .merge(mesh_snomed_disease.loc[:, [mesh_col, snomed_disease_col]], how="left", on=mesh_col)
        .loc[:, [rxcui_col, snomed_disease_col]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_rxcui_readcodev3_map(
    rxcui_snomed_disease: pd.DataFrame,
    snomed_disease_readcodev3: pd.DataFrame,
    *,
    rxcui_col: str = "rxcui",
    snomed_disease_col: str = "snomed.disease",
    readcodev3_col: str = "readcodev3",
) -> pd.DataFrame:
    """Build the RxCUI to Read code v3 map from SNOMED disease links."""
    _require_columns(rxcui_snomed_disease, {rxcui_col, snomed_disease_col}, context="RxCUI to SNOMED disease mapping")
    _require_columns(
        snomed_disease_readcodev3,
        {snomed_disease_col, readcodev3_col},
        context="SNOMED disease to Read code v3 mapping",
    )

    return (
        rxcui_snomed_disease.loc[:, [rxcui_col, snomed_disease_col]]
        .merge(
            snomed_disease_readcodev3.loc[:, [snomed_disease_col, readcodev3_col]], how="left", on=snomed_disease_col
        )
        .loc[:, [rxcui_col, readcodev3_col]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_rxcui_readcodev2_map(
    rxcui_readcodev3: pd.DataFrame,
    readcodev3_readcodev2: pd.DataFrame,
    *,
    rxcui_col: str = "rxcui",
    readcodev3_col: str = "readcodev3",
    readcodev2_col: str = "readcodev2",
) -> pd.DataFrame:
    """Build the RxCUI to Read code v2 map used before CPRD medcode joins."""
    _require_columns(rxcui_readcodev3, {rxcui_col, readcodev3_col}, context="RxCUI to Read code v3 mapping")
    _require_columns(
        readcodev3_readcodev2,
        {readcodev3_col, readcodev2_col},
        context="Read code v3 to v2 mapping",
    )

    return (
        rxcui_readcodev3.loc[:, [rxcui_col, readcodev3_col]]
        .merge(readcodev3_readcodev2.loc[:, [readcodev3_col, readcodev2_col]], how="left", on=readcodev3_col)
        .loc[:, [rxcui_col, readcodev2_col]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_rxcui_medcode_map(
    rxcui_readcodev2: pd.DataFrame,
    readcodev2_medcode: pd.DataFrame,
    *,
    rxcui_col: str = "rxcui",
    readcodev2_col: str = "readcodev2",
    medcode_col: str = "medcode",
    fallback_prefix_len: int | None = 4,
) -> pd.DataFrame:
    """Build the disease-side RxCUI to medcode map with the original 4-char fallback."""
    _require_columns(rxcui_readcodev2, {rxcui_col, readcodev2_col}, context="RxCUI to Read code v2 mapping")
    _require_columns(readcodev2_medcode, {readcodev2_col, medcode_col}, context="Read code v2 to medcode mapping")

    direct = (
        rxcui_readcodev2.loc[:, [rxcui_col, readcodev2_col]]
        .merge(readcodev2_medcode.loc[:, [readcodev2_col, medcode_col]], how="left", on=readcodev2_col)
        .loc[:, [rxcui_col, medcode_col]]
        .dropna()
        .drop_duplicates()
    )

    if not fallback_prefix_len:
        return direct.reset_index(drop=True)

    lookup_codes = set(readcodev2_medcode[readcodev2_col].dropna())
    unmatched = rxcui_readcodev2.loc[
        ~rxcui_readcodev2[readcodev2_col].isin(lookup_codes), [rxcui_col, readcodev2_col]
    ].copy()
    if unmatched.empty:
        return direct.reset_index(drop=True)

    fallback_lookup = readcodev2_medcode.loc[:, [readcodev2_col, medcode_col]].dropna().copy()
    fallback_lookup[readcodev2_col] = fallback_lookup[readcodev2_col].astype("string").str.slice(0, fallback_prefix_len)
    fallback_lookup = fallback_lookup.drop_duplicates()

    unmatched[readcodev2_col] = unmatched[readcodev2_col].astype("string").str.slice(0, fallback_prefix_len)
    fallback = (
        unmatched.merge(fallback_lookup, how="left", on=readcodev2_col)
        .loc[:, [rxcui_col, medcode_col]]
        .dropna()
        .drop_duplicates()
    )

    return pd.concat([direct, fallback], ignore_index=True).drop_duplicates().reset_index(drop=True)


def build_bnfcode_prodcode_map(
    product_df: pd.DataFrame,
    *,
    prodcode_col: str = "prodcode",
    bnfcode_col: str = "bnfcode",
    prefix_len: int = 6,
) -> pd.DataFrame:
    """Build the prodcode to truncated BNF code map from CPRD product rows."""
    _require_columns(product_df, {prodcode_col, bnfcode_col}, context="Product to BNF mapping")

    exploded = product_df.loc[:, [prodcode_col, bnfcode_col]].dropna().copy()
    exploded[bnfcode_col] = exploded[bnfcode_col].astype("string").str.split("/")
    exploded = exploded.explode(bnfcode_col)
    exploded[bnfcode_col] = exploded[bnfcode_col].astype("string").str.strip().str.slice(0, prefix_len)
    return exploded.dropna().drop_duplicates().reset_index(drop=True)


def build_rxcui_prodcode_map(
    rxcui_ingredient: pd.DataFrame,
    snomed_ingredient_rxcui: pd.DataFrame,
    snomed_ingredient_uk: pd.DataFrame,
    snomed_bnfcode: pd.DataFrame,
    bnfcode_prodcode: pd.DataFrame,
    *,
    rxcui_col: str = "rxcui",
    ingredient_rxcui_col: str = "rxcui.ingredient",
    snomed_ingredient_col: str = "snomed.ingredient",
    snomed_drug_col: str = "snomed.drug.uk",
    bnfcode_col: str = "bnfcode",
    prodcode_col: str = "prodcode",
) -> pd.DataFrame:
    """Build the drug-side RxCUI to prodcode map.

    This mirrors the merge order used in ``drug indication.R``. In particular, it
    joins ``rxcui_ingredient`` to the SNOMED ingredient table by the base ``rxcui``
    column, matching the original script exactly.
    """
    _require_columns(rxcui_ingredient, {rxcui_col, ingredient_rxcui_col}, context="RxCUI ingredient mapping")
    _require_columns(
        snomed_ingredient_rxcui,
        {snomed_ingredient_col, rxcui_col},
        context="SNOMED ingredient to RxCUI mapping",
    )
    _require_columns(
        snomed_ingredient_uk,
        {snomed_ingredient_col, snomed_drug_col},
        context="SNOMED drug to ingredient mapping",
    )
    _require_columns(snomed_bnfcode, {snomed_drug_col, bnfcode_col}, context="SNOMED drug to BNF mapping")
    _require_columns(bnfcode_prodcode, {bnfcode_col, prodcode_col}, context="BNF to prodcode mapping")

    return (
        rxcui_ingredient.merge(snomed_ingredient_rxcui, how="left", on=rxcui_col)
        .merge(snomed_ingredient_uk, how="left", on=snomed_ingredient_col)
        .merge(snomed_bnfcode, how="left", on=snomed_drug_col)
        .merge(bnfcode_prodcode, how="left", on=bnfcode_col)
        .loc[:, [rxcui_col, prodcode_col]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_prodcode_medcode_map(
    rxcui_prodcode: pd.DataFrame,
    rxcui_medcode: pd.DataFrame,
    *,
    rxcui_col: str = "rxcui",
    prodcode_col: str = "prodcode",
    medcode_col: str = "medcode",
) -> pd.DataFrame:
    """Build the final prodcode to medcode indication map."""
    _require_columns(rxcui_prodcode, {rxcui_col, prodcode_col}, context="RxCUI to prodcode mapping")
    _require_columns(rxcui_medcode, {rxcui_col, medcode_col}, context="RxCUI to medcode mapping")

    return (
        rxcui_prodcode.loc[:, [rxcui_col, prodcode_col]]
        .merge(rxcui_medcode.loc[:, [rxcui_col, medcode_col]], how="inner", on=rxcui_col)
        .loc[:, [prodcode_col, medcode_col]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_disease_indication_map(
    rxcui_mesh: pd.DataFrame,
    mesh_snomed_disease: pd.DataFrame,
    snomed_disease_readcodev3: pd.DataFrame,
    readcodev3_readcodev2: pd.DataFrame,
    medical: pd.DataFrame,
    *,
    rxcui_col: str = "rxcui",
    mesh_col: str = "mesh",
    snomed_disease_col: str = "snomed.disease",
    readcodev3_col: str = "readcodev3",
    readcodev2_col: str = "readcodev2",
    medcode_col: str = "medcode",
    readcode_col: str = "readcode",
    fallback_prefix_len: int | None = 4,
) -> pd.DataFrame:
    """Run the disease-side mapping workflow from RxCUI to CPRD medcode."""
    rxcui_snomed_disease = build_rxcui_snomed_disease_map(
        rxcui_mesh,
        mesh_snomed_disease,
        rxcui_col=rxcui_col,
        mesh_col=mesh_col,
        snomed_disease_col=snomed_disease_col,
    )
    rxcui_readcodev3 = build_rxcui_readcodev3_map(
        rxcui_snomed_disease,
        snomed_disease_readcodev3,
        rxcui_col=rxcui_col,
        snomed_disease_col=snomed_disease_col,
        readcodev3_col=readcodev3_col,
    )
    normalized_v3_v2 = normalize_readcodev3_to_v2_map(
        readcodev3_readcodev2,
        readcodev3_col=readcodev3_col,
        readcodev2_col=readcodev2_col,
    )
    rxcui_readcodev2 = build_rxcui_readcodev2_map(
        rxcui_readcodev3,
        normalized_v3_v2,
        rxcui_col=rxcui_col,
        readcodev3_col=readcodev3_col,
        readcodev2_col=readcodev2_col,
    )
    normalized_medical = normalize_readcodev2_medcode_map(
        medical,
        medcode_col=medcode_col,
        readcode_col=readcode_col,
        output_readcode_col=readcodev2_col,
    )
    return build_rxcui_medcode_map(
        rxcui_readcodev2,
        normalized_medical,
        rxcui_col=rxcui_col,
        readcodev2_col=readcodev2_col,
        medcode_col=medcode_col,
        fallback_prefix_len=fallback_prefix_len,
    )


def build_drug_indication_map(
    rxcui_ingredient: pd.DataFrame,
    snomed_ingredient_rxcui: pd.DataFrame,
    snomed_ingredient_uk: pd.DataFrame,
    snomed_bnfcode: pd.DataFrame,
    product_df: pd.DataFrame,
    *,
    rxcui_col: str = "rxcui",
    ingredient_rxcui_col: str = "rxcui.ingredient",
    snomed_ingredient_col: str = "snomed.ingredient",
    snomed_drug_col: str = "snomed.drug.uk",
    bnfcode_col: str = "bnfcode",
    prodcode_col: str = "prodcode",
) -> pd.DataFrame:
    """Run the drug-side mapping workflow from RxCUI to CPRD prodcode."""
    normalized_snomed_bnf = normalize_snomed_bnf_map(
        snomed_bnfcode,
        bnfcode_col=bnfcode_col,
        snomed_drug_col=snomed_drug_col,
    )
    bnfcode_prodcode = build_bnfcode_prodcode_map(
        product_df,
        prodcode_col=prodcode_col,
        bnfcode_col=bnfcode_col,
    )
    return build_rxcui_prodcode_map(
        rxcui_ingredient,
        snomed_ingredient_rxcui,
        snomed_ingredient_uk,
        normalized_snomed_bnf,
        bnfcode_prodcode,
        rxcui_col=rxcui_col,
        ingredient_rxcui_col=ingredient_rxcui_col,
        snomed_ingredient_col=snomed_ingredient_col,
        snomed_drug_col=snomed_drug_col,
        bnfcode_col=bnfcode_col,
        prodcode_col=prodcode_col,
    )


def build_indication_map(
    rxcui_mesh: pd.DataFrame,
    mesh_snomed_disease: pd.DataFrame,
    snomed_disease_readcodev3: pd.DataFrame,
    readcodev3_readcodev2: pd.DataFrame,
    medical: pd.DataFrame,
    rxcui_ingredient: pd.DataFrame,
    snomed_ingredient_rxcui: pd.DataFrame,
    snomed_ingredient_uk: pd.DataFrame,
    snomed_bnfcode: pd.DataFrame,
    product_df: pd.DataFrame,
    *,
    rxcui_col: str = "rxcui",
    mesh_col: str = "mesh",
    snomed_disease_col: str = "snomed.disease",
    readcodev3_col: str = "readcodev3",
    readcodev2_col: str = "readcodev2",
    medcode_col: str = "medcode",
    readcode_col: str = "readcode",
    ingredient_rxcui_col: str = "rxcui.ingredient",
    snomed_ingredient_col: str = "snomed.ingredient",
    snomed_drug_col: str = "snomed.drug.uk",
    bnfcode_col: str = "bnfcode",
    prodcode_col: str = "prodcode",
    fallback_prefix_len: int | None = 4,
) -> pd.DataFrame:
    """Run the full indication-mapping workflow to build ``prodcode_medcode``."""
    rxcui_medcode = build_disease_indication_map(
        rxcui_mesh,
        mesh_snomed_disease,
        snomed_disease_readcodev3,
        readcodev3_readcodev2,
        medical,
        rxcui_col=rxcui_col,
        mesh_col=mesh_col,
        snomed_disease_col=snomed_disease_col,
        readcodev3_col=readcodev3_col,
        readcodev2_col=readcodev2_col,
        medcode_col=medcode_col,
        readcode_col=readcode_col,
        fallback_prefix_len=fallback_prefix_len,
    )
    rxcui_prodcode = build_drug_indication_map(
        rxcui_ingredient,
        snomed_ingredient_rxcui,
        snomed_ingredient_uk,
        snomed_bnfcode,
        product_df,
        rxcui_col=rxcui_col,
        ingredient_rxcui_col=ingredient_rxcui_col,
        snomed_ingredient_col=snomed_ingredient_col,
        snomed_drug_col=snomed_drug_col,
        bnfcode_col=bnfcode_col,
        prodcode_col=prodcode_col,
    )
    return build_prodcode_medcode_map(
        rxcui_prodcode,
        rxcui_medcode,
        rxcui_col=rxcui_col,
        prodcode_col=prodcode_col,
        medcode_col=medcode_col,
    )


def _as_sequence(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _mapping_get(value: object, key: str) -> object:
    return value.get(key) if isinstance(value, Mapping) else None


def _require_columns(frame: pd.DataFrame, columns: set[str], *, context: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns for {context}: {missing_str}")
