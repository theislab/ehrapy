from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from thefuzz import process

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData

    try:
        from medcat.cat import CAT

    except ModuleNotFoundError:
        pass


def _format_df_column(df: pd.DataFrame, column_name: str) -> list[tuple[int, str]]:
    """Format the df to match: formatted_data = [(row_id, row_text), (row_id, row_text), ...]
    as this is required by MedCAT's multiprocessing annotation step

    """
    formatted_data = []
    for id, row in df.iterrows():
        text = row[column_name]
        formatted_data.append((id, text))
    return formatted_data


def _flatten_annotated_results(annotation_results: dict) -> dict:
    """Flattens the nested set (usually 5 level nested) of annotation results.

    annotation_results is just a simple flattened dict with infos on all entities found
    """
    flattened_annotated_dict = {}
    entry_nr = 0

    # row numbers where the text column is located in the original data
    for row_id in annotation_results.keys():
        # all entities extracted from a given row
        entities = annotation_results[row_id]["entities"]
        for entity_id in entities.keys():
            # tokens are currently ignored, as they will not appear with the current basic model used by ehrapy from MedCAT
            if entity_id != "tokens":
                single_entity = {"row_nr": row_id}
                entity = entities[entity_id]
                # iterate over all info attributes of a single entity found in a specific row
                for entity_key in entity.keys():
                    if entity_key in ["pretty_name", "cui", "type_ids", "types"]:
                        single_entity[entity_key] = entities[entity_id][entity_key]
                    elif entity_key == "meta_anns":
                        single_entity[entity_key] = entities[entity_id][entity_key]["Status"]["value"]
                flattened_annotated_dict[entry_nr] = single_entity
                entry_nr += 1
    return flattened_annotated_dict


def annotate_text(
    adata: AnnData,
    cat: CAT,
    text_column: str,
    key_added: str = "medcat_annotations",
    n_proc: int = 2,
    batch_size_chars: int = 500000,
    copy: bool = False,
) -> AnnData | None:
    """Annotate the original free text data. Note this will only annotate non null rows.

    The result is a DataFrame.
    This DataFrame serves as the base for all further analyses, for example coloring UMAPs by specific diseases.

    Args:
        adata: AnnData object that holds the data to annotate.
        cat: MedCAT object.
        text_column: Name of the column that should be annotated.
        key_added: Key to add to adata.uns for the annotated results.
        n_proc: Number of processors to use.
        batch_size_chars: batch size to use for CAT's multiprocessing method.
        copy: Whether to copy adata or not.

    Returns:
        Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following fields;

        `adata.uns[key_added]` : :class:`pandas.DataFrame`
            DataFrame with the annotated results.

    """
    if copy:
        adata = adata.copy()
    non_null_text = pd.DataFrame(adata.obs[text_column][~adata.obs[text_column].isnull()])
    formatted_text_column = _format_df_column(non_null_text, text_column)
    results = cat.multiprocessing(formatted_text_column, batch_size_chars=batch_size_chars, nproc=n_proc)
    flattened_res = _flatten_annotated_results(results)

    # flatten annotated results into a Pandas DataFrame and remove duplicate entries; for example when a single entity like a disease is mentioned multiple times without any meaningful context changes
    # Example: The patient suffers from Diabetes. Cause of the Diabetes, he receives drug X.
    flattened_res_df = pd.DataFrame.from_dict(flattened_res, orient="index").drop_duplicates(
        subset=["cui", "row_nr", "meta_anns"]
    )

    # sort for row number in ascending order and reset index to keep index updated
    adata.uns[key_added] = flattened_res_df.sort_values(by=["row_nr"]).reset_index(drop=True)

    return adata if copy else None


def _filter_df_by_status(df: pd.DataFrame, status: str) -> pd.DataFrame:
    """Util function to filter passed dataframe by status."""
    df_res = df
    if status != "Both":
        if status not in {"Affirmed", "Other"}:
            raise StatusNotSupportedError(f"{status} is not available. Please use either Affirmed, Other or Both!")
        mask = df["meta_anns"].values == status
        df_res = df[mask]
    return df_res


def get_medcat_annotation_overview(
    adata: AnnData, status: str = "Affirmed", use_key: str = "medcat_annotations"
) -> pd.DataFrame:
    """Provide an overview for the annotation results. An overview will look like the following:

    cui (the CUI), nsubjects (from how many rows this one got extracted), type_ids (TUIs), name(name of the entitiy), perc_subjects (how many rows relative
    to absolute number of rows)

    Args:
        medcat_obj: The current MedCAT object which holds all infos on NLP analysis with MedCAT and ehrapy.
        n: Basically the parameter for head() of pandas Dataframe. How many of the most common entities should be shown?
        status: One of "Affirmed" (default), "Other" or "Both". Displays stats for either only affirmed entities, negated ones or both.
        use_key: Key to use for the annotated results.

    Returns:
        A Pandas DataFrame with the overview stats.
    """
    df = _filter_df_by_status(adata.uns[use_key], status)
    # group by CUI as this is a unique identifier per entity
    grouped = df.groupby("cui")
    # get absolute number of rows with this entity
    # note for overview, only one TUI and type is shown (there shouldn't be much situations were multiple are even possible or useful)
    res = grouped.agg(
        {
            "pretty_name": (lambda x: next(iter(set(x)))),
            "type_ids": (lambda x: next(iter(x))[0]),
            "types": (lambda x: next(iter(x))[0]),
            "row_nr": "nunique",
        }
    )
    res = res.rename(columns={"row_nr": "n_patient_visit"})
    # relative amount of patient visits with the specific entity to all patient visits (or rows in the original data)
    res["n_patient_visit_percent"] = (res["n_patient_visit"] / df["row_nr"].nunique()) * 100
    res.round({"n_patient_visit_percent": 1})

    return res


def _check_valid_name(df: pd.DataFrame, name: Iterable[str]) -> None:
    """Check whether the name is in the extracted entities to inform about possible typos.
    Currently, only the pretty_name column is supported.
    """
    invalid_names = []
    suggested_names = []

    for nm in name:
        pretty_names = df["pretty_name"].unique()

        if nm not in pretty_names:
            invalid_names.append(nm)
            try:
                new_name, _ = process.extractOne(query=nm, choices=pretty_names, score_cutoff=50)
                suggested_names.append(new_name)
            except EntitiyNotFoundError:
                pass

    if invalid_names:
        suggested_str = f" Do you mean {suggested_names}?" if suggested_names else ""
        msg = f"Did not find {invalid_names} in MedCAT's extracted entities and added them not to .obs.{suggested_str}"
        raise EntitiyNotFoundError(msg)


def add_medcat_annotation_to_obs(
    adata: AnnData,
    name: Iterable[str] | str,
    use_key: str = "medcat_annotations",
    added_colname: Iterable[str] | str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Add info extracted from free text as a binary column to obs.

    Indicates whether the specific entity to color by has been found in that row or not.


    Args:
        adata: AnnData object that holds the data to annotate.
        name: Name of the entity to add as a column to obs.
        use_key: Key to use for the annotated results.
        added_colname: Name of the column to add to obs. If None, name will be used.
        copy: Whether to copy adata or not.

    Returns:
        Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following fields;

        `adata.obs[name | added_coname]` : :class:`pandas.DataFrame`
            Added column(s) `to adata.obs`, indicating whether the specific entity to color by has been found in that row or not.

    """
    if use_key not in adata.uns.keys():
        raise ValueError(f"Key {use_key} not found in adata.uns. Please run ep.tl.annotate_text first.")

    if copy:
        adata = adata.copy()

    if isinstance(name, str):
        annotation_names = [name]
    else:
        annotation_names = list(name)

    if added_colname is None:
        added_colname = annotation_names
    elif isinstance(added_colname, str):
        added_colname = [added_colname]

    added_colnames = list(added_colname)
    if len(added_colnames) != len(annotation_names):
        raise ValueError(
            f"Length of added_colname ({len(added_colnames)}) does not match length of name ({len(annotation_names)})."
        )

    _check_valid_name(adata.uns[use_key], annotation_names)

    # only extract affirmed entities
    df = _filter_df_by_status(adata.uns[use_key], "Affirmed")

    # check whether the name is in the extracted entities to inform about possible typos
    # currently, only the pretty_name column is supported
    for i, annotation_name in enumerate(annotation_names):
        adata.obs[added_colnames[i]] = df.groupby("row_nr").agg(
            {
                "pretty_name": (
                    lambda row_pretty_names, annotation_name=annotation_name: any(
                        row_pretty_names.isin([annotation_name])
                    )
                )
            }
        )

    return adata if copy else None


class StatusNotSupportedError(Exception):
    pass


class EntitiyNotFoundError(Exception):
    pass
