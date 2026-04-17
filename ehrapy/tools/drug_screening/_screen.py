from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from ehrapy.tools.drug_screening._exposure import (
    build_exposure_episodes_from_prescriptions,
    prepare_exposure_windows,
    prepare_prescriptions_from_therapy,
)
from ehrapy.tools.drug_screening._grouping import (
    assign_grouping_labels,
    count_ever_users_by_group,
    group_known_pairs,
    validate_grouping_level,
)
from ehrapy.tools.drug_screening._self_controlled_cohort import rate_ratio_test

DEFAULT_AGE_GROUPS: tuple[tuple[str, float, float] | tuple[str], ...] = (
    ("0-20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-80", 60.0, 80.0),
    ("80-100", 80.0, 100.0),
    ("100-120", 100.0, 120.0),
    ("all",),
)
SCREENING_WORKFLOW_FOLLOWUP_DAYS: dict[str, int | None] = {
    "actual": None,
    "30days": 30,
    "365days": 365,
}


def screen_substance_cohort(
    prescriptions: pd.DataFrame,
    patients: pd.DataFrame,
    events: pd.DataFrame,
    *,
    patient_col: str = "patid",
    drug_col: str = "drug",
    duration_col: str = "duration",
    prescription_start_col: str = "start_date",
    event_date_col: str = "disease_eventdate",
    disease_col: str = "disease",
    fixed_followup_days: int | None = None,
    workflow: str = "actual",
    gap_days: int = 0,
    keep_first_only: bool = True,
    known_drug_disease_pairs: pd.DataFrame | None = None,
    ever_user_counts: pd.DataFrame | None = None,
    ever_user_drug_col: str = "drug",
    ever_user_count_col: str = "N.everuser",
    min_total_events: int = 100,
    age_groups: Sequence[tuple[str, float, float] | tuple[str]] = DEFAULT_AGE_GROUPS,
) -> pd.DataFrame:
    """Screen one prescription table end-to-end using the first substance-level port.

    This convenience wrapper mirrors the high-level structure of the R workflow:
    prescription rows with durations are collapsed into episodes, patient-specific
    exposed/unexposed windows are prepared, and the self-controlled cohort screen
    is run over diagnosis events.

    Args:
        prescriptions: Normalized prescription table with ``start_date`` and
            ``duration`` columns.
        patients: Patient table with demographics and observation-window dates.
        events: Disease-event table.
        patient_col: Patient identifier column.
        drug_col: Drug column in the prescription table.
        duration_col: Prescription duration column in days.
        prescription_start_col: Prescription start-date column.
        event_date_col: Disease-event date column.
        disease_col: Disease grouping column.
        fixed_followup_days: Optional fixed follow-up window for each exposure.
            If provided together with ``workflow``, it must match the workflow's
            configured window.
        workflow: One of ``actual``, ``30days``, or ``365days``. This mirrors
            the original script variants.
        gap_days: Maximum gap used while collapsing prescription episodes.
        keep_first_only: If ``True``, keep only the earliest exposure episode per
            patient and drug.
        known_drug_disease_pairs: Optional known indications/safety pairs to
            exclude from screening.
        ever_user_counts: Optional per-drug ever-user counts to merge into the
            result table.
        ever_user_drug_col: Drug column in ``ever_user_counts``.
        ever_user_count_col: Count column in ``ever_user_counts``.
        min_total_events: Minimum number of disease events required to report a
            drug-disease pair.
        age_groups: Age strata used during screening.

    Returns:
        A screening result table with incidence-rate ratios and event summaries.
    """
    fixed_followup_days = _resolve_screening_workflow(workflow, fixed_followup_days=fixed_followup_days)
    episodes = build_exposure_episodes_from_prescriptions(
        prescriptions,
        patient_col=patient_col,
        start_col=prescription_start_col,
        duration_col=duration_col,
        group_cols=(drug_col,),
        gap_days=gap_days,
        keep_first_only=keep_first_only,
    )
    exposure_windows = prepare_exposure_windows(
        episodes,
        patients,
        patient_col=patient_col,
        start_col=prescription_start_col,
        stop_col="stop_date",
        group_cols=(drug_col,),
        fixed_followup_days=fixed_followup_days,
        gap_days=0,
        keep_first_only=keep_first_only,
    )
    return screen_drugs(
        exposure_windows,
        events,
        patient_col=patient_col,
        drug_col=drug_col,
        disease_col=disease_col,
        event_date_col=event_date_col,
        start_col=prescription_start_col,
        stop_col="stop_date",
        unexposed_start_col="unexposed_start_date",
        exposure_length_col="exposure_length",
        age_col="prescription_age",
        age_groups=age_groups,
        min_total_events=min_total_events,
        known_drug_disease_pairs=known_drug_disease_pairs,
        ever_user_counts=ever_user_counts,
        ever_user_drug_col=ever_user_drug_col,
        ever_user_count_col=ever_user_count_col,
    )


def screen_substance_therapy(
    therapy: pd.DataFrame,
    patients: pd.DataFrame,
    events: pd.DataFrame,
    *,
    dosage_lookup: pd.DataFrame | None = None,
    min_max_lookup: pd.DataFrame | None = None,
    drugprepr_decisions: Sequence[str] | None = None,
    patient_col: str = "patid",
    drug_col: str = "drugsubstance",
    event_date_col: str = "eventdate",
    disease_col: str = "disease",
    disease_event_date_col: str = "disease_eventdate",
    fixed_followup_days: int | None = None,
    workflow: str = "actual",
    gap_days: int = 0,
    keep_first_only: bool = True,
    known_drug_disease_pairs: pd.DataFrame | None = None,
    ever_user_counts: pd.DataFrame | None = None,
    ever_user_drug_col: str = "drug",
    ever_user_count_col: str = "N.everuser",
    min_total_events: int = 100,
    age_groups: Sequence[tuple[str, float, float] | tuple[str]] = DEFAULT_AGE_GROUPS,
) -> pd.DataFrame:
    """Run the substance-level screen directly from raw therapy-like records.

    This is the main public entry point for the current drug-screening port. It
    accepts raw therapy rows, normalizes them into prescription episodes, and
    evaluates drug-disease associations with the self-controlled cohort design.

    Args:
        therapy: Therapy-like prescription table.
        patients: Patient table with dates of birth and observation-window dates.
        events: Disease-event table.
        dosage_lookup: Optional dosage lookup providing structured duration
            inputs such as ``dose_duration``.
        min_max_lookup: Optional plausible-range lookup for ``drugprepr``-style
            cleaning.
        drugprepr_decisions: Optional seven- or ten-step ``drugprepr`` decision
            vector.
        patient_col: Patient identifier column.
        drug_col: Drug grouping column in the therapy table.
        event_date_col: Prescription event-date column in the therapy table.
        disease_col: Disease grouping column in the event table.
        disease_event_date_col: Disease-event date column.
        fixed_followup_days: Optional fixed follow-up window for each exposure.
            If provided together with ``workflow``, it must match the workflow's
            configured window.
        workflow: One of ``actual``, ``30days``, or ``365days``. This mirrors
            the original script variants.
        gap_days: Maximum gap used while collapsing exposure episodes.
        keep_first_only: If ``True``, keep only the earliest exposure episode per
            patient and drug.
        known_drug_disease_pairs: Optional known indications/safety pairs to
            exclude from screening.
        ever_user_counts: Optional per-drug ever-user counts to merge into the
            result table.
        ever_user_drug_col: Drug column in ``ever_user_counts``.
        ever_user_count_col: Count column in ``ever_user_counts``.
        min_total_events: Minimum number of disease events required to report a
            drug-disease pair.
        age_groups: Age strata used during screening.

    Returns:
        A drug-disease screening result table.

    Examples:
        >>> import pandas as pd
        >>> import ehrapy as ep
        >>> therapy = pd.DataFrame(
        ...     {
        ...         "patid": [1, 2],
        ...         "eventdate": ["2020-01-10", "2020-01-10"],
        ...         "drugsubstance": ["drug_a", "drug_a"],
        ...         "duration": [5, 5],
        ...     }
        ... )
        >>> patients = pd.DataFrame(
        ...     {
        ...         "patid": [1, 2],
        ...         "dob": ["1970-01-01", "1970-01-01"],
        ...         "frd": ["2019-01-01", "2019-01-01"],
        ...         "tod": ["2021-01-01", "2021-01-01"],
        ...         "lcd": ["2021-01-01", "2021-01-01"],
        ...         "deathdate": ["2021-01-01", "2021-01-01"],
        ...     }
        ... )
        >>> events = pd.DataFrame(
        ...     {
        ...         "patid": [1, 2],
        ...         "disease": ["disease_x", "disease_x"],
        ...         "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06"]),
        ...     }
        ... )
        >>> result = ep.tl.screen_substance_therapy(therapy, patients, events, min_total_events=1)
        >>> "IRR" in result.columns
        True
    """
    prescriptions = prepare_prescriptions_from_therapy(
        therapy,
        patients=patients,
        dosage_lookup=dosage_lookup,
        min_max_lookup=min_max_lookup,
        drugprepr_decisions=drugprepr_decisions,
        patient_col=patient_col,
        event_date_col=event_date_col,
        drug_col=drug_col,
    )
    return screen_substance_cohort(
        prescriptions,
        patients,
        events.rename(columns={disease_event_date_col: "disease_eventdate"}) if disease_event_date_col != "disease_eventdate" else events,
        patient_col=patient_col,
        drug_col="drug",
        duration_col="duration",
        prescription_start_col="start_date",
        event_date_col="disease_eventdate",
        disease_col=disease_col,
        fixed_followup_days=fixed_followup_days,
        workflow=workflow,
        gap_days=gap_days,
        keep_first_only=keep_first_only,
        known_drug_disease_pairs=known_drug_disease_pairs,
        ever_user_counts=ever_user_counts,
        ever_user_drug_col=ever_user_drug_col,
        ever_user_count_col=ever_user_count_col,
        min_total_events=min_total_events,
        age_groups=age_groups,
    )


def screen_grouped_therapy(
    therapy: pd.DataFrame,
    patients: pd.DataFrame,
    events: pd.DataFrame,
    *,
    level: str,
    grouping: pd.DataFrame | None = None,
    grouping_col: str | None = None,
    level_label_col: str | None = None,
    dosage_lookup: pd.DataFrame | None = None,
    min_max_lookup: pd.DataFrame | None = None,
    drugprepr_decisions: Sequence[str] | None = None,
    patient_col: str = "patid",
    prodcode_col: str = "prodcode",
    drug_col: str = "drugsubstance",
    event_date_col: str = "eventdate",
    disease_col: str = "disease",
    disease_event_date_col: str = "disease_eventdate",
    fixed_followup_days: int | None = None,
    workflow: str = "actual",
    gap_days: int = 0,
    keep_first_only: bool = True,
    known_drug_disease_pairs: pd.DataFrame | None = None,
    ever_user_counts: pd.DataFrame | None = None,
    ever_user_drug_col: str = "drug",
    ever_user_count_col: str = "N.everuser",
    min_total_events: int = 100,
    age_groups: Sequence[tuple[str, float, float] | tuple[str]] = DEFAULT_AGE_GROUPS,
) -> pd.DataFrame:
    """Run the screening workflow at chapter, section, paragraph, or substance level.

    The grouped workflow prepares prescriptions at the original therapy row level,
    maps those prepared rows onto a requested grouping hierarchy, and reuses the
    substance-level cohort screening engine on the grouped labels.

    Args:
        therapy: Therapy-like prescription table.
        patients: Patient table with dates of birth and observation-window dates.
        events: Disease-event table.
        level: Grouping level. One of ``substance``, ``paragraph``, ``section``,
            or ``chapter``.
        grouping: Optional product-code mapping table for chapter, section, and
            paragraph workflows.
        grouping_col: Optional mapping column to use instead of the default for
            the requested ``level``.
        level_label_col: Optional extra output column to populate with the
            grouped label. This is added in addition to the canonical ``drug``
            result column, for example ``chapter`` or ``section``.
        dosage_lookup: Optional dosage lookup providing structured duration
            inputs such as ``dose_duration``.
        min_max_lookup: Optional plausible-range lookup for ``drugprepr``-style
            cleaning.
        drugprepr_decisions: Optional seven- or ten-step ``drugprepr`` decision
            vector.
        patient_col: Patient identifier column.
        prodcode_col: Product-code column used for grouping mappings.
        drug_col: Drug grouping column in the therapy table.
        event_date_col: Prescription event-date column in the therapy table.
        disease_col: Disease grouping column in the event table.
        disease_event_date_col: Disease-event date column.
        fixed_followup_days: Optional fixed follow-up window for each exposure.
            If provided together with ``workflow``, it must match the workflow's
            configured window.
        workflow: One of ``actual``, ``30days``, or ``365days``. This mirrors
            the original script variants.
        gap_days: Maximum gap used while collapsing exposure episodes.
        keep_first_only: If ``True``, keep only the earliest exposure episode per
            patient and grouped drug label.
        known_drug_disease_pairs: Optional known indications/safety pairs to
            exclude from screening. For grouped workflows this may be provided at
            prodcode level and will be aggregated through ``grouping``.
        ever_user_counts: Optional per-group ever-user counts to merge into the
            result table. If omitted, counts are computed from the grouped
            prescription table.
        ever_user_drug_col: Group-label column in ``ever_user_counts``.
        ever_user_count_col: Count column in ``ever_user_counts``.
        min_total_events: Minimum number of disease events required to report a
            drug-disease pair.
        age_groups: Age strata used during screening.

    Returns:
        A grouped drug-disease screening result table.
    """
    validated_level = validate_grouping_level(level)
    prescriptions = prepare_prescriptions_from_therapy(
        therapy,
        patients=patients,
        dosage_lookup=dosage_lookup,
        min_max_lookup=min_max_lookup,
        drugprepr_decisions=drugprepr_decisions,
        patient_col=patient_col,
        event_date_col=event_date_col,
        drug_col=drug_col,
    )
    grouped_prescriptions = assign_grouping_labels(
        prescriptions,
        level=validated_level,
        mapping=grouping,
        prodcode_col=prodcode_col,
        grouping_col=grouping_col,
        source_drug_col="drug",
        output_col="drug",
    )
    if grouped_prescriptions.empty:
        return pd.DataFrame()

    grouped_known_pairs = group_known_pairs(
        known_drug_disease_pairs,
        level=validated_level,
        mapping=grouping,
        prodcode_col=prodcode_col,
        disease_col=disease_col,
        grouping_col=grouping_col,
        output_col="drug",
    )

    if ever_user_counts is None:
        ever_user_counts = count_ever_users_by_group(
            grouped_prescriptions,
            patient_col=patient_col,
            drug_col="drug",
            output_count_col=ever_user_count_col,
        )
        ever_user_drug_col = "drug"

    normalized_events = (
        events.rename(columns={disease_event_date_col: "disease_eventdate"})
        if disease_event_date_col != "disease_eventdate"
        else events
    )
    result = screen_substance_cohort(
        grouped_prescriptions,
        patients,
        normalized_events,
        patient_col=patient_col,
        drug_col="drug",
        duration_col="duration",
        prescription_start_col="start_date",
        event_date_col="disease_eventdate",
        disease_col=disease_col,
        fixed_followup_days=fixed_followup_days,
        workflow=workflow,
        gap_days=gap_days,
        keep_first_only=keep_first_only,
        known_drug_disease_pairs=grouped_known_pairs,
        ever_user_counts=ever_user_counts,
        ever_user_drug_col=ever_user_drug_col,
        ever_user_count_col=ever_user_count_col,
        min_total_events=min_total_events,
        age_groups=age_groups,
    )
    if level_label_col is not None and not result.empty and level_label_col != "drug":
        result[level_label_col] = result["drug"]
    return result


def screen_drugs(
    exposure_windows: pd.DataFrame,
    events: pd.DataFrame,
    *,
    patient_col: str = "patid",
    drug_col: str = "drug",
    disease_col: str = "disease",
    event_date_col: str = "disease_eventdate",
    start_col: str = "start_date",
    stop_col: str = "stop_date",
    unexposed_start_col: str = "unexposed_start_date",
    exposure_length_col: str = "exposure_length",
    age_col: str = "prescription_age",
    age_groups: Sequence[tuple[str, float, float] | tuple[str]] = DEFAULT_AGE_GROUPS,
    min_total_events: int = 100,
    known_drug_disease_pairs: pd.DataFrame | None = None,
    ever_user_counts: pd.DataFrame | None = None,
    ever_user_drug_col: str = "drug",
    ever_user_count_col: str = "N.everuser",
) -> pd.DataFrame:
    """Run the prepared-data self-controlled cohort screen.

    This is the first substantive port of the substance-level R workflow. It
    assumes the exposure intervals have already been prepared into balanced
    exposed and unexposed windows via :func:`prepare_exposure_windows`.

    Args:
        exposure_windows: Prepared exposure/unexposed comparison windows.
        events: Disease-event table.
        patient_col: Patient identifier column.
        drug_col: Drug column in the exposure table.
        disease_col: Disease grouping column in the event table.
        event_date_col: Disease-event date column.
        start_col: Exposed-window start column.
        stop_col: Exposed-window stop column.
        unexposed_start_col: Matched unexposed-window start column.
        exposure_length_col: Exposure window length column in days.
        age_col: Age-at-prescription column.
        age_groups: Age strata used during screening.
        min_total_events: Minimum number of disease events required to report a
            drug-disease pair.
        known_drug_disease_pairs: Optional known indications/safety pairs to
            exclude from screening.
        ever_user_counts: Optional per-drug ever-user counts to merge into the
            result table.
        ever_user_drug_col: Drug column in ``ever_user_counts``.
        ever_user_count_col: Count column in ``ever_user_counts``.

    Returns:
        A screening result table with incidence-rate ratios and event summaries.
    """
    if exposure_windows.empty or events.empty:
        return pd.DataFrame()

    required_exposure_columns = {
        patient_col,
        drug_col,
        start_col,
        stop_col,
        unexposed_start_col,
        exposure_length_col,
        age_col,
    }
    missing_exposure = required_exposure_columns.difference(exposure_windows.columns)
    if missing_exposure:
        missing_str = ", ".join(sorted(missing_exposure))
        raise KeyError(f"Missing required exposure columns: {missing_str}")

    required_event_columns = {patient_col, disease_col, event_date_col}
    missing_events = required_event_columns.difference(events.columns)
    if missing_events:
        missing_str = ", ".join(sorted(missing_events))
        raise KeyError(f"Missing required event columns: {missing_str}")

    exposure_frame = exposure_windows.copy()
    event_frame = events.copy()
    exposure_frame[start_col] = pd.to_datetime(exposure_frame[start_col])
    exposure_frame[stop_col] = pd.to_datetime(exposure_frame[stop_col])
    exposure_frame[unexposed_start_col] = pd.to_datetime(exposure_frame[unexposed_start_col])
    event_frame[event_date_col] = pd.to_datetime(event_frame[event_date_col])

    excluded_pairs = _make_known_pair_set(known_drug_disease_pairs, drug_col=drug_col, disease_col=disease_col)
    results: list[dict] = []

    for drug, drug_exposures in exposure_frame.groupby(drug_col, sort=False):
        diseases = event_frame[disease_col].dropna().unique()
        for age_group in age_groups:
            age_label, age_exposures = _filter_age_group(drug_exposures, age_group, age_col=age_col)
            if age_exposures.empty:
                continue

            person_time = float(age_exposures[exposure_length_col].sum())
            mean_exposure = float(age_exposures[exposure_length_col].mean())
            sd_exposure = float(age_exposures[exposure_length_col].std(ddof=1))

            for disease in diseases:
                if (drug, disease) in excluded_pairs:
                    continue

                disease_events = event_frame[event_frame[disease_col] == disease]
                disease_drug = age_exposures.merge(disease_events, how="left", on=patient_col)
                total_events = int(disease_drug[event_date_col].notna().sum())
                if total_events < min_total_events:
                    continue

                unexposed_mask = disease_drug[event_date_col].between(
                    disease_drug[unexposed_start_col], disease_drug[start_col], inclusive="both"
                )
                exposed_mask = disease_drug[event_date_col].gt(disease_drug[start_col]) & disease_drug[event_date_col].le(
                    disease_drug[stop_col]
                )

                irr = rate_ratio_test(
                    x=[int(exposed_mask.sum()), int(unexposed_mask.sum())],
                    n=[person_time, person_time],
                    alternative="two.sided",
                )

                result = {
                    drug_col: drug,
                    disease_col: disease,
                    "age.group": age_label,
                    "IRR": irr.rate_ratio,
                    "IRR.lower.95": irr.conf_int[0],
                    "IRR.higher.95": irr.conf_int[1],
                    "p.value": irr.p_value,
                    "N.everuser": np.nan,
                    "N.exposed": int(len(age_exposures)),
                    "exposed.mean": mean_exposure,
                    "exposed.sd": sd_exposure,
                    "cond.median.length.before.exposure": _timedelta_summary(
                        disease_drug.loc[unexposed_mask, start_col] - disease_drug.loc[unexposed_mask, event_date_col],
                        reducer="median",
                    ),
                    "cond.mean.length.before.exposure": _timedelta_summary(
                        disease_drug.loc[unexposed_mask, start_col] - disease_drug.loc[unexposed_mask, event_date_col],
                        reducer="mean",
                    ),
                    "cond.median.length.after.exposure": _timedelta_summary(
                        disease_drug.loc[exposed_mask, event_date_col] - disease_drug.loc[exposed_mask, start_col],
                        reducer="median",
                    ),
                    "cond.mean.length.after.exposure": _timedelta_summary(
                        disease_drug.loc[exposed_mask, event_date_col] - disease_drug.loc[exposed_mask, start_col],
                        reducer="mean",
                    ),
                    "N.disease.C.during.exposed": int(exposed_mask.sum()),
                    "N.disease.B.during.unexposed": int(unexposed_mask.sum()),
                    "sum.exposed.person.time": person_time,
                    "N.disease.ABCD.all": total_events,
                    "N.disease.BC.during.followup": int(
                        disease_drug[event_date_col]
                        .between(disease_drug[unexposed_start_col], disease_drug[stop_col], inclusive="both")
                        .sum()
                    ),
                    "N.disease.AD.during.followup": int(
                        (
                            disease_drug[event_date_col].lt(disease_drug[unexposed_start_col])
                            | disease_drug[event_date_col].gt(disease_drug[stop_col])
                        ).sum()
                    ),
                    "N.disease.A.before.exposed": int(disease_drug[event_date_col].lt(disease_drug[unexposed_start_col]).sum()),
                    "N.disease.D.after.exposed": int(disease_drug[event_date_col].gt(disease_drug[stop_col]).sum()),
                }
                results.append(result)

    result_frame = pd.DataFrame(results)
    if result_frame.empty:
        return result_frame

    if ever_user_counts is not None:
        if ever_user_drug_col not in ever_user_counts.columns or ever_user_count_col not in ever_user_counts.columns:
            raise KeyError("ever_user_counts must contain the configured drug and count columns")
        ever_user_map = ever_user_counts.set_index(ever_user_drug_col)[ever_user_count_col]
        result_frame["N.everuser"] = result_frame[drug_col].map(ever_user_map)

    return result_frame.reset_index(drop=True)


def _filter_age_group(
    exposures: pd.DataFrame,
    age_group: tuple[str, float, float] | tuple[str],
    *,
    age_col: str,
) -> tuple[str, pd.DataFrame]:
    label = age_group[0]
    if label == "all":
        return label, exposures

    _, lower, upper = age_group
    filtered = exposures[(exposures[age_col] > lower) & (exposures[age_col] < upper)]
    return label, filtered


def _timedelta_summary(values: Iterable[pd.Timedelta], *, reducer: str) -> float:
    series = pd.Series(values)
    if series.empty:
        return float("nan")
    days = series.dt.total_seconds() / 86400
    if reducer == "median":
        return float(days.median())
    return float(days.mean())


def _make_known_pair_set(
    known_drug_disease_pairs: pd.DataFrame | None,
    *,
    drug_col: str,
    disease_col: str,
) -> set[tuple[object, object]]:
    if known_drug_disease_pairs is None:
        return set()
    if drug_col not in known_drug_disease_pairs.columns or disease_col not in known_drug_disease_pairs.columns:
        raise KeyError("known_drug_disease_pairs must contain the configured drug and disease columns")
    return {
        (row[drug_col], row[disease_col])
        for _, row in known_drug_disease_pairs.loc[:, [drug_col, disease_col]].dropna().iterrows()
    }


def _resolve_screening_workflow(
    workflow: str,
    *,
    fixed_followup_days: int | None,
) -> int | None:
    normalized_workflow = workflow.lower()
    if normalized_workflow not in SCREENING_WORKFLOW_FOLLOWUP_DAYS:
        valid = ", ".join(sorted(SCREENING_WORKFLOW_FOLLOWUP_DAYS))
        raise ValueError(f"Unsupported screening workflow: {workflow}. Expected one of: {valid}")

    workflow_followup_days = SCREENING_WORKFLOW_FOLLOWUP_DAYS[normalized_workflow]
    if fixed_followup_days is None:
        return workflow_followup_days
    if workflow_followup_days is None or int(fixed_followup_days) == workflow_followup_days:
        return int(fixed_followup_days)
    raise ValueError(
        f"fixed_followup_days={fixed_followup_days} does not match workflow '{normalized_workflow}' "
        f"which expects {workflow_followup_days}"
    )
