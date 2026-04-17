from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "half": 0.5,
    "quarter": 0.25,
}
_FREQUENCY_WORDS = {
    "once": 1.0,
    "twice": 2.0,
    "thrice": 3.0,
}
_DOSE_UNIT_PATTERN = (
    r"tablets?|tabs?|capsules?|caps?|puffs?|drops?|patch(?:es)?|"
    r"suppositor(?:y|ies)|sprays?|ml|units?|spoonfuls?|teaspoonfuls?|"
    r"applications?|applicators?|ampoules?|lozenges?|sachets?|packets?|pills?"
)
_OPTIONAL_PATTERNS = (
    ("as required", r"\bas required\b"),
    ("when required", r"\bwhen required\b"),
    ("as needed", r"\bas needed\b"),
    ("prn", r"\bprn\b"),
    ("up to", r"\bup to\b"),
    ("as directed", r"\bas directed\b"),
)
_FREQUENCY_ABBREVIATIONS = {
    "od": 1.0,
    "qd": 1.0,
    "daily": 1.0,
    "nocte": 1.0,
    "qds": 4.0,
    "qid": 4.0,
    "tds": 3.0,
    "tid": 3.0,
    "bd": 2.0,
    "bid": 2.0,
}


def compute_ndd_from_text(
    data: pd.DataFrame,
    *,
    text_col: str = "text",
    dose_fn: str | Callable[[Sequence[float]], float] = "mean",
    freq_fn: str | Callable[[Sequence[float]], float] = "mean",
    interval_fn: str | Callable[[Sequence[float]], float] = "mean",
) -> pd.DataFrame:
    """Compute approximate numerical daily dose from free-text dosage strings.

    This ports the role of ``drugprepr::compute_ndd()`` into Python for common
    CPRD-style prescription instructions. The parser is intentionally bounded:
    it handles common patterns such as dose ranges, daily frequencies, hourly
    frequencies, and every-other-day intervals, but it is not a full
    free-text NLP replacement for ``doseminer``.

    Args:
        data: Table containing free-text dosage instructions.
        text_col: Column containing the raw instruction text.
        dose_fn: Summary rule for dose ranges. Supports ``"min"``, ``"max"``,
            ``"mean"``, or a callable.
        freq_fn: Summary rule for frequency ranges.
        interval_fn: Summary rule for interval ranges.

    Returns:
        The input table with parsed dosage columns appended: ``dose``, ``freq``,
        ``itvl``, ``ndd``, and ``optional``.
    """
    if text_col not in data.columns:
        raise KeyError(f"{text_col} is required to compute ndd from text")

    if data.empty:
        result = data.copy()
        for column in ["dose", "freq", "itvl", "ndd", "optional"]:
            result[column] = pd.Series(dtype=float if column != "optional" else object)
        return result

    dose_reducer = _resolve_summary_function(dose_fn)
    freq_reducer = _resolve_summary_function(freq_fn)
    interval_reducer = _resolve_summary_function(interval_fn)

    unique_text = pd.DataFrame({text_col: pd.Series(data[text_col].dropna().unique(), dtype=object)})
    if unique_text.empty:
        result = data.copy()
        result["dose"] = np.nan
        result["freq"] = np.nan
        result["itvl"] = np.nan
        result["ndd"] = np.nan
        result["optional"] = pd.NA
        return result

    parsed_rows = unique_text[text_col].apply(
        lambda value: _extract_text_dosage_components(
            value,
            dose_reducer=dose_reducer,
            freq_reducer=freq_reducer,
            interval_reducer=interval_reducer,
        )
    )
    parsed = pd.concat([unique_text, parsed_rows.apply(pd.Series)], axis=1)
    return data.merge(parsed, how="left", on=text_col)


def prepare_prescriptions_from_therapy(
    therapy: pd.DataFrame,
    *,
    patients: pd.DataFrame | None = None,
    dosage_lookup: pd.DataFrame | None = None,
    min_max_lookup: pd.DataFrame | None = None,
    drugprepr_decisions: Sequence[str] | None = None,
    patient_col: str = "patid",
    practice_col: str = "pracid",
    prodcode_col: str = "prodcode",
    event_date_col: str = "eventdate",
    drug_col: str = "drugsubstance",
    dosage_id_col: str = "dosageid",
    dosage_text_col: str = "text",
    qty_col: str = "qty",
    ndd_col: str = "ndd",
    duration_col: str = "duration",
    numdays_col: str = "numdays",
    dose_duration_col: str = "dose_duration",
    registration_start_col: str = "frd",
    observation_end_cols: Sequence[str] = ("tod", "lcd", "deathdate"),
    compute_ndd: bool = True,
    dose_fn: str | Callable[[Sequence[float]], float] = "mean",
    freq_fn: str | Callable[[Sequence[float]], float] = "mean",
    interval_fn: str | Callable[[Sequence[float]], float] = "mean",
) -> pd.DataFrame:
    """Normalize raw therapy-like records into prescription rows with durations.

    This helper supports two levels of preparation:

    1. a lightweight path where durations are inferred from one of
       `duration`, `numdays`, `dose_duration`, or `qty / ndd`;
    2. a fuller `drugprepr`-style path when a 10-element decision vector is
       provided, covering same-day clashes, overlap shifting, and gap closing.

    When free-text dosage instructions are available, this helper can also
    compute approximate ``ndd`` values for common instruction patterns before
    inferring durations.

    Args:
        therapy: Therapy-like table containing one row per prescription event.
        patients: Optional patient table used to restrict prescriptions to the
            observation window.
        dosage_lookup: Optional lookup table providing structured dosage fields
            such as ``dose_duration``.
        min_max_lookup: Optional plausible-range lookup for ``drugprepr``-style
            cleaning.
        drugprepr_decisions: Optional decision vector. Seven decisions apply the
            subset used in the original paper workflow; ten decisions apply the
            fuller structured ``drugprepr`` path.
        patient_col: Patient identifier column.
        practice_col: Practice identifier column used for grouped imputations.
        prodcode_col: Product code column used by ``drugprepr``.
        event_date_col: Prescription start-date column in the raw therapy table.
        drug_col: Drug grouping column, typically a substance-level name.
        dosage_id_col: Dosage lookup key.
        dosage_text_col: Free-text dosage instruction column.
        qty_col: Quantity column.
        ndd_col: Numeric daily dose column.
        duration_col: Precomputed duration column.
        numdays_col: CPRD-style number-of-days column.
        dose_duration_col: Lookup-derived duration column.
        registration_start_col: Patient registration-start column.
        observation_end_cols: Patient observation-end columns. The earliest date
            is used as the effective observation end.
        compute_ndd: Whether to derive ``ndd`` from free-text dosage instructions
            when possible.
        dose_fn: Summary rule for dose ranges in free-text parsing.
        freq_fn: Summary rule for frequency ranges in free-text parsing.
        interval_fn: Summary rule for interval ranges in free-text parsing.

    Returns:
        A normalized prescription table with ``start_date``, ``duration``, and
        ``drug`` columns ready for cohort construction.

    Examples:
        >>> import pandas as pd
        >>> import ehrapy as ep
        >>> therapy = pd.DataFrame(
        ...     {
        ...         "patid": [1],
        ...         "eventdate": ["2020-01-10"],
        ...         "drugsubstance": ["drug_a"],
        ...         "duration": [7],
        ...     }
        ... )
        >>> patients = pd.DataFrame(
        ...     {
        ...         "patid": [1],
        ...         "frd": ["2019-01-01"],
        ...         "tod": ["2021-01-01"],
        ...         "lcd": ["2021-01-01"],
        ...         "deathdate": ["2021-01-01"],
        ...     }
        ... )
        >>> prescriptions = ep.tl.prepare_prescriptions_from_therapy(therapy, patients=patients)
        >>> prescriptions.loc[0, "duration"]
        7.0
    """
    if therapy.empty:
        return therapy.copy()

    required_columns = {patient_col, event_date_col, drug_col}
    missing = required_columns.difference(therapy.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required therapy columns: {missing_str}")

    frame = therapy.copy()

    if min_max_lookup is not None and drugprepr_decisions is not None:
        if len(drugprepr_decisions) >= 10:
            frame = prepare_prescriptions_with_drugprepr(
                frame,
                dosage_lookup=dosage_lookup,
                min_max_lookup=min_max_lookup,
                decisions=drugprepr_decisions,
                patient_col=patient_col,
                practice_col=practice_col,
                prodcode_col=prodcode_col,
                event_date_col=event_date_col,
                dosage_id_col=dosage_id_col,
                dosage_text_col=dosage_text_col,
                qty_col=qty_col,
                ndd_col=ndd_col,
                duration_col=duration_col,
                numdays_col=numdays_col,
                dose_duration_col=dose_duration_col,
                compute_ndd=compute_ndd,
                dose_fn=dose_fn,
                freq_fn=freq_fn,
                interval_fn=interval_fn,
            )
        else:
            if dosage_lookup is not None:
                if dosage_id_col not in frame.columns or dosage_id_col not in dosage_lookup.columns:
                    raise KeyError("dosage_lookup merge requires dosageid in both therapy and lookup tables")
                frame = frame.merge(dosage_lookup, how="left", on=dosage_id_col, suffixes=("", "_dosage"))

            if compute_ndd:
                frame = _fill_ndd_from_text(
                    frame,
                    text_col=dosage_text_col,
                    ndd_col=ndd_col,
                    dose_fn=dose_fn,
                    freq_fn=freq_fn,
                    interval_fn=interval_fn,
                )

            frame = apply_drugprepr_decisions(
                frame,
                min_max_lookup=min_max_lookup,
                decisions=drugprepr_decisions,
                patient_col=patient_col,
                practice_col=practice_col,
                prodcode_col=prodcode_col,
                qty_col=qty_col,
                ndd_col=ndd_col,
                numdays_col=numdays_col,
                dose_duration_col=dose_duration_col,
                duration_col=duration_col,
            )
    elif dosage_lookup is not None:
        if dosage_id_col not in frame.columns or dosage_id_col not in dosage_lookup.columns:
            raise KeyError("dosage_lookup merge requires dosageid in both therapy and lookup tables")
        frame = frame.merge(dosage_lookup, how="left", on=dosage_id_col, suffixes=("", "_dosage"))

    if compute_ndd:
        frame = _fill_ndd_from_text(
            frame,
            text_col=dosage_text_col,
            ndd_col=ndd_col,
            dose_fn=dose_fn,
            freq_fn=freq_fn,
            interval_fn=interval_fn,
        )

    frame[event_date_col] = pd.to_datetime(frame[event_date_col], errors="coerce")
    frame = frame.dropna(subset=[event_date_col, drug_col])

    if patients is not None:
        required_patient_columns = {patient_col, registration_start_col, *observation_end_cols}
        missing_patient = required_patient_columns.difference(patients.columns)
        if missing_patient:
            missing_str = ", ".join(sorted(missing_patient))
            raise KeyError(f"Missing required patient columns: {missing_str}")

        patient_frame = patients.loc[:, list(required_patient_columns)].copy()
        for column in [registration_start_col, *observation_end_cols]:
            patient_frame[column] = pd.to_datetime(patient_frame[column], errors="coerce")

        frame = frame.merge(patient_frame, how="left", on=patient_col)
        observation_end = frame.loc[:, list(observation_end_cols)].min(axis=1)
        valid_mask = frame[registration_start_col].lt(frame[event_date_col]) & observation_end.gt(frame[event_date_col])
        frame = frame.loc[valid_mask].copy()

    frame["duration"] = infer_prescription_duration(
        frame,
        duration_col=duration_col,
        numdays_col=numdays_col,
        dose_duration_col=dose_duration_col,
        qty_col=qty_col,
        ndd_col=ndd_col,
    )
    frame = frame.dropna(subset=["duration"]).copy()
    frame = frame[frame["duration"] > 0].copy()
    if "start_date" in frame.columns:
        frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
    else:
        frame["start_date"] = frame[event_date_col]
    frame["drug"] = frame[drug_col]

    keep_columns = [column for column in frame.columns if column not in {registration_start_col, *observation_end_cols}]
    return frame.loc[:, keep_columns].reset_index(drop=True)


def prepare_prescriptions_with_drugprepr(
    therapy: pd.DataFrame,
    *,
    min_max_lookup: pd.DataFrame,
    decisions: Sequence[str],
    dosage_lookup: pd.DataFrame | None = None,
    patient_col: str = "patid",
    practice_col: str = "pracid",
    prodcode_col: str = "prodcode",
    event_date_col: str = "eventdate",
    dosage_id_col: str = "dosageid",
    dosage_text_col: str = "text",
    qty_col: str = "qty",
    ndd_col: str = "ndd",
    duration_col: str = "duration",
    numdays_col: str = "numdays",
    dose_duration_col: str = "dose_duration",
    start_col: str = "start_date",
    stop_col: str = "stop_date",
    compute_ndd: bool = True,
    dose_fn: str | Callable[[Sequence[float]], float] = "mean",
    freq_fn: str | Callable[[Sequence[float]], float] = "mean",
    interval_fn: str | Callable[[Sequence[float]], float] = "mean",
) -> pd.DataFrame:
    """Run the structured part of the `drugprepr::drug_prep()` algorithm.

    This ports decisions 1-10 once the therapy table already contains structured
    quantity and daily-dose inputs. When dosage text is available, it can
    compute approximate ``ndd`` values before running the structured decisions.

    Args:
        therapy: Therapy-like table with structured quantity and dose inputs.
        min_max_lookup: Plausible-range lookup keyed by product code.
        decisions: Ten-element structured ``drugprepr`` decision vector.
        dosage_lookup: Optional dosage lookup merged before processing.
        patient_col: Patient identifier column.
        practice_col: Practice identifier column.
        prodcode_col: Product code column.
        event_date_col: Raw therapy start-date column.
        dosage_id_col: Dosage lookup key.
        dosage_text_col: Free-text dosage instruction column.
        qty_col: Quantity column.
        ndd_col: Numeric daily dose column.
        duration_col: Duration column to create/update.
        numdays_col: Raw ``numdays`` column.
        dose_duration_col: Raw or lookup ``dose_duration`` column.
        start_col: Output start-date column.
        stop_col: Output stop-date column.
        compute_ndd: Whether to derive ``ndd`` from free-text dosage instructions
            when possible.
        dose_fn: Summary rule for dose ranges in free-text parsing.
        freq_fn: Summary rule for frequency ranges in free-text parsing.
        interval_fn: Summary rule for interval ranges in free-text parsing.

    Returns:
        A prescription table with structured durations and stop dates after the
        ten-step preparation sequence.
    """
    if len(decisions) < 10:
        raise ValueError("Ten drugprepr decisions are required for the full preparation path")

    frame = therapy.copy()
    if dosage_lookup is not None:
        if dosage_id_col not in frame.columns or dosage_id_col not in dosage_lookup.columns:
            raise KeyError("dosage_lookup merge requires dosageid in both therapy and lookup tables")
        frame = frame.merge(dosage_lookup, how="left", on=dosage_id_col, suffixes=("", "_dosage"))

    if compute_ndd:
        frame = _fill_ndd_from_text(
            frame,
            text_col=dosage_text_col,
            ndd_col=ndd_col,
            dose_fn=dose_fn,
            freq_fn=freq_fn,
            interval_fn=interval_fn,
        )

    if start_col not in frame.columns:
        if event_date_col not in frame.columns:
            raise KeyError(f"Either {start_col} or {event_date_col} is required for drugprepr preparation")
        frame[start_col] = pd.to_datetime(frame[event_date_col], errors="coerce")
    else:
        frame[start_col] = pd.to_datetime(frame[start_col], errors="coerce")

    frame = frame.dropna(subset=[start_col, prodcode_col]).copy()
    frame = apply_drugprepr_decisions(
        frame,
        min_max_lookup=min_max_lookup,
        decisions=decisions[:7],
        patient_col=patient_col,
        practice_col=practice_col,
        prodcode_col=prodcode_col,
        qty_col=qty_col,
        ndd_col=ndd_col,
        numdays_col=numdays_col,
        dose_duration_col=dose_duration_col,
        duration_col=duration_col,
    )
    frame = frame.dropna(subset=[duration_col]).copy()
    frame = frame[pd.to_numeric(frame[duration_col], errors="coerce") > 0].copy()
    frame[duration_col] = pd.to_numeric(frame[duration_col], errors="coerce")

    frame = disambiguate_same_day_prescriptions(
        frame,
        decision=decisions[7],
        patient_col=patient_col,
        prodcode_col=prodcode_col,
        start_col=start_col,
        duration_col=duration_col,
    )
    frame[stop_col] = frame[start_col] + pd.to_timedelta(frame[duration_col], unit="D")
    frame = resolve_overlapping_prescriptions(
        frame,
        decision=decisions[8],
        patient_col=patient_col,
        prodcode_col=prodcode_col,
        start_col=start_col,
        stop_col=stop_col,
    )
    frame = close_small_gaps_in_prescriptions(
        frame,
        decision=decisions[9],
        patient_col=patient_col,
        prodcode_col=prodcode_col,
        start_col=start_col,
        stop_col=stop_col,
    )
    frame[duration_col] = (frame[stop_col] - frame[start_col]).dt.days.astype(float)
    return frame.reset_index(drop=True)


def apply_drugprepr_decisions(
    therapy: pd.DataFrame,
    *,
    min_max_lookup: pd.DataFrame,
    decisions: Sequence[str],
    patient_col: str = "patid",
    practice_col: str = "pracid",
    prodcode_col: str = "prodcode",
    qty_col: str = "qty",
    ndd_col: str = "ndd",
    numdays_col: str = "numdays",
    dose_duration_col: str = "dose_duration",
    duration_col: str = "duration",
) -> pd.DataFrame:
    """Apply the subset of `drugprepr` decisions used by the original workflow.

    Supported decisions:
    - 1: implausible qty
    - 2: missing qty
    - 3: implausible ndd
    - 4: missing ndd
    - 5: implausible duration
    - 6: duration source selection
    - 7: missing duration
    """
    if len(decisions) < 7:
        raise ValueError("At least seven decisions are required for the drugprepr subset")
    if prodcode_col not in therapy.columns or prodcode_col not in min_max_lookup.columns:
        raise KeyError("Both therapy and min_max_lookup must contain prodcode")

    min_max_columns = {prodcode_col, "min_qty", "max_qty", "min_ndd", "max_ndd"}
    missing = min_max_columns.difference(min_max_lookup.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required min/max columns: {missing_str}")

    frame = therapy.merge(
        min_max_lookup.loc[:, list(min_max_columns)],
        how="left",
        on=prodcode_col,
        suffixes=("", "_limits"),
    ).copy()

    frame[qty_col] = _apply_implausible_imputation(
        frame,
        value_col=qty_col,
        lower_col="min_qty",
        upper_col="max_qty",
        decision=decisions[0],
        patient_col=patient_col,
        practice_col=practice_col,
        prodcode_col=prodcode_col,
    )
    frame[qty_col] = _apply_missing_imputation(
        frame,
        value_col=qty_col,
        decision=decisions[1],
        patient_col=patient_col,
        practice_col=practice_col,
        prodcode_col=prodcode_col,
    )
    frame[ndd_col] = _apply_implausible_imputation(
        frame,
        value_col=ndd_col,
        lower_col="min_ndd",
        upper_col="max_ndd",
        decision=decisions[2],
        patient_col=patient_col,
        practice_col=practice_col,
        prodcode_col=prodcode_col,
    )
    frame[ndd_col] = _apply_missing_imputation(
        frame,
        value_col=ndd_col,
        decision=decisions[3],
        patient_col=patient_col,
        practice_col=practice_col,
        prodcode_col=prodcode_col,
    )

    frame[duration_col] = _calculate_duration_from_decision(
        frame,
        decision=decisions[5],
        qty_col=qty_col,
        ndd_col=ndd_col,
        numdays_col=numdays_col,
        dose_duration_col=dose_duration_col,
    )
    frame[duration_col] = _clean_duration(frame[duration_col], decisions[4])
    frame[duration_col] = _apply_missing_duration_imputation(
        frame,
        duration_col=duration_col,
        decision=decisions[6],
        patient_col=patient_col,
        prodcode_col=prodcode_col,
    )

    return frame.drop(columns=["min_qty", "max_qty", "min_ndd", "max_ndd"])


def infer_prescription_duration(
    therapy: pd.DataFrame,
    *,
    duration_col: str = "duration",
    numdays_col: str = "numdays",
    dose_duration_col: str = "dose_duration",
    qty_col: str = "qty",
    ndd_col: str = "ndd",
) -> pd.Series:
    """Infer prescription durations from the best available fields."""
    result = pd.Series(np.nan, index=therapy.index, dtype=float)

    for column in [duration_col, numdays_col, dose_duration_col]:
        if column in therapy.columns:
            values = pd.to_numeric(therapy[column], errors="coerce")
            result = result.fillna(values)

    if qty_col in therapy.columns and ndd_col in therapy.columns:
        qty = pd.to_numeric(therapy[qty_col], errors="coerce")
        ndd = pd.to_numeric(therapy[ndd_col], errors="coerce")
        ratio = qty / ndd.replace(0, np.nan)
        result = result.fillna(ratio)

    return result


def _fill_ndd_from_text(
    frame: pd.DataFrame,
    *,
    text_col: str,
    ndd_col: str,
    dose_fn: str | Callable[[Sequence[float]], float],
    freq_fn: str | Callable[[Sequence[float]], float],
    interval_fn: str | Callable[[Sequence[float]], float],
) -> pd.DataFrame:
    if text_col not in frame.columns:
        return frame

    text_values = frame[text_col]
    if text_values.isna().all():
        return frame

    parsed = compute_ndd_from_text(
        frame.loc[:, [text_col]].copy(),
        text_col=text_col,
        dose_fn=dose_fn,
        freq_fn=freq_fn,
        interval_fn=interval_fn,
    )
    result = frame.copy()

    for source_col in ["dose", "freq", "itvl", "optional"]:
        if source_col in result.columns:
            result[source_col] = result[source_col].where(result[source_col].notna(), parsed[source_col])
        else:
            result[source_col] = parsed[source_col]

    computed_ndd = pd.to_numeric(parsed["ndd"], errors="coerce")
    if ndd_col in result.columns:
        existing = pd.to_numeric(result[ndd_col], errors="coerce")
        result[ndd_col] = existing.fillna(computed_ndd)
    else:
        result[ndd_col] = computed_ndd

    return result


def _resolve_summary_function(summary: str | Callable[[Sequence[float]], float]) -> Callable[[Sequence[float]], float]:
    if callable(summary):
        return summary

    if summary == "min":
        return min
    if summary == "max":
        return max
    if summary == "mean":
        return lambda values: float(sum(values) / len(values))
    raise ValueError("Summary function must be one of {'min', 'max', 'mean'} or a callable")


def _extract_text_dosage_components(
    text: object,
    *,
    dose_reducer: Callable[[Sequence[float]], float],
    freq_reducer: Callable[[Sequence[float]], float],
    interval_reducer: Callable[[Sequence[float]], float],
) -> dict[str, object]:
    normalized = _normalize_prescription_text(text)
    if not normalized:
        return {"dose": np.nan, "freq": np.nan, "itvl": np.nan, "ndd": np.nan, "optional": pd.NA}

    dose_values = _extract_dose_values(normalized)
    freq_values = _extract_frequency_values(normalized)
    interval_values = _extract_interval_values(normalized)

    dose = _summarize_values(dose_values, dose_reducer)
    freq = _summarize_values(freq_values, freq_reducer)
    interval = _summarize_values(interval_values, interval_reducer)
    if pd.notna(dose) and pd.notna(freq) and pd.notna(interval) and interval > 0:
        ndd = float(dose * freq / interval)
    else:
        ndd = np.nan

    optional_flags = [label for label, pattern in _OPTIONAL_PATTERNS if re.search(pattern, normalized)]
    optional = "; ".join(optional_flags) if optional_flags else pd.NA
    return {"dose": dose, "freq": freq, "itvl": interval, "ndd": ndd, "optional": optional}


def _normalize_prescription_text(text: object) -> str:
    if not isinstance(text, str):
        return ""

    normalized = text.lower().strip()
    normalized = normalized.replace("½", " 0.5 ").replace("¼", " 0.25 ").replace("¾", " 0.75 ")
    normalized = normalized.replace("–", "-").replace("—", "-")
    for word, value in _NUMBER_WORDS.items():
        normalized = re.sub(rf"\b{word}\b", str(value), normalized)
    normalized = re.sub(r"(\d+)\s*/\s*(\d+)", lambda m: str(float(m.group(1)) / float(m.group(2))), normalized)
    normalized = re.sub(r"(?<=\d)\s+to\s+(?=\d)", "-", normalized)
    normalized = re.sub(r"(?<=\d)\s+or\s+(?=\d)", "-", normalized)
    normalized = re.sub(r"\bevery day\b", " daily ", normalized)
    normalized = re.sub(r"\bon alternate days\b", " every other day ", normalized)
    normalized = re.sub(r"\balt(?:ernate)? days?\b", " every other day ", normalized)
    normalized = re.sub(r"\btimes?\s*/\s*day\b", " times per day ", normalized)
    normalized = re.sub(r"\btimes?\s*/\s*week\b", " times per week ", normalized)
    normalized = re.sub(r"\bq\.?d\.?\b", " qd ", normalized)
    normalized = re.sub(r"\bo\.?d\.?\b", " od ", normalized)
    normalized = re.sub(r"\bb\.?d\.?\b", " bd ", normalized)
    normalized = re.sub(r"\bt\.?d\.?s\.?\b", " tds ", normalized)
    normalized = re.sub(r"\bq\.?d\.?s\.?\b", " qds ", normalized)
    normalized = re.sub(r"\bq\.?i\.?d\.?\b", " qid ", normalized)
    normalized = re.sub(r"\bb\.?i\.?d\.?\b", " bid ", normalized)
    normalized = re.sub(r"\bt\.?i\.?d\.?\b", " tid ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _extract_dose_values(text: str) -> list[float]:
    patterns = (
        rf"(?:take|give|use|apply|insert|inhale|instil|instill)\s+([0-9]+(?:\.[0-9]+)?(?:\s*-\s*[0-9]+(?:\.[0-9]+)?)?)\s*(?:{_DOSE_UNIT_PATTERN})\b",
        rf"([0-9]+(?:\.[0-9]+)?(?:\s*-\s*[0-9]+(?:\.[0-9]+)?)?)\s*(?:{_DOSE_UNIT_PATTERN})\b",
        (
            r"(?:take|give|use|apply|insert|inhale|instil|instill)\s+"
            r"([0-9]+(?:\.[0-9]+)?(?:\s*-\s*[0-9]+(?:\.[0-9]+)?)?)"
            r"(?=\s+(?:up to\s+)?(?:[0-9]+(?:\.[0-9]+)?\s+times?\s+(?:a|per)\s+day|"
            r"(?:once|twice|thrice)\s+(?:a\s+)?day|daily|every\b|bd\b|bid\b|tds\b|tid\b|qds\b|qid\b)|$)"
        ),
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _parse_numeric_range(match.group(1))
    return []


def _extract_frequency_values(text: str) -> list[float]:
    hourly_range_match = re.search(
        r"every\s+([0-9]+(?:\.[0-9]+)?(?:\s*-\s*[0-9]+(?:\.[0-9]+)?)?)\s+hours?\b",
        text,
    )
    if hourly_range_match:
        hours_values = _parse_numeric_range(hourly_range_match.group(1))
        return [24.0 / value for value in hours_values if value > 0]

    hourly_match = re.search(r"every\s+([0-9]+(?:\.[0-9]+)?)\s+hours?\b", text)
    if hourly_match:
        hours = float(hourly_match.group(1))
        if hours > 0:
            return [24.0 / hours]

    upto_match = re.search(r"up to\s+([0-9]+(?:\.[0-9]+)?)\s+times?\s+(?:a|per)\s+day\b", text)
    if upto_match:
        return [1.0, float(upto_match.group(1))]

    range_match = re.search(
        r"([0-9]+(?:\.[0-9]+)?(?:\s*-\s*[0-9]+(?:\.[0-9]+)?)?)\s+times?\s+(?:a|per)\s+day\b",
        text,
    )
    if range_match:
        return _parse_numeric_range(range_match.group(1))

    weekly_upto_match = re.search(r"up to\s+([0-9]+(?:\.[0-9]+)?)\s+times?\s+(?:a|per)\s+week\b", text)
    if weekly_upto_match:
        return [1.0, float(weekly_upto_match.group(1))]

    weekly_range_match = re.search(
        r"([0-9]+(?:\.[0-9]+)?(?:\s*-\s*[0-9]+(?:\.[0-9]+)?)?)\s+times?\s+(?:a|per)\s+week\b",
        text,
    )
    if weekly_range_match:
        return _parse_numeric_range(weekly_range_match.group(1))

    daily_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+daily\b", text)
    if daily_match:
        return [float(daily_match.group(1))]

    for word, value in _FREQUENCY_WORDS.items():
        if re.search(rf"\b{word}\s+(?:a\s+)?day\b", text) or re.search(rf"\b{word}\s+daily\b", text):
            return [value]
        if re.search(rf"\b{word}\s*/\s*day\b", text):
            return [value]

    for abbreviation, value in _FREQUENCY_ABBREVIATIONS.items():
        if re.search(rf"\b{abbreviation}\b", text):
            return [value]

    if "morning and night" in text or "morning and evening" in text:
        return [2.0]
    if "morning, noon and night" in text:
        return [3.0]
    if "weekly" in text or re.search(r"\bonce (?:a|per) week\b", text):
        return [1.0]
    if re.search(r"\btwice (?:a|per) week\b", text):
        return [2.0]
    if "monthly" in text or re.search(r"\bonce (?:a|per) month\b", text):
        return [1.0]
    if re.search(r"\b(at night|every night|daily|each day|per day)\b", text):
        return [1.0]

    return [1.0]


def _extract_interval_values(text: str) -> list[float]:
    if "every other day" in text:
        return [2.0]

    if re.search(r"every\s+[0-9]+(?:\.[0-9]+)?(?:\s*-\s*[0-9]+(?:\.[0-9]+)?)?\s+hours?\b", text):
        return [1.0]

    day_match = re.search(r"every\s+([0-9]+(?:\.[0-9]+)?)\s+days?\b", text)
    if day_match:
        return [float(day_match.group(1))]

    day_range_match = re.search(
        r"every\s+([0-9]+(?:\.[0-9]+)?(?:\s*-\s*[0-9]+(?:\.[0-9]+)?)?)\s+days?\b",
        text,
    )
    if day_range_match:
        return _parse_numeric_range(day_range_match.group(1))

    week_match = re.search(r"every\s+([0-9]+(?:\.[0-9]+)?)\s+weeks?\b", text)
    if week_match:
        return [7.0 * float(week_match.group(1))]

    week_range_match = re.search(
        r"every\s+([0-9]+(?:\.[0-9]+)?(?:\s*-\s*[0-9]+(?:\.[0-9]+)?)?)\s+weeks?\b",
        text,
    )
    if week_range_match:
        return [7.0 * value for value in _parse_numeric_range(week_range_match.group(1))]

    if re.search(r"\bweekly\b", text):
        return [7.0]
    if re.search(r"\bmonthly\b", text):
        return [30.0]

    return [1.0]


def _parse_numeric_range(fragment: str) -> list[float]:
    values = [part.strip() for part in re.split(r"\s*-\s*", fragment) if part.strip()]
    parsed = [float(value) for value in values]
    return parsed if parsed else []


def _summarize_values(
    values: Sequence[float],
    reducer: Callable[[Sequence[float]], float],
) -> float:
    if not values:
        return np.nan
    return float(reducer(values))


def disambiguate_same_day_prescriptions(
    prescriptions: pd.DataFrame,
    *,
    decision: str = "a",
    patient_col: str = "patid",
    prodcode_col: str = "prodcode",
    start_col: str = "start_date",
    duration_col: str = "duration",
) -> pd.DataFrame:
    """Apply `drugprepr` decision 8 to same-day prescription clashes."""
    if decision == "a" or prescriptions.empty:
        return prescriptions.copy()

    reducer_map = {"b": "mean", "c": "min", "d": "max", "e": "sum"}
    reducer = reducer_map.get(decision)
    if reducer is None:
        raise NotImplementedError(f"Unsupported same-day clash decision: {decision}")

    frame = prescriptions.copy()
    frame[start_col] = pd.to_datetime(frame[start_col], errors="coerce")
    frame[duration_col] = pd.to_numeric(frame[duration_col], errors="coerce")
    frame["_row_order"] = np.arange(len(frame))

    group_cols = [prodcode_col, patient_col, start_col]
    group_sizes = frame.groupby(group_cols, dropna=False, sort=False)[duration_col].transform("size")
    summaries = frame.groupby(group_cols, dropna=False, sort=False)[duration_col].transform(reducer)
    frame.loc[group_sizes.gt(1), duration_col] = summaries[group_sizes.gt(1)]
    frame = frame.sort_values("_row_order").drop_duplicates(subset=group_cols, keep="first")
    return frame.drop(columns="_row_order").reset_index(drop=True)


def resolve_overlapping_prescriptions(
    prescriptions: pd.DataFrame,
    *,
    decision: str = "a",
    patient_col: str = "patid",
    prodcode_col: str = "prodcode",
    start_col: str = "start_date",
    stop_col: str = "stop_date",
) -> pd.DataFrame:
    """Apply `drugprepr` decision 9 to overlapping prescription periods."""
    if decision == "a" or prescriptions.empty:
        return prescriptions.copy()
    if decision != "b":
        raise NotImplementedError(f"Unsupported overlap decision: {decision}")

    frame = prescriptions.copy()
    frame[start_col] = pd.to_datetime(frame[start_col], errors="coerce")
    frame[stop_col] = pd.to_datetime(frame[stop_col], errors="coerce")
    frame["_row_order"] = np.arange(len(frame))
    frame = frame.sort_values([patient_col, prodcode_col, start_col, stop_col, "_row_order"]).reset_index(drop=True)

    adjusted_rows: list[dict] = []
    for _, group in frame.groupby([patient_col, prodcode_col], dropna=False, sort=False):
        current_stop: pd.Timestamp | None = None
        for row in group.to_dict("records"):
            start = row[start_col]
            stop = row[stop_col]
            if pd.isna(start) or pd.isna(stop):
                continue

            duration = stop - start
            if current_stop is not None and start < current_stop:
                start = current_stop
                stop = start + duration

            row[start_col] = start
            row[stop_col] = stop
            adjusted_rows.append(row)
            current_stop = stop

    adjusted = pd.DataFrame(adjusted_rows)
    return adjusted.drop(columns="_row_order").reset_index(drop=True)


def close_small_gaps_in_prescriptions(
    prescriptions: pd.DataFrame,
    *,
    decision: str = "a",
    patient_col: str = "patid",
    prodcode_col: str = "prodcode",
    start_col: str = "start_date",
    stop_col: str = "stop_date",
) -> pd.DataFrame:
    """Apply `drugprepr` decision 10 to short gaps between prescriptions."""
    if decision == "a" or prescriptions.empty:
        return prescriptions.copy()
    if not decision.startswith("b_"):
        raise NotImplementedError(f"Unsupported gap-closing decision: {decision}")

    min_gap = int(decision.split("_", 1)[1])
    frame = prescriptions.copy()
    frame[start_col] = pd.to_datetime(frame[start_col], errors="coerce")
    frame[stop_col] = pd.to_datetime(frame[stop_col], errors="coerce")
    frame["_row_order"] = np.arange(len(frame))
    frame = frame.sort_values([patient_col, prodcode_col, start_col, stop_col, "_row_order"]).reset_index(drop=True)

    closed_rows: list[dict] = []
    for _, group in frame.groupby([patient_col, prodcode_col], dropna=False, sort=False):
        records = group.to_dict("records")
        for index, row in enumerate(records[:-1]):
            next_start = records[index + 1][start_col]
            gap = next_start - row[stop_col]
            if pd.notna(gap) and pd.Timedelta(0) <= gap < pd.Timedelta(days=min_gap):
                row[stop_col] = next_start
        closed_rows.extend(records)

    closed = pd.DataFrame(closed_rows)
    return closed.drop(columns="_row_order").reset_index(drop=True)


def _apply_implausible_imputation(
    frame: pd.DataFrame,
    *,
    value_col: str,
    lower_col: str,
    upper_col: str,
    decision: str,
    patient_col: str,
    practice_col: str,
    prodcode_col: str,
) -> pd.Series:
    values = pd.to_numeric(frame[value_col], errors="coerce")
    lower = pd.to_numeric(frame[lower_col], errors="coerce")
    upper = pd.to_numeric(frame[upper_col], errors="coerce")
    mask = values.lt(lower) | values.gt(upper)
    method = {"a": "ignore", "b": "replace", "c": "mean", "d": "median", "e": "mode"}.get(decision[0])
    if method is None:
        raise NotImplementedError(f"Unsupported implausible-value decision: {decision}")
    group = _decision_group(decision, patient_col=patient_col, practice_col=practice_col)
    return _impute_by_group(
        frame, values, mask, method=method, group=group, prodcode_col=prodcode_col, replace_with=np.nan
    )


def _apply_missing_imputation(
    frame: pd.DataFrame,
    *,
    value_col: str,
    decision: str,
    patient_col: str,
    practice_col: str,
    prodcode_col: str,
) -> pd.Series:
    values = pd.to_numeric(frame[value_col], errors="coerce")
    method = {"a": "ignore", "b": "mean", "c": "median", "d": "mode"}.get(decision[0])
    if method is None:
        raise NotImplementedError(f"Unsupported missing-value decision: {decision}")
    group = _decision_group(decision, patient_col=patient_col, practice_col=practice_col)
    return _impute_by_group(frame, values, values.isna(), method=method, group=group, prodcode_col=prodcode_col)


def _apply_missing_duration_imputation(
    frame: pd.DataFrame,
    *,
    duration_col: str,
    decision: str,
    patient_col: str,
    prodcode_col: str,
) -> pd.Series:
    values = pd.to_numeric(frame[duration_col], errors="coerce")
    if decision == "a":
        return values
    if decision == "b":
        return _impute_by_group(
            frame, values, values.isna(), method="mean", group=patient_col, prodcode_col=prodcode_col
        )
    if decision == "c":
        return _impute_by_group(
            frame, values, values.isna(), method="mean", group="population", prodcode_col=prodcode_col
        )
    if decision == "d":
        individual = _impute_by_group(
            frame, values, values.isna(), method="mean", group=patient_col, prodcode_col=prodcode_col
        )
        temp = frame.copy()
        temp[duration_col] = individual
        return _impute_by_group(
            temp, individual, individual.isna(), method="mean", group="population", prodcode_col=prodcode_col
        )
    raise NotImplementedError(f"Unsupported missing-duration decision: {decision}")


def _calculate_duration_from_decision(
    frame: pd.DataFrame,
    *,
    decision: str,
    qty_col: str,
    ndd_col: str,
    numdays_col: str,
    dose_duration_col: str,
) -> pd.Series:
    if decision == "a":
        if numdays_col not in frame.columns:
            raise KeyError(f"{numdays_col} is required for duration decision 'a'")
        return pd.to_numeric(frame[numdays_col], errors="coerce")
    if decision == "b":
        if dose_duration_col not in frame.columns:
            raise KeyError(f"{dose_duration_col} is required for duration decision 'b'")
        return pd.to_numeric(frame[dose_duration_col], errors="coerce")
    if decision == "c":
        if qty_col not in frame.columns or ndd_col not in frame.columns:
            raise KeyError(f"{qty_col} and {ndd_col} are required for duration decision 'c'")
        qty = pd.to_numeric(frame[qty_col], errors="coerce")
        ndd = pd.to_numeric(frame[ndd_col], errors="coerce")
        return qty / ndd.replace(0, np.nan)
    raise NotImplementedError(f"Unsupported duration-source decision: {decision}")


def _clean_duration(values: pd.Series, decision: str) -> pd.Series:
    if decision == "a":
        return values
    mode, months = decision.split("_", 1)
    max_days = round(int(months) * 365 / 12)
    cleaned = values.copy()
    if mode == "b":
        cleaned = cleaned.mask(cleaned > max_days)
    elif mode == "c":
        cleaned = cleaned.mask(cleaned > max_days, max_days)
    else:
        raise NotImplementedError(f"Unsupported duration-cleaning decision: {decision}")
    return cleaned


def _decision_group(decision: str, *, patient_col: str, practice_col: str) -> str:
    if len(decision) < 2:
        return "population"
    group_key = decision[1]
    return {"1": patient_col, "2": practice_col, "3": "population"}.get(group_key, "population")


def _impute_by_group(
    frame: pd.DataFrame,
    values: pd.Series,
    mask: pd.Series,
    *,
    method: str,
    group: str,
    prodcode_col: str,
    replace_with: float | None = None,
) -> pd.Series:
    if method == "ignore":
        return values
    if method == "replace":
        return values.mask(mask, replace_with)

    if group == "population":
        group_cols = [prodcode_col]
    else:
        group_cols = [prodcode_col, group]

    grouped = pd.DataFrame({"_value": values, **{column: frame[column] for column in group_cols}})
    if method == "mean":
        aggregates = grouped.groupby(group_cols, dropna=False)["_value"].transform("mean")
    elif method == "median":
        aggregates = grouped.groupby(group_cols, dropna=False)["_value"].transform("median")
    elif method == "mode":
        aggregates = grouped.groupby(group_cols, dropna=False)["_value"].transform(_first_mode)
    else:
        raise NotImplementedError(f"Unsupported imputation method: {method}")
    return values.mask(mask, aggregates)


def _first_mode(values: pd.Series) -> float:
    non_missing = values.dropna()
    if non_missing.empty:
        return np.nan
    modes = non_missing.mode(dropna=True)
    return float(modes.iloc[0])


def build_exposure_episodes_from_prescriptions(
    prescriptions: pd.DataFrame,
    *,
    patient_col: str = "patid",
    start_col: str = "start_date",
    duration_col: str = "duration",
    group_cols: Sequence[str] = ("drug",),
    gap_days: int = 0,
    keep_first_only: bool = True,
    stop_col: str = "stop_date",
) -> pd.DataFrame:
    """Build exposure episodes from prescription rows with precomputed durations.

    This ports the part of the original R workflow that starts after the dosage
    inference decisions from `drugprepr` have already produced a `duration`
    column. The function:

    1. aggregates repeated prescriptions on the same start date,
    2. converts durations into stop dates,
    3. merges overlapping or near-adjacent intervals, and
    4. optionally keeps the earliest episode per patient/group.
    """
    if prescriptions.empty:
        return prescriptions.copy()

    required_columns = {patient_col, start_col, duration_col, *group_cols}
    missing = required_columns.difference(prescriptions.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required prescription columns: {missing_str}")

    frame = prescriptions.loc[:, [patient_col, *group_cols, start_col, duration_col]].copy()
    frame[start_col] = pd.to_datetime(frame[start_col])
    frame[duration_col] = pd.to_numeric(frame[duration_col], errors="coerce")
    frame = frame.dropna(subset=[start_col, duration_col])
    frame = frame[frame[duration_col] > 0].copy()

    grouped = (
        frame.groupby([patient_col, *group_cols, start_col], as_index=False, sort=False)[duration_col]
        .sum()
        .rename(columns={duration_col: "episode_duration"})
    )
    grouped[stop_col] = grouped[start_col] + pd.to_timedelta(grouped["episode_duration"], unit="D")

    episodes = collapse_exposure_episodes(
        grouped.loc[:, [patient_col, *group_cols, start_col, stop_col]],
        patient_col=patient_col,
        start_col=start_col,
        stop_col=stop_col,
        group_cols=group_cols,
        gap_days=gap_days,
        keep_first_only=keep_first_only,
    )
    episodes["episode_duration"] = (episodes[stop_col] - episodes[start_col]).dt.days.astype(float)
    return episodes.reset_index(drop=True)


def collapse_exposure_episodes(
    exposures: pd.DataFrame,
    *,
    patient_col: str = "patid",
    start_col: str = "start_date",
    stop_col: str = "stop_date",
    group_cols: Sequence[str] = ("drug",),
    gap_days: int = 0,
    keep_first_only: bool = True,
) -> pd.DataFrame:
    """Merge overlapping or near-adjacent exposure episodes.

    The original R workflow collapses repeated prescriptions into one exposure
    interval and then keeps the earliest interval per patient and drug. This
    helper ports that prepared-data step without the CPRD-specific dosage logic.
    """
    if exposures.empty:
        return exposures.copy()

    required_columns = {patient_col, start_col, stop_col, *group_cols}
    missing = required_columns.difference(exposures.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required exposure columns: {missing_str}")

    frame = exposures.copy()
    frame[start_col] = pd.to_datetime(frame[start_col])
    frame[stop_col] = pd.to_datetime(frame[stop_col])
    frame = frame.sort_values([patient_col, *group_cols, start_col, stop_col]).reset_index(drop=True)

    merge_keys = [patient_col, *group_cols]
    merged_rows: list[dict] = []

    for keys, group in frame.groupby(merge_keys, dropna=False, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        current_start = None
        current_stop = None

        for row in group.itertuples(index=False):
            start = getattr(row, start_col)
            stop = getattr(row, stop_col)
            if pd.isna(start) or pd.isna(stop):
                continue

            if current_start is None:
                current_start, current_stop = start, stop
                continue

            if start <= current_stop + pd.Timedelta(days=gap_days):
                current_stop = max(current_stop, stop)
            else:
                merged_rows.append(
                    _merge_row_dict(merge_keys, key_values, start_col, current_start, stop_col, current_stop)
                )
                current_start, current_stop = start, stop

        if current_start is not None:
            merged_rows.append(
                _merge_row_dict(merge_keys, key_values, start_col, current_start, stop_col, current_stop)
            )

    merged = pd.DataFrame(merged_rows)
    if keep_first_only and not merged.empty:
        merged = (
            merged.sort_values([patient_col, *group_cols, start_col])
            .groupby([patient_col, *group_cols], as_index=False, sort=False)
            .first()
        )

    return merged.reset_index(drop=True)


def prepare_exposure_windows(
    exposures: pd.DataFrame,
    patients: pd.DataFrame,
    *,
    patient_col: str = "patid",
    start_col: str = "start_date",
    stop_col: str = "stop_date",
    birth_col: str = "dob",
    registration_start_col: str = "frd",
    observation_end_cols: Sequence[str] = ("tod", "lcd", "deathdate"),
    fixed_followup_days: int | None = None,
    group_cols: Sequence[str] = ("drug",),
    gap_days: int = 0,
    keep_first_only: bool = True,
) -> pd.DataFrame:
    """Prepare balanced exposed and unexposed windows for drug screening.

    Args:
        exposures: Exposure intervals with patient, drug/grouping, start, and stop columns.
        patients: Patient table with date of birth, registration start, and observation end columns.
        fixed_followup_days: If provided, cap the balanced windows to this many days.
            If omitted, use the observed exposure duration after clipping to patient timelines.
        group_cols: Columns that define the screening grouping. For the first slice this is usually ``("drug",)``.
        gap_days: Maximum gap that is still considered the same exposure episode.
        keep_first_only: Match the original workflow and keep the earliest merged episode per patient/group.
    """
    collapsed = collapse_exposure_episodes(
        exposures,
        patient_col=patient_col,
        start_col=start_col,
        stop_col=stop_col,
        group_cols=group_cols,
        gap_days=gap_days,
        keep_first_only=keep_first_only,
    )
    if collapsed.empty:
        return collapsed

    required_patient_columns = {patient_col, birth_col, registration_start_col, *observation_end_cols}
    missing = required_patient_columns.difference(patients.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required patient columns: {missing_str}")

    patient_frame = patients.loc[:, list(required_patient_columns)].copy()
    for column in [birth_col, registration_start_col, *observation_end_cols]:
        patient_frame[column] = pd.to_datetime(patient_frame[column])

    prepared = collapsed.merge(patient_frame, how="left", on=patient_col)
    prepared[start_col] = pd.to_datetime(prepared[start_col])
    prepared[stop_col] = pd.to_datetime(prepared[stop_col])

    observation_end = prepared.loc[:, list(observation_end_cols)].min(axis=1)
    raw_duration = (prepared[stop_col] - prepared[start_col]).dt.days.astype(float)
    available_before = (prepared[start_col] - prepared[registration_start_col]).dt.days.astype(float)
    available_after = (observation_end - prepared[start_col]).dt.days.astype(float)

    if fixed_followup_days is None:
        exposure_length = pd.concat([raw_duration, available_before, available_after], axis=1).min(axis=1)
    else:
        exposure_length = pd.concat(
            [pd.Series(float(fixed_followup_days), index=prepared.index), available_before, available_after],
            axis=1,
        ).min(axis=1)

    prepared["exposure_length"] = exposure_length
    prepared = prepared[prepared["exposure_length"] > 0].copy()
    prepared["stop_date"] = prepared[start_col] + pd.to_timedelta(prepared["exposure_length"], unit="D")
    prepared["unexposed_start_date"] = prepared[start_col] - pd.to_timedelta(prepared["exposure_length"], unit="D")
    prepared["prescription_age"] = (prepared[start_col] - prepared[birth_col]).dt.days / 365.25

    return prepared.reset_index(drop=True)


def _merge_row_dict(
    merge_keys: Sequence[str],
    key_values: Sequence[object],
    start_col: str,
    start_value: pd.Timestamp,
    stop_col: str,
    stop_value: pd.Timestamp,
) -> dict:
    row = dict(zip(merge_keys, key_values, strict=False))
    row[start_col] = start_value
    row[stop_col] = stop_value
    return row
