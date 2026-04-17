from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable


class LLMCallable(Protocol):
    """Callable protocol for optional LLM review backends."""

    def __call__(self, prompt: str) -> str: ...


def build_repurposing_indication_prompt(drug: object, disease: object) -> str:
    """Build the repurposing indication prompt from the original R workflow."""
    return f"is {drug} used to treat {disease}? Just answer yes or no"


def build_drug_indications_prompt(drug: object) -> str:
    """Build the indication-summary prompt from the original R workflow."""
    return f"which diseases are {drug} used to treat? Limit answer within eight words"


def build_repurposing_risk_factor_prompt(indications: object, disease: object) -> str:
    """Build the repurposing risk-factor prompt from the original R workflow."""
    return f"is any disease in {indications} a risk factor of {disease}? Just answer yes or no"


def build_safety_symptom_prompt(drug: object, disease: object) -> str:
    """Build the safety symptom prompt from the original R workflow."""
    return f"is {disease} a symptom of any disorder treated by {drug}? Just answer yes or no"


def build_safety_indication_prompt(drug: object, disease: object) -> str:
    """Build the safety indication prompt from the original R workflow."""
    return f"is {disease} caused by any indication of {drug}? Just answer yes or no"


def build_aging_prompt(disease: object) -> str:
    """Build the aging prompt from the original R workflow."""
    return f"is {disease} more common as people age? Just answer yes or no"


def review_repurposing_indications(
    data: pd.DataFrame,
    llm_call: LLMCallable,
    *,
    drug_col: str = "drug",
    disease_col: str = "disease",
    output_col: str = "indication",
    normalize_binary: bool = True,
    max_attempts: int = 3,
) -> pd.DataFrame:
    """Review whether each drug-disease pair is an existing treatment indication."""
    return _review_binary_pairs(
        data,
        llm_call,
        prompt_builder=build_repurposing_indication_prompt,
        left_col=drug_col,
        right_col=disease_col,
        output_col=output_col,
        normalize_binary=normalize_binary,
        max_attempts=max_attempts,
    )


def summarize_drug_indications(
    data: pd.DataFrame,
    llm_call: LLMCallable,
    *,
    drug_col: str = "drug",
    output_col: str = "indication.of.drug",
    normalize_summary: bool = True,
    max_attempts: int = 3,
) -> pd.DataFrame:
    """Summarize the known indications of each drug."""
    _require_columns(data, {drug_col}, context="drug indication review")

    frame = data.copy()
    responses = [
        _call_with_retries(llm_call, build_drug_indications_prompt(drug), max_attempts=max_attempts)
        for drug in frame[drug_col]
    ]
    frame[output_col] = responses
    if normalize_summary:
        frame[output_col] = frame[output_col].apply(normalize_summary_answer)
    return frame


def review_repurposing_risk_factors(
    data: pd.DataFrame,
    llm_call: LLMCallable,
    *,
    indications_col: str = "indication.of.drug",
    disease_col: str = "disease",
    output_col: str = "risk.factor",
    normalize_binary: bool = True,
    max_attempts: int = 3,
) -> pd.DataFrame:
    """Review whether any indication of the drug is a risk factor for the disease."""
    return _review_binary_pairs(
        data,
        llm_call,
        prompt_builder=build_repurposing_risk_factor_prompt,
        left_col=indications_col,
        right_col=disease_col,
        output_col=output_col,
        normalize_binary=normalize_binary,
        max_attempts=max_attempts,
    )


def review_safety_symptoms(
    data: pd.DataFrame,
    llm_call: LLMCallable,
    *,
    drug_col: str = "drug",
    disease_col: str = "disease",
    output_col: str = "symptom",
    normalize_binary: bool = True,
    max_attempts: int = 3,
) -> pd.DataFrame:
    """Review whether the disease is a symptom of any disorder treated by the drug."""
    return _review_binary_pairs(
        data,
        llm_call,
        prompt_builder=build_safety_symptom_prompt,
        left_col=drug_col,
        right_col=disease_col,
        output_col=output_col,
        normalize_binary=normalize_binary,
        max_attempts=max_attempts,
    )


def review_safety_indications(
    data: pd.DataFrame,
    llm_call: LLMCallable,
    *,
    drug_col: str = "drug",
    disease_col: str = "disease",
    output_col: str = "indication",
    normalize_binary: bool = True,
    max_attempts: int = 3,
) -> pd.DataFrame:
    """Review whether the disease is caused by any indication of the drug."""
    return _review_binary_pairs(
        data,
        llm_call,
        prompt_builder=build_safety_indication_prompt,
        left_col=drug_col,
        right_col=disease_col,
        output_col=output_col,
        normalize_binary=normalize_binary,
        max_attempts=max_attempts,
    )


def review_safety_aging(
    data: pd.DataFrame,
    llm_call: LLMCallable,
    *,
    disease_col: str = "disease",
    exposed_mean_col: str = "exposed.mean",
    output_col: str = "aging",
    aging_threshold_days: float = 367.0,
    short_label: str = "short",
    normalize_binary: bool = True,
    max_attempts: int = 3,
) -> pd.DataFrame:
    """Review whether the disease becomes more common with age.

    The original R workflow only asks this question when ``exposed.mean > 367`` and
    assigns the label ``"short"`` otherwise.
    """
    _require_columns(data, {disease_col, exposed_mean_col}, context="safety aging review")

    frame = data.copy()
    answers: list[object] = []
    for _, row in frame.iterrows():
        if pd.isna(row[exposed_mean_col]) or float(row[exposed_mean_col]) <= aging_threshold_days:
            answers.append(short_label)
            continue
        response = _call_with_retries(llm_call, build_aging_prompt(row[disease_col]), max_attempts=max_attempts)
        answers.append(normalize_binary_answer(response) if normalize_binary else response)
    frame[output_col] = answers
    return frame


def normalize_binary_answer(answer: object) -> object:
    """Normalize free-form LLM output to ``yes``/``no`` when possible."""
    if not isinstance(answer, str):
        return answer
    normalized = answer.strip().lower()
    normalized = re.sub(r"\s.*$", "", normalized)
    normalized = re.sub(r"[^\w]+", "", normalized)
    if normalized in {"yes", "no", "unknown"}:
        return normalized
    return normalized or pd.NA


def normalize_summary_answer(answer: object) -> object:
    """Normalize the short indication-summary answer like the original R script."""
    if not isinstance(answer, str):
        return answer
    normalized = answer.strip().lower()
    normalized = re.sub(r"[^\w\s]+$", "", normalized)
    normalized = re.sub(r"\band\b", "or", normalized)
    return normalized or pd.NA


def prepare_unique_drug_disease_pairs(
    data: pd.DataFrame,
    *,
    drug_col: str = "drug",
    disease_col: str = "disease",
) -> pd.DataFrame:
    """Prepare the unique drug-disease pairs used by the original review loops."""
    _require_columns(data, {drug_col, disease_col}, context="drug-disease pair preparation")
    return data.loc[:, [drug_col, disease_col]].drop_duplicates().reset_index(drop=True)


def _review_binary_pairs(
    data: pd.DataFrame,
    llm_call: LLMCallable,
    *,
    prompt_builder: Callable[[object, object], str],
    left_col: str,
    right_col: str,
    output_col: str,
    normalize_binary: bool,
    max_attempts: int,
) -> pd.DataFrame:
    _require_columns(data, {left_col, right_col}, context=f"{output_col} review")

    frame = data.copy()
    responses = [
        _call_with_retries(llm_call, prompt_builder(left, right), max_attempts=max_attempts)
        for left, right in zip(frame[left_col], frame[right_col], strict=False)
    ]
    frame[output_col] = [normalize_binary_answer(response) if normalize_binary else response for response in responses]
    return frame


def _call_with_retries(
    llm_call: LLMCallable,
    prompt: str,
    *,
    max_attempts: int = 3,
) -> str:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    last_response = ""
    for _ in range(max_attempts):
        try:
            response = llm_call(prompt)
            last_response = "" if response is None else str(response)
        except Exception as exc:
            last_response = f"Error: {exc}"
        if _is_acceptable_response(last_response):
            return last_response
    return last_response


def _is_acceptable_response(response: str) -> bool:
    if not response:
        return False
    return not bool(re.search(r"^error", response, flags=re.IGNORECASE))


def _require_columns(frame: pd.DataFrame, columns: set[str], *, context: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns for {context}: {missing_str}")
