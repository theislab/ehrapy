import pandas as pd

import ehrapy as ep


class FakeLLM:
    def __init__(self, responses: dict[str, list[str] | str]):
        self.responses = {prompt: value if isinstance(value, list) else [value] for prompt, value in responses.items()}
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        values = self.responses.get(prompt)
        if not values:
            raise AssertionError(f"Unexpected prompt: {prompt}")
        response = values.pop(0) if len(values) > 1 else values[0]
        if isinstance(response, Exception):
            raise response
        return response


def test_prompt_builders_match_original_workflow_wording():
    assert ep.tl.build_repurposing_indication_prompt("drug_a", "disease_x") == (
        "is drug_a used to treat disease_x? Just answer yes or no"
    )
    assert ep.tl.build_drug_indications_prompt("drug_a") == (
        "which diseases are drug_a used to treat? Limit answer within eight words"
    )
    assert ep.tl.build_repurposing_risk_factor_prompt("disease_a or disease_b", "disease_x") == (
        "is any disease in disease_a or disease_b a risk factor of disease_x? Just answer yes or no"
    )
    assert ep.tl.build_safety_symptom_prompt("drug_a", "disease_x") == (
        "is disease_x a symptom of any disorder treated by drug_a? Just answer yes or no"
    )
    assert ep.tl.build_safety_indication_prompt("drug_a", "disease_x") == (
        "is disease_x caused by any indication of drug_a? Just answer yes or no"
    )
    assert ep.tl.build_aging_prompt("disease_x") == "is disease_x more common as people age? Just answer yes or no"


def test_normalize_binary_answer_reduces_free_text_to_yes_no_unknown():
    assert ep.tl.normalize_binary_answer("Yes, definitely.") == "yes"
    assert ep.tl.normalize_binary_answer("NO.") == "no"
    assert ep.tl.normalize_binary_answer("unknown maybe") == "unknown"


def test_normalize_summary_answer_matches_original_cleanup():
    assert ep.tl.normalize_summary_answer("Asthma, and COPD.") == "asthma, or copd"


def test_prepare_unique_drug_disease_pairs_matches_original_unique_subset():
    data = pd.DataFrame(
        {
            "drug": ["drug_a", "drug_a", "drug_b"],
            "disease": ["disease_x", "disease_x", "disease_y"],
            "score": [1, 2, 3],
        }
    )

    pairs = ep.tl.prepare_unique_drug_disease_pairs(data)

    assert pairs.to_dict("records") == [
        {"drug": "drug_a", "disease": "disease_x"},
        {"drug": "drug_b", "disease": "disease_y"},
    ]


def test_review_repurposing_indications_retries_error_like_responses():
    data = pd.DataFrame({"drug": ["drug_a"], "disease": ["disease_x"]})
    prompt = ep.tl.build_repurposing_indication_prompt("drug_a", "disease_x")
    llm = FakeLLM({prompt: ["Error: timeout", "Yes."]})

    reviewed = ep.tl.review_repurposing_indications(data, llm, max_attempts=3)

    assert reviewed.loc[0, "indication"] == "yes"
    assert llm.calls == [prompt, prompt]


def test_review_repurposing_indications_retries_backend_exceptions():
    data = pd.DataFrame({"drug": ["drug_a"], "disease": ["disease_x"]})
    prompt = ep.tl.build_repurposing_indication_prompt("drug_a", "disease_x")
    llm = FakeLLM({prompt: [RuntimeError("timeout"), "No"]})

    reviewed = ep.tl.review_repurposing_indications(data, llm, max_attempts=3)

    assert reviewed.loc[0, "indication"] == "no"
    assert llm.calls == [prompt, prompt]


def test_summarize_drug_indications_normalizes_short_text_output():
    data = pd.DataFrame({"drug": ["drug_a"]})
    prompt = ep.tl.build_drug_indications_prompt("drug_a")
    llm = FakeLLM({prompt: "Asthma and COPD."})

    reviewed = ep.tl.summarize_drug_indications(data, llm)

    assert reviewed.loc[0, "indication.of.drug"] == "asthma or copd"


def test_review_repurposing_risk_factors_uses_indication_summary_column():
    data = pd.DataFrame({"indication.of.drug": ["asthma or copd"], "disease": ["pneumonia"]})
    prompt = ep.tl.build_repurposing_risk_factor_prompt("asthma or copd", "pneumonia")
    llm = FakeLLM({prompt: "No"})

    reviewed = ep.tl.review_repurposing_risk_factors(data, llm)

    assert reviewed.loc[0, "risk.factor"] == "no"


def test_review_safety_workflows_match_original_columns():
    safety_pairs = pd.DataFrame({"drug": ["drug_a"], "disease": ["disease_x"]})
    symptom_prompt = ep.tl.build_safety_symptom_prompt("drug_a", "disease_x")
    indication_prompt = ep.tl.build_safety_indication_prompt("drug_a", "disease_x")
    llm = FakeLLM({symptom_prompt: "yes", indication_prompt: "no"})

    symptoms = ep.tl.review_safety_symptoms(safety_pairs, llm)
    indications = ep.tl.review_safety_indications(safety_pairs, llm)

    assert symptoms.loc[0, "symptom"] == "yes"
    assert indications.loc[0, "indication"] == "no"


def test_review_safety_aging_short_circuits_short_exposures():
    data = pd.DataFrame(
        {
            "disease": ["disease_short", "disease_long"],
            "exposed.mean": [100.0, 500.0],
        }
    )
    prompt = ep.tl.build_aging_prompt("disease_long")
    llm = FakeLLM({prompt: "Yes"})

    reviewed = ep.tl.review_safety_aging(data, llm)

    assert reviewed.loc[0, "aging"] == "short"
    assert reviewed.loc[1, "aging"] == "yes"
    assert llm.calls == [prompt]
