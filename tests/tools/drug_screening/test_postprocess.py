import pandas as pd

import ehrapy as ep


def test_combine_screening_results_normalizes_first_column_and_tags_levels():
    chapter = pd.DataFrame({"chapter": ["cardiology"], "disease": ["condition_a"], "IRR.higher.95": [0.8]})
    substance = pd.DataFrame({"drug": ["aspirin"], "disease": ["condition_b"], "IRR.higher.95": [0.7]})

    combined = ep.tl.combine_screening_results({"chapter": chapter, "substance": substance})

    assert list(combined["drug"]) == ["cardiology", "aspirin"]
    assert list(combined["drug.level"]) == ["chapter", "substance"]


def test_rank_repurposing_hits_filters_and_sorts_lowest_upper_bound_first():
    results = pd.DataFrame(
        {
            "drug": ["drug_a", "drug_b", "drug_c"],
            "disease": ["d1", "d2", "d3"],
            "age.group": ["actual", "actual", "actual"],
            "N.disease.B.during.unexposed": [50, 50, 20],
            "N.disease.C.during.exposed": [40, 60, 40],
            "IRR.higher.95": [0.9, 0.7, 0.6],
        }
    )

    ranked = ep.tl.rank_repurposing_hits(results)

    assert list(ranked["drug"]) == ["drug_b", "drug_a"]
    assert list(ranked["IRR.higher.95"]) == [0.7, 0.9]


def test_rank_safety_hits_filters_and_sorts_highest_lower_bound_first():
    results = pd.DataFrame(
        {
            "drug": ["drug_a", "drug_b", "drug_c"],
            "disease": ["d1", "d2", "d3"],
            "age.group": ["actual", "actual", "actual"],
            "N.disease.B.during.unexposed": [50, 20, 50],
            "N.disease.C.during.exposed": [60, 60, 60],
            "IRR.lower.95": [1.2, 1.8, 0.9],
        }
    )

    ranked = ep.tl.rank_safety_hits(results)

    assert list(ranked["drug"]) == ["drug_a"]
    assert list(ranked["IRR.lower.95"]) == [1.2]
