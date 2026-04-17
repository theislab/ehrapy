import pandas as pd
import pytest

import ehrapy as ep


def test_extract_rxclass_may_treat_returns_mesh_rows():
    payload = {
        "rxclassDrugInfoList": {
            "rxclassDrugInfo": [
                {
                    "minConcept": {"name": "drug_a"},
                    "rxclassMinConceptItem": {"classId": "D001", "className": "Disease A"},
                },
                {
                    "minConcept": {"name": "drug_a"},
                    "rxclassMinConceptItem": {"classId": "D002", "className": "Disease B"},
                },
            ]
        }
    }

    mapping = ep.tl.extract_rxclass_may_treat(payload, rxcui="123")

    assert mapping.to_dict("records") == [
        {"rxcui": "123", "drug": "drug_a", "mesh": "D001", "disease": "Disease A"},
        {"rxcui": "123", "drug": "drug_a", "mesh": "D002", "disease": "Disease B"},
    ]


def test_extract_readcodev3_from_snomedbrowser_uses_eighth_text_node():
    text_nodes = ["", "", "", "", "", "", "", "Xa123 Xb456"]

    mapping = ep.tl.extract_readcodev3_from_snomedbrowser(text_nodes, snomed_disease="900000")

    assert mapping.to_dict("records") == [
        {"snomed.disease": "900000", "readcodev3": "Xa123"},
        {"snomed.disease": "900000", "readcodev3": "Xb456"},
    ]


def test_extract_snomed_ingredient_links_handles_substance_identity():
    attr_concepts = pd.DataFrame(
        {
            "sourceId": ["111"],
            "sourceDesc": ["Example drug (substance)"],
            "destinationId": [None],
            "typeDesc": [None],
        }
    )

    mapping = ep.tl.extract_snomed_ingredient_links(attr_concepts, snomed_drug="111")

    assert mapping.to_dict("records") == [{"snomed.drug.uk": "111", "snomed.ingredient": "111"}]


def test_extract_snomed_ingredient_links_filters_active_ingredient_relationships():
    attr_concepts = pd.DataFrame(
        {
            "sourceId": ["222", "222", "222"],
            "sourceDesc": ["Example drug", "Example drug", "Example drug"],
            "destinationId": ["3001", "3002", "3003"],
            "typeDesc": [
                "Has specific active ingredient (attribute)",
                "Has precise active ingredient (attribute)",
                "Is a (attribute)",
            ],
        }
    )

    mapping = ep.tl.extract_snomed_ingredient_links(attr_concepts, snomed_drug="222")

    assert mapping.to_dict("records") == [
        {"snomed.drug.uk": "222", "snomed.ingredient": "3001"},
        {"snomed.drug.uk": "222", "snomed.ingredient": "3002"},
    ]


def test_extract_rxnav_ingredient_links_filters_in_and_adds_identity():
    payload = {
        "allRelatedGroup": {
            "conceptGroup": [
                {
                    "conceptProperties": [
                        {"rxcui": "5001", "tty": "IN"},
                        {"rxcui": "5002", "tty": "PIN"},
                    ]
                }
            ]
        }
    }

    mapping = ep.tl.extract_rxnav_ingredient_links(payload, rxcui="4000")

    assert mapping.to_dict("records") == [
        {"rxcui": "4000", "rxcui.ingredient": "5001"},
        {"rxcui": "4000", "rxcui.ingredient": "4000"},
    ]


def test_build_bnfcode_prodcode_map_splits_and_truncates_codes():
    product_df = pd.DataFrame({"prodcode": [10, 20], "bnfcode": ["010203/0405067", "999999"]})

    mapping = ep.tl.build_bnfcode_prodcode_map(product_df)

    assert mapping.to_dict("records") == [
        {"prodcode": 10, "bnfcode": "010203"},
        {"prodcode": 10, "bnfcode": "040506"},
        {"prodcode": 20, "bnfcode": "999999"},
    ]


def test_build_rxcui_medcode_map_applies_four_character_fallback():
    rxcui_readcodev2 = pd.DataFrame({"rxcui": ["100", "200"], "readcodev2": ["A1234", "B2345"]})
    medical = pd.DataFrame({"medcode": ["10", "20"], "readcode": ["A1234Z", "B234X"]})

    readcodev2_medcode = ep.tl.normalize_readcodev2_medcode_map(medical)
    mapping = ep.tl.build_rxcui_medcode_map(rxcui_readcodev2, readcodev2_medcode)

    assert mapping.to_dict("records") == [
        {"rxcui": "100", "medcode": "10"},
        {"rxcui": "200", "medcode": "20"},
    ]


def test_build_rxcui_prodcode_map_follows_original_merge_chain():
    rxcui_ingredient = pd.DataFrame(
        {"rxcui": ["100", "200"], "rxcui.ingredient": ["5001", "5002"]}
    )
    snomed_ingredient_rxcui = pd.DataFrame(
        {"snomed.ingredient": ["3001", "3002"], "rxcui": ["100", "200"]}
    )
    snomed_ingredient_uk = pd.DataFrame(
        {"snomed.drug.uk": ["7001", "7002"], "snomed.ingredient": ["3001", "3002"]}
    )
    snomed_bnfcode = pd.DataFrame(
        {"snomed.drug.uk": ["7001", "7002"], "bnfcode": ["111111", "222222"]}
    )
    bnfcode_prodcode = pd.DataFrame({"prodcode": [10, 20], "bnfcode": ["111111", "222222"]})

    mapping = ep.tl.build_rxcui_prodcode_map(
        rxcui_ingredient,
        snomed_ingredient_rxcui,
        snomed_ingredient_uk,
        snomed_bnfcode,
        bnfcode_prodcode,
    )

    assert mapping.to_dict("records") == [
        {"rxcui": "100", "prodcode": 10},
        {"rxcui": "200", "prodcode": 20},
    ]


def test_build_prodcode_medcode_map_returns_unique_pairs():
    rxcui_prodcode = pd.DataFrame({"rxcui": ["100", "100", "200"], "prodcode": [10, 10, 20]})
    rxcui_medcode = pd.DataFrame({"rxcui": ["100", "200", "200"], "medcode": ["m1", "m2", "m2"]})

    mapping = ep.tl.build_prodcode_medcode_map(rxcui_prodcode, rxcui_medcode)

    assert mapping.to_dict("records") == [
        {"prodcode": 10, "medcode": "m1"},
        {"prodcode": 20, "medcode": "m2"},
    ]


def test_build_disease_indication_map_runs_full_disease_side_chain():
    rxcui_mesh = pd.DataFrame({"rxcui": ["100", "200"], "mesh": ["M1", "M2"]})
    mesh_snomed_disease = pd.DataFrame({"mesh": ["M1", "M2"], "snomed.disease": ["S1", "S2"]})
    snomed_disease_readcodev3 = pd.DataFrame(
        {"snomed.disease": ["S1", "S2"], "readcodev3": ["V31", "V32"]}
    )
    readcodev3_readcodev2 = pd.DataFrame({"readcodev3": ["V31", "V32"], "readcodev2": ["A1234", "B2345"]})
    medical = pd.DataFrame({"medcode": ["10", "20"], "readcode": ["A1234Z", "B234X"]})

    mapping = ep.tl.build_disease_indication_map(
        rxcui_mesh,
        mesh_snomed_disease,
        snomed_disease_readcodev3,
        readcodev3_readcodev2,
        medical,
    )

    assert mapping.to_dict("records") == [
        {"rxcui": "100", "medcode": "10"},
        {"rxcui": "200", "medcode": "20"},
    ]


def test_build_indication_map_runs_end_to_end_workflow():
    rxcui_mesh = pd.DataFrame({"rxcui": ["100"], "mesh": ["M1"]})
    mesh_snomed_disease = pd.DataFrame({"mesh": ["M1"], "snomed.disease": ["S1"]})
    snomed_disease_readcodev3 = pd.DataFrame({"snomed.disease": ["S1"], "readcodev3": ["V31"]})
    readcodev3_readcodev2 = pd.DataFrame({"readcodev3": ["V31"], "readcodev2": ["A1234"]})
    medical = pd.DataFrame({"medcode": ["10"], "readcode": ["A1234Z"]})
    rxcui_ingredient = pd.DataFrame({"rxcui": ["100"], "rxcui.ingredient": ["5001"]})
    snomed_ingredient_rxcui = pd.DataFrame({"snomed.ingredient": ["3001"], "rxcui": ["100"]})
    snomed_ingredient_uk = pd.DataFrame({"snomed.drug.uk": ["7001"], "snomed.ingredient": ["3001"]})
    snomed_bnfcode = pd.DataFrame({"snomed.drug.uk": ["7001"], "bnfcode": ["111111"]})
    product_df = pd.DataFrame({"prodcode": [10], "bnfcode": ["111111/222222"]})

    mapping = ep.tl.build_indication_map(
        rxcui_mesh,
        mesh_snomed_disease,
        snomed_disease_readcodev3,
        readcodev3_readcodev2,
        medical,
        rxcui_ingredient,
        snomed_ingredient_rxcui,
        snomed_ingredient_uk,
        snomed_bnfcode,
        product_df,
    )

    assert mapping.to_dict("records") == [{"prodcode": 10, "medcode": "10"}]


def test_build_rxcui_prodcode_map_requires_expected_columns():
    with pytest.raises(KeyError, match="RxCUI ingredient mapping"):
        ep.tl.build_rxcui_prodcode_map(
            pd.DataFrame({"rxcui": ["100"]}),
            pd.DataFrame({"snomed.ingredient": ["3001"], "rxcui": ["100"]}),
            pd.DataFrame({"snomed.drug.uk": ["7001"], "snomed.ingredient": ["3001"]}),
            pd.DataFrame({"snomed.drug.uk": ["7001"], "bnfcode": ["111111"]}),
            pd.DataFrame({"prodcode": [10], "bnfcode": ["111111"]}),
        )
