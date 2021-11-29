from math import isclose
from pathlib import Path

from pyhpo import HPOSet, HPOTerm, Ontology

from ehrapy.api.tools import HPO

CURRENT_DIR = Path(__file__).parent
_TEST_PATH = f"{CURRENT_DIR}/test_data_encode"


class TestHPO:
    def setup_method(self):
        _ = Ontology()

    def test_patient_hpo_similarity(self):
        patient_1 = HPOSet.from_queries(["HP:0002943", "HP:0008458", "HP:0100884", "HP:0002944", "HP:0002751"])
        patient_2 = HPOSet.from_queries(["HP:0002650", "HP:0010674", "HP:0000925", "HP:0009121"])
        similarity = HPO.patient_hpo_similarity(patient_1, patient_2)

        assert isclose(similarity, 0.7583849517696274)

    def test_hpo_term_similarity(self):
        term_1 = Ontology.get_hpo_object("Scoliosis")
        term_2 = Ontology.get_hpo_object("Abnormal axial skeleton morphology")
        path = HPO.hpo_term_similarity(term_1, term_2)

        assert path == (
            3,
            (
                HPOTerm(
                    id="HP:0002650",
                    name="Scoliosis",
                    is_a=["HP:0010674 ! Abnormality of the curvature of the vertebral column"],
                ),
                HPOTerm(
                    id="HP:0010674",
                    name="Abnormality of the curvature of the vertebral column",
                    is_a=["HP:0000925 ! Abnormality of the vertebral column"],
                ),
                HPOTerm(
                    id="HP:0000925",
                    name="Abnormality of the vertebral column",
                    is_a=["HP:0009121 ! Abnormal axial skeleton morphology"],
                ),
                HPOTerm(
                    id="HP:0009121",
                    name="Abnormal axial skeleton morphology",
                    is_a=["HP:0011842 ! Abnormality of skeletal morphology"],
                ),
            ),
            3,
            0,
        )
