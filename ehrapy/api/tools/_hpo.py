from typing import List, Optional, Tuple, Union

from anndata import AnnData
from pyhpo import HPOSet, HPOTerm, Ontology


class HPO:
    """Wrapper class for PyHPO. Documentation: https://centogene.github.io/pyhpo/index.html"""

    def __init__(self, ontology: Ontology = None):
        self.ontology = ontology if ontology else Ontology()

    @staticmethod
    def patient_hpo_similarity(patient_1_set: HPOSet, patient_2_set: HPOSet) -> float:
        """Determines the similarity between two sets of Human Phenotype Ontologies.

        Args:
            patient_1_set: First set of HPO terms to compare.
            patient_2_set: Second set of HPO terms to compare against.

        Returns:
            A similarity score between 0 (no similarity) and 1 (perfect similarity).
        """
        return patient_1_set.similarity(patient_2_set)

    @staticmethod
    def hpo_term_similarity(term_1: HPOTerm, term_2: HPOTerm) -> Tuple[int, Tuple[HPOTerm, ...], int, int]:
        """Calculates the similarity between two single HPO terms.

        Args:
            term_1: First HPO term.
            term_2: Second HPO term to compare against.

        Returns:
            The length of the shortest path between the terms,
            a tuple of the passed terms,
            the number of steps of term 1 to the common parent
            the number of steps of term 2 to the common parent
        """
        return term_1.path_to_other(term_2)

    @staticmethod
    def map_to_hpo(adata: AnnData, var_names: Union[str, List[str]], copy: bool = False) -> Optional[AnnData]:
        """Maps a single column of an AnnData object into HPO terms.

        Args:
            adata: AnnData object containing a column of non-HPO terms.
            var_names: The column(s) containing non-HPO terms to map.
            copy: Whether to return a copy.

        Returns:
            An AnnData object containing new columns named var_name_hpo with HPO terms.
        """
        if copy:
            adata = adata.copy()
        if not isinstance(var_names, List):
            var_names = list(var_names)
        return None
