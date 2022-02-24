import difflib
from typing import Dict, List, Optional, Sequence, Tuple

from anndata import AnnData
from pyhpo import HPOSet, HPOTerm, Ontology


class HPOMapper:
    """Wrapper class for PyHPO. Documentation: https://centogene.github.io/pyhpo/index.html"""

    ontology_values: Optional[Dict[str, List[HPOTerm]]] = None

    def __init__(self, ontology=None):
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

    @classmethod
    def _load_full_ontology(cls) -> None:
        """Loads the full ontology with synonyms.

        The full ontology is useful to match queries against it.
        """
        cls.ontology_values = {}

        def name_to_term(name: str, hpo_term: HPOTerm):
            if name not in cls.ontology_values:
                cls.ontology_values[name] = [hpo_term]

        for term in Ontology:
            name_to_term(term.name, term)
            for syn in term.synonym:
                name_to_term(syn, term)

    @staticmethod
    def map_to_hpo(
        adata: AnnData, obs_key: str = None, key: str = "hpo_terms", strict: bool = False, copy: bool = False
    ) -> Optional[AnnData]:
        """Maps a single column of an AnnData object into HPO terms.

        Adds the determined HPO terms as a new obs column.

        Args:
            adata: AnnData object containing a column of non-HPO terms.
            obs_key: Key in adata.obs which contains the non-HPO terms.
            key: Name of the new obs column to add.
            strict: Returns only the closest HPO term match at the cost of a much higher runtime.
            copy: Whether to return a copy.

        Returns:
            An AnnData object containing new columns named var_name_hpo with HPO terms.
        """
        if copy:
            adata = adata.copy()

        # Implemented after https://github.com/Centogene/pyhpo/issues/2#issuecomment-896510747
        # Gets all terms of the ontology and their synonyms. Then finds the closest match.
        if strict:
            if HPOMapper.ontology_values is None:
                HPOMapper._load_full_ontology()

                def _get_closest_HPOterm_match(name) -> Sequence:
                    closest_matches = difflib.get_close_matches(name, HPOMapper.ontology_values, 1)
                    closest_match: str = closest_matches[0].strip()  # type: ignore
                    return closest_match

                adata.obs[key] = adata.obs[obs_key].apply(_get_closest_HPOterm_match)
        else:

            def _get_all_HPOterm_matches(name) -> Sequence:
                result = []
                for term in Ontology.search(name):
                    result.append(term.name)
                return result

            adata.obs[key] = adata.obs[obs_key].apply(_get_all_HPOterm_matches)

        return adata
