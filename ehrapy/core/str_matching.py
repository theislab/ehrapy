from __future__ import annotations

from numpy import ndarray


class StrMatcher:
    def __init__(self, references: list[str] | ndarray):
        self.reference_bigrams = self.calculate_bigrams_list_str(references)

    @classmethod
    def calculate_bigrams_single_str(cls, string: str) -> list[str]:
        """Take a string and return a list of bigrams."""
        string_lower = string.lower()

        return [string_lower[i : i + 2] for i in range(len(string_lower) - 1)]

    @classmethod
    def calculate_bigrams_list_str(cls, all_strings: list[str] | ndarray) -> dict[str, tuple[set[str], int]]:
        """Calculates all bigrams for a list of strings.

        Args:
            all_strings: List of strings to calculate the bigrams for.

        Returns:
            A dictionary associating input strings against sets of bigrams and their amounts.
        """
        # could be optimized for ndarrays
        all_bigrams: dict[str, tuple[set[str], int]] = {}
        for string in all_strings:
            bigrams = cls.calculate_bigrams_single_str(string)
            all_bigrams[string] = (set(bigrams), len(bigrams))

        return all_bigrams

    def best_match(self, query: str, threshold: float = 0.2) -> tuple[float, str] | None:
        """Calculates the best match of a given query against the references.

        Args:
            query: The query to check against the references.
            threshold: Minimum required matching confidence score of the bigrams.
                       0 = low requirements, 1 = high requirements.

        Returns:
            The best matching string and a corresponding confidence score.
        """
        pairs1 = self.calculate_bigrams_single_str(query)
        pairs1_len = len(pairs1)

        score, str2 = max(
            (2.0 * sum(x in pairs2 for x in pairs1) / (pairs1_len + pairs2_len), str2)
            for str2, (pairs2, pairs2_len) in self.reference_bigrams.items()
        )
        if score < threshold:
            str2 = None

        return score, str2

    def first_match(self, query: str, threshold: float = 0.2) -> tuple[float, str] | None:
        """Calculates the first match of a given query against the references.

        Args:
            query: The query to check against the references.
            threshold: Minimum required matching confidence score of the bigrams.
                       0 = low requirements, 1 = high requirements.

        Returns:
            The first matching string and a corresponding confidence score.
        """
        pairs1 = self.calculate_bigrams_single_str(query)
        pairs1_len = len(pairs1)

        for str2, (pairs2, pairs2_len) in self.reference_bigrams.items():
            score = 2.0 * sum(x in pairs2 for x in pairs1) / (pairs1_len + pairs2_len)
            if score >= threshold:
                return score, str2

        return None

    def get_closest_matches(
        self, queries: list[str], first_matches: bool = False, threshold: float = 0.2
    ) -> list[tuple[float, str] | None | tuple[float, str | None]]:
        """Determines the closest matches for a list of queries.

        Args:
            queries: A list of queries to get matches for.
            first_matches: Whether to return the first matches. Else, gets the best matches (default: false)
            threshold: Minimum required matching confidence score of the bigrams.
                       0 = low requirements, 1 = high requirements.

        Returns:
            List of matches and their associated matching score.
        """
        matches = []

        for query in queries:
            if first_matches:
                matches.append(self.first_match(query, threshold=threshold))
            else:
                matches.append(self.best_match(query, threshold=threshold))

        return matches
