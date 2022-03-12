from ehrapy.core.str_matching import StrMatcher


class TestStrMatcher:
    def setup_method(self):
        reference_lab_measurement_names = [
            "Activated partial thromboplastin time (APTT)",
            "α-Aminobutyric acid",
            "Apolipoprotein A",
            "Chloramphenicol (therapeutic)",
            "Chlordiazepoxide (therapeutic)",
        ]

        self.str_matcher = StrMatcher(reference_lab_measurement_names)

        self.queries = [
            "APTT",
            "aminubutyric acid",
            "apolipoprotein",
            "chloramphenicol",
            "chlordiazepoxide",
        ]

    def test_calculate_bigrams_single_str(self):
        expected_bigrams = ["so", "om", "me", "es", "st", "tr", "ri", "in", "ng"]
        calculated_bigrams = StrMatcher.calculate_bigrams_single_str("somestring")

        assert expected_bigrams == calculated_bigrams

    def test_calculate_bigrams_list_str(self):
        expected_bigrams = {"in", "me", "ng", "st", "es", "so", "ri", "tr", "om"}
        expected_number_bigrams = 9

        calculated_bigrams_counts = StrMatcher.calculate_bigrams_list_str(["somestring"])
        calculated_bigrams = calculated_bigrams_counts["somestring"][0]
        calculated_bigrams_number = calculated_bigrams_counts["somestring"][1]

        assert expected_bigrams == calculated_bigrams
        assert expected_number_bigrams == calculated_bigrams_number

    def test_best_match(self):
        expected_match = (0.6818181818181818, "Chlordiazepoxide (therapeutic)")
        calculated_best_match = self.str_matcher.best_match(query=self.queries[-1])

        assert expected_match == calculated_best_match

    def test_first_match(self):
        expected_match = (0.6818181818181818, "Chlordiazepoxide (therapeutic)")
        calculated_best_match = self.str_matcher.first_match(query=self.queries[-1])

        assert expected_match == calculated_best_match

    def test_get_closest_matches(self):
        expected_match = [
            (0.13043478260869565, None),
            (0.8235294117647058, "α-Aminobutyric acid"),
            (0.9285714285714286, "Apolipoprotein A"),
            (0.6666666666666666, "Chloramphenicol (therapeutic)"),
            (0.6818181818181818, "Chlordiazepoxide (therapeutic)"),
        ]
        calculated_matches = self.str_matcher.get_closest_matches(self.queries)

        assert expected_match == calculated_matches
