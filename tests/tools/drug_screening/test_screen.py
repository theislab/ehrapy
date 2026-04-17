import pandas as pd
import pytest

import ehrapy as ep


def test_compute_ndd_from_text_parses_common_prescription_patterns():
    dosage = pd.DataFrame(
        {
            "text": [
                "Take 1 tablet once daily",
                "Take 1-2 tablets 2 times a day",
                "Take 1 tablet every other day",
                "Take 2 capsules every 12 hours as required",
            ]
        }
    )

    parsed = ep.tl.compute_ndd_from_text(dosage)

    assert list(parsed["ndd"]) == [1.0, 3.0, 0.5, 4.0]
    assert parsed.loc[3, "optional"] == "as required"


def test_compute_ndd_from_text_handles_ranges_abbreviations_and_non_daily_intervals():
    dosage = pd.DataFrame(
        {
            "text": [
                "Take 2-4 tablets up to 3 times per day",
                "Take one tablet bd",
                "Take 1 capsule every 4-6 hours",
                "Take 1 tablet weekly",
            ]
        }
    )

    parsed_min = ep.tl.compute_ndd_from_text(dosage, dose_fn="min", freq_fn="min", interval_fn="mean")
    parsed_max = ep.tl.compute_ndd_from_text(dosage, dose_fn="max", freq_fn="max", interval_fn="mean")

    assert parsed_min.loc[0, "ndd"] == 2.0
    assert parsed_max.loc[0, "ndd"] == 12.0
    assert parsed_min.loc[1, "ndd"] == 2.0
    assert parsed_min.loc[2, "ndd"] == 4.0
    assert parsed_max.loc[2, "ndd"] == 6.0
    assert parsed_min.loc[3, "ndd"] == pytest.approx(1.0 / 7.0)


def test_compute_ndd_from_text_matches_drugprepr_example_patterns():
    dosage = pd.DataFrame(
        {
            "text": ["TAKE 1 OR 2 4 TIMES/DAY"] * 6
            + ["TAKE 1-2 THREE TIMES A DAY"] * 8
            + [""] * 4
        }
    )

    parsed_min = ep.tl.compute_ndd_from_text(dosage, dose_fn="min", freq_fn="min", interval_fn="mean")
    parsed_min_max = ep.tl.compute_ndd_from_text(dosage, dose_fn="min", freq_fn="max", interval_fn="mean")
    parsed_max = ep.tl.compute_ndd_from_text(dosage, dose_fn="max", freq_fn="max", interval_fn="mean")

    assert parsed_min.loc[0, "ndd"] == 4.0
    assert parsed_min.loc[6, "ndd"] == 3.0
    assert pd.isna(parsed_min.loc[14, "ndd"])
    assert parsed_min["ndd"].mean() == pytest.approx(3.4285714286)
    assert parsed_min_max["ndd"].mean() == pytest.approx(3.4285714286)
    assert parsed_max["ndd"].mean() == pytest.approx(6.8571428571)


def test_prepare_prescriptions_from_therapy_infers_duration_from_lookup_and_filters_observation_window():
    therapy = pd.DataFrame(
        {
            "patid": [1, 1, 2],
            "eventdate": ["2020-01-10", "2018-01-01", "2020-01-11"],
            "drugsubstance": ["drug_a", "drug_a", "drug_b"],
            "dosageid": [100, 100, 200],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2],
            "frd": ["2019-01-01", "2019-01-01"],
            "tod": ["2021-01-01", "2021-01-01"],
            "lcd": ["2021-01-01", "2021-01-01"],
            "deathdate": ["2021-01-01", "2021-01-01"],
        }
    )
    dosage_lookup = pd.DataFrame({"dosageid": [100, 200], "dose_duration": [7, 14]})

    prescriptions = ep.tl.prepare_prescriptions_from_therapy(therapy, patients=patients, dosage_lookup=dosage_lookup)

    assert len(prescriptions) == 2
    assert list(prescriptions["duration"]) == [7.0, 14.0]
    assert list(prescriptions["drug"]) == ["drug_a", "drug_b"]


def test_prepare_prescriptions_from_therapy_can_compute_ndd_from_dosage_text():
    therapy = pd.DataFrame(
        {
            "patid": [1],
            "eventdate": ["2020-01-10"],
            "drugsubstance": ["drug_a"],
            "qty": [14.0],
            "dosageid": [100],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1],
            "frd": ["2019-01-01"],
            "tod": ["2021-01-01"],
            "lcd": ["2021-01-01"],
            "deathdate": ["2021-01-01"],
        }
    )
    dosage_lookup = pd.DataFrame({"dosageid": [100], "text": ["Take 1 tablet twice daily"]})

    prescriptions = ep.tl.prepare_prescriptions_from_therapy(therapy, patients=patients, dosage_lookup=dosage_lookup)

    assert prescriptions.loc[0, "ndd"] == 2.0
    assert prescriptions.loc[0, "duration"] == 7.0


def test_prepare_prescriptions_from_therapy_supports_text_summary_rules():
    therapy = pd.DataFrame(
        {
            "patid": [1],
            "eventdate": ["2020-01-10"],
            "drugsubstance": ["drug_a"],
            "qty": [24.0],
            "dosageid": [100],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1],
            "frd": ["2019-01-01"],
            "tod": ["2021-01-01"],
            "lcd": ["2021-01-01"],
            "deathdate": ["2021-01-01"],
        }
    )
    dosage_lookup = pd.DataFrame({"dosageid": [100], "text": ["Take 2-4 tablets up to 3 times per day"]})

    prescriptions = ep.tl.prepare_prescriptions_from_therapy(
        therapy,
        patients=patients,
        dosage_lookup=dosage_lookup,
        dose_fn="max",
        freq_fn="max",
    )

    assert prescriptions.loc[0, "ndd"] == 12.0
    assert prescriptions.loc[0, "duration"] == 2.0


def test_apply_drugprepr_decisions_matches_original_policy_shape():
    therapy = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "prodcode": [10, 10, 10],
            "qty": [10.0, None, 10000.0],
            "ndd": [2.0, None, 2.0],
        }
    )
    min_max_lookup = pd.DataFrame(
        {
            "prodcode": [10],
            "min_qty": [1.0],
            "max_qty": [5000.0],
            "min_ndd": [1.0],
            "max_ndd": [50.0],
        }
    )

    cleaned = ep.tl.apply_drugprepr_decisions(
        therapy,
        min_max_lookup=min_max_lookup,
        decisions=("c3", "b3", "c3", "b3", "c_12", "c", "d"),
    )

    assert list(cleaned["qty"]) == [10.0, 2507.5, 5005.0]
    assert list(cleaned["ndd"]) == [2.0, 2.0, 2.0]
    assert list(cleaned["duration"]) == [5.0, 365.0, 365.0]


def test_apply_drugprepr_decisions_requires_seven_decisions():
    therapy = pd.DataFrame({"patid": [1], "prodcode": [10], "qty": [10.0], "ndd": [2.0]})
    min_max_lookup = pd.DataFrame(
        {"prodcode": [10], "min_qty": [1.0], "max_qty": [5000.0], "min_ndd": [1.0], "max_ndd": [50.0]}
    )

    with pytest.raises(ValueError, match="At least seven decisions"):
        ep.tl.apply_drugprepr_decisions(therapy, min_max_lookup=min_max_lookup, decisions=("a", "a", "a"))


def test_apply_drugprepr_decisions_requires_min_max_columns():
    therapy = pd.DataFrame({"patid": [1], "prodcode": [10], "qty": [10.0], "ndd": [2.0]})
    min_max_lookup = pd.DataFrame({"prodcode": [10], "min_qty": [1.0], "max_qty": [5000.0]})

    with pytest.raises(KeyError, match="Missing required min/max columns"):
        ep.tl.apply_drugprepr_decisions(
            therapy,
            min_max_lookup=min_max_lookup,
            decisions=("a", "a", "a", "a", "a", "c", "a"),
        )


def test_apply_drugprepr_decisions_requires_duration_source_inputs():
    therapy = pd.DataFrame({"patid": [1], "prodcode": [10], "qty": [10.0], "ndd": [2.0]})
    min_max_lookup = pd.DataFrame(
        {"prodcode": [10], "min_qty": [1.0], "max_qty": [5000.0], "min_ndd": [1.0], "max_ndd": [50.0]}
    )

    with pytest.raises(KeyError, match="numdays is required"):
        ep.tl.apply_drugprepr_decisions(
            therapy,
            min_max_lookup=min_max_lookup,
            decisions=("a", "a", "a", "a", "a", "a", "a"),
        )


def test_build_exposure_episodes_from_prescriptions_aggregates_and_merges():
    prescriptions = pd.DataFrame(
        {
            "patid": [1, 1, 1, 2],
            "drug": ["drug_a", "drug_a", "drug_a", "drug_a"],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-05", "2020-02-01"],
            "duration": [2, 3, 5, 4],
        }
    )

    episodes = ep.tl.build_exposure_episodes_from_prescriptions(prescriptions, gap_days=0, keep_first_only=False)

    assert len(episodes) == 2
    first = episodes[episodes["patid"] == 1].iloc[0]
    assert first["start_date"] == pd.Timestamp("2020-01-01")
    assert first["stop_date"] == pd.Timestamp("2020-01-10")
    assert first["episode_duration"] == 9


def test_prepare_exposure_windows_caps_followup_and_computes_age():
    exposures = pd.DataFrame(
        {
            "patid": [1, 1],
            "drug": ["drug_a", "drug_a"],
            "start_date": ["2020-01-10", "2020-01-15"],
            "stop_date": ["2020-01-20", "2020-01-25"],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1],
            "dob": ["1980-01-01"],
            "frd": ["2019-01-01"],
            "tod": ["2021-01-01"],
            "lcd": ["2021-01-01"],
            "deathdate": ["2021-01-01"],
        }
    )

    prepared = ep.tl.prepare_exposure_windows(exposures, patients, gap_days=7, fixed_followup_days=5)

    assert len(prepared) == 1
    assert prepared.loc[0, "exposure_length"] == 5
    assert prepared.loc[0, "unexposed_start_date"] == pd.Timestamp("2020-01-05")
    assert prepared.loc[0, "stop_date"] == pd.Timestamp("2020-01-15")
    assert prepared.loc[0, "prescription_age"] > 39


def test_screen_drugs_returns_substance_level_summary():
    prescriptions = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "drug": ["drug_a", "drug_a", "drug_a"],
            "start_date": ["2020-01-10", "2020-01-10", "2020-01-10"],
            "duration": [5, 5, 5],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "dob": ["1970-01-01", "1970-01-01", "1970-01-01"],
            "frd": ["2019-01-01", "2019-01-01", "2019-01-01"],
            "tod": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "lcd": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "deathdate": ["2021-01-01", "2021-01-01", "2021-01-01"],
        }
    )
    episodes = ep.tl.build_exposure_episodes_from_prescriptions(prescriptions)
    exposure_windows = ep.tl.prepare_exposure_windows(episodes, patients)
    events = pd.DataFrame(
        {
            "patid": [1, 2, 3, 1],
            "disease": ["disease_x", "disease_x", "disease_x", "disease_y"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06", "2020-01-13", "2020-01-12"]),
        }
    )

    result = ep.tl.screen_drugs(exposure_windows, events, min_total_events=2)

    assert len(result) == 2
    disease_x = result[result["disease"] == "disease_x"].iloc[0]
    assert disease_x["drug"] == "drug_a"
    assert disease_x["age.group"] == "40-60"
    assert disease_x["N.exposed"] == 3
    assert disease_x["N.disease.C.during.exposed"] == 2
    assert disease_x["N.disease.B.during.unexposed"] == 1
    assert disease_x["sum.exposed.person.time"] == 15


def test_screen_drugs_skips_known_pairs():
    prescriptions = pd.DataFrame(
        {
            "patid": [1, 2],
            "drug": ["drug_a", "drug_a"],
            "start_date": ["2020-01-10", "2020-01-10"],
            "duration": [5, 5],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2],
            "dob": ["1970-01-01", "1970-01-01"],
            "frd": ["2019-01-01", "2019-01-01"],
            "tod": ["2021-01-01", "2021-01-01"],
            "lcd": ["2021-01-01", "2021-01-01"],
            "deathdate": ["2021-01-01", "2021-01-01"],
        }
    )
    exposure_windows = ep.tl.prepare_exposure_windows(ep.tl.build_exposure_episodes_from_prescriptions(prescriptions), patients)
    events = pd.DataFrame(
        {
            "patid": [1, 2],
            "disease": ["disease_x", "disease_x"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-12"]),
        }
    )
    known_pairs = pd.DataFrame({"drug": ["drug_a"], "disease": ["disease_x"]})

    result = ep.tl.screen_drugs(exposure_windows, events, min_total_events=1, known_drug_disease_pairs=known_pairs)

    assert result.empty


def test_screen_substance_cohort_runs_end_to_end_from_prescriptions():
    prescriptions = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "drug": ["drug_a", "drug_a", "drug_a"],
            "start_date": ["2020-01-10", "2020-01-10", "2020-01-10"],
            "duration": [5, 5, 5],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "dob": ["1970-01-01", "1970-01-01", "1970-01-01"],
            "frd": ["2019-01-01", "2019-01-01", "2019-01-01"],
            "tod": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "lcd": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "deathdate": ["2021-01-01", "2021-01-01", "2021-01-01"],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "disease": ["disease_x", "disease_x", "disease_x"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06", "2020-01-13"]),
        }
    )

    result = ep.tl.screen_substance_cohort(prescriptions, patients, events, min_total_events=2)

    assert len(result) == 2
    assert set(result["age.group"]) == {"40-60", "all"}


def test_screen_substance_therapy_runs_from_raw_therapy_records():
    therapy = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "eventdate": ["2020-01-10", "2020-01-10", "2020-01-10"],
            "drugsubstance": ["drug_a", "drug_a", "drug_a"],
            "duration": [5, 5, 5],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "dob": ["1970-01-01", "1970-01-01", "1970-01-01"],
            "frd": ["2019-01-01", "2019-01-01", "2019-01-01"],
            "tod": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "lcd": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "deathdate": ["2021-01-01", "2021-01-01", "2021-01-01"],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "disease": ["disease_x", "disease_x", "disease_x"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06", "2020-01-13"]),
        }
    )

    result = ep.tl.screen_substance_therapy(therapy, patients, events, min_total_events=2)

    assert len(result) == 2
    assert set(result["age.group"]) == {"40-60", "all"}


def test_screen_substance_therapy_supports_named_followup_workflows():
    therapy = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "eventdate": ["2020-01-10", "2020-01-10", "2020-01-10"],
            "drugsubstance": ["drug_a", "drug_a", "drug_a"],
            "duration": [5, 5, 5],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "dob": ["1970-01-01", "1970-01-01", "1970-01-01"],
            "frd": ["2018-01-01", "2018-01-01", "2018-01-01"],
            "tod": ["2023-01-01", "2023-01-01", "2023-01-01"],
            "lcd": ["2023-01-01", "2023-01-01", "2023-01-01"],
            "deathdate": ["2023-01-01", "2023-01-01", "2023-01-01"],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "disease": ["disease_x", "disease_x", "disease_x"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06", "2020-01-13"]),
        }
    )

    actual = ep.tl.screen_substance_therapy(therapy, patients, events, workflow="actual", min_total_events=2)
    thirty_days = ep.tl.screen_substance_therapy(therapy, patients, events, workflow="30days", min_total_events=2)

    actual_all = actual[(actual["disease"] == "disease_x") & (actual["age.group"] == "all")].iloc[0]
    thirty_all = thirty_days[(thirty_days["disease"] == "disease_x") & (thirty_days["age.group"] == "all")].iloc[0]

    assert actual_all["sum.exposed.person.time"] == 15
    assert thirty_all["sum.exposed.person.time"] == 90
    assert thirty_all["exposed.mean"] == 30


def test_screen_substance_therapy_supports_drugprepr_style_cleaning():
    therapy = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "eventdate": ["2020-01-10", "2020-01-10", "2020-01-10"],
            "drugsubstance": ["drug_a", "drug_a", "drug_a"],
            "prodcode": [10, 10, 10],
            "qty": [10.0, None, 10000.0],
            "ndd": [2.0, None, 2.0],
        }
    )
    min_max_lookup = pd.DataFrame(
        {
            "prodcode": [10],
            "min_qty": [1.0],
            "max_qty": [5000.0],
            "min_ndd": [1.0],
            "max_ndd": [50.0],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "dob": ["1970-01-01", "1970-01-01", "1970-01-01"],
            "frd": ["2019-01-01", "2019-01-01", "2019-01-01"],
            "tod": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "lcd": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "deathdate": ["2021-01-01", "2021-01-01", "2021-01-01"],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "disease": ["disease_x", "disease_x", "disease_x"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06", "2020-01-13"]),
        }
    )

    result = ep.tl.screen_substance_therapy(
        therapy,
        patients,
        events,
        min_max_lookup=min_max_lookup,
        drugprepr_decisions=("c3", "b3", "c3", "b3", "c_12", "c", "d"),
        min_total_events=2,
    )

    assert len(result) == 2


def test_screen_substance_therapy_rejects_conflicting_followup_configuration():
    therapy = pd.DataFrame(
        {
            "patid": [1, 2],
            "eventdate": ["2020-01-10", "2020-01-10"],
            "drugsubstance": ["drug_a", "drug_a"],
            "duration": [5, 5],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2],
            "dob": ["1970-01-01", "1970-01-01"],
            "frd": ["2018-01-01", "2018-01-01"],
            "tod": ["2023-01-01", "2023-01-01"],
            "lcd": ["2023-01-01", "2023-01-01"],
            "deathdate": ["2023-01-01", "2023-01-01"],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1, 2],
            "disease": ["disease_x", "disease_x"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06"]),
        }
    )

    with pytest.raises(ValueError, match="does not match workflow"):
        ep.tl.screen_substance_therapy(
            therapy,
            patients,
            events,
            workflow="30days",
            fixed_followup_days=365,
            min_total_events=1,
        )


def test_prepare_prescriptions_with_drugprepr_supports_decisions_8_to_10():
    therapy = pd.DataFrame(
        {
            "patid": [1, 1, 1, 1],
            "prodcode": [10, 10, 10, 10],
            "eventdate": ["2020-01-01", "2020-01-01", "2020-01-05", "2020-01-20"],
            "qty": [10.0, 4.0, 2.0, 1.0],
            "ndd": [1.0, 1.0, 1.0, 1.0],
        }
    )
    min_max_lookup = pd.DataFrame(
        {
            "prodcode": [10],
            "min_qty": [1.0],
            "max_qty": [5000.0],
            "min_ndd": [1.0],
            "max_ndd": [50.0],
        }
    )

    prepared = ep.tl.prepare_prescriptions_with_drugprepr(
        therapy,
        min_max_lookup=min_max_lookup,
        decisions=("a", "a", "a", "a", "a", "c", "a", "e", "b", "b_15"),
    )

    assert list(prepared["start_date"]) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-15"),
        pd.Timestamp("2020-01-20"),
    ]
    assert list(prepared["stop_date"]) == [
        pd.Timestamp("2020-01-15"),
        pd.Timestamp("2020-01-20"),
        pd.Timestamp("2020-01-21"),
    ]
    assert list(prepared["duration"]) == [14.0, 5.0, 1.0]


def test_prepare_prescriptions_with_drugprepr_requires_ten_decisions():
    therapy = pd.DataFrame({"patid": [1], "prodcode": [10], "eventdate": ["2020-01-01"], "qty": [10.0], "ndd": [1.0]})
    min_max_lookup = pd.DataFrame(
        {"prodcode": [10], "min_qty": [1.0], "max_qty": [5000.0], "min_ndd": [1.0], "max_ndd": [50.0]}
    )

    with pytest.raises(ValueError, match="Ten drugprepr decisions"):
        ep.tl.prepare_prescriptions_with_drugprepr(
            therapy,
            min_max_lookup=min_max_lookup,
            decisions=("a", "a", "a", "a", "a", "c", "a"),
        )


def test_prepare_prescriptions_from_therapy_uses_full_drugprepr_when_ten_decisions_are_provided():
    therapy = pd.DataFrame(
        {
            "patid": [1, 1, 1],
            "eventdate": ["2020-01-01", "2020-01-01", "2020-01-05"],
            "drugsubstance": ["drug_a", "drug_a", "drug_a"],
            "prodcode": [10, 10, 10],
            "qty": [10.0, 4.0, 2.0],
            "ndd": [1.0, 1.0, 1.0],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1],
            "frd": ["2019-01-01"],
            "tod": ["2021-01-01"],
            "lcd": ["2021-01-01"],
            "deathdate": ["2021-01-01"],
        }
    )
    min_max_lookup = pd.DataFrame(
        {
            "prodcode": [10],
            "min_qty": [1.0],
            "max_qty": [5000.0],
            "min_ndd": [1.0],
            "max_ndd": [50.0],
        }
    )

    prescriptions = ep.tl.prepare_prescriptions_from_therapy(
        therapy,
        patients=patients,
        min_max_lookup=min_max_lookup,
        drugprepr_decisions=("a", "a", "a", "a", "a", "c", "a", "e", "b", "a"),
    )

    assert list(prescriptions["start_date"]) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-15"),
    ]
    assert list(prescriptions["duration"]) == [14.0, 2.0]


def test_prepare_prescriptions_with_drugprepr_can_compute_ndd_from_text():
    therapy = pd.DataFrame(
        {
            "patid": [1],
            "prodcode": [10],
            "eventdate": ["2020-01-01"],
            "qty": [14.0],
            "dosageid": [100],
        }
    )
    dosage_lookup = pd.DataFrame({"dosageid": [100], "text": ["Take 1 tablet twice daily"]})
    min_max_lookup = pd.DataFrame(
        {
            "prodcode": [10],
            "min_qty": [1.0],
            "max_qty": [5000.0],
            "min_ndd": [1.0],
            "max_ndd": [50.0],
        }
    )

    prepared = ep.tl.prepare_prescriptions_with_drugprepr(
        therapy,
        dosage_lookup=dosage_lookup,
        min_max_lookup=min_max_lookup,
        decisions=("a", "a", "a", "a", "a", "c", "a", "a", "a", "a"),
    )

    assert prepared.loc[0, "ndd"] == 2.0
    assert prepared.loc[0, "duration"] == 7.0


def test_disambiguate_same_day_prescriptions_supports_shortest_and_longest():
    prescriptions = pd.DataFrame(
        {
            "patid": [1, 1, 1],
            "prodcode": [10, 10, 10],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-10"],
            "duration": [10.0, 4.0, 3.0],
        }
    )

    shortest = ep.tl.disambiguate_same_day_prescriptions(prescriptions, decision="c")
    longest = ep.tl.disambiguate_same_day_prescriptions(prescriptions, decision="d")

    assert list(shortest["duration"]) == [4.0, 3.0]
    assert list(longest["duration"]) == [10.0, 3.0]


def test_disambiguate_same_day_prescriptions_rejects_unknown_decision():
    prescriptions = pd.DataFrame({"patid": [1], "prodcode": [10], "start_date": ["2020-01-01"], "duration": [10.0]})

    with pytest.raises(NotImplementedError, match="Unsupported same-day clash decision"):
        ep.tl.disambiguate_same_day_prescriptions(prescriptions, decision="z")


def test_resolve_overlapping_prescriptions_shifts_cascading_overlaps():
    prescriptions = pd.DataFrame(
        {
            "patid": [1, 1, 1],
            "prodcode": [10, 10, 10],
            "start_date": ["2020-01-01", "2020-01-03", "2020-01-04"],
            "stop_date": ["2020-01-06", "2020-01-08", "2020-01-09"],
        }
    )

    shifted = ep.tl.resolve_overlapping_prescriptions(prescriptions, decision="b")

    assert list(shifted["start_date"]) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-06"),
        pd.Timestamp("2020-01-11"),
    ]
    assert list(shifted["stop_date"]) == [
        pd.Timestamp("2020-01-06"),
        pd.Timestamp("2020-01-11"),
        pd.Timestamp("2020-01-16"),
    ]


def test_resolve_overlapping_prescriptions_rejects_unknown_decision():
    prescriptions = pd.DataFrame(
        {"patid": [1], "prodcode": [10], "start_date": ["2020-01-01"], "stop_date": ["2020-01-06"]}
    )

    with pytest.raises(NotImplementedError, match="Unsupported overlap decision"):
        ep.tl.resolve_overlapping_prescriptions(prescriptions, decision="z")


def test_close_small_gaps_in_prescriptions_respects_strict_gap_boundary():
    prescriptions = pd.DataFrame(
        {
            "patid": [1, 1, 2, 2],
            "prodcode": [10, 10, 10, 10],
            "start_date": ["2020-01-01", "2020-01-19", "2020-01-01", "2020-01-20"],
            "stop_date": ["2020-01-05", "2020-01-22", "2020-01-05", "2020-01-22"],
        }
    )

    closed = ep.tl.close_small_gaps_in_prescriptions(prescriptions, decision="b_15")

    patient_one = closed[closed["patid"] == 1].reset_index(drop=True)
    patient_two = closed[closed["patid"] == 2].reset_index(drop=True)

    assert patient_one.loc[0, "stop_date"] == pd.Timestamp("2020-01-19")
    assert patient_two.loc[0, "stop_date"] == pd.Timestamp("2020-01-05")


def test_close_small_gaps_in_prescriptions_rejects_unknown_decision():
    prescriptions = pd.DataFrame(
        {"patid": [1], "prodcode": [10], "start_date": ["2020-01-01"], "stop_date": ["2020-01-05"]}
    )

    with pytest.raises(NotImplementedError, match="Unsupported gap-closing decision"):
        ep.tl.close_small_gaps_in_prescriptions(prescriptions, decision="z")


def test_screen_drugs_requires_exposure_columns():
    exposure_windows = pd.DataFrame(
        {
            "patid": [1],
            "drug": ["drug_a"],
            "start_date": ["2020-01-10"],
            "stop_date": ["2020-01-15"],
            "unexposed_start_date": ["2020-01-05"],
            "prescription_age": [50.0],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1],
            "disease": ["disease_x"],
            "disease_eventdate": pd.to_datetime(["2020-01-12"]),
        }
    )

    with pytest.raises(KeyError, match="Missing required exposure columns"):
        ep.tl.screen_drugs(exposure_windows, events, min_total_events=1)


def test_screen_drugs_requires_event_columns():
    exposure_windows = pd.DataFrame(
        {
            "patid": [1],
            "drug": ["drug_a"],
            "start_date": ["2020-01-10"],
            "stop_date": ["2020-01-15"],
            "unexposed_start_date": ["2020-01-05"],
            "exposure_length": [5.0],
            "prescription_age": [50.0],
        }
    )
    events = pd.DataFrame({"patid": [1], "disease": ["disease_x"]})

    with pytest.raises(KeyError, match="Missing required event columns"):
        ep.tl.screen_drugs(exposure_windows, events, min_total_events=1)


def test_screen_drugs_rejects_malformed_known_pair_table():
    exposure_windows = pd.DataFrame(
        {
            "patid": [1],
            "drug": ["drug_a"],
            "start_date": ["2020-01-10"],
            "stop_date": ["2020-01-15"],
            "unexposed_start_date": ["2020-01-05"],
            "exposure_length": [5.0],
            "prescription_age": [50.0],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1],
            "disease": ["disease_x"],
            "disease_eventdate": pd.to_datetime(["2020-01-12"]),
        }
    )
    known_pairs = pd.DataFrame({"drug": ["drug_a"]})

    with pytest.raises(KeyError, match="known_drug_disease_pairs must contain the configured drug and disease columns"):
        ep.tl.screen_drugs(
            exposure_windows,
            events,
            min_total_events=1,
            known_drug_disease_pairs=known_pairs,
        )


def test_screen_drugs_rejects_malformed_ever_user_counts():
    exposure_windows = pd.DataFrame(
        {
            "patid": [1],
            "drug": ["drug_a"],
            "start_date": ["2020-01-10"],
            "stop_date": ["2020-01-15"],
            "unexposed_start_date": ["2020-01-05"],
            "exposure_length": [5.0],
            "prescription_age": [50.0],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1],
            "disease": ["disease_x"],
            "disease_eventdate": pd.to_datetime(["2020-01-12"]),
        }
    )
    ever_user_counts = pd.DataFrame({"drug": ["drug_a"]})

    with pytest.raises(KeyError, match="ever_user_counts must contain the configured drug and count columns"):
        ep.tl.screen_drugs(
            exposure_windows,
            events,
            min_total_events=1,
            ever_user_counts=ever_user_counts,
        )


def test_public_drug_screening_workflow_from_raw_therapy():
    therapy = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "eventdate": ["2020-01-10", "2020-01-10", "2020-01-10"],
            "drugsubstance": ["drug_a", "drug_a", "drug_a"],
            "duration": [5, 5, 5],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "dob": ["1970-01-01", "1970-01-01", "1970-01-01"],
            "frd": ["2019-01-01", "2019-01-01", "2019-01-01"],
            "tod": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "lcd": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "deathdate": ["2021-01-01", "2021-01-01", "2021-01-01"],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1, 2, 3, 1],
            "disease": ["disease_x", "disease_x", "disease_x", "disease_y"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06", "2020-01-13", "2020-01-12"]),
        }
    )
    ever_user_counts = pd.DataFrame({"drug": ["drug_a"], "N.everuser": [3]})
    known_pairs = pd.DataFrame({"drug": ["drug_a"], "disease": ["disease_y"]})

    prescriptions = ep.tl.prepare_prescriptions_from_therapy(therapy, patients=patients)
    result = ep.tl.screen_substance_cohort(
        prescriptions,
        patients,
        events,
        known_drug_disease_pairs=known_pairs,
        ever_user_counts=ever_user_counts,
        min_total_events=2,
    )

    assert len(prescriptions) == 3
    assert set(result["disease"]) == {"disease_x"}
    assert set(result["age.group"]) == {"40-60", "all"}
    assert set(result["N.everuser"]) == {3}
    assert {"IRR", "p.value", "N.exposed"}.issubset(result.columns)


def test_screen_grouped_therapy_runs_chapter_level_workflow():
    therapy = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "prodcode": [10, 11, 20],
            "eventdate": ["2020-01-10", "2020-01-10", "2020-01-10"],
            "drugsubstance": ["drug_a", "drug_b", "drug_c"],
            "duration": [5, 5, 5],
        }
    )
    grouping = pd.DataFrame(
        {
            "prodcode": [10, 11, 20],
            "bnf.chapter": ["chapter_1", "chapter_1", "chapter_2"],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "dob": ["1970-01-01", "1970-01-01", "1970-01-01"],
            "frd": ["2019-01-01", "2019-01-01", "2019-01-01"],
            "tod": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "lcd": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "deathdate": ["2021-01-01", "2021-01-01", "2021-01-01"],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1, 2, 3, 3],
            "disease": ["disease_x", "disease_x", "disease_x", "disease_y"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06", "2020-01-13", "2020-01-12"]),
        }
    )
    known_pairs = pd.DataFrame({"prodcode": [20], "disease": ["disease_y"]})

    result = ep.tl.screen_grouped_therapy(
        therapy,
        patients,
        events,
        level="chapter",
        grouping=grouping,
        known_drug_disease_pairs=known_pairs,
        min_total_events=1,
    )

    assert set(result["drug"]) == {"chapter_1", "chapter_2"}
    assert set(result["disease"]) == {"disease_x"}
    assert set(result[result["drug"] == "chapter_1"]["N.everuser"]) == {2}
    assert set(result[result["drug"] == "chapter_2"]["N.everuser"]) == {1}


def test_screen_grouped_therapy_supports_named_followup_workflows():
    therapy = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "prodcode": [10, 11, 20],
            "eventdate": ["2020-01-10", "2020-01-10", "2020-01-10"],
            "drugsubstance": ["drug_a", "drug_b", "drug_c"],
            "duration": [5, 5, 5],
        }
    )
    grouping = pd.DataFrame(
        {
            "prodcode": [10, 11, 20],
            "bnf.chapter": ["chapter_1", "chapter_1", "chapter_2"],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "dob": ["1970-01-01", "1970-01-01", "1970-01-01"],
            "frd": ["2018-01-01", "2018-01-01", "2018-01-01"],
            "tod": ["2023-01-01", "2023-01-01", "2023-01-01"],
            "lcd": ["2023-01-01", "2023-01-01", "2023-01-01"],
            "deathdate": ["2023-01-01", "2023-01-01", "2023-01-01"],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "disease": ["disease_x", "disease_x", "disease_x"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06", "2020-01-13"]),
        }
    )

    result = ep.tl.screen_grouped_therapy(
        therapy,
        patients,
        events,
        level="chapter",
        grouping=grouping,
        workflow="365days",
        min_total_events=1,
    )

    chapter_one = result[(result["drug"] == "chapter_1") & (result["disease"] == "disease_x") & (result["age.group"] == "all")].iloc[0]
    assert chapter_one["sum.exposed.person.time"] == 730


def test_screen_grouped_therapy_can_add_level_specific_output_column():
    therapy = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "prodcode": [10, 11, 20],
            "eventdate": ["2020-01-10", "2020-01-10", "2020-01-10"],
            "drugsubstance": ["drug_a", "drug_b", "drug_c"],
            "duration": [5, 5, 5],
        }
    )
    grouping = pd.DataFrame(
        {
            "prodcode": [10, 11, 20],
            "bnf.chapter": ["chapter_1", "chapter_1", "chapter_2"],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "dob": ["1970-01-01", "1970-01-01", "1970-01-01"],
            "frd": ["2019-01-01", "2019-01-01", "2019-01-01"],
            "tod": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "lcd": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "deathdate": ["2021-01-01", "2021-01-01", "2021-01-01"],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "disease": ["disease_x", "disease_x", "disease_x"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06", "2020-01-13"]),
        }
    )

    result = ep.tl.screen_grouped_therapy(
        therapy,
        patients,
        events,
        level="chapter",
        grouping=grouping,
        level_label_col="chapter",
        min_total_events=1,
    )

    assert "drug" in result.columns
    assert "chapter" in result.columns
    assert set(result["chapter"]) == set(result["drug"])


def test_screen_grouped_therapy_runs_section_level_workflow():
    therapy = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "prodcode": [10, 11, 20],
            "eventdate": ["2020-01-10", "2020-01-10", "2020-01-10"],
            "drugsubstance": ["drug_a", "drug_b", "drug_c"],
            "duration": [5, 5, 5],
        }
    )
    grouping = pd.DataFrame(
        {
            "prodcode": [10, 11, 20],
            "bnf.section": ["section_1", "section_1", "section_2"],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2, 3],
            "dob": ["1970-01-01", "1970-01-01", "1970-01-01"],
            "frd": ["2019-01-01", "2019-01-01", "2019-01-01"],
            "tod": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "lcd": ["2021-01-01", "2021-01-01", "2021-01-01"],
            "deathdate": ["2021-01-01", "2021-01-01", "2021-01-01"],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1, 2, 3, 1],
            "disease": ["disease_x", "disease_x", "disease_x", "disease_y"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06", "2020-01-13", "2020-01-12"]),
        }
    )
    known_pairs = pd.DataFrame({"prodcode": [10], "disease": ["disease_y"]})

    result = ep.tl.screen_grouped_therapy(
        therapy,
        patients,
        events,
        level="section",
        grouping=grouping,
        known_drug_disease_pairs=known_pairs,
        min_total_events=1,
    )

    assert set(result["drug"]) == {"section_1", "section_2"}
    assert set(result["disease"]) == {"disease_x"}
    assert set(result[result["drug"] == "section_1"]["N.everuser"]) == {2}
    assert set(result[result["drug"] == "section_2"]["N.everuser"]) == {1}


def test_screen_grouped_therapy_runs_paragraph_level_workflow():
    therapy = pd.DataFrame(
        {
            "patid": [1, 2, 3, 4],
            "prodcode": [10, 11, 20, 21],
            "eventdate": ["2020-01-10", "2020-01-10", "2020-01-10", "2020-01-10"],
            "drugsubstance": ["drug_a", "drug_b", "drug_c", "drug_d"],
            "duration": [5, 5, 5, 5],
        }
    )
    grouping = pd.DataFrame(
        {
            "prodcode": [10, 11, 20, 21],
            "bnf.paragraph": ["paragraph_1", "paragraph_1", "paragraph_2", "paragraph_2"],
        }
    )
    patients = pd.DataFrame(
        {
            "patid": [1, 2, 3, 4],
            "dob": ["1970-01-01", "1970-01-01", "1970-01-01", "1970-01-01"],
            "frd": ["2019-01-01", "2019-01-01", "2019-01-01", "2019-01-01"],
            "tod": ["2021-01-01", "2021-01-01", "2021-01-01", "2021-01-01"],
            "lcd": ["2021-01-01", "2021-01-01", "2021-01-01", "2021-01-01"],
            "deathdate": ["2021-01-01", "2021-01-01", "2021-01-01", "2021-01-01"],
        }
    )
    events = pd.DataFrame(
        {
            "patid": [1, 2, 3, 4, 4],
            "disease": ["disease_x", "disease_x", "disease_x", "disease_x", "disease_y"],
            "disease_eventdate": pd.to_datetime(["2020-01-12", "2020-01-06", "2020-01-13", "2020-01-14", "2020-01-12"]),
        }
    )
    known_pairs = pd.DataFrame({"prodcode": [21], "disease": ["disease_y"]})

    result = ep.tl.screen_grouped_therapy(
        therapy,
        patients,
        events,
        level="paragraph",
        grouping=grouping,
        known_drug_disease_pairs=known_pairs,
        min_total_events=1,
    )

    assert set(result["drug"]) == {"paragraph_1", "paragraph_2"}
    assert set(result["disease"]) == {"disease_x"}
    assert set(result[result["drug"] == "paragraph_1"]["N.everuser"]) == {2}
    assert set(result[result["drug"] == "paragraph_2"]["N.everuser"]) == {2}
