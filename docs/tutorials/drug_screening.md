# Drug Screening

This tutorial shows the current substance-level drug-screening workflow in
`ehrapy`. The API ports the first end-to-end slice from the `original/`
research code into `ep.tl`.

## What the current port covers

- normalize raw therapy rows into prescription rows
- optionally apply structured `drugprepr`-style duration cleaning
- collapse prescriptions into exposure episodes
- build matched exposed and unexposed windows
- screen disease events with a self-controlled cohort design

The current implementation works with structured dosage fields such as
`duration`, `numdays`, `dose_duration`, or `ndd`, and it can also derive
approximate `ndd` values from common free-text dosage instructions such as
`1-2 tablets twice daily` or `every other day`. For more complex instructions,
prefer precomputed structured dosage fields.

The screening wrappers also expose the three workflow variants used in
`original/`: `workflow="actual"`, `workflow="30days"`, and `workflow="365days"`.

## Minimal workflow

```python
import pandas as pd
import ehrapy as ep

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

result = ep.tl.screen_substance_therapy(
    therapy,
    patients,
    events,
    workflow="actual",
    min_total_events=2,
)

result[["drug", "disease", "age.group", "IRR", "p.value"]]
```

## Matching the original workflow variants

Use the `workflow` argument to match the original script families:

```python
actual_result = ep.tl.screen_substance_therapy(
    therapy,
    patients,
    events,
    workflow="actual",
    min_total_events=2,
)

thirty_day_result = ep.tl.screen_substance_therapy(
    therapy,
    patients,
    events,
    workflow="30days",
    min_total_events=2,
)

year_result = ep.tl.screen_substance_therapy(
    therapy,
    patients,
    events,
    workflow="365days",
    min_total_events=2,
)
```

## Grouped workflow examples

Use `ep.tl.screen_grouped_therapy(...)` when you want to aggregate prepared
prescriptions by BNF hierarchy instead of screening at the substance level.

```python
grouping = pd.DataFrame(
    {
        "prodcode": [10, 11, 20],
        "bnf.section": ["section_1", "section_1", "section_2"],
        "bnf.paragraph": ["paragraph_1", "paragraph_1", "paragraph_2"],
    }
)

section_result = ep.tl.screen_grouped_therapy(
    therapy_with_prodcode,
    patients,
    events,
    level="section",
    grouping=grouping,
    level_label_col="section",
    min_total_events=2,
)

paragraph_result = ep.tl.screen_grouped_therapy(
    therapy_with_prodcode,
    patients,
    events,
    level="paragraph",
    grouping=grouping,
    level_label_col="paragraph",
    min_total_events=2,
)
```

The grouped result table always includes the canonical `drug` column. Set
`level_label_col` if you also want a level-specific output column such as
`section`, `paragraph`, or `chapter`.

## When to use the lower-level functions

Use [`ep.tl.compute_ndd_from_text`](../api/tools_index.md) when you want to
inspect parsed dosage text directly. Use
[`ep.tl.prepare_prescriptions_from_therapy`](../api/tools_index.md) when you want
to inspect or customize prescription preparation before screening. Use
`ep.tl.screen_substance_cohort` when your prescription table already has
`start_date` and `duration`.

## Current scope

This port currently targets the substance-level workflow from `original/`.
Grouped chapter-, section-, and paragraph-level workflows are available through
`ep.tl.screen_grouped_therapy(...)` when a `prodcode`-to-group mapping table is
provided. The current text parser covers common CPRD-style dosage instructions,
but not the full `doseminer` feature set from `drugprepr`.
