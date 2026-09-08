- Source: Kaggle dataset version 118, downloaded using `kagglehub`.

- Coverage: 2015-01-01 through 2026-05-27.

- Records: 1,851,285 rows after excluding one imported header row.

- `race_id` is reused across different events and cannot identify races by itself.

- Candidate key: `(date, course, off)`, producing 189,043 groups.

- Key checks: no missing or blank components, repeated horses within groups, or conflicting race names.

- Winner counts: 188,742 groups with one winner and 301 with two. Five inspected pairs support dead heats; the remainder are unverified.

- Unresolved: eight zero finishing positions, mixed value types, missing-value markers, and availability of potential predictors before race time.

- Geographic scope includes races outside the UK and Ireland.

## Initial model scope

Predict win probabilities for races labeled exactly `Flat`.

Group runners using (date, course, off). Retain groups only when:
- Exactly one runner has pos = 1.
- Every runner has a non-NULL ran value.
- All ran values agree and equal the observed row count.

Version 118 results:
- Retained: 126,136 race groups and 1,265,660 runner rows.
- Excluded from the flat subset: 255 groups and 2,569 rows.

These are initial eligibility checks, not complete data validation.
Races with multiple recorded winners are outside this first model's scope.
Raw source data remains unchanged.