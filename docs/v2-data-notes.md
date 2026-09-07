\- Source: Kaggle dataset version 118, downloaded using `kagglehub`.

\- Coverage: 2015-01-01 through 2026-05-27.

\- Records: 1,851,285 rows after excluding one imported header row.

\- `race\_id` is reused across different events and cannot identify races by itself.

\- Candidate key: `(date, course, off)`, producing 189,043 groups.

\- Key checks: no missing or blank components, repeated horses within groups, or conflicting race names.

\- Winner counts: 188,742 groups with one winner and 301 with two. Five inspected pairs support dead heats; the remainder are unverified.

\- Unresolved: eight zero finishing positions, mixed value types, missing-value markers, and availability of potential predictors before race time.

\- Geographic scope includes races outside the UK and Ireland.

