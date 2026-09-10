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

## Chronological evaluation splits

Assign each race to a split using its race date. All runners in the
same race belong to the same split.

| Split | Date rule | Races in version 118 | Purpose |
|---|---|---:|---|
| Training | date < 2024-01-01 | 98,625 | Fit models and preprocessing |
| Validation | 2024-01-01 <= date < 2025-01-01 | 11,637 | Select features, hyperparameters, and model |
| Test | date >= 2025-01-01 | 15,872 | Evaluate the finalized approach |

The prepared version 118 dataset spans 2015-01-01 through 2026-05-27.
The test period therefore contains all available 2025 races and a
partial 2026. Counts refer to eligible flat race groups.

### Evaluation rules

- Do not randomly shuffle races across splits.
- Fit learned preprocessing on training data only.
- Construct historical features using information available before
  each race; chronological splits alone do not prevent feature leakage.
- Use validation performance to guide development.
- Reserve test performance for final evaluation, without using it
  to select features or tune the model.
- Compare models and baselines on identical evaluation races, using
  mean race-level log loss with equal weight per race.

The uniform baseline log loss of 2.243584 was calculated over the
entire prepared dataset. It is descriptive, not a validation or test
benchmark; split-specific baselines will be calculated separately.

### Validation baseline

| Baseline | Period | Races | Mean race-level log loss |
|---|---|---:|---:|
| Uniform probability (1 / field size) | 2024 | 11,637 | 2.253199 |

Lower loss is better. This baseline assigns equal probability to every
runner within a race and requires no training.

### First historical-form model

- Feature history: earlier dates only, using the prepared race subset.
- Horse identity: exact horse-name string.
- Score: (prior_wins + alpha * reference_rate) / (prior_starts + alpha).
- Reference rate: training-period wins divided by training-period starts
  (approximately 0.099715; full precision used in code).
- Initial smoothing strength: alpha = 10.
- Race probabilities: each runner's score divided by the race's total score.
- Validation histories update with results from earlier validation dates.
- Validation: 11,637 races in 2024.
- Model race log loss: 2.247187.
- Uniform race log loss: 2.253199.
- Improvement: 0.006012; lower loss is better.
- Test performance remains unevaluated.

Selected alpha = 30 from the fixed candidate set {1, 3, 10, 30, 100}
using the lowest 2024 validation race log loss.

Selected validation loss: 2.240013.
Improvement over uniform: 0.013186 (approximately 0.59%).
This is the best tested setting, not a proven global optimum.
Test performance remains unevaluated.