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

### Recent-history experiment

Compared all recorded history with the preceding 365 days, holding
alpha = 30 and the validation races fixed.

| History | Validation race log loss |
|---|---:|
| All recorded history | 2.240013 |
| Previous 365 days | 2.241627 |

The 365-day window includes results exactly 365 days earlier and
excludes same-day results. Its loss was 0.001614 higher.

Retain all-history with alpha = 30 as the selected baseline.
This comparison does not establish the best smoothing strength
for the 365-day model. Test performance remains unevaluated.

### History-coverage limitation

In 2024 validation, 685 of 11,637 races (5.9%) had a majority of
runners with no earlier history in the prepared subset.

| Group | Races | Model log loss | Uniform log loss | Improvement |
|---|---:|---:|---:|---:|
| Majority without history | 685 | 2.219470 | 2.222409 | 0.002939 |
| Other races | 10,952 | 2.241298 | 2.255125 | 0.013827 |

Settings: all-history, alpha = 30.

The model's observed advantage was smaller in the sparse-history
group. These descriptive results do not establish causality or
statistical significance.

No recorded history does not imply a career debut. For three
inspected horses, the raw source also contained no earlier records
under their exact names.

### Logistic regression baseline

- Inputs: prior starts, prior wins, and prior win rate.
- Missing rates: filled with zero, with a missing-value indicator added.
- Preprocessing: imputer and standard scaler fitted on training only.
- Model: scikit-learn 1.9.1 LogisticRegression,
  solver="lbfgs", C=1.0, max_iter=1000.
- Training objective: regularized runner-level binary log loss.
- Evaluation: predicted win probabilities normalized within each race;
  mean race-level log loss, with equal weight per race.
- Validation period: 2024.

| Model | Validation race log loss |
|---|---:|
| Uniform | 2.253199 |
| Smoothed all-history, alpha = 30 | 2.240013 |
| Logistic regression | 2.225778 |

Logistic regression is the best validation model evaluated so far.
Test performance remains unevaluated.

### Days-since-run experiment

Added raw days since the previous recorded race on an earlier date.
Same-day starts are excluded. Missing gaps receive zero imputation
and a missing-value indicator.

Training gaps ranged from 1 to 3,020 days, with a mean of 59.1 days
among observed values. These describe the prepared subset, not
necessarily complete career histories.

Model settings and validation races were unchanged.

| Model | Validation race log loss |
|---|---:|
| Original logistic regression | 2.225778 |
| Logistic regression with days since run | 2.222540 |

Observed improvement over the original logistic model: 0.003238.
The missing-win-rate and missing-gap indicators are identical in
this all-history dataset. Test performance remains unevaluated.

### Validation calibration: logistic model with days since run

Calibration was inspected using race-normalized probabilities on the
2024 validation set (117,378 runners across 11,637 races).

| Probability band | Runners | Mean predicted | Observed win rate |
|---|---:|---:|---:|
| 0–5% | 4,649 | 4.17% | 4.17% |
| 5–10% | 67,582 | 7.71% | 7.37% |
| 10–15% | 33,533 | 11.97% | 12.40% |
| 15–20% | 8,296 | 16.92% | 17.10% |
| 20–30% | 2,959 | 23.21% | 25.38% |
| 30–50% | 353 | 34.58% | 37.39% |
| 50–100% | 6 | 57.17% | 33.33% |

Bands include their lower bound and exclude their upper bound,
except the final band, which includes 100%.

Predicted and observed rates were close in the larger lower-probability
bands. The 20–30% and 30–50% bands showed underprediction. The final
band contained too few runners for a reliable conclusion.

Overall mean predicted probability equalled the observed win rate
because probabilities sum to one per race and each race has one winner.
That equality alone does not demonstrate calibration.

No calibration adjustment was applied. The test set remains unevaluated.

### Log-transformed days since run

Replaced the raw days-since-run input with log1p(days_since_run)
inside the pipeline, before imputation and scaling.

All other inputs, model settings, and data splits stayed the same.

| Gap representation | Validation race log loss |
|---|---:|
| Raw days | 2.222540 |
| log1p(days) | 2.220082 |

Validation log loss improved by 0.002458.
Reloading the saved pipeline reproduced the score of 2.220082.

The log-gap model is the current preferred candidate based on
validation performance. The test set remains unevaluated.

Artifacts:
- Report: outputs/reports/v2_logistic_log_gap.json
- Model: outputs/models/v2_logistic_log_gap.pkl

The saved pipeline uses the function in feature_transforms.py,
which must remain available when loading the model.