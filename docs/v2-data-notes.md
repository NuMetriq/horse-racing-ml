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

### Previous recorded finishing position

Added the horse's previous recorded position from an earlier date
within the prepared dataset. Same-day results are excluded.

If the most recent earlier date contains multiple records, the
previous position is treated as unknown.

Positive numeric positions become a numeric input. Result codes
become a missing numeric position plus a separate code flag.
Missing values and position "0" become a missing numeric position
with the code flag set to zero.

The pipeline imputes missing numeric positions and adds a missing
indicator. Training and saved-model evaluation use the same
previous_position_v1 encoding.

| Model | Validation race log loss |
|---|---:|
| Logistic with log gap | 2.220082 |
| Logistic with log gap and previous position | 2.171974 |

Validation log loss improved by 0.048108.
Saved-model evaluation reproduced 2.171974.
The test set remains unevaluated.

History refers to available records in the prepared dataset,
not necessarily the horse's complete racing career.


| Current field size | Races | Log-gap model log loss | Previous-position model log loss | Loss reduction |
|---|---:|---:|---:|---:|
| 2–7 runners | 2,762 | 1.739591 | 1.708877 | 0.030714 |
| 8–12 runners | 6,310 | 2.253233 | 2.204126 | 0.049107 |
| 13+ runners | 2,565 | 2.655922 | 2.591545 | 0.064377 |


The previous-position model improved validation race log loss
across all three current-field-size groups. The largest absolute
reduction occurred in races with 13 or more runners.

Both saved models were evaluated on the same validation races.
These groups describe the current race's field size, not the
horse's previous race. The test set remains unevaluated.

### Previous race field size

Added previous_runner_count alongside previous finishing position.
Both refer to the same most recent recorded race on an earlier date
within the prepared dataset.

Same-day results are excluded. If the most recent earlier date has
multiple records, both previous position and field size are unknown.

Training and saved-model evaluation use the
previous_position_field_v1 encoding.

| Model | Validation race log loss |
|---|---:|
| Previous position | 2.171974 |
| Previous position + previous runner count | 2.165894 |

Validation log loss improved by 0.006080.
Saved-model evaluation reproduced 2.165894.
The test set remains unevaluated.

### Paired comparison: adding previous field size

Compared both models on the same 117,378 validation runners across
11,637 races. Runner identities and winner labels matched.

Improvement is baseline race log loss minus candidate race log loss.
Positive values favor the model with previous runner count.

| Measure | Result |
|---|---:|
| Baseline mean race log loss | 2.171974 |
| Candidate mean race log loss | 2.165894 |
| Mean improvement, calculated before rounding | 0.006081 |
| Median improvement | 0.004193 |
| Races improved | 6,297 |
| Races worsened | 5,192 |
| Races effectively unchanged | 148 |

The positive median and improvement in 54.1% of races indicate
that gains were not confined to a small number of races.
These are descriptive validation results, not a significance test.
The reserved test set remains unevaluated.

### Previous relative finish

Added (previous_position - 1) / (previous_runner_count - 1)
alongside the existing position and field-size inputs.

The value is 0 for first place and 1 when position equals runner
count. Missing positions, result codes, and invalid combinations
produce a missing value.

| Measure | Result |
|---|---:|
| Baseline race log loss | 2.165894 |
| Relative-finish race log loss | 2.160138 |
| Mean paired improvement | 0.005756 |
| Median paired improvement | 0.005967 |
| Races improved | 6,323 |
| Races worsened | 5,166 |
| Races effectively unchanged | 148 |

Saved-model evaluation reproduced 2.160138 and exported 117,378
validation predictions. The test set remains unevaluated.

Encoding: previous_relative_finish_v1.

Verified artifacts:
- Model: outputs/models/v2_logistic_relative_finish_corrected.pkl
- Report: outputs/reports/v2_logistic_relative_finish_corrected.json

The original relative-finish artifacts had an incorrect encoding
label and are superseded by these corrected artifacts.

### Monthly validation comparison: relative finish

Compared the previous-field model (baseline) with the relative-finish
model (candidate) on identical races within each month.

| Month | Races | Baseline log loss | Candidate log loss | Improvement |
|---|---:|---:|---:|---:|
| 2024-01 | 667 | 2.159586 | 2.154240 | 0.005346 |
| 2024-02 | 654 | 2.129737 | 2.125943 | 0.003794 |
| 2024-03 | 760 | 2.184486 | 2.179418 | 0.005068 |
| 2024-04 | 909 | 2.188207 | 2.185192 | 0.003016 |
| 2024-05 | 1,227 | 2.151848 | 2.145659 | 0.006190 |
| 2024-06 | 1,224 | 2.160143 | 2.151185 | 0.008958 |
| 2024-07 | 1,152 | 2.136321 | 2.127814 | 0.008507 |
| 2024-08 | 1,206 | 2.105682 | 2.098880 | 0.006802 |
| 2024-09 | 1,213 | 2.146502 | 2.141356 | 0.005146 |
| 2024-10 | 1,185 | 2.202920 | 2.199546 | 0.003374 |
| 2024-11 | 780 | 2.222891 | 2.217878 | 0.005013 |
| 2024-12 | 660 | 2.256171 | 2.250678 | 0.005493 |

Positive improvement favors the candidate. Differences are calculated
before rounding. Each race receives equal weight; the overall result
is weighted by monthly race counts, not an equal average of months.

The candidate improved mean log loss in every validation month.
These are descriptive validation findings. The test set remains
unevaluated.

### First frozen v2 baseline: test evaluation

The relative-finish logistic model was frozen after development on
training data before 2024 and validation data from 2024.

The unchanged saved model was evaluated on eligible test races from
2025-01-01 through 2026-05-27, using Kaggle dataset version 118.

| Measure | Result |
|---|---:|
| Test races | 15,872 |
| Test runner rows | 159,207 |
| Model race log loss | 2.149611 |
| Uniform race log loss | 2.245443 |
| Improvement over uniform | 0.095832 |

| Current field size | Races | Model log loss | Uniform log loss | Improvement |
|---|---:|---:|---:|---:|
| 2–7 runners | 3,969 | 1.682587 | 1.768776 | 0.086189 |
| 8–12 runners | 8,285 | 2.179909 | 2.281252 | 0.101343 |
| 13+ runners | 3,618 | 2.592561 | 2.686354 | 0.093793 |

The model beat uniform in all three field-size groups.

Model parameters and preprocessing remained fixed. Historical
features used only earlier dates; results from earlier test dates
could inform features for later test dates. Same-day results were
excluded.

This establishes performance against a uniform baseline, not
profitability or superiority to betting-market probabilities.

The test period has now been evaluated. Further development informed
by these results requires a new untouched period for an independent
final assessment.

Frozen model:
outputs/models/v2_logistic_relative_finish_corrected.pkl

### Prediction from a supplied runner list

predict_racecard.py accepts a race date and repeated --horse arguments.
It builds all-history features from the prepared race database using
only records dated strictly before the target date.

For the ten-runner Ascot (AUS) example on 2024-01-01, predictions
matched the prepared-feature prediction at displayed precision.
Race probabilities summed to one.

Horse names must match the historical dataset exactly. No matching
earlier history does not establish that a horse is a debutant.

The supplied list must contain the complete race field. The script
does not verify racecard completeness or download current racecards.
History freshness depends on the supplied database; the current
version-118 snapshot ends on 2026-05-27.