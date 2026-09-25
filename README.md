# Horse Racing Outcome Prediction

An end-to-end data science project for estimating each runner’s
probability of winning a horse race.

The v2 rebuild emphasizes data auditing, chronological feature
construction, reproducible experiments, probability calibration,
and honest reporting of model limitations.

**Current development model:** histogram gradient boosting with
ten encoded inputs and a frozen race-level calibration adjustment.

**2024 development race log loss: 2.130549**, compared with
2.253199 for uniform probabilities. Lower is better. This is not
an untouched test result or evidence of betting profitability.

## Start here

The rebuild is on the `v2-rebuild-master` branch.

- [V2 quickstart](docs/v2-quickstart.md): environment setup,
  preparation, training, evaluation, and calibration commands.
- [Data notes](docs/v2-data-notes.md): data structure, selection
  rules, historical experiments, and limitations.
- [Feature and model audit](docs/v2-feature-audit.md): feature
  investigations, temporal comparisons, tuning, calibration,
  and rejected experiments.
- [Race-type review](docs/v2-race-type-review.csv): reviewed
  race classifications and supporting evidence.

V2 uses the root-level Python scripts and `requirements-v2.txt`.
The older `src/hrml` implementation belongs to v1.

## Prediction task

For each eligible race, estimate a win probability for every
runner, with probabilities summing to one across the field.

Models are trained as binary runner classifiers. Their outputs
are normalized within each race, then evaluated using **race
log loss**: the mean negative natural logarithm of the probability
assigned to the recorded winner.

Training therefore uses a runner-level objective; model selection
and evaluation use a race-level metric.

## Data and scope

The documented experiments use version 118 of the Kaggle
[Horse Racing Results dataset](https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025).

Despite its title, the dataset includes international courses.
The snapshot ends on 2026-05-27.

Eligible race groups must:

- Be labelled `Flat`, subject to documented review exclusions.
- Have at least two runners and exactly one recorded winner.
- Have observed runner counts matching a consistent reported
  field size.

Dead heats and incomplete or inconsistent groups are excluded.
A manual review excluded 25 eligible race groups containing
267 runner rows that were outside the intended flat-racing scope.

| Stored split | Dates | Races | Runners |
|---|---|---:|---:|
| Training | 2015–2023 | 98,600 | 988,806 |
| Validation | 2024 | 11,637 | 117,378 |
| Previously examined test period | 2025-01-01 through 2026-05-27 | 15,872 | 159,207 |
| Total | | 126,109 | 1,265,391 |

Data auditing identified reused source race IDs, an embedded
header row, missing-value conventions, ambiguous outcomes,
race-type inconsistencies, and incomplete historical coverage.

Race groups use `(date, course, off)` rather than assuming source
race IDs are unique. This remains a dataset-specific identifier,
not a universal race identity.

## Features and chronological evaluation

The selected model uses:

- Prior recorded starts, wins, and win rate.
- Days since the previous recorded race.
- Previous finish position and a nonnumeric result-code indicator.
- Previous field size and relative finishing position.
- Age.
- Signed distance change from the previous recorded race.

Historical features use strictly earlier dates. Same-day results
are withheld, and ambiguous previous-race information remains
missing. History is limited to the selected dataset; it is not
necessarily a horse’s complete career.

Annual comparisons train on preceding years and evaluate the next
calendar year. Earlier evaluation-period results may inform later
historical features, while the fitted model stays fixed.

The boosting model handles missing inputs natively. It does not
use the imputation, scaling, or logarithmic gap transformation
used by the logistic benchmark.

## Model development

Under fixed initial boosting settings, boosting improved over
logistic regression in all four annual comparisons.

| Evaluation year | Logistic regression | Initial boosting |
|---|---:|---:|
| 2021 | 2.134419 | 2.116284 |
| 2022 | 2.129207 | 2.111592 |
| 2023 | 2.152865 | 2.132037 |
| 2024 | 2.156699 | 2.135187 |

A bounded Optuna search evaluated 20 parameter combinations across
2021–2023. The selected configuration improved 2024 loss slightly,
but its paired date-bootstrap interval included zero. It remains
a comparison candidate rather than the selected configuration.

Adding current field size was also tested. It worsened mean
2021–2023 loss and 2024 loss, so it was not adopted.

## Probability calibration

A single exponent adjusts the initial boosting probabilities:

`q_i = p_i^gamma / sum_j(p_j^gamma)`

The exponent was fitted using chronological predictions for
2021–2023, then frozen before application to 2024.
Its value is approximately **1.237228322**.

| 2024 comparison | Race log loss |
|---|---:|
| Uniform probabilities | 2.253199 |
| Logistic regression | 2.156699 |
| Initial boosting | 2.135187 |
| Optuna-tuned boosting | 2.134307 |
| Initial boosting with frozen calibration | **2.130549** |

Calibration improved initial boosting by **0.004638**.
A paired bootstrap over 363 race dates, using 10,000 resamples
and seed 42, gave a 95% percentile interval of
**[+0.003014, +0.006257]**. Eleven of twelve months improved.

The adjustment preserves runner rankings, so it does not improve
winner-selection accuracy. Pooled probability bins showed closer
agreement with observed win rates, but subgroup limitations remain.

## Error analysis and limitations

For runners without earlier recorded history, mean calibrated
probability was **8.65%**, versus an observed win rate of **7.86%**.
Good pooled calibration therefore does not establish calibration
within every subgroup.

Other limitations include:

- Historical coverage and exact-name matching affect features.
- Manual race-type review is not an exhaustive correctness guarantee.
- Historical result data do not establish when every field became
  available in a live prediction workflow.
- 2024 has been examined repeatedly during development.
- The period labelled `test` was examined in earlier experiments
  and is no longer an untouched holdout.
- Date-bootstrap intervals do not account for model selection,
  fitting uncertainty, or dependence between different dates.

An earlier frozen model underperformed normalized Betfair starting
prices on a matched subset. The current model has not established
superiority to market odds or betting profitability.

## Reproduce the current workflow

Follow the [quickstart](docs/v2-quickstart.md) to:

1. Install the pinned v2 dependencies.
2. Prepare version-118 data with the reviewed exclusions.
3. Build chronological features.
4. Train and save the initial boosting model.
5. Export 2024 race-normalized predictions.
6. Apply the saved calibration exponent.
7. Run the focused tests.

Selected JSON experiment reports and the exported Optuna trial
history are tracked in Git. Generated databases, model files,
and prediction CSVs remain local and are generally ignored.

Earlier supplied-racecard prediction scripts were built for older
feature configurations. The documented current workflow evaluates
historical predictions; live inference with the selected model
still requires integration and validation.

## Legacy v1

The earlier implementation is preserved on
`archive/pre-rebuild-work` and under the `pre-v2-rebuild` tag.

Its reported metrics and feature-availability claims have not been
revalidated in this rebuild and should not be compared directly
with v2 results.

This repository is an educational and portfolio project.