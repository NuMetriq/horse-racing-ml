# V2 baseline quickstart

Run commands from the repository’s top folder in PowerShell.

## Create and activate the environment

Use Python 3.11. From the repository's top folder, create the
environment once:

```powershell
py -3.11 -m venv .venv-v2
```

Activate it and install dependencies:

```powershell
.\.venv-v2\Scripts\Activate.ps1
python -m pip install -r requirements-v2.txt
```

Skip environment creation if .venv-v2 already exists.

## Download source data

To reproduce the dataset version used for the frozen baseline:

```powershell
python download_data.py --version 118
```

To request the latest available version:

```powershell
python download_data.py
```

The script prints the downloaded or cached dataset directory.
Downloading does not rebuild prepared databases or retrain the model.

The frozen baseline used version 118, with recorded results through
2026-05-27. Preserve its prepared databases when refreshing history.
Prepare any newer version into a separate output database.

## Prepare the data

Use Kaggle dataset version 118 for the documented results.
Set this variable to the downloaded raceform.db file:

```powershell
$rawDatabase = Join-Path $env:USERPROFILE ".cache\kagglehub\datasets\deltaromeo\horse-racing-results-ukireland-2015-2025\versions\118\form_2015-present\form_2015-present\raceform.db"
```

Create the prepared dataset using the version-controlled race review:

```powershell
python prepare_data.py $rawDatabase --race-type-review docs/v2-race-type-review.csv --output data/processed/v2_flat_reviewed_distance_v2.db
```

Expected preparation results:

- Reviewed race exclusions: 25
- Eligible races removed: 25
- Runner rows removed: 267
- Exported races: 126,109
- Exported runner rows: 1,265,391

Build all-history features, including age, previous-race information,
and distance change:

```powershell
python build_features.py data/processed/v2_flat_reviewed_distance_v2.db --alpha 30 --features-output data/processed/v2_features_reviewed_distance_v2.db
```

Expected feature rows: 1,265,391.

| Stored split | Dates | Runner rows |
|---|---|---:|
| train | 2015-01-01 through 2023-12-31 | 988,806 |
| validation | 2024-01-01 through 2024-12-31 | 117,378 |
| test | 2025-01-01 through 2026-05-27 | 159,207 |

Historical features use only records from earlier dates.
The `--alpha 30` setting controls the feature builder's smoothed-form
diagnostic; it is not a boosting hyperparameter.

The training commands below select rows by explicit date boundaries.
The presence of later rows in the feature database does not make
them training inputs.

The period labelled `test` has already been examined in earlier
development and should not be described as an untouched holdout.

These commands require new output paths. Reuse existing exports
only when they were created from the same dataset version,
review decisions, and preparation code.

## Train and save

These commands require the reviewed feature database:
`data/processed/v2_features_reviewed_distance_v2.db`.

Train the initial boosting configuration on dates before 2024:

```powershell
python train_boosting.py data/processed/v2_features_reviewed_distance_v2.db --train-end 2024-01-01 --evaluation-end 2025-01-01 --feature-set relative_finish_age_distance_change --max-iter 200 --model-output outputs/models/v2_hist_boosting_initial.pkl
```

Omit `--parameters` to use the initial boosting configuration.
The model-output path must not already exist. Reuse an existing
matching model to skip retraining.

## Evaluate without retraining

Evaluate the saved model and export its uncalibrated,
race-normalized validation probabilities:

```powershell
python evaluate_saved.py data/processed/v2_features_reviewed_distance_v2.db outputs/models/v2_hist_boosting_initial.pkl --evaluation-start 2024-01-01 --evaluation-end 2025-01-01 --predictions-output outputs/predictions/v2_hist_boosting_initial_validation.csv
```

Expected 2024 results:

- Races: 11,637
- Runner predictions: 117,378
- Uncalibrated boosting race log loss: 2.135187
- Uniform race log loss: 2.253199

The prediction-output path must not already exist. Reuse an
existing matching export to skip this step.

Only load trusted pickle files. Use the same package environment
that created the saved model.

## Apply frozen probability calibration

The repository includes:
`outputs/reports/v2_initial_boosting_calibration.json`.

This file contains the full-precision exponent fitted using
2021–2023 chronological predictions from initial-settings
boosting models. Apply it unchanged to the 2024 predictions:

```powershell
python apply_calibration.py outputs/predictions/v2_hist_boosting_initial_validation.csv outputs/reports/v2_initial_boosting_calibration.json --year 2024 --output outputs/predictions/v2_hist_boosting_calibrated_validation.csv
```

Expected results:

- Original race log loss: 2.135187
- Calibrated race log loss: 2.130549
- Improvement: +0.004638
- Exported runner predictions: 117,378

The destination must not already exist. The CSV contains race
identifiers, horse names, calibrated win probabilities, and
actual winner indicators.

Calibration preserves runner rankings and normalizes probabilities
to sum to one within each race. The calibration file belongs to
the initial boosting model; it has not been validated for the
Optuna-tuned model.

2024 was excluded from calibration fitting but was examined during
development. These results are not an untouched final assessment.
See `v2-feature-audit.md` for comparisons and limitations.

## Run focused tests

```powershell
python -m unittest test_feature_transforms test_prior_form test_race_metrics
```

The first frozen baseline has been evaluated on the reserved test
period. See v2-data-notes.md for results and evaluation rules.

The model requires feature_transforms.py when training or evaluating.
It supplies the previous-position encoding and the pipeline's
logarithmic gap transformation.

## Compare prediction exports (OPTIONAL)

This optional comparison requires the archived previous-field model's
validation export, v2_logistic_previous_field_validation.csv.
The current training command does not recreate that earlier model.
Skip this section on a fresh setup unless that export is available.

After exporting both models' validation predictions, run:

```powershell
python compare_predictions.py outputs/predictions/v2_logistic_previous_field_validation.csv outputs/predictions/v2_logistic_relative_finish_validation.csv
```

The first file is the baseline; the second is the candidate.
Both must contain the same runners and winner labels.

Expected matched races: 11,637.
Expected mean improvement: 0.005756.

Positive improvement means the candidate assigned a higher
probability to the recorded winner.

## Predict one historical race

```powershell
python predict_race.py data/processed/v2_features_all_splits.db outputs/models/v2_logistic_relative_finish_corrected.pkl --date 2024-01-01 --course "Ascot (AUS)" --off "7:50"
```

Requires the all-splits feature database. The script loads prepared
historical features and applies the frozen model without reading the
selected race's outcome.

Probabilities sum to one within the race. Horses with identical
inputs receive identical predictions; ordering within a tie does
not indicate preference.

This predicts an existing race from prepared features. It does not
yet construct features for upcoming racecards.

## Predict from a runner-list file

Supply a UTF-8 text file with one exact horse name per line.
Blank lines are ignored; duplicate names are rejected.
Use either --runners-file or repeated --horse arguments, not both.

Historical example:

```powershell
python predict_racecard.py data/processed/v2_flat_competitive.db outputs/models/v2_logistic_relative_finish_corrected.pkl --date 2024-01-01 --runners-file examples/ascot_2024-01-01_0750.txt
```

Expected: 10 runners, with probabilities summing to one.
The list must include the complete race field. Historical features
use only records dated before the supplied race date.

To save a JSON prediction report, add --report:

```powershell
python predict_racecard.py data/processed/v2_flat_competitive.db outputs/models/v2_logistic_relative_finish_corrected.pkl --date 2024-01-01 --runners-file examples/ascot_2024-01-01_0750.txt --report outputs/predictions/ascot_2024-01-01_0750.json
```

The report destination must not already exist. The report includes
full-precision probabilities, each runner's source features, history
coverage, and the model and database paths.

## Reproduce the recorded test evaluation

The test period has already been evaluated for this frozen baseline.
Rerunning it reproduces that assessment; it is not a new holdout.

```powershell
python evaluate_saved.py data/processed/v2_features_all_splits.db outputs/models/v2_logistic_relative_finish_corrected.pkl --split test
```

Expected: 15,872 races, model log loss 2.149611, and uniform
log loss 2.245443.