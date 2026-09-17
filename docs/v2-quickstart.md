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

Create the prepared dataset and all-history features:

```powershell
python prepare_data.py $rawDatabase --output data/processed/v2_flat_competitive.db
python build_features.py data/processed/v2_flat_competitive.db --alpha 30 --features-output data/processed/v2_features_all_splits.db
```

This creates 1,265,658 feature rows across train, validation, and test.
Training selects only training rows; evaluation defaults to validation.

These commands require new output paths. Skip completed exports when
reusing existing files from the same dataset version and preparation rules.

## Train and save

```powershell
python train_logistic.py data/processed/v2_features_all_splits.db --model-output outputs/models/v2_logistic_relative_finish_corrected.pkl
```

The model-output path must not already exist.

## Evaluate without retraining

```powershell
python evaluate_saved.py data/processed/v2_features_all_splits.db outputs/models/v2_logistic_relative_finish_corrected.pkl
```

Expected 2024 validation results:
- Races: 11,637
- Logistic race log loss: 2.160138
- Uniform race log loss: 2.253199

Only load trusted pickle files. Use the same package environment
that created the saved model.

To export validation probabilities for each runner:

```powershell
python evaluate_saved.py data/processed/v2_features_all_splits.db outputs/models/v2_logistic_relative_finish_corrected.pkl --predictions-output outputs/predictions/v2_logistic_relative_finish_validation.csv
```

The destination must not already exist. The CSV contains race
identifiers, horse names, race-normalized win probabilities, and
actual winner indicators. Expected prediction rows: 117,378.

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