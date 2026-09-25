\# V2 feature availability audit



\## Purpose



Evaluate candidate features before adding them to the frozen

horse-history baseline.



Prediction time: before the race starts, using information available

when the prediction is generated.



Raw result-table values are not automatically verified pre-race inputs.

Their timing and meaning must be checked.



\## Initial candidates



| Field | Intended meaning | Pre-race availability | Data checks | Proposed encoding | Open questions |

|---|---|---|---|---|---|

| dist | Race distance | Scheduled distance is normally known beforehand; source timing unverified | Formats, units, missing values | Numeric distance in a consistent unit | Does the source record scheduled or final distance? |

| going | Track condition | Published beforehand but may change; source timing unverified | Categories, missing values, course differences | Categorical initially | Was this value available at our prediction time or revised later? |

| age | Horse age at the race | Generally known beforehand; source definition unverified | Numeric range, missing values, unusual values | Numeric initially | Are age conventions consistent across countries? |



\## Inspection scope



Begin with raw flat-racing records dated before 2024-01-01.

Repeat relevant checks on the eligible prepared subset before modeling.



\## Modeling considerations



Distance and going are shared by runners in the same race.

Their usefulness may depend on interactions with each horse's

previous performance under similar conditions.



Age varies between runners and can be evaluated as a direct input.



\## Status



No candidate has been added to the model yet.

Observed distributions and availability evidence will be recorded here.



\## Initial training-data inspection



Inspected 991,051 raw flat-racing runner rows dated before 2024-01-01.

The eligible prepared training subset contains 989,073 runners.



| Field | Null or blank rows | Observations | Next check |

|---|---:|---|---|

| dist | 0 | Strings combine miles, furlongs, and fractions, such as 1m2f and 1m1½f | Inspect all distinct formats before implementing a parser |

| going | 45 | Multiple categorical labels, including Standard, Good, Heavy, Fast, and Sloppy | Inspect all categories and their surface/course context |

| age | 0 | The 15 most frequent values are integers from 2 through 16 | Check all values and SQLite storage types |



Zero null or blank values does not establish validity: placeholders,

unexpected formats, and implausible values still need checking.



These observations do not verify when the source values became available.



\### Complete age-value inspection



All raw flat training ages are stored as SQLite integers.

Observed values range from 1 to 16, with no nulls or blanks.

Five rows have age 1 and require investigation.

No age-based exclusions or corrections have been applied.



The five age-1 rows all concern Anthem Of Peace (AUS), racing between

2018-04-30 and 2018-06-18 in races labeled 2yo.



An external Sporting Life racecard lists age 2 for the Goodwood race

on 2018-06-10. The reason for the source disagreement is unresolved.

Raw values remain unchanged pending further investigation.



\### Initial age-cleaning decision



Anthem Of Peace (AUS) has only the five previously identified records

in the training-period query, so no later age progression is available.



For the first age experiment:

\- Preserve age in the raw source.

\- Encode ages below 2 as missing in the model input.

\- Retain the affected runners and races.

\- Do not infer individual age from the race's age band.



This is a provisional treatment of unresolved values, not a claim

that the source values have been proven incorrect. It affects five

raw training rows; the count in the eligible subset remains to be checked.



\### Eligible training subset



The prepared training subset contains 989,073 runners.

All ages are stored as integers, ranging from 1 to 16.

All five unresolved age-1 records remain after race eligibility filtering.



The provisional encoding rule will mark those five ages as missing

while retaining their runners and races.



\## Age experiment: validation results



Added current-race age to the relative-finish logistic model.

The existing features and model settings were retained.



Source ages below 2 were encoded as missing. This affected five

training runners. No runners or races were removed.



| Measure | Result |

|---|---:|

| Validation races | 11,637 |

| Validation runners | 117,378 |

| Baseline race log loss | 2.160138 |

| Age-model race log loss | 2.157301 |

| Mean paired improvement | 0.002837 |

| Median paired improvement | 0.001616 |

| Races improved | 6,163 |

| Races worsened | 5,338 |

| Races effectively unchanged | 136 |



Mean log loss improved in all twelve validation months.

Monthly improvements ranged from 0.000445 in November to 0.006115

in July.



Saved-model evaluation reproduced 2.157301.



Decision: retain age as a development candidate, pending evaluation

across additional chronological development folds. Repeated use

of 2024 means these results are model-selection evidence, not a

new independent test.



The original frozen baseline and its recorded test results remain

unchanged. The age experiment has not been evaluated on the test split.



Artifacts:

\- Model: outputs/models/v2\_logistic\_age.pkl

\- Report: outputs/reports/v2\_logistic\_age.json

\- Predictions: outputs/predictions/v2\_logistic\_age\_validation.csv

\- Encoding: relative\_finish\_age\_v1



\### Age: chronological development comparison



Each model was fitted on data from 2015 through the year before

evaluation. Both feature sets used identical training and evaluation

rows, preprocessing rules, and logistic regression settings.



| Evaluation year | Training years | Without age | With age | Log-loss improvement |

|---|---|---:|---:|---:|

| 2021 | 2015–2020 | 2.137236 | 2.135275 | +0.001961 |

| 2022 | 2015–2021 | 2.131581 | 2.129857 | +0.001724 |

| 2023 | 2015–2022 | 2.154748 | 2.153240 | +0.001508 |

| 2024 | 2015–2023 | 2.160138 | 2.157301 | +0.002837 |



Lower race log loss is better. Improvement is calculated as

without-age loss minus with-age loss.



Age improved performance in all four evaluation years. Retain age

as a model input. The gains are small, and statistical uncertainty

has not yet been quantified.



These are development comparisons: feature design was already

informed by the existing data and earlier experiments. They are

not independent confirmation on an untouched holdout.



Model parameters remained fixed within each evaluation year.

Earlier race results could contribute to features for later dates,

with same-day and future results excluded from horse history.



\### Uncertainty in the 2024 age comparison



The mean race log-loss improvement from adding age was 0.002837.



A paired bootstrap resampled the 363 represented race dates with

replacement, retaining all races on each selected date. Each

resample calculated total improvement divided by total races.



\- Resamples: 10,000

\- Random seed: 42

\- 95% percentile interval: \[0.002058, 0.003611]



The interval lies entirely above zero, supporting a small improvement

within this development sample. Together with positive improvements

in all four chronological comparisons, this supports retaining age.



The bootstrap treats dates as independent clusters. It does not

capture dependence across dates, model-training uncertainty, or

selection effects from earlier experiments. These results do not

establish performance on untouched future data or betting profitability.



\### Race-type correction and model impact



Review of 43 training-period candidates produced 22 non-flat

exclusions and 21 retained flat races. All 22 exclusions matched

previously eligible races, removing 233 runner rows.



The corrected dataset contains 126,112 races and 1,265,425 runners.

Historical features were rebuilt after exclusions.



With the same age-model settings and date boundaries, 2024

validation race log loss changed from 2.157301 to 2.157347

(a deterioration of 0.000046, based on displayed values).

The evaluation population remained 117,378 runners in 11,637 races.



Retain the correction because the excluded races fall outside

the intended flat-racing scope. No performance benefit is claimed.



This targeted review covered flagged races from 2015–2023.

It does not establish that all classification errors have been

found or that later periods have been audited.



\### Distance-change experiment



Added signed distance change in furlongs: current race distance

minus the horse's previous recorded race distance. Previous-race

information uses only earlier dates; unavailable or ambiguous

previous distances remain missing.



Both models use the same reviewed dataset, with 25 race exclusions.

Each model is fitted on data from 2015 through the year preceding

its evaluation year.



| Evaluation year | Age baseline log loss | With distance change | Improvement |

|---|---:|---:|---:|

| 2021 | 2.135242 | 2.134419 | +0.000823 |

| 2022 | 2.129812 | 2.129207 | +0.000605 |

| 2023 | 2.153239 | 2.152865 | +0.000374 |

| 2024 | 2.157350 | 2.156699 | +0.000651 |



Positive improvement means lower race log loss for the candidate.



For 2024, a paired bootstrap by race date used 363 dates,

10,000 resamples, and seed 42. The 95% percentile interval for

mean improvement was \[+0.000139, +0.001174].

Eight months improved and four worsened.



Decision: retain signed distance change. Its contribution is

small but positive across all four annual evaluations.



These are development comparisons, not a new untouched test.

The bootstrap does not account for repeated feature selection,

dependence across dates, or model-fitting uncertainty.

No betting-profitability claim follows from these results.



\### Initial histogram gradient boosting comparison



Compared histogram gradient boosting against logistic regression

using the same reviewed dataset, ten encoded input features,

and annual evaluation periods. Each model was trained on dates

from 2015 up to the start of its evaluation year.



Boosting used native missing-value handling, without scaling or

the logarithmic gap transformation used by logistic regression.

This comparison therefore evaluates the two complete pipelines.



Fixed boosting settings:

\- Learning rate: 0.05

\- Iterations: 200

\- Maximum leaves per tree: 15

\- Minimum samples per leaf: 50

\- L2 regularization: 1.0

\- Early stopping: disabled

\- Random seed: 42



| Evaluation year | Logistic log loss | Boosting log loss | Improvement |

|---|---:|---:|---:|

| 2021 | 2.134419 | 2.116284 | +0.018135 |

| 2022 | 2.129207 | 2.111592 | +0.017615 |

| 2023 | 2.152865 | 2.132037 | +0.020828 |

| 2024 | 2.156699 | 2.135187 | +0.021512 |



Probabilities were normalized within each race before evaluation.



The 2024 comparison matched 117,378 runners across 11,637 races.

All 12 months improved. A paired bootstrap over 363 race dates

used 10,000 resamples and seed 42, giving a 95% percentile interval

of \[+0.018763, +0.024281] for mean log-loss improvement.



Decision: select boosting as the current development model and

retain logistic regression as a benchmark.



These are development results, not an untouched final assessment.

The bootstrap does not account for repeated model selection,

dependence between dates, or model-fitting uncertainty.

No betting-profitability claim follows from this comparison.



\### Optuna hyperparameter search



Ran 20 trials, including the initial boosting configuration,

using a seeded TPE sampler. Each trial fitted three chronological

folds and minimized the equally weighted mean race log loss for

2021, 2022, and 2023. All trials evaluated all three years.



Trial 10 achieved the lowest tuning objective.



| Parameter | Selected value |

|---|---:|

| learning\_rate | 0.030902793355826564 |

| max\_iter | 400 |

| max\_leaf\_nodes | 30 |

| min\_samples\_leaf | 120 |

| l2\_regularization | 0.02107434007660055 |



| Evaluation year | Initial boosting | Tuned boosting | Improvement |

|---|---:|---:|---:|

| 2021 | 2.116284 | 2.115281 | +0.001003 |

| 2022 | 2.111592 | 2.110814 | +0.000778 |

| 2023 | 2.132037 | 2.130949 | +0.001088 |

| 2024 | 2.135187 | 2.134307 | +0.000880 |



The selected settings were frozen before evaluating 2024.

That year was outside the optimization objective but had already

been examined during development; it is not an untouched test.



The 2024 comparison matched 117,378 runners across 11,637 races.

Seven months improved and five worsened. A paired bootstrap by

race date, using 363 dates, 10,000 resamples, and seed 42, produced

a 95% percentile interval of \[-0.000047, +0.001803] for mean

improvement.



Interpretation: the tuned model has a slightly better observed

score, but the interval includes zero. The evidence for its

incremental advantage is weaker than the evidence for switching

from logistic regression to boosting.



Decision: preserve the tuned model as a candidate and retain

initial boosting as the established benchmark. Stop this search

at the planned budget. The bootstrap does not account for model

selection, dependence across dates, or fitting uncertainty.



\### Frozen race-probability calibration



Fitted one power exponent using chronological predictions for

2021–2023. Each year's predictions came from an initial-settings

boosting model trained only on preceding years.



The adjustment is:



q\_i = p\_i^gamma / sum\_j(p\_j^gamma)



The objective equally weights each year's mean race log loss.

Gamma was searched within \[0.5, 2.0]; the fitted value was

approximately 1.237228322. The full-precision value is saved in

outputs/reports/v2\_initial\_boosting\_calibration.json.



The exponent was frozen before application to 2024.



| 2024 measure | Result |

|---|---:|

| Original boosting race log loss | 2.135187 |

| Calibrated race log loss | 2.130549 |

| Improvement | +0.004638 |

| 95% paired date-bootstrap interval | \[+0.003014, +0.006257] |

| Months improved | 11 of 12 |



The bootstrap used 363 race dates, 10,000 resamples, and seed 42.



| Calibrated probability bin | Runners | Mean predicted | Observed wins |

|---|---:|---:|---:|

| 0–5% | 23,973 | 3.63% | 3.58% |

| 5–10% | 48,465 | 7.33% | 7.29% |

| 10–15% | 25,442 | 12.17% | 12.29% |

| 15–20% | 10,987 | 17.15% | 17.17% |

| 20–30% | 6,749 | 23.65% | 23.48% |

| 30–50% | 1,699 | 35.40% | 36.14% |

| 50–100% | 63 | 56.55% | 50.79% |



Decision: use initial boosting with frozen calibration as the

current development model. Preserve the uncalibrated and tuned

models as comparison artifacts.



The adjustment preserves runner rankings and therefore does not

change winner-selection accuracy. The highest probability bin is

small, and pooled calibration does not establish subgroup calibration.



2024 was excluded from calibration fitting but had already been

examined during development. These results are not an untouched

final assessment. The bootstrap does not account for selection,

dependence across dates, or fitting uncertainty.



\### History-coverage diagnostics



Evaluated the initial boosting model with frozen calibration on

2024 predictions, matched to the reviewed feature database.



| Race group | Races | Model loss | Uniform loss | Improvement |

|---|---:|---:|---:|---:|

| Majority without history | 685 | 2.130063 | 2.222409 | +0.092347 |

| Other races | 10,952 | 2.130579 | 2.255125 | +0.124546 |



"Majority without history" means more than half the runners have

prior\_starts = 0.



| Runner group | Runners | Mean predicted | Observed wins | Observed minus predicted |

|---|---:|---:|---:|---:|

| No earlier recorded history | 11,937 | 8.65% | 7.86% | -0.79 pp |

| Has earlier recorded history | 105,441 | 10.06% | 10.15% | +0.09 pp |



The model overestimates win probabilities for runners without

recorded history on average. Good pooled calibration does not

guarantee calibration within history groups.



Missing history refers to the selected dataset, not necessarily

a horse's racing debut. These descriptive results do not identify

the cause or establish statistical uncertainty.



Exact runner keys and winner labels were checked against the

feature database. No subgroup correction was fitted to 2024.



\### Current field-size experiment



Added current\_runner\_count as an eleventh model input.

Compared uncalibrated models using the same reviewed data and

fixed initial boosting settings, including 200 iterations.



| Evaluation year | Baseline loss | With current field size | Improvement |

|---|---:|---:|---:|

| 2021 | 2.116284 | 2.118864 | -0.002580 |

| 2022 | 2.111592 | 2.111168 | +0.000424 |

| 2023 | 2.132037 | 2.131543 | +0.000494 |

| 2024 | 2.135187 | 2.136522 | -0.001335 |



Positive improvement means lower loss for the candidate.



The feature improved two years and worsened two. Equally weighted

mean loss across 2021–2023 worsened by approximately 0.000554.

The 2024 comparison also worsened.



Decision: do not adopt current field size under this configuration.

Retain the ten-input initial boosting model and its frozen

calibration. Preserve this experiment as a negative result.



This does not establish that field size is universally unhelpful.

No additional tuning or candidate-specific calibration was performed.



The historical field count represents actual starters. Any future

use would require the active field available at prediction time,

with explicit handling of withdrawals.



\### Supplied-racecard inference replay



Integrated the selected ten-input initial boosting model and frozen

calibration into predict\_boosting\_racecard.py.



The predictor accepts a race date, distance, and CSV containing

exact horse names and current ages. Historical features use only

records from dates before the supplied race date.



A replay of 2024-01-01, Ascot (AUS), 7:50 verified:



\- All eight source features matched the prepared feature table

&#x20; for all ten runners: 80 comparisons passed.

\- Uncalibrated probabilities matched the validation export exactly.

\- Calibrated probabilities matched the calibrated export exactly.

\- Final probabilities summed to one.



The replay used racecard attributes extracted from historical data.

It establishes agreement for this example, not general live-data

availability or comprehensive inference validation.



Runner lists must contain the complete active field. The saved

calibration applies to the selected initial boosting model.



\### Broader inference replay checks



Selected four 2024 races by input characteristics rather than

prediction errors:



| Case | Date | Course | Off | Runners |

|---|---|---|---|---:|

| Small field | 2024-01-01 | Santa Anita (USA) | 11:43 | 5 |

| Large field | 2024-01-01 | Ascot (AUS) | 9:02 | 16 |

| Previous result code | 2024-01-13 | Chelmsford (AW) | 4:45 | 9 |

| All runners have history | 2024-01-01 | Newcastle (AW) | 2:00 | 11 |



Across these 41 runners:



\- All 328 source-feature comparisons passed.

\- All 82 probability comparisons matched exactly: uncalibrated

&#x20; and calibrated probabilities for every runner.

\- Prediction checks invoked the actual racecard CLI using

&#x20; temporary CSV inputs and JSON reports.



These checks establish agreement with the historical preparation

and evaluation workflow for the selected cases. They do not

establish predictive performance on new data or rule out errors

shared by both workflows.



The replay checks require local data, models, and prediction

exports and are separate from the small fixture-based CI tests.

