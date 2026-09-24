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

