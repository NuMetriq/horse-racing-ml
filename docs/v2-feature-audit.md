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

