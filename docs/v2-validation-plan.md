\# V2 chronological development plan



\## Purpose



Compare feature and model changes across multiple years rather than

relying solely on the repeatedly inspected 2024 validation period.



These folds are development evaluations, not untouched final tests.

Feature design has already been informed by this dataset.



\## Expanding-window folds



| Fold | Training dates | Evaluation dates |

|---|---|---|

| 2021 | 2015-01-01 through 2020-12-31 | 2021-01-01 through 2021-12-31 |

| 2022 | 2015-01-01 through 2021-12-31 | 2022-01-01 through 2022-12-31 |

| 2023 | 2015-01-01 through 2022-12-31 | 2023-01-01 through 2023-12-31 |

| 2024 | 2015-01-01 through 2023-12-31 | 2024-01-01 through 2024-12-31 |



\## Evaluation rules



\- Fit a new model and preprocessing pipeline for each fold.

\- Fit imputation, scaling, and model parameters using that fold's

&#x20; training rows only.

\- Calculate historical features using strictly earlier dates.

\- Earlier evaluation-period results may inform later historical

&#x20; features, but do not update the fitted model.

\- Exclude same-day results from historical features.

\- Keep every race entirely within one period.

\- Compare candidates on identical runners and races.

\- Keep eligibility rules, age cleaning, and model settings fixed

&#x20; for the initial comparison.

\- Do not use the 2025–2026 period for this development comparison.



\## Initial experiment



Compare two input sets using the same logistic-regression settings:



1\. Relative-finish baseline.

2\. The same inputs plus current-race age.



Retrain both input sets separately within each fold. The frozen

baseline model itself is not suitable for evaluating 2021–2023,

because it was trained on those years.



\## Reporting



Report race count, model race log loss, uniform race log loss, and

paired improvement for each fold.



Also report the overall race-weighted improvement and whether the

direction of improvement is consistent across years.



The original frozen baseline and its recorded test results remain

unchanged. A future independent final assessment requires new,

untouched data.

