# Model Card: Race Outcome Models (v1.2.x)



## Summary

This repository contains models for predicting **horse race outcomes** using historical race and runner data. The primary outputs are:



- **Win probability** for each runner *within a race* (normalized across the field)

- **Race-aware ranking scores** (ordering quality evaluated per race)

- (Optional) **expected rank** / **place probability** estimates when supported by the model/objective



These models are intended for **research, evaluation, and methodological demonstration** of race-level prediction under strict leakage controls and calibration discipline.



---



## Data Source

- Data is derived from a historical UK/IRE-style horse racing dataset (races + runners) processed into:

    - `data/processed/races.parquet`

    - `data/processed/runners.parquet`

    - `data/processed/model_frame.parquet`

- The pipeline constructs race-level context (e.g., distance, going, class, field size) and runner-level historical form features using **only prior information** relative to the race date.



**Important:** Dataset coverage, labeling conventions, and completeness may vary across time periods and race types.



---



## Target Definition

### Primary target (winner-only)

- `is_winner` ∈ {0,1} per runner.

- Exactly one winner is expected per race.

- Race-level normalization is used when producing probabilities (i.e., probabilities sum to 1 within each race).



### Ranking / outcome-structure targets

Depending on the model:

- **Pairwise ranking:** learns to rank the winner above non-winners (winner vs others).

- **Plackett–Luce (top-K):** learns an ordered outcome likelihood using `finish_position`, typically using only the top-K finishers for stability.



---



## Modeling Approaches in Scope

### Race Softmax (winner probability baseline)

- Trains a race-level softmax objective (custom XGBoost objective).

- Outputs calibrated-ish `p_win` by construction (race-normalized), with strong observed calibration in OOS evaluation.



### Pairwise Learning-to-Rank baseline

- Trains XGBoost with `rank:pairwise` grouped by race.

- Outputs a **ranking score**. Any “probability-like” normalization is strictly a convenience for comparisons and is **not** guaranteed calibrated.



### Plackett–Luce race model

- Trains a top-K Plackett–Luce objective (custom XGBoost objective) using finish order information.

- Outputs race-normalized `p_win` from the model scores (stage-1 probability under the PL formulation).

- Typically benefits from **post-hoc calibration** (e.g., temperature scaling) to maintain calibration discipline.



---



## Leakage Controls

Leakage prevention is treated as a first-class requirement.



### Temporal discipline

- Historical features (horse/jockey/trainer form) are computed **only from races prior to the current race date**.

- Updates to historical state occur **after** feature computation for a race.



### Feature allowlist enforcement

- Training uses an explicit **feature allowlist/prefix policy**.

- A strict “unknown numeric feature” check fails fast if new numeric columns appear unexpectedly.

- Known post-race labels are explicitly forbidden (e.g., `is_winner`, `finish_position`).



### Race completeness filtering (when required)

- Plackett–Luce training requires complete and valid finish position labels.

- Races with missing or invalid finish order data are excluded from PL training/evaluation.



---



## Validation Methodology

### Out-of-sample (OOS) temporal split

- Data is split chronologically into train/validation/test windows.

- The held-out test set represents **future races** relative to training.



### Metrics reported

Two classes of evaluation are used:



#### Win-probability quality (when applicable)

- **Logloss**

- **Brier score**

- **ECE (Expected Calibration Error)** overall and stratified by race context



#### Race-aware ranking quality (per race)

- **MRR** (winner reciprocal rank)

- **NDCG@K** (K = 3, 5)

- **Mean winner rank**

- **Winner-in-topK** hit rates



### Calibration stratification diagnostics

Calibration is evaluated not only overall, but across race context buckets such as:

- Field size buckets

- Distance buckets (e.g., sprint / middle / staying)

- Race class buckets



---



## Known Limitations

- **Dataset limitations:** Missingness, inconsistent labeling, and coverage shifts over time can degrade performance.

- **Domain shift:** Model quality may not transfer across jurisdictions, surfaces, race conditions, or eras not represented in training.

- **Non-stationarity:** Racing evolves (trainers, breeding, track conditions, rules), so historical patterns may not hold.

- **Partial observability:** Many important drivers (fitness, injuries, strategy, stable intent) are not in the data.

- **Ranking vs probability tension:** Some ranking-optimized models may improve ordering while degrading probability calibration unless calibrated explicitly.

- **Plackett–Luce label dependency:** PL requires finish order labels and can exclude races where the order is incomplete or unreliable.



---



## Non-Goals

These models are explicitly **not** designed for:

- Betting strategy optimization

- Profitability estimation

- Market inefficiency exploitation

- Advice to place wagers

- Real-time production deployment without additional monitoring and governance



This project is about **predictive structure**, **calibration**, and **methodological clarity**.



---



## Ethical / Misuse Considerations

Even when framed as a research project, race outcome prediction can be misused.



- **Gambling harm:** Predictions could be used to encourage or intensify gambling behavior. This repo does not provide betting advice or bankroll strategy.

- **False certainty:** Well-calibrated probabilities can still be wrong on individual races; users may over-trust outputs.

- **Misleading transfer:** Applying the model to different countries, eras, or race types without evaluation can produce unreliable results.

- **Responsible communication:** Any public-facing use should emphasize uncertainty, limitations, and that this is not financial advice.



If you extend this work, consider adding safeguards such as:

- Prominent disclaimers in any UI

- Calibration monitoring over time

- Clear documentation of data coverage and evaluation windows



---



## Changelog Notes (v1.2.x focus)

- Expanded evaluation from win-only metrics to **race-aware ranking metrics**.

- Added **stratified calibration diagnostics** (ECE + reliability plots by race strata).

- Introduced ranking-oriented baselines and race outcome structure objectives while preserving leakage and calibration discipline.

