# Horse Racing Outcome Prediction (v1.0.1)

See `CHANGELOG.md` for a detailed history of releases and validation steps.

Leakage-safe, time-aware machine learning pipeline for predicting horse race outcomes using historical UK & Ireland racing data.



---



## Overview



This project implements an end-to-end machine learning pipeline that predicts **win probabilities for each runner in a horse race**, using only **pre-race information** and **strict temporal validation**.



Horse racing presents several structural challenges for modeling:



- each race contains multiple dependent observations (runners)

- only one winner exists per race

- public ratings and markets already encode strong prior information

- naive random cross-validation introduces severe data leakage



This repository focuses on **methodological correctness**, **probability calibration**, and **race-aware evaluation**, rather than betting or profitability claims.



---



## Model Validation Philosophy

This project treats validation as a first-class concern, not an afterthought.

Rather than optimizing metrics alone, the pipeline explicitly tests whether strong performance could arise from:
- target leakage
- train/test contamination
- dominant proxy features
- evaluation artifacts

Validation steps include:
- within-race label shuffle controls
- time-based splits aligned with real deployment
- systematic feature ablation with logged performance deltas
- comparison against naïve and odds-implied baselines

The goal is not to claim economic arbitrage, but to demonstrate that predictive performance reflects real, pre-race structure in the data and degrades appropriately when signal is removed.

This distinction—between predictive power and economic exploitability—is intentional.



---



## Key Features



- **Leakage-safe feature construction**

	- All historical aggregates (horse, jockey, trainer form) are computed strictly from past races only

- **Time-aware evaluation**

	- Train / validation / test splits are based on race date (no random shuffling)

- **Race-normalized probabilities**

	- Predicted win probabilities are normalized within each race to sum to 1

- **Proper scoring evaluation**

	- Log loss, Brier score, top-k hit rates, and calibration curves

- **Reproducible pipeline**

	- Modular ingestion, feature building, training, and evaluation scripts



---



## Data Source



The model is trained on historical UK & Ireland race results from:



**Kaggle Dataset:**  

*Horse Racing Results – UK & Ireland (2015–2025)*  

https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025



The raw dataset is provided as a SQLite database and is **not included in this repository** due to size and licensing constraints.



---



## Feature Summary



Only information available **before post time** is used.



### Race context

- course

- distance (converted to furlongs)

- going (track condition)

- field size

- race class / type



### Runner attributes

- post position (draw)

- carried weight

- age / sex



### Ratings (pre-race)

- Official Rating (OR)

- Racing Post Rating (RPR)

- Topspeed (TS)



### Historical form (leakage-safe)

- recent finishing position trends

- win rates over last N starts

- days since last run

- jockey and trainer expanding win rates



No post-race information (e.g. margins, comments, times) is used.



---



## Modeling Approach



- **Model:** XGBoost (binary classification)

- **Target:** winner indicator (`is_winner`)

- **Hyperparameter tuning:** Optuna

- **Early stopping:** validation-based

- **Probability handling:** per-race normalization



Each runner is modeled independently, but predictions are normalized within each race to respect the single-winner structure.



---



## Train / Validation / Test Split



All splits are **time-based**:



- **Training:** races up to 2022-12-31

- **Validation:** races during 2023–2024

- **Test:** races after 2024-12-31



This avoids forward-looking leakage and simulates a real deployment scenario.



---



## Evaluation Metrics



Evaluation is performed at the **race level** using:



- **Log loss** (proper scoring rule)

- **Brier score**

- **Top-1 accuracy per race**

- **Top-3 hit rate**

- **Calibration curve**



Example outputs (generated on the held-out test set):



- `outputs/figures/calibration_model.png`

- `outputs/reports/metrics.md`



Baseline comparisons (uniform, ratings-only, implied odds) are intentionally deferred to **v1.0.1** to keep v1.0.0 focused on pipeline correctness.



---



## Repository Structure



```

horse-racing-ml/

├── src/

│ └── hrml/

│ ├── ingest/ # raw data inspection & normalization

│ ├── features/ # leakage-safe feature construction

│ ├── models/ # training & hyperparameter tuning

│ └── eval/ # evaluation & plotting

├── configs/

├── notebooks/

├── requirements.txt

├── pyproject.toml

└── README.md

```



Raw data (`data/raw`) and large derived artifacts (`data/processed`) are excluded via `.gitignore`.



---



## How to Reproduce



### 1. Create a virtual environment

```bash

python -m venv .venv

source .venv/bin/activate   # Windows: .\\.venv\\Scripts\\Activate

```



### 2. Install dependencies

```bash

pip install -r requirements.txt

```



### 3. Download the dataset


Manually download the Kaggle dataset and place the SQLite file in:

```bash

data/raw/

```



### 4. Run the pipeline

```bash

python -m hrml.ingest.inspect_raw

python -m hrml.ingest.normalize

python -m hrml.features.build_features

python -m hrml.models.train_xgb_optuna

python -m hrml.eval.run_eval

```



---



## Limitations & Future Work

- No profitability or betting simulation is performed

- Odds-based baselines are not included in v1.0.0

- Future versions will introduce:

	- ratings-only and market baselines

	- learning-to-rank objectives

	- deeper calibration analysis



---



## Versioning

- v1.0.0 -- Leakage-safe pipeline, probabilistic modeling, and proper evaluation

- v1.0.1 -- Sanity checks, feature ablations, and validation artifacts

- v1.1.0 (planned) -- Evaluation refinement and modeling extensions beyond initial validation phase



---



## License

This project is provided for educational and portfolio purposes only.

