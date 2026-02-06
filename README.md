# Horse Racing Outcome Prediction (v1.2.0-dev)

Leakage-safe, time-aware ML pipeline for predicting horse race outcomes using historical UK & Ireland racing data — with an explicit focus on **calibration**, **race-aware ranking quality**, and **methodological correctness**.

See `CHANGELOG.md` for a detailed history of releases and validation steps.

> **Non-goal:** This repo is not betting advice and does not attempt to optimize profitability.

---

## Overview

This project predicts **win probabilities for each runner in a race** using only **pre-race information** and **strict temporal validation**.

Horse racing presents several structural challenges for modeling:

- Each race contains multiple dependent observations (runners)
- Exactly one winner exists per race
- Public markets already encode strong prior information
- Naive random cross-validation introduces severe data leakage

This repository focuses on **correct methodology**, **probability calibration**, and **race-level evaluation**, not wagering or profit claims.

---

## Key Features

- **Leakage-safe feature construction**
  - All historical aggregates (horse, jockey, trainer form) are computed strictly from past races only
- **Time-aware evaluation**
  - Train / validation / test splits are based on race date (no random shuffling)
- **Race-aware objectives**
  - Winner only probability modeling (race-softmax)
  - Race-level ordering models (pairwise ranking, Plackett-Luce top-K)
- **Proper evaluation**
  - Log loss, Brier, ECE, plus race-aware ranking metrics (MRR, NDCG@K, mean winner rank)
- **Reproducible pipeline**
  - Modular ingestion, feature building, training, and evaluation scripts
- **Guardrails against leakage**
  - Deterministic feature allowlist + strict "unknown numeric feature" checks

---

## Modeling Approaches

This repo currently supports three complementary approaches:

1. **Race Softmax (winner probability; v1.1.x+)**
   - Custom race-level objective that directly optimizes a multinomial-like win target per race.
   - Outputs `p_win` that sums to 1 within a race by construction.

2. **Pairwise Learning-to-Rank**
   - XGBoost `rank:pairwise` grouped by race.
   - Produces a **ranking score** (and a probability-like normalization `p_rank` for diagnostics).

3. **Plackett–Luce (race outcome structure; v1.2.x)**
   - Custom top-K Plackett–Luce objective using finish-order labels.
   - Produces stage-1 `p_win_raw` via race softmax on scores, with optional **temperature scaling** for calibration discipline.

A short comparison write-up is in:
- `outputs/reports/model_comparison.md`

A model card (assumptions/scope/limitations) is in:
- `docs/model_card.md`

---

## Data Source

**Kaggle Dataset:** *Horse Racing Results – UK & Ireland (2015–2025)*  
https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025

The raw dataset is a SQLite database and is **not included** in this repository due to size and licensing constraints.

Place the SQLite file in:

```bash
data/raw/
```

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

No post-race information (margins, comments, times, etc.) is used.

---

## Train / Validation / Test Split

All splits are **time-based**:
- **Training:** races up to 2022-12-31
- **Validation:** races during 2023–2024 (up to **2024-12-31**)
- **Test:** races after **2024-12-31**

This avoids forward-looking leakage and simulates real deployment.

---

## Evaluation

Evaluation is performed at the **race level** (not per-runner independently).

**Win-probability quality (when applicable)**
- Log loss
- Brier score
- ECE (Expected Calibration Error)

**Race-aware ranking quality**
- MRR
- NDCG@K (K = 3, 5)
- Mean winner rank
- Winner-in-topK hit rates

Common artifacts:
- `outputs/reports/metrics.md` / `outputs/reports/metrics.json` (when generated)
- `outputs/figures/ calibration plots (when generated)
- `outputs/reports/metrics_ranking_*.md|json`

---

## Results (Held-out Test Set)

### **Race SoftMax (winner probability baseline)**

- Logloss: **0.0845**
- Brier: **0.0246**
- ECE: **0.0036**
- Top-1 accuracy (per race): **~0.832**
- Top-3 hit rate: **~0.972**
- Mean winner rank: **1.304**
- NDCG@3: **0.9171**
- NDCG@5: **0.9239**
- MRR: **0.9031**

### **Plackett-Luce (race outcome model; calibrated)

- Mean winner rank: **1.346**
- MRR: **0.8994**
- NDCG@3: **0.9120**
- NDCG@5: **0.9186**
- Winner top-3: **0.9655**
- Winner top-5: **0.9816**
- Overall ECE: **0.0052**

(See `outputs/reports/model_comparison.md` for a side-by-side summary.)

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
├── docs/
│ └── model_card.md
├── outputs/ # generated artifacts (mostly ignored; reports allowlisted)
├── requirements.txt
├── pyproject.toml
└── README.md

Raw data (`data/raw`) and most derived artifacts are excluded via `.gitignore`. Selected small reports are allowlisted under `outputs/reports/`.

```

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

```

Then train one (or all) models:

**Race SoftMax (winner probabilities)**

```bash
# full training
python -m hrml.models.train_xgb_race_softmax

# fast iteration (fewer rounds; skips ablations)
python -m hrml.models.train_xgb_race_softmax --fast-dev

# base model only (skip ablations)
python -m hrml.models.train_xgb_race_softmax --base-only

# reuse existing artifacts (prevents accidental full retrain)
python -m hrml.models.train_xgb_race_softmax --reuse-existing
```

**Pairwise Ranking**

```bash
python -m hrml.models.train_xgb_pairwise_rank
python -m hrml.models.train_xgb_pairwise_rank --fast-dev
python -m hrml.models.train_xgb_pairwise_rank --reuse-existing

# CLI consistency only (no-op for pairwise)
python -m hrml.models.train_xgb_pairwise_rank --base-only
```

**Plackett-Luce (top-K race outcome model)**

```bash
# full training (defaults: --top-k 3, --mc-samples 200, temperature scaling enabled)
python -m hrml.models.train_xgb_plackett_luce

# fast iteration
python -m hrml.models.train_xgb_plackett_luce --fast-dev

# reuse existing artifacts (prevents accidental full retrain)
python -m hrml.models.train_xgb_plackett_luce --reuse-existing

# customize PL objective & Monte Carlo extras
python -m hrml.models.train_xgb_plackett_luce --top-k 3 --mc-samples 0
```

---

## Validation & Robustness

Robustness is validated via:
- Feature ablation experiments (race-softmax)
- Strict OOS evaluation on held-out races
- Probability calibration diagnostics
- Race-aware ranking metrics

Ablation metrics:
- `outputs/reports/ablation_softmax_metrics.md`

---

## Limitations & Future Work

- No betting or profitability simulation is performed
- Market baselines are treated as diagnostics, not trading signals.
- Expected rank / place probabilities (PL MC sampling) are approximate and may be disabled (`--mc-samples 0`).

Possible next steps:
- richer baselines (ratings-only, odds-only)
- more extensive calibration stratification + reliability plots
- automated "model comparison" report generation from metrics JSONs
- stability checks across field size / distance / class regimes

---

## Versioning

- v1.0.0 -- Leakage-safe pipeline, probabilistic modeling, and proper evaluation
- v1.0.1 -- Sanity checks, feature ablations, and validation artifacts
- v1.1.x -- Race-SoftMax objective + improved evaluation discipline
- v1.2.x **(dev)** -- Outcome-structure models (pairwise ranking, Plackett-Luce), ranking metrics, unified trainer CLI

---

## License

This project is provided for educational and portfolio purposes only.

