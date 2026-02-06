# Changelog

All notable changes to this project are documented in this file.

The format follows [Semantic Versioning](https://semver.org/).

---

## v1.2.0 -- Race-Aware Ranking & Training Guardrails

### Added

- **Plackett-Luce race outcome model**
  - Custom top-K Plackett-Luce objective using full finish-order labels
  - Supports configurable `top_k` loss truncation for robustness
  - Produces stage-1 race-level win probabilities via softmax on scores
- **Temperature scaling for probability calibration
  - Validation-set-only temperature fitting to minimize winner log loss
  - Explicit separation between `p_win_raw` (uncalibrated) and `p_win` (calibrated)
- **Pairwise learning-to-rank baseline**
  - XGBoost `rank:pairwise` grouped by race
  - Race-aware ranking metrics (MRR, NDCG@K, mean winner rank)
- **Model card documentation**
  - Formalized assumptions, scope, non-goals, and known limitations
  - Explicit ethical / misuse considerations (`docs/model_card.md`)

### Changed

- **Unified training CLI across all models**
  - Standardized flags: `--reuse-existing`, `--fast-dev`, `--features-only`, `--base-only`
  - Consistent behavior across race-softmax, pairwise, and Plackett-Luce trainers
- **Safe reuse guardrails**
  - Canonical artifacts are reused only when all required outputs exist
  - Prevents accidental full retraining or silent partial reuse
- **Evaluation emphasis shifted toward race-aware ranking quality**
  - Metrics reported at the race level (MRR, NDCG@K, mean winner rank)
  - Clear distinction between ranking scores and calibrated probabilities
- **Repository documentation refreshed**
  - README updated to reflect multiple modeling approaches
  - Explicit non-goal statements around betting and profitability

### Verified

- No target leakage introduced by finish-order modeling
- Plackett-Luce training excludes incomplete or invalid finish-position races
- Calibration fitted strictly on validation data (no test leakage)
- Fast-dev mode reliably reduces runtime without changing semantics
- All training scripts exit cleanly when reusable artifacts already exist

### Notes

- This release expands **methodological breadth**, not performance claims.
- No betting simulation, bankroll strategy, or profitability optimization is included or implied.

## v1.1.0 — Race-Level Multinomial Softmax Model

- Introduced a race-level multinomial (softmax) XGBoost model that explicitly enforces one winner per race
- Implemented a custom race-level cross-entropy objective
- Added feature ablation experiments to assess robustness and leakage risk
- Achieved strong out-of-sample calibration (ECE ≈ 0.0036, slope ≈ 1.01)
- Model substantially outperforms uniform and implied-odds baselines on log loss and Brier score
- Added CLI flags to support fast reuse of existing models and predictions

---

## [v1.0.1] – Sanity Checks & Feature Robustness

### Added

- Automated sanity-check pipeline:
	- leakage keyword scan
	- within-race label shuffle control
	- train/test duplicate detection
- Feature snapshot artifacts for auditability:
	- `outputs/features/train_features.parquet`
	- `outputs/features/test_features.parquet`
- Systematic feature ablation runs with logged deltas:
	- base model
	- no post position
	- no recent-form feature
	- combined ablation
- Ablation reports saved as CSV, JSON, and Markdown.

### Changed

- Evaluation workflow now explicitly validates that predictive performance collapses under shuffled labels.
- Model robustness verified against removal of highly predictive pre-race features.
- Feature selection hardened to reduce accidental leakage risk.

### Verified

- No target leakage or train/test contamination detected.
- Model performance remains stable under feature ablation.
- Predictive strength arises from aggregation of multiple weak pre-race signals rather than any single dominant proxy.
- Model calibration evaluated via reliability curves, ECE, and slope/intercept diagnostics.
- Confirmed slight underconfidence but stable, well-behaved probability estimates.
- Odds-implied baseline exhibits expected overconfidence consistent with known market biases.

---

## [v1.0.0] – Initial Public Release

### Added

- End-to-end horse race modeling pipeline:
	- data ingestion and normalization
	- feature engineering
	- Optuna-tuned XGBoost model
	- race-level probability normalization
- Time-based train/validation/test split.
- Evaluation metrics and reporting (logloss, Brier, top-k accuracy).
- Baseline comparisons (uniform and odds-implied where available).

---

