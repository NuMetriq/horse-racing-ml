# Changelog



All notable changes to this project are documented in this file.



The format follows [Semantic Versioning](https://semver.org/).



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

