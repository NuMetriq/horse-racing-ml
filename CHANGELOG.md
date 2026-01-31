\# Changelog



All notable changes to this project are documented in this file.



The format follows \[Semantic Versioning](https://semver.org/).



---



\## \[v1.0.1] – Sanity Checks \& Feature Robustness

\### Added

\- Automated sanity-check pipeline:

&nbsp; - leakage keyword scan

&nbsp; - within-race label shuffle control

&nbsp; - train/test duplicate detection

\- Feature snapshot artifacts for auditability:

&nbsp; - `outputs/features/train\_features.parquet`

&nbsp; - `outputs/features/test\_features.parquet`

\- Systematic feature ablation runs with logged deltas:

&nbsp; - base model

&nbsp; - no post position

&nbsp; - no recent-form feature

&nbsp; - combined ablation

\- Ablation reports saved as CSV, JSON, and Markdown.



\### Changed

\- Evaluation workflow now explicitly validates that predictive performance collapses under shuffled labels.

\- Model robustness verified against removal of highly predictive pre-race features.

\- Feature selection hardened to reduce accidental leakage risk.



\### Verified

\- No target leakage or train/test contamination detected.

\- Model performance remains stable under feature ablation.

\- Predictive strength arises from aggregation of multiple weak pre-race signals rather than any single dominant proxy.



---



\## \[v1.0.0] – Initial Public Release

\### Added

\- End-to-end horse race modeling pipeline:

&nbsp; - data ingestion and normalization

&nbsp; - feature engineering

&nbsp; - Optuna-tuned XGBoost model

&nbsp; - race-level probability normalization

\- Time-based train/validation/test split.

\- Evaluation metrics and reporting (logloss, Brier, top-k accuracy).

\- Baseline comparisons (uniform and odds-implied where available).



---

