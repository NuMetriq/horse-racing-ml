# Pipeline Contract



| Stage       | Inputs                        | Outputs                 | Must NOT Know About |
|-------------|-------------------------------|-------------------------|---------------------|
| ingest      | raw data sources              | canonical race table    | labels, odds        |
| features    | canonical table               | X, race_ids, runner_ids | outcomes            |
| model       | X                             | scores or logits        | odds, policies      |
| calibration | scores, outcomes (train only) | calibrated p_win        | odds, policies      |
| policy      | p_win, odds, race structure   | bets (runner, stake)    | labels              |
| evaluation  | bets, outcomes                | metrics, plots          | model internals     |


---

## ingest

### `src/hrml/ingest/inspect_raw.py`

#### Purpose
Human-oriented inspection utility for raw SQLite files. Lists tables, columns, row counts, and small samples to understand schema and data shape before normalization.

#### Stage role
- **ingest (diagnostics / inspection)**
- Does **not** transform data or emit canonical datasets.

#### Reads/Writes
- **Reads**: SQLite files in `--raw-dir` (default: `data/raw/`): `*.db`, `*.sqlite`, `*.sqlite3`
- **Writes**: none (stdout only)

#### Entry points
- `inspect_sqlite(db_path: Path) -> None`
- `main() -> None` (CLI utility; supports `--raw-dir`)

#### Inputs
- `--raw-dir` (Path, optional; default `data/raw`)

#### Outputs
-Printed to stdout:
  - table names
  - column names per table
  - row counts (best-effort)
  - sample rows (LIMIT 5)

#### Invariants / Guardrails
- Must not modify the raw database.
- Must not write to `data/processed/`.
- Must not depend on any downstream schema assumptions (no "canonical" column names required)

#### Must NOT know about
- labels/outcomes semantics (e.g., `is_winner`, `finish_position`)
- odds semantics (e.g., implied probabilities)
- feature/model/calibration/policy logic


### `src/hrml/ingest/normalize.py`

#### Purpose
Read the raw SQLite dataset and produce **canonical, analysis-ready parquet tables**:
- `races.parquet` (one row per race)
- `runners.parquet` (one row per runner entry)
Includes schema normalization, type coercion, and parsing of common racing encodings (distance, weight, odds).

#### Stage role
- **ingest (normalization / canonicalization)**
- Produces the canonical tables consumed by downstream stages.

#### Reads/Writes
- **Reads**: `data/raw/raceform.db` (SQLite)
  - Table: `TABLE_NAME="table"`
- **Writes**: `data/processed/`
  - `races.parquet`
  - `runners.parquet`

#### Entry points
- `load_table(con: sqlite3.Connection) -> pd.DataFrame`
- `normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`
- `main() -> None` (CLI utility; reads raw DB and writes parquet)

#### Inputs
- Raw SQLite DB file at `RAW_DB`
- Raw schema must include (minimum viable):
  - race identifiers: `race_id`
  - race date: `date`
  - race context: `course`, `dist`, `going`, `type`, `class` (best-effort; missing allowed)
  - runner info: `horse`, `jockey`, `trainer`
  - outcome: `pos` (finish position) **if available**
  - odds: `sp` **if available**

#### Outputs
`races.parquett` (canonical races table)
One row per `race_id`, containing (best-effort):
- `race_id` (str)
- `race_date` (datetime)
- race context: `course`, `off`, `race_name`, `race_class`, `pattern`, `rating_band`, `age_band`, `sex_rest`
- normalized race fields:
  - `dist_f` (float, furlongs)
  - `going` (str)
  - `field_size` (int/float)
  - `prize` (numeric if paresable)

`runners.parquet` (canonical runners table)
One row per runner entry, containing (best-effort):
- identifiers: `race_id` (str), `horse_id` (str), `jockey_id` (str), `trainer_id` (str)
- race_date: `race_date` (datetime)
- pre-race fields:
  - `post_position` (float)
  - weight_lbs (float)
  - ratings: `or_rating`, `rpr_rating`, `ts_rating` (float)
  - `age`, `sex` (as present)
  - `num` (runner number, as present)
  - `sp_decimal` (float, decimal odds)
  - `field_size` (computed per race)
- outcome fields (if present in raw)
  - `finish_position` (float)
  - `is_winner` (int: 1 if finish_position==1 else 0)

#### Normalization rules
- Date parsing: `date` parsed with `errors="coerce"`.
- Key types: `race_id` cast to `str`.
- IDs: `horse_id`, `jockey_id`, `trainer_id` derived from raw name columns (v1.0.0 convention).
- Distance normalization: `dist` → `dist_f` via `parse_dist_to_furlongs`.
- Weight normalization: `wgt` → `weight_lbs` via `parse_wgt_to_lbs`.
- Odds normalization: `sp` → `sp_decimal` via `parse_sp_to_decimal`.
- Field size: computed as `count(runners) per race_id`.

#### Invariants / Guardrails
- Ingest outputs must be **canonical and complete enough` for features to run without re-reading raw sources.
- Output tables must be deterministic given the same raw DB.
- `field_size` must match the number of runner rows per `race_id` in `runners.parquet`.

#### Must NOT know about
- feature engineering logic
- model training logic
- calibration logic
- policy/betting logic
- evaluation metrics / profit calculations
*(Note: This module may include outcome and odds columns as raw-normalized fields, but it must not compute any derived "decision" quantities like implied probabilities, edges, or EV.)*

#### Failure modes
- If `RAW_DB` is missing: raise `FileNotFoundError`.
- If SQLite read fails: propagate exception (or print and exit, depending on desired CLI behavior).
- If parsing fails for individual field: produce `NaN` (best-effort parsing functions already do this).

#### IMPORTANT NOTES!!!
- **Outcomes** (`finish_position`, `is_winner`) **live in canonical runners**.
  - Downstream stages must treat them as labels, not features.
- **Odds* (`sp_decimal`) **live in canonical runners**.

---

## features
### `src/hrml/features/build_features.py`
#### Purpose
Build a runner-level, model-ready feature frame using only **pre-race information** plus **history from prior races**. Enforces strict ordering by race date and updates historical aggregates only after features are computed for a race.

#### Inputs
- `races.parquet`, `runners.parquet` (canonical tables)
- May contain outcomes (`finish_position`, `is_winner`) used **only** for history updates after feature generation.

#### Outputs
- Runner-level frame including ids + race context + engineered features (e.g., recent horse finishes, expanding jockey/trainer rates)

#### Invariants / Guardrails
- Features for race *R* computed using only data from races strictly earlier than *R*
- No same-race leakage: store updates occur only after feature computation for the entire race

#### Must NOT know about
- policy logic (betting, staking)
- evaluation/reporting
- calibration fitting logic


### `src/hrml/features/allowlist.py`
#### Purpose
Define a strict, deterministic **feature selection policy** to prevent silent leakage regressions. Provides:
- a curated allowlist of safe pre-race numeric fields
- prefix-based inclusion rules for engineered features
- a blocklist of identifiers, labels, and known leakage-prone fields
- a strict mode that fails if unknown numeric columns appear

#### Stage role
- **features (feature selection / guardrails)**
- Used by modeling code to select feature columns consistently and safely.

#### Key components
- `DEFAULT_BLOCKLIST`: columns that are never features (ids, labels, outcomes, odds-baseline columns, etc.)
- `DEFAULT_LEAKAGE_PATTERNS`: token patterns that indicate likely post-race/outcome leakage
- `FeatureAllowlist`: feature selection policy object
  - `explicit`: exact column names allowed as "safe pre-race" numeric features
  - `prefixes`: allowed engineered feature prefixes (e.g., `horse_`, `jockey_`)
  - `blocklist`: forbidden columns (frozen)
  - `strict_unknown_numeric`: if `True`, raise if numeric columns exist that are not allowed or blocklisted
  - `leakage_token_block`: tokens used to flag suspicious unknown numeric columns
- `write_feature_manifest(feature_cols, out_path) -> dict`: writes a stable manifest with SHA256 hash of sorted feature names

#### Inputs
- `FeatureAllowList.select(df: pd.DataFrame)`: takes a runner-level frame that includes candidate feature columns (e.g., model frame)

#### Outputs
- Returns `List[str]` of feature column names:
  - deterministic ordering (sorted)
- Optional manifest JSON written by `write_feature_manifest`:
  - `{ n_features, sha256, features }

#### Invariants / Guardrails
- Feature columns must be stable and reproducible (sorted unique list).
- If `strict_unknown_numeric=True`, selection fails if the DataFrame contains any numeric/bool column that is neither:
  - explicitly allowed, nor
  - prefix-allowed, nor
  - blocklisted
- Error message must enumerate unknown numeric columns and separately highlight suspicious leakage-like names

#### Must NOT know about
- training splits or calibration
- betting policies
- evaluation metrics

#### Contract notes
- Odds-derived columns (`sp_decimal`, `odds_implied*`) are currently in the blocklist to prevent market leakage unless explicitly re-enabled.
- Any new numeric engineered feature should either:
  - adopt an allowed prefix, or
  - be added to the explicit allowlist intentionally (preferred)

### `src/hrml/features/history_aggregates.py`
#### Purpose
Provide reusable, in-memore **history aggregation primitives** used to compute leakage-safe "form" features (recent horse finished, expanding jockey/trainer win rates, and jockey-trainer combo rates) with strict chronological updates.

#### Stage role
- **features (history aggreagates / feature primitives)**
- Pure Python stateful utilities; no IO.

#### Key components
- `RecentStats`
  - Stores recent `finishes` and `dates` (ordinal ints) with bounded lookback (`deque(maxlen=10)`).
  - Provides:
    - `mean_finish_last(n)`
    - `win_rate_last(n)`
    - `days_since_last(current_date_ordinal)`
- `ExpandingRate`
  - Tracks expanding `wins` and `starts`
  - Provides `rate()`
- `HistoryStore`
  - Maintains per-entity histories:
    - `horse_recent: Dict[str, RecentStats]`
    - `jockey_exp: Dict[str, ExpandingRate]`
    - `trainer_exp: Dict[str, ExpandingRate]`
    - `combo_exp: Dict[(jockey_id, trainer_id), ExpandingRate]`
  - Provides:
    - `features_for_runner(horse_id, jockey_id, trainer_id, race_date_ordinal) -> dict`
    - `update_after_race(horse_id, jockey_id, trainer_id, finish_pos, race_date_ordinal) -> None`

#### Inputs
- Runner identifiers: `horse_id`, `jockey_id`, `trainer_id`
- `race_date_ordinal` (int; date ordinal)
- `finish_pos` (int) for updates

#### Outputs
- `features_for_runner(...)` returns a dict of engineered feature values (may contain `None` when history is insufficient):
  - `horse_mean_finish_last5`
  - `horse_win_rate_last10`
  - `horse_days_since_last`
  - `jockey_win_rate_exp`
  - `trainer_win_rate_exp`
  - `jockey_trainer_win_rate_exp`

#### Invariant / Guardrails
- Features must reflect **history strictly prior to the current race**:
  - Call `features_for_runner` before any `update_after_race` for the same race.
- No within-race leakage:
  - `update_after_race` must occur only after outcomes are known (post-race) and after current-race features have been computed.
- Returned feature keys are stable and intended to match allowlist prefix rules (e.g., `horse_`, `jockey_`, `trainer_`).

#### Must NOT know about
- dataset splits
- odds / market features
- model training, calibration, policies, evaluation
- file paths / parquet IO

#### Failure/edge behavior
- When no history exists, methods return `None` rather than numeric defaults; downstream stages must handle imputation.

---

## model

### `train_xgb_optuna.py` (v1.0.0 baseline)
#### Purpose
Train a baseline runner-level XGBoost classifier (winner vs not) with Optuna hyperparameter search, using a time-based split. Produces per-runner win probabilities and race-normalized probabilities for evaluation and reporting.
#### Stage role
- Primarily **model** stage (training + inference)
- Also performs some **evaluation/reporting** (writes predictions and ablation reports)
*(Contract note: in later versions, evaluation/report writing should move to the evaluation stage, but this script currently bundles them.)*

#### Inputs
##### Required artifacts
- `data/processed/model_frame.parquet` (runner-level frame)
  - Required columns:
    - identifiers: `race_id`, `race_date`, `horse_id`
    - label: `is_winner` (binary)
  - Required for race normalization:
    - `race_id`
  - Feature candidates:
    - numeric columns not in the drop list
  - Optional columns (passed through to predictions):
    - `field_size`
    - `sp_decimal`
    - `odds_implied_norm` (if present)
##### Parameters / constants
- `RANDOM_SEED=42`
- Time split cutoffs:
  - train: `race_date <= 2022-12-31`
  - valid: `2023-01-01 .. 2024-12-31`
  - test: `race_date > 2024-12-31`
- Optuna trials: `n_trials=30`
- XGBoost:
  - `n_estimators=5000`
  - `early_stopping_rounds=200`
  - `tree_method="hist"`

#### Outputs
##### Model artifact
- `outputs/models/xgb_optuna.json` (XGBoost model saved via `save_model`)
##### Predictions artifact
- `outputs/reports/pred_test.parquet`
  - Columns (minimum):
    - `race_id`, `race_date`, `horse_id`, `is_winner`, `field_size` (if present)
    - `p_win_raw` (uncalibrated per-runner probability)
    - `p_win` (race-normalized; sums to ~1 per race)
  - Optional passthrough:
    - `p_odds` if `odds_implied_norm` exists
    - `sp_decimal` if exists
##### Feature dump artifacts (debut/inspection)
- `outputs/features/train_features.parquet`
- `outputs/features/test_features.parquet`
  - Contain: ids + label + selected feature columns
##### Ablation reports
- `outputs/reports/ablation_metrics.csv`
- `outputs/reports/ablation_metrics.json`
- `outputs/reports/ablation_metrics.md`

#### Core behaviors / invariants
##### Feature selection rule
- Uses numeric columns excluding:
  - identifiers (`race_id`, `race_date`, `course`, `horse_id`, `jockey_id`, `trainer_id`)
  - labels (`finish_position`, `is_winner`)
  - odds baseline columns (`odds_implied`, `odds_implied_norm`)
  - ordering helper (`race_date_ord`)
##### Missing value handling
- Median imputation computed from **train split only**
- Same medians applied to valid/test
##### Race normalization
- Converts `predict_proba` outputs into per-race probabilities by dividing by the per-race sum.
- Output `p_win` is clipped to `[1e-12, 1-1e-12]`.
##### Early stopping
- Fit uses validation set for early stopping and Optuna objective.

#### Guardrails
##### Temporal split
- Splitting must be based on `race_date` to prevent temporal leakage.
##### No outcome leakage into features
- The model must not use `is_winner`/`finish_position` as feature inputs (enforced by feature drop list).
- Race normalization must not use labels.
##### Reproducibility
- `random_state` set to `RANDOM_SEED`
- Note: full determinism can still vary due to multithreading / XGBoost internals unless additional settings are pinned.

#### Must NOT know about (ideal, even if this script currently violates it)
- Betting policies / staking
- Decision metrics (EV/profit)
- Calibration fitting beyond producing `p_win` (race-normalization is not calibration)
*(Contract note: this script writes and reports metrics; in v1.3.0+ those belong to evaluation.)*

#### Failure Modes
- If `data/processed/model_frame.parque` missing: raises `FileNotFoundError` instructing to build features first.
- If required columns missing or `race_date` unparsable: may raise during split or model fit.
- If a race has all-zero predicted probs (unlikely): race normalization protects via `sums <= 0 => 1.0`.


### `src/hrml/models/train_xgb_softmax.py`
#### Purpose
Train a race-aware XGBoost model using a **custom race-level softmax cross-entropy objective** (one winner per race). Produces per-runner scores and **race-normalized win probabilities** via a stable softmax within each race. Also writes a feature manifest and optional ablation reports.

#### Stage role
- Primary: **model (training + inference)**
- Secondary (side effects): writes feature manifest and ablation reports under `outputs/reports/`

#### Reads/Writes
- **Reads**: `data/processed/model_frame.parquet`
- **Writes**:
  - model artifact: `outputs/models/xgb_race_softmax.json`
  - predictions: `outputs/reports/pred_test.parquet`
  - feature manifest: `outputs/reports/feature_list.json`
  - feature snapshots (debug): `outputs/features/train_features.parquet`, `outputs/features/test_features.parquet`
  - optional ablation reports:
    - `outputs/reports/ablation_softmax_metrics.csv`
    - `outputs/reports/ablation_softmax_metrics.json`
    - `outputs/reports/ablation_softmax_metrics.md`

#### CLI/ entry point
- `main() -> None`
- Flags:
  - `--reuse-existing`: if model + pred exist, exit without training
  - `--features-only`: only compute feature list + manifest, then exit
  - `--base-only`: skip ablations, train only base model
  - `--fast-dev`: reduce boosting rounds and skip ablations

#### Inputs
- `model_frame.parquet` must include:
  - identifiers: `race_id`, `race_date`, `horse_id`
  - label: `is_winner` (binary)
  - optional: `final_position`, `field_size`
  - optional market columns (passed through to predictions if present):
    - `sp_decimal`
    - `odds_implied_norm`
- Feature columns are selected via:
  - `FeatureAllowList.select(df)` (strict mode default: fail on unknown numeric columns)
  - Additional numeric coercion check `_assert_features_numeric`

#### Outputs
- `pred_test.parquet` schema (minimum)
  - `race_id`, `race_date`, `horse_id`
  - `is_winner` (truth label, for evaluation only)
  - optional: `finish_position`, `field_size`
  - `score` (raw model score per runner)
  - `p_win` (race-softmax probability per runner; sums to ~1 per race)
  - optional passthrough:
    - `p_odds` if `odds_implied_norm` exists
    - `sp_decimal` if exists
- `feature_list.json` schema:
  - `n_features`
  - `sha256` of sorted feature names
  - `features` (sorted list)

#### Core behaviors / invariants
- **One-winner constraint enforcement**: races must have exactly one `is_winner==1` (filtered by `_ensure_one_winner_per_race`).
- **Group construction**: DMatrix includes group sizes per race; ordering is stable by `race_date`, `race_id`.
- **Race probability mapping**: probabilities computed by applying softmax to scores within each race.
- **Missing value handling**: median imputation computed from train split only and applied to train/valid/test.
- **Strict leakage guardrails**: unknown numeric columns cause failure when `strict=True` (default).

#### Training split
- Time-based split:
  - train: `race_date <= 2022-12-31`
  - valid: `2023-01-01 .. 2024-12-31`
  - test: `race_date > 2024-12-31`

#### Training objective
- Custom objective `race_softmax_obj(preds, dtrain)`:
  - uses race-level softmax to compute gradients/hessians as `p - y` and `p(1-p)` within each group
- Custom eval metric `race_nll_eval`:
  - computes negative log likelihood of the winner probability per race

#### Must NOT know about (contract intent)
- betting policies / staking
- decision metrics (EV/profit) and reporting beyond prediction artifacts
- calibration fitting (this produces `p_win` but does not perform probability calibration)

#### Failure modes
- Missing `model_frame.parquet`: raises `FileNotFoundError` instructing to build features first.
- Any selected feature non-numeric/coercion failure: raises `TypeError` with sample values.
- If unknow numeric columns appear (strict allowlist): raises `AssertionError` listing unknown and suspicious leakage-like columns.
- If races violate one-winner rule: they are filtered out; training/eval uses only "good" races.


### `src/hrml/models/train_xgb_pairwise_rank.py`
#### Purpose
Train an XGBoost Learning-to-Rank model using `rank:pairwise` grouped by `race_id` as a race-aware baseline. Produces runner-level ranking scores and a within-race softmax-normalized "probability-like" column (`p_rank`) for diagnostics only (not calibrated). Also writes a feature manifest and a lightweight ranking-metrics summary.

#### Stage role
- Primary: **model (training + inference**
- Secondary: invokes **evaluation** (`hrml.eval.ranking`) to write ranking reports for this baseline

#### Reads/Writes
- **Reads**: `data/processed/model_frame.parquet`
- **Writes**:
  - model artifact: `outputs/models/xgb_rank_pairwise.json`
  - predictions: `outputs/reports/pred_test_pairwise.parquet`
  - feature manifest: `outputs/reports/feature_list_pairwise.json`
  - ranking reports:
    - `outputs/reports/metrics_pairwise.json`
    - `outputs/reports/metrics_pairwise.md`
  - quick metrics summary
    - `outputs/reports/metrics_pairwise.json`

#### CLI / entry point
- `main() -> None`
- Flags:
  - `--reuse-existing`: reuse model + predictions if present
  - `--features-only`: only build feature list + manifest then exit
  - `--fast-dev`: fewer boosting rounds
  - `--base-only`: no-op (kept for CLI consistency

#### Inputs
- `model_frame.parquet` must include:
  - identifiers: `race_id`, `race_date`, `horse_id`
  - label: `is_winner`
  - optional: `finish_position`, `field_size`
- Feature columns selected via:
  - `FeatureAllowList(strict_unknown_numeric=True).select(df)`
  - plus numeric-coercion guard (`_assert_features_numeric`)
- Only races with exactly one winner are kept via `_ensure_one_winner_per_race`.

#### Outputs
##### Predictions: `pred_test_pairwise.parquet`
Minimum schema:
- identifiers/context: `race_id`, `race_date`, `horse_id`
- truth label (for eval only): `is_winner`
- optional passthrough: `finish_position`, `field_size`
- model outputs:
  - `score` (pairwise rank score; higher = better)
  - `p_rank` (race-softmax of `score`; diagnostic only, **not calibrated**)

##### Ranking evaluation artifacts (race-aware)
- `metrics_ranking_pairwise.json` / `.md` generated by `hrml.eval.ranking` using:
  - `race_col="race_id"`
  - `score_col="score"`
  - `winner_col="is_winner"`
  - `k_values=(3,5)`
- `metrics_pairwise.json`: a small summary subset (n_races, mean_winner_rank, mrr, ndcg@3, ndcg@5)

##### Feature manifest
- `feature_list_pairwise.json` containing sorted feature list and sha256 hash.

#### Training split
- Time-based:
  - train: `race_date <= 2022-12-31`
  - valid: `2023-01-01 .. 2024-12-31`
  - test: `race_date > 2024-12-31`

#### Modeling details
- XGBoost params:
  - `objective="rank:pairwise"`
  - `eval_metric="ndcg@5"`
  - tree method: `hist`
  - seeded with `RANDOM_SEED`
- Early stopping uses validation set.
- Missing value handling:
  - median imputation computed from tain split only and applied to train/valid/test.

#### Core invariants / guardrails
- Grouping must be consistent: group sizes derived from sorted `race_date`, `race_id`.
- Feature leakage prevention enforced via allowlist + forbidden overlap check (`is_winner`, `finish_position`).
- `p_rank` must be treated as diagnostic (sums to ~1 per race) but not interpreted as calibrated probability

#### Must NOT know about (contract intent)
- betting policies / staking
- decision metrics (EV/profit)
- probability calibration fitting

#### Failure modes
- Missing `model_frame.parquet`: raises `FileNotFoundError`.
- Feature coercion failure: raises `TypeError` with sample values.
- Unknown numeric columns when strict allowlist enabled: raises `AssertionError`.
- Races with invalid winner counts are filtered out before training/eval.


### `train_xgb_plackett_luce.py`
#### Purpose
Train a race-aware XGBoost model using a **custom top-K Plackett–Luce negative log-likelihood objective** over full finish-order labels. Produces runner-level scores plus race-softmax win probabilities. Optionally computes Monte Carlo–based expected rank and place probabilities. Fits a **single temperature scalar** on the validation split for calibration of win probabilities.

#### Stage role
- Primary: **model (training + inference)**
- Secondary:
  - **calibration** (temperature scaling fit on validation)
  - invokes **evaluation** (`hrml.eval.ranking`) to write ranking reports

#### Reads/Writes
- **Reads**: `data/processed/model_frame.parquet`
- **Writes**:
  - model artifact: `outputs/models/xgb_plackett_luce.json`
  - predictions: `outputs/reports/pred_test_plackett_luce.parquet`
  - feature manifest: `outputs/reports/feature_list_plackett_luce.json`
  - calibration artifact (if enabled): `outputs/models/xgb_plackett_luce_temperature.json`
  - ranking reports:
    - `outputs/reports/metrics_ranking_plackett_luce.json`
    - `outputs/reports/metrics_ranking_plackett_luce.md`
  - quick metrics summary:
    - `outputs/reports/metrics_plackett_luce.json`

#### CLI / entry point
- `main() -> None`
- Flags:
  - `--resuse-existing`: reuse model + predictions (+ temperature if calibration enabled)
  - `--features-only`: only build feature list + manifest then exit
  - `--fast-dev`: fewer boosting rounds
  - `--top-k`: PL loss truncation K (default 3)
  - `--mc-samples`: MC samples per race for `expected_rank` + `p_place_le_k` (0 disables)
  - `--place-k`: place threshold for `p_place_le_k` (default 3)
  - `--no-calibrate`: disable temperature scaling (sets `p_win == p_win_raw`)
  - `--base-only`: disables calibration and MC extras (training unchanged)

#### Inputs
- `model_frame.parquet` must include:
  - identifiers: `race_id`, `race_date`, `horse_id`
  - labels:
    - `is_winner` (required for filtering and calibration loss)
    - `finish_position` (**required** for PL objective; must be complete per race)
  - optional: `field_size`
- Feature columns selected via:
  - `FeatureAllowlist(strict_unknown_numeric=True).select(df)`
  - numeric-coercion guard `(_assert_features_numeric)`
- Data filtering:
  - races must have exactly one winner: `_ensure_one_winner_per_race`
  - races must have **complete, finite finish positions for all runners**: `_ensure_complete_finish_positions` (drops entire races if any row invalid)

#### Training split
- Time-based:
  - train: `race_date <= 2022-12-31`
  - valid: `2023-01-01 .. 2024-12-31`
  - test: `race_date > 2024-12-31`

#### Modeling details
- Grouped XGBoost via `DMatrix.set_group` with group sizes per `race_id`
- **Custom objective**: `plackett_luce_obj_factory(cfg)`
  - Uses `finish_position` stored in `DMatrix.label`
  - Optimizes truncated PL NLL over the first `top_k` finishers (stops early if positions missing/duplicated)
- **Custom metric**: `pl_nll_eval_factory(cfg)` reports mean PL NLL@K

#### Probability outputs
- `score`: raw model score per runner
- `p_win_raw`: stage-1 PL win probabilities computed as race softmax over scores (`softmax(score)`)
- `p_win`: calibrated win probabilities via temperature scaling if enabled (`softmax(score / T)`)

#### Optional MC diagnostics (if `mc_samples > 0`)
- Adds:
  - `expected rank`
  - `p_place_le_{place_k}`
- MC sampling uses `RANDOM_SEED` for reproducibility

#### Calibration behavior (temperature scaling)
- Fits scalar temperature **on validation split only** by minimizing winner logloss computed from per-race softmax(score/T) vs `is_winner`
- Search method: log-space grid + local refinement (no scipy dependency)
- Writes `outputs/models/xgb_plackett_luce_temperature.json` containing:
  - `temperature`
  - `valid_logloss`

#### Outputs
##### Predictions: `pred_test_plackett_luce.parquet`
Minimum schema:
- identifiers/context: `race_id`, `race_date`, `horse_id`
- truth labels (for evaluation only): `is_winner` (+ `finish_position` if present)
- optional: `field_size`
- model outputs:
  - `score`
  - `p_win_raw`
  - `p_win`
- optional MC outputs (if enabled):
  - `expected_rank`
  - `p_place_le_{place_k}`

##### Ranking evaluation artifacts
- Generated by `hrml.eval.ranking` using:
  - `race_col="race_id"`, `score_col="score"`, `winner_col="is_winner"`, `k_values=(3,5)`
- `metrics_plackett_luce.json` contains summary stats plus config values (`pl_top_k`, `mc_samples`, `temperature`).

##### Feature Manifest
- `feature_list_plackett_luce.json` contains sorted feature list and sha256 hash

#### Core invariants / guardrails
- No leakage through feature selection: allowlist + forbidden overlap check for `is_winner`/`finish_position`
- PL training requires complete finish order labels; races with any missing `finish_position` are dropped wholesale
- Calibration fit must use validation split only (never test)

#### Must NOT know about (contract intent)
- betting policies / staking
- decision metrics (EV/profit)
- post-hoc evaluation beyond writing prediction artifacts and ranking reports

#### Failure modes
- Missing `model_frame.parquet`: raises `FileNotFoundError`.
- Missing `finish_position`: raises `ValueError` (PL requires it).
- Any race with invalid/missing finish positions: dropped entirely by `_ensure_complete_finish_positions`.
- Feature coercion failure: raises `TypeError` with sample values.
- Unknown numeric columns under strict allowlist: raises `AssertionError`.
- Calibration:
  - invalid temperature (non-finite or <= 0) raises `ValueError`.

#### Calibration (cross-reference entry)
##### Temperature scaling implemented in `train_xgb_plackett_luce.py`
###### Purpose
Fit scalar temperature `T` on validation set to calibrate race-SoftMax probabilities.

###### Inputs
- validation scores trained from PL booster
- `race_id`, `is_winner` labels

###### Outputs
- `outputs/models/xgb_plackett_luce_temperature.json` with `temperature`, `valid_logloss`
- `p_win` defined as `softmax(score / T)` per race

###### Guardrail
- Must fit on validation only (not test).

---

## Calibration

---

## Policy

---

## Evaluation
### `src/hrml/eval/ranking.py`
#### Purpose
Compute **race-aware ranking metrics** from a predictions parquet. Metrics are computed **per race** (grouped by race id), then aggregated across races. Writes both machine-readable JSON and a human-readable Markdown report.

#### Stage role
- **evaluation** only (reads predictions + labels, writes reports)
- No model training, no feature building.

#### Reads/Writes
- **Reads**: `cfg.pred_path` (default: `outputs/pred_test.parquet`) as parquet
- **Writes**:
  - `cfg.out_json` (default: `outputs/reports/metrics_ranking.json`)
  - `cfg.out_md` (default: `outputs/reports/metrics_ranking.md`)

#### Entry points:
- `run_ranking_eval(cfg: RankingEvalConfig) -> dict`
- Supporting functions:
  - `infer_columns(df, cfg) -> (race_col, score_col, winner_col)`
  - `compute_race_ranking_metrics(...) -> dict`

#### Inputs
##### Required content (conceptual)
Prediction table must contain, per runner row:
- a race identifier column (race grouping key)
- a score/probability columns where **higher = better** (more likely winner)
- a winner indicator (binary, truth label)

##### Column naming (inference)
If not explicitly provided in `RankingEvalConfig`, columns are inferred by first-match among:
- `race_col`: one of `["race_id", "raceID", "race", "event_id", "race_key"]'
- `score_col`: one of `["p_win", "pred_win_prob", "prob_win", "y_pred", "pred", "score", "logit"]`
- `winner_col`: one of `["is_winner", "winner", "y_true", "target", "label", "won", "win"]`
If any cannot be inferred, raises `ValueError` with guidance to pass explicit names.

#### Outputs
##### Report JSON schema (written to `out_json`)
A dict with:
- `columns`: `{ race_col, score_col, winner_col }`
- `aggregate`:
  - counts: `n_races_total`, `n_races_used`, `n_races_skipped_no_winner`, `n_races_multi_winner_label`
  - ranking stats: `mean_winner_rank`, median_winner_rank`, `mrr`
  - NDCG cutoffs: `ndcg@k` for each requested `k`
  - distribution cuts: `pct_winner_top1`, `pct_winner_in_top3`, `pct_winner_in_top5`
- `per_race_preview`: list of up to 10 "worst winner ranks" rows (debug preview)

##### Markdown report (written to `out_md`)
- Documents which columns were used
- Presents aggregate metrics in a stable table
- Includes a per-race preview table

#### Metrics definitions / invariants
- Winner rank is computed by sorting within each race by `score` descending.
- If multiple winners exist in a race, the best (lowest) winner rank is used, and the race is counted as "multi_winner_label".
- Races with **no winner label** are skipped.
- NDCG@k assumes **a single relevant item** (the true winner) with binary relevance:
  - NDCG@k = 1/log2(rank+1) if rank ≤ k else 0
- MRR = 1/rank

#### Must NOT know about
- model internals or training configuration
- feature selection or leakage rules
- betting/policy logic

#### Failure modes
- Missing required columns (unable to inger): `ValueError`
- No valid races after skipping missing-winner races: `ValueError`
- Non-numeric scores or labels that can't be coerced: will raise during `.astype(float)` conversions

#### Contract note (important for pipeline)
Because ranking.py supports column inference, the **contract for prediction artifacts** should still specify canonical column names you intend to standardize on (recommended: `race_id`, `score`, and `is_winner` for ranking; `p_win` for calibrated probabilities). The inference is a convenience, not a substitute for a stable schema.

### `src/hrml/eval/calibration.py`
#### Purpose
Evaluate and report `probability calibration diagnostics` for model predictions (`p_win`) and (optionally) an odds baseline (`p_odds`) using:
- log loss
- Brier score
- ECE (expected calibration error) from equal-mass bins
- OLS calibration line proxy (intercept/slope)
- reliability plots
- calibration stratified by **field size** buckets
This module does **not** fit a calibrator; it evaluates calibration quality of existing probability columns.

#### Stage role
- **evaluation** (diagnostics + reporting)

#### Reads/Writes
- **Reads**: `outputs/reports/pred_test.parquet` (note:module constant `PRED_PATH`)
- **Writes**:
  - Summary JSON:
    - `outputs/reports/calibration_summary.json`
  - Bin tables:
    - `outputs/reports/calibration_bins_model.csv`
    - `outputs/reports/calibration_bins_odds.csv` *(only if `p_odds` present and sufficiently dense)*
  - Field-size tables:
    - `outputs/reports/calibration_by_field_size_model.csv`
    - `outputs/reports/calibration_by_field_size_odds.csv`
  - Figures:
    - `outputs/figures/calibration_reliability_model.png`
    - `outputs/figures/calibration_fieldsize_model.png`
    - `outputs/figures/calibration_reliability_odds.png` *(conditional)*
    - `outputs/figures/calibration_fieldsize_odds.png` *(conditional)*

#### Entry point
- `main() -> None`

#### Required inputs / schema
`pred_test.parquet` must contain:
- `race_id`
- `is_winner` (binary label)
- `p_win` (model predicted probability)
Optional:
- `field_size` (for field-size stratification; otherwise bin = `"unknown"`)
- `p_odds` (odds baseline probability; analyzed if present and not too sparse)
If any required columns are missing, raises `ValueError`.

#### Core computations
- Probability clipping: all p are clipped to `[1e-12, 1 - 1e-12]`
- Reliability bins:
  - uses `pd.qcut` to create **equal-mass (quantile) bins**
  - bins may be fewer than requested if many duplicate probabilities (`duplicates="drop"`)
- ECE:
  - computed as Σ_k (n_k / N) * |mean_y_k − mean_p_k|
- OLS calibration proxy:
  - fits `y ≈ a + b*p` by least squares (diagnostic slope/intercept)

#### Field-size stratification
- Creates `field_bin` via:
  - bins: `[0, 7, 10, 14, 1000]`
  - labels: `["<=7", "8-10", "11-14", "15+"]`
- Computes per-bin: logloss, brier, ece, OLS intercept/slope

#### Odds baseline handling
- If `p_odds` exists:
  - records `odds_coverage = mean(p_odds not null)`
  - runs odds calibration only if `m.sum() > 1000` rows; otherwise records a note and skips plots/tables
- If `p_odds` absent: 
  - records `odds_note = "No p_odds column found."`

#### Outputs
##### `calibration_summary.json` schema (high-level)
- `model`: `{ n_rows, n_races, logloss, brier, ece, cal_intercept_ols, cal_slope_ols, bins_used, label }`
- optionally `odds`: same metric block if odds analyzed
- `odds_coverage` and/or `odds_note` as applicable

#### Must NOT know about
- training internals
- how probabilities were produced (softmax, temperature scaling, etc.)
- betting policies / evaluation of profit

#### Failure modes
- Missing `pred_test.parquet`: `FileNotFoundError`
- Missing required columns: `ValueError`
- Non-numeric/coercion issus in `p_win`/`p_odds`/`is_winner`: can raise during `.astype(float)` or metric caluclations
- Matplotlib backend issues can surface in headless environments unless configured (plots are always attempted)

#### Contract note (important)
This module assumes the canonical predictions file path is `outputs/reports/pred_test.parquet`, while other parts of the project sometimes use `outputs/pred_test.parquet`. In the pipeline contract, standardize **one canonical location** for "the" test prediction artifact, and treat other paths as legacy.


### `src/hrml/eval/calibration_strata.py`
#### Purpose
Evaluate calibratin (ECE + reliability plots) **across race strata**. Reads a predictions table (with probabilities + labels) and optionally joins additional strata columns (`dist_f`, `race_class`) from `model_frame.parquet`.
Produces:
- overall ECE + reliability plot
- per-stratum ECE + reliability plots
- JSON artifact containing per-bin tables for every stratum bucket
- Markdown summary tables pointing to plots

#### Stage role
- **evaluation** (diagnostics + reporting)
- Does not train models or fit calibration parameters.

#### Reads/Writes
- **Reads**:
  - `cfg.pred_path` (default: `outputs/reports/pred_test.parquet)
  - `cfg.model_frame_path` (default: `data/processed/model_frame.parquet`) *(only used if join columns missing in pred table)*
- **Writes**:
  - `cfg.out_json` (default: `outputs/reports/calibration_strata.json`)
  - `cfg.out_md` (default: `outputs/reports/calibration_strata.md`)
  - figures under `cfg.fig_dir` (default: `outputs/figures/`)
    - `reliability_overall.png`
    - `reliability_field_size_<bucket>.png`
    - optionally:
      - `reliability_distance_<bucket>.png`
      - `reliability_race_class_<bucket>.png`

#### Entry points
- `run_calibration_strata(cfg: CalibrationStrataConfig) -> dict
- Key helpers:
  - `load_and_join(cfg) -> pd.DataFrame`
  - `ece_table(y, p, n_bins) -> (ece, tbl)`
  - `reliability_plot(tbl, title, out_path)`

#### Required inputs/schema
Predictions parquet (`pred_path`) must contain:
- `race_id` (configurable via `cfg.race_col`)
- `horse_id` (configurable via `cfg.horse_col`)
- `p_win` (configurable via `cfg.prob_col`)
- `is_winner` (configurable via `cfg.label_col`)
- `field_size` is required for field-size strata (this module raises if missing)
If any required columns are missing, raises `ValueError`.

#### Optional joined strata columns
If not present in `pred_path`, the module attempts to join from `model_frame.parquet` on:
- `[race_id, horse_id]` (configurable via cfg)
Join targets:
- `dist_f`
- `race_class`

Join behavior:
- `mf2 = model_frame[[race_id, horse_id] + need_join_cols].drop_duplicates(...)`
- merges with `validate="many_to_one"`
If join yields no usable values (all missing), that stratum is skipped with a note.

#### Strata definitions
- **Field size buckets** (always attempted; required):
  - edges: `[0] + list(cfg.field_size_bins)` where default `field_size_bins=(6,8,10,12,14,16,999)`
  - lables generated as `1-6`, `7-8`, `9-10`, ..., `17-999`
- **Distance buckets** (optional):
  - requires `dist_f`
  - buckets:
    - `sprint` if `dist_f <= sprint_max_f` (default 8.0)
    - `middle` if `<= middle_max_f` (default 12.0)
    - else `staying`
- **Race class buckets** (optional):
  - requires `race_class`
  - if numeric dtype: qcut into up to `max_class_group` (default 8), `duplicates="drop"`, NA bucket
  - if string/object: keep top `max_class_groups-1` by frequency, rest -> `OTHER`

#### Calibration metric
- ECE computed using **equal-width bins** on `[0,1]` (different from `eval/calibration.py` which uses quantile bins):
  - `edges = linspace(0,1,n_bins+1)
  - per bin: `p_mean`, `y_mean`, `abs_gap`, weighted sum by bin support
- Probabilities are clipped to `[1e-12, 1-1e-12]`.

#### Outputs
##### JSON (`calibration_strata.json`) schema (high-level)
- `paths`: `{ pred_path, model_frame_path, out_json, out_md, fig_dir }`
- `overall`: `{ ece, n_rows, n_races, n_bins, reliability_plot }`
- `notes`: `{ distance_included: bool, race_class_included: bool }`
- `strata`: mapping of strata name → list of buckets
  - each bucket entry includes:
    - `stratum` (string)
    - `n_rows`, `n_races`
    - `ece`
    - `plot` (path)
    - `bin_table`: list of records with `{bin, bin_lo, bin_hi, n, p_mean, y_mean, abs_gap}`

##### Markdown (`calibration_strata.md`)
- Overall summary
- One table per included stratum with: stratum, n_rows, n_races, ece, plot path
- Quick check: counts buckets with `ece < 0.02`

#### Must NOT know about
- training internals / how probabilities were produced
- calibration fitting parameters (temperature, isotonic, etc.)
- betting policies / profit

#### Failure modes
- Missing required pred columns: `ValueError`
- Missing `field_size`: `ValueError` (cannot build mandatory field size strata)
- Join issues (e.g., unexpected duplicates in model_frame on race_id+horse_id): merge validation error
- Empty bins for a bucket: plot function returns without writing a plot for that bucket
- Headless matplotlib issues can surface if backend not configured (plots are always attempted when data exists)