from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

DATA_PATH = Path("data/processed/model_frame.parquet")
PRED_PATH = Path("outputs/reports/pred_test.parquet")
OUT_DIR = Path("outputs/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


LEAK_KEYWORDS = [
    "finish", "position", "margin", "time", "payout", "mutuel", "official", "result", "placed"
]


def _keyword_flags(cols: List[str]) -> List[str]:
    bad = []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in LEAK_KEYWORDS):
            bad.append(c)
    return bad


def _hash_rows(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    # stable hash of selected columns
    tmp = df[cols].copy()
    for c in cols:
        if pd.api.types.is_float_dtype(tmp[c]):
            tmp[c] = tmp[c].round(8)
    return pd.util.hash_pandas_object(tmp, index=False)


def race_prob_sums(pred: pd.DataFrame, prob_col: str) -> pd.Series:
    return pred.groupby("race_id")[prob_col].sum()


def shuffle_within_race(pred: pd.DataFrame, prob_col: str) -> float:
    rng = np.random.default_rng(RANDOM_SEED)

    y = pred["is_winner"].astype(int).to_numpy(copy=True)
    p = np.clip(pred[prob_col].astype(float).to_numpy(), 1e-12, 1 - 1e-12)

    # shuffle y within each race
    for _, idx in pred.groupby("race_id", sort=False).indices.items():
        idx = np.asarray(idx)
        y[idx] = y[idx][rng.permutation(len(idx))]

    return float(log_loss(y, p))


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing data/processed/model_frame.parquet (run feature build).")
    if not PRED_PATH.exists():
        raise FileNotFoundError("Missing outputs/reports/pred_test.parquet (run training).")

    df = pd.read_parquet(DATA_PATH)
    pred = pd.read_parquet(PRED_PATH)

    report: Dict[str, object] = {}

    # ---- 1) Column leakage scan on full frame ----
    flags = _keyword_flags(list(df.columns))
    report["leakage_keyword_columns"] = flags

    # ---- 2) Ensure obvious leakage cols are excluded from model features ----
    # We can't perfectly infer feat_cols here, but we can check that pred contains post-race cols
    # and warn if they exist in df (they will).
    must_never_model = [c for c in df.columns if c in {"finish_position", "is_winner"}]
    report["must_never_model_cols_present_in_frame"] = must_never_model

    # ---- 3) Prediction sanity: per-race probability sums ----
    if "p_win" in pred.columns:
        sums = race_prob_sums(pred, "p_win")
        report["p_win_sum_min"] = float(sums.min())
        report["p_win_sum_median"] = float(sums.median())
        report["p_win_sum_max"] = float(sums.max())

    if "p_win_raw" in pred.columns:
        sums_raw = race_prob_sums(pred, "p_win_raw")
        report["p_win_raw_sum_min"] = float(sums_raw.min())
        report["p_win_raw_sum_median"] = float(sums_raw.median())
        report["p_win_raw_sum_max"] = float(sums_raw.max())

    # ---- 4) Control experiment: shuffle labels within race ----
    report["logloss_model"] = float(
        log_loss(pred["is_winner"].astype(int).values, np.clip(pred["p_win"].astype(float).values, 1e-12, 1-1e-12))
    )
    report["logloss_shuffle_within_race"] = shuffle_within_race(pred, "p_win")

    # ---- 5) Duplicate leakage check (train/test overlap) ----
    # Use time split boundaries consistent with training.
    d = pd.to_datetime(df["race_date"])
    train = df[d <= pd.to_datetime("2022-12-31")].copy()
    test = df[d > pd.to_datetime("2024-12-31")].copy()

    # Candidate columns for duplicate hashing: numeric cols excluding target and obvious ids
    drop = {
        "race_id", "race_date", "course",
        "horse_id", "jockey_id", "trainer_id",
        "finish_position", "is_winner",
        "odds_implied", "odds_implied_norm",
        "race_date_ord",
    }
    feat_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    report["n_feature_cols_inferred"] = int(len(feat_cols))

    # Hash rows of features to detect exact duplicates across split
    if feat_cols:
        h_tr = set(_hash_rows(train, feat_cols).values.tolist())
        h_te = _hash_rows(test, feat_cols)
        overlap = int(sum(1 for x in h_te.values.tolist() if x in h_tr))
        report["feature_row_hash_overlap_train_test"] = overlap

    # ---- 6) Odds baseline sanity ----
    if "p_odds" in pred.columns:
        m = pred["p_odds"].notna()
        if m.any():
            y = pred.loc[m, "is_winner"].astype(int).values
            p = np.clip(pred.loc[m, "p_odds"].astype(float).values, 1e-12, 1-1e-12)
            report["logloss_odds_subset"] = float(log_loss(y, p))
            report["odds_coverage"] = float(m.mean())

    # Save report
    out_path = OUT_DIR / "sanity_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved:", out_path)

    # Print key lines
    print("\n== Sanity summary ==")
    print("Leakage keyword columns (frame):", report["leakage_keyword_columns"][:25])
    print("Inferred feature cols:", report["n_feature_cols_inferred"])
    print("Train/Test exact feature overlap:", report.get("feature_row_hash_overlap_train_test"))
    print("Logloss (model):", f"{report['logloss_model']:.6f}")
    print("Logloss (shuffle within race):", f"{report['logloss_shuffle_within_race']:.6f}")
    if "logloss_odds_subset" in report:
        print("Logloss (odds subset):", f"{report['logloss_odds_subset']:.6f}", "coverage:", f"{report['odds_coverage']:.3f}")

    # Hard-fail heuristics (tune thresholds as needed)
    # If shuffled performance is still extremely good, something is very wrong.
    if report["logloss_shuffle_within_race"] < 0.55:
        raise RuntimeError("Sanity check failed: shuffle-within-race logloss is too good (possible leakage or metric bug).")


if __name__ == "__main__":
    main()