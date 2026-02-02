from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from hrml.features.allowlist import FeatureAllowlist, write_feature_manifest

DATA_PATH = Path("data/processed/model_frame.parquet")

OUT_DIR = Path("outputs/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


# -----------------------------
# Helpers
# -----------------------------
def time_split(df: pd.DataFrame, train_end: str, valid_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = pd.to_datetime(df["race_date"])
    train = df[d <= pd.to_datetime(train_end)].copy()
    valid = df[(d > pd.to_datetime(train_end)) & (d <= pd.to_datetime(valid_end))].copy()
    test = df[d > pd.to_datetime(valid_end)].copy()
    return train, valid, test


def _ensure_one_winner_per_race(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("race_id")["is_winner"].sum()
    good = grp[grp == 1].index
    return df[df["race_id"].isin(good)].copy()


def impute_from_train(train_df: pd.DataFrame, other: pd.DataFrame, feat_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    med = train_df[feat_cols].median(numeric_only=True)
    out = other.copy()
    out[feat_cols] = out[feat_cols].fillna(med)
    return out, med


def _race_softmax(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=float)
    tmp = pd.DataFrame({"race_id": race_ids, "s": scores})
    for _, idx in tmp.groupby("race_id", sort=False).indices.items():
        s = scores[idx].astype(float)
        s = s - np.max(s)
        ex = np.exp(s)
        denom = np.sum(ex)
        if denom <= 0 or not np.isfinite(denom):
            out[idx] = 1.0 / len(idx)
        else:
            out[idx] = ex / denom
    return np.clip(out, 1e-12, 1 - 1e-12)


def _assert_features_numeric(df: pd.DataFrame, feat_cols: List[str]) -> None:
    """
    Guard against mixed/extension dtypes where pandas "looks numeric" but contains strings
    (e.g. race_class = 'Class 6'). XGBoost requires float-convertible features.
    """
    bad: List[str] = []
    for c in feat_cols:
        try:
            pd.to_numeric(df[c], errors="raise")
        except Exception:
            bad.append(c)
    if bad:
        sample = {c: df[c].dropna().astype(str).head(3).tolist() for c in bad}
        raise TypeError(
            "Some selected feature columns are not coercible to numeric. "
            "Remove/encode them explicitly.\n"
            f"bad_features={bad}\n"
            f"samples={sample}"
        )


def make_grouped_dmatrix(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[xgb.DMatrix, pd.DataFrame, np.ndarray]:
    d = df.copy()
    d["race_id"] = d["race_id"].astype(str)
    d = d.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    sizes = d.groupby("race_id", sort=False).size().to_numpy()

    # Ensure features are numeric before handing to XGBoost
    _assert_features_numeric(d, feat_cols)
    X = d[feat_cols].apply(pd.to_numeric, errors="raise")

    # winner=1, others=0 works fine for rank:pairwise
    y = d["is_winner"].astype(int).to_numpy()

    dm = xgb.DMatrix(X, label=y)
    dm.set_group(sizes)
    return dm, d, sizes


def get_feature_cols(df: pd.DataFrame, *, strict: bool = True) -> List[str]:
    policy = FeatureAllowlist(strict_unknown_numeric=strict)
    cols = policy.select(df)

    # extra hard-stop
    forbidden = {"is_winner", "finish_position"}
    overlap = forbidden.intersection(cols)
    if overlap:
        raise AssertionError(f"Forbidden leakage columns present in features: {sorted(overlap)}")

    # additional guard: ensure selected features are numerically coercible
    _assert_features_numeric(df, cols)
    return cols


def predict_table(booster: xgb.Booster, test_df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    dte, te_s, _ = make_grouped_dmatrix(test_df, feat_cols)
    score = booster.predict(dte)

    # Provide a normalized per-race probability-like output for downstream diagnostics
    p_rank = _race_softmax(score, te_s["race_id"].astype(str).to_numpy())

    keep_cols = ["race_id", "race_date", "horse_id", "finish_position", "is_winner", "field_size"]
    keep_cols = [c for c in keep_cols if c in te_s.columns]
    pred = te_s[keep_cols].copy()
    pred["score"] = score
    pred["p_rank"] = p_rank  # not a true calibrated probability; useful for comparisons

    return pred


def train_pairwise(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feat_cols: List[str],
    params: Dict,
    *,
    num_boost_round: int,
    early_stopping_rounds: int,
    verbose_eval: bool | int,
) -> xgb.Booster:
    dtr, _, _ = make_grouped_dmatrix(train_df, feat_cols)
    dva, _, _ = make_grouped_dmatrix(valid_df, feat_cols)

    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=num_boost_round,
        evals=[(dtr, "train"), (dva, "valid")],
        maximize=True,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    return booster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse canonical model+predictions if present.")
    parser.add_argument("--features-only", action="store_true", help="Only build feature list + manifest then exit.")
    parser.add_argument("--fast-dev", action="store_true", help="Fast dev mode: fewer rounds for quick iteration.")
    args = parser.parse_args()

    model_path = OUT_DIR / "xgb_rank_pairwise.json"
    pred_path = REPORT_DIR / "pred_test_pairwise.parquet"
    metrics_path = REPORT_DIR / "metrics_pairwise.json"

    if args.reuse_existing and not args.features_only and model_path.exists() and pred_path.exists():
        print("Reusing existing model and predictions.")
        print("Model:", model_path)
        print("Predictions:", pred_path)
        return

    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python -m hrml.features.build_features` first.")

    df = pd.read_parquet(DATA_PATH)
    df = df[pd.notnull(df["race_date"]) & pd.notnull(df["is_winner"])].copy()
    df = _ensure_one_winner_per_race(df)

    feat_cols = get_feature_cols(df, strict=True)
    print("Feature columns:", len(feat_cols))
    manifest = write_feature_manifest(feat_cols, REPORT_DIR / "feature_list_pairwise.json")
    print("Feature manifest:", manifest["n_features"], "features | sha256:", manifest["sha256"])

    if args.features_only:
        print("Features-only mode: exiting before training.")
        return

    train, valid, test = time_split(df, train_end="2022-12-31", valid_end="2024-12-31")

    train_imp, _ = impute_from_train(train, train, feat_cols)
    valid_imp, _ = impute_from_train(train, valid, feat_cols)
    test_imp, _ = impute_from_train(train, test, feat_cols)

    # Pairwise rank objective (fast)
    params: Dict = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@5",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 5.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "tree_method": "hist",
        "seed": RANDOM_SEED,
        "verbosity": 0,
    }

    if args.fast_dev:
        num_boost_round = 600
        early_stop = 50
        verbose_eval = 50
    else:
        # intentionally smaller than softmax training to meet your "<50%" bar
        num_boost_round = 1500
        early_stop = 100
        verbose_eval = 200

    booster = train_pairwise(
        train_imp,
        valid_imp,
        feat_cols,
        params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stop,
        verbose_eval=verbose_eval,
    )

    booster.save_model(model_path.as_posix())
    print("Saved model:", model_path)

    pred = predict_table(booster, test_imp, feat_cols)
    pred.to_parquet(pred_path, index=False)
    print("Saved predictions:", pred_path)

    # lightweight metrics summary (race-aware via winner rank)
    # You can do full eval via hrml.cli eval-ranking
    from hrml.eval.ranking import RankingEvalConfig, run_ranking_eval

    cfg = RankingEvalConfig(
        pred_path=pred_path,
        out_json=REPORT_DIR / "metrics_ranking_pairwise.json",
        out_md=REPORT_DIR / "metrics_ranking_pairwise.md",
        race_col="race_id",
        score_col="score",
        winner_col="is_winner",
        k_values=(3, 5),
    )
    rep = run_ranking_eval(cfg)

    metrics = {
        "n_races": rep["aggregate"]["n_races_used"],
        "mean_winner_rank": rep["aggregate"]["mean_winner_rank"],
        "mrr": rep["aggregate"]["mrr"],
        "ndcg@3": rep["aggregate"].get("ndcg@3"),
        "ndcg@5": rep["aggregate"].get("ndcg@5"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved quick metrics:", metrics_path)


if __name__ == "__main__":
    main()