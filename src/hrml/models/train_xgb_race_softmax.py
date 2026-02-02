from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from hrml.features.allowlist import FeatureAllowlist, write_feature_manifest

DATA_PATH = Path("data/processed/model_frame.parquet")

OUT_DIR = Path("outputs/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = Path("outputs/reports")
PRED_DIR.mkdir(parents=True, exist_ok=True)

FEAT_DIR = Path("outputs/features")
FEAT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


# -----------------------------
# Basic metrics (race-aware top-k)
# -----------------------------
def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def topk_hit(df: pd.DataFrame, prob_col: str, k: int) -> float:
    hits = []
    for _, g in df.groupby("race_id"):
        g2 = g.sort_values(prob_col, ascending=False).head(k)
        hits.append(int(g2["is_winner"].max() == 1))
    return float(np.mean(hits)) if hits else float("nan")


def top1_acc(df: pd.DataFrame, prob_col: str) -> float:
    hits = []
    for _, g in df.groupby("race_id"):
        pick = g.sort_values(prob_col, ascending=False).iloc[0]
        hits.append(int(pick["is_winner"] == 1))
    return float(np.mean(hits)) if hits else float("nan")


def eval_pred_table(pred: pd.DataFrame, prob_col: str = "p_win") -> Dict[str, float]:
    y = pred["is_winner"].astype(int).to_numpy()
    p = np.clip(pred[prob_col].astype(float).to_numpy(), 1e-12, 1 - 1e-12)
    return {
        "n_races": int(pred["race_id"].nunique()),
        "logloss": float(log_loss(y, p)),
        "brier": brier(y, p),
        "top1_acc": top1_acc(pred, prob_col),
        "top3_hit": topk_hit(pred, prob_col, 3),
    }


def write_ablation_report(rows: List[Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    base = df.loc[df["label"] == "base"].iloc[0]
    for col in ["logloss", "brier", "top1_acc", "top3_hit"]:
        df[f"delta_{col}"] = df[col] - float(base[col])

    csv_path = out_dir / "ablation_softmax_metrics.csv"
    df.to_csv(csv_path, index=False)

    json_path = out_dir / "ablation_softmax_metrics.json"
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    cols = [
        "label",
        "n_races",
        "n_features",
        "logloss",
        "delta_logloss",
        "brier",
        "delta_brier",
        "top1_acc",
        "delta_top1_acc",
        "top3_hit",
        "delta_top3_hit",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, r in df[cols].iterrows():
        lines.append(
            "| " + " | ".join(
                [
                    str(r["label"]),
                    str(int(r["n_races"])),
                    str(int(r["n_features"])),
                    f"{r['logloss']:.6f}",
                    f"{r['delta_logloss']:+.6f}",
                    f"{r['brier']:.6f}",
                    f"{r['delta_brier']:+.6f}",
                    f"{r['top1_acc']:.6f}",
                    f"{r['delta_top1_acc']:+.6f}",
                    f"{r['top3_hit']:.6f}",
                    f"{r['delta_top3_hit']:+.6f}",
                ]
            )
            + " |"
        )
    md_path = out_dir / "ablation_softmax_metrics.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", csv_path)
    print("Saved:", json_path)
    print("Saved:", md_path)


# -----------------------------
# Split helpers
# -----------------------------
def time_split(df: pd.DataFrame, train_end: str, valid_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = pd.to_datetime(df["race_date"])
    train = df[d <= pd.to_datetime(train_end)].copy()
    valid = df[(d > pd.to_datetime(train_end)) & (d <= pd.to_datetime(valid_end))].copy()
    test = df[d > pd.to_datetime(valid_end)].copy()
    return train, valid, test


# -----------------------------
# Feature selection (Issue #3)
# Deterministic allowlist + strict unknown numeric check
# -----------------------------
def get_feature_cols(df: pd.DataFrame, *, strict: bool = True) -> List[str]:
    policy = FeatureAllowlist(strict_unknown_numeric=strict)
    cols = policy.select(df)

    # Extra hard-stop: never allow obvious post-race labels as features
    forbidden = {"is_winner", "finish_position"}
    overlap = forbidden.intersection(cols)
    if overlap:
        raise AssertionError(f"Forbidden leakage columns present in features: {sorted(overlap)}")

    return cols


# -----------------------------
# Grouping utilities
# -----------------------------
def _ensure_one_winner_per_race(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("race_id")["is_winner"].sum()
    good = grp[grp == 1].index
    return df[df["race_id"].isin(good)].copy()


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


def make_grouped_dmatrix(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[xgb.DMatrix, pd.DataFrame, np.ndarray]:
    d = df.copy()
    d["race_id"] = d["race_id"].astype(str)
    d = d.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    sizes = d.groupby("race_id", sort=False).size().to_numpy()

    X = d[feat_cols]
    y = d["is_winner"].astype(int).to_numpy()

    dm = xgb.DMatrix(X, label=y)
    dm.set_group(sizes)
    return dm, d, sizes


# -----------------------------
# Custom objective: race-level softmax cross-entropy
# -----------------------------
def race_softmax_obj(preds: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label().astype(int)
    group = dtrain.get_group().astype(int)

    grad = np.zeros_like(preds, dtype=float)
    hess = np.zeros_like(preds, dtype=float)

    start = 0
    for gsize in group:
        end = start + gsize
        s = preds[start:end].astype(float)

        s = s - np.max(s)
        ex = np.exp(s)
        denom = np.sum(ex)
        if denom <= 0 or not np.isfinite(denom):
            p = np.ones_like(ex) / len(ex)
        else:
            p = ex / denom

        yg = y[start:end].astype(float)

        grad[start:end] = p - yg
        hess[start:end] = p * (1.0 - p)

        start = end

    return grad, hess


def race_nll_eval(preds: np.ndarray, dmat: xgb.DMatrix):
    y = dmat.get_label().astype(int)
    group = dmat.get_group().astype(int)

    losses = []
    start = 0
    for gsize in group:
        end = start + gsize
        s = preds[start:end].astype(float)
        s = s - np.max(s)
        ex = np.exp(s)
        denom = np.sum(ex)
        p = ex / denom if denom > 0 and np.isfinite(denom) else (np.ones_like(ex) / len(ex))

        yg = y[start:end]
        w_idx = np.where(yg == 1)[0]
        if len(w_idx) == 1:
            pw = float(np.clip(p[w_idx[0]], 1e-12, 1 - 1e-12))
            losses.append(-np.log(pw))

        start = end

    return "race_nll", float(np.mean(losses)) if losses else float("nan")


def train_booster(
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
        obj=race_softmax_obj,
        custom_metric=race_nll_eval,
        maximize=False,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    return booster


def predict_table(booster: xgb.Booster, test_df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    dte, te_s, _ = make_grouped_dmatrix(test_df, feat_cols)
    score = booster.predict(dte)
    p_win = _race_softmax(score, te_s["race_id"].astype(str).to_numpy())

    keep_cols = ["race_id", "race_date", "horse_id", "finish_position", "is_winner", "field_size"]
    keep_cols = [c for c in keep_cols if c in te_s.columns]
    pred = te_s[keep_cols].copy()
    pred["score"] = score
    pred["p_win"] = p_win

    if "odds_implied_norm" in te_s.columns:
        pred["p_odds"] = te_s["odds_implied_norm"].to_numpy()
    if "sp_decimal" in te_s.columns:
        pred["sp_decimal"] = te_s["sp_decimal"].to_numpy()

    return pred


def impute_from_train(train_df: pd.DataFrame, other: pd.DataFrame, feat_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    med = train_df[feat_cols].median(numeric_only=True)
    out = other.copy()
    out[feat_cols] = out[feat_cols].fillna(med)
    return out, med


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-only", action="store_true", help="Skip ablations and train base model only.")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="If canonical model + predictions exist, reuse them and exit.",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Only compute feature columns + write manifest, then exit (no training).",
    )
    args = parser.parse_args()

    model_path = OUT_DIR / "xgb_race_softmax.json"
    pred_path = PRED_DIR / "pred_test.parquet"

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

    # Log final feature list deterministically (Issue #3)
    manifest = write_feature_manifest(feat_cols, PRED_DIR / "feature_list.json")
    print("Feature manifest:", manifest["n_features"], "features | sha256:", manifest["sha256"])
    print("Saved feature manifest:", PRED_DIR / "feature_list.json")

    if args.features_only:
        print("Features-only mode: exiting before training.")
        return

    train, valid, test = time_split(df, train_end="2022-12-31", valid_end="2024-12-31")

    # Base imputation & snapshots
    train_imp, med = impute_from_train(train, train, feat_cols)
    valid_imp, _ = impute_from_train(train, valid, feat_cols)
    test_imp, _ = impute_from_train(train, test, feat_cols)

    train_imp[["race_id", "race_date", "horse_id", "is_winner"] + feat_cols].to_parquet(
        FEAT_DIR / "train_features.parquet", index=False
    )
    test_imp[["race_id", "race_date", "horse_id", "is_winner"] + feat_cols].to_parquet(
        FEAT_DIR / "test_features.parquet", index=False
    )
    print("Saved feature snapshots:", FEAT_DIR)

    params: Dict = {
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 5.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "objective": "reg:squarederror",  # ignored (custom obj)
        "tree_method": "hist",
        "seed": RANDOM_SEED,
        "verbosity": 0,
    }

    # -----------------------------
    # Optional ablations (fast)
    # -----------------------------
    if not args.base_only:
        def drop(cols: List[str], names: List[str]) -> List[str]:
            s = set(names)
            return [c for c in cols if c not in s]

        ablations = [
            ("base", feat_cols),
            ("no_post_position", drop(feat_cols, ["post_position"])),
            ("no_horse_mean_finish_last5", drop(feat_cols, ["horse_mean_finish_last5"])),
            ("no_post_and_form", drop(feat_cols, ["post_position", "horse_mean_finish_last5"])),
        ]

        rows: List[Dict[str, float]] = []
        for label, cols in ablations:
            print(f"[ablation:{label}] n_features={len(cols)}")

            tr_a, _ = impute_from_train(train, train, cols)
            va_a, _ = impute_from_train(train, valid, cols)
            te_a, _ = impute_from_train(train, test, cols)

            booster_a = train_booster(
                tr_a,
                va_a,
                cols,
                params,
                num_boost_round=1500,
                early_stopping_rounds=100,
                verbose_eval=False,
            )
            pred_a = predict_table(booster_a, te_a, cols)
            m = eval_pred_table(pred_a, "p_win")
            m["label"] = label
            m["n_features"] = int(len(cols))
            rows.append(m)

        write_ablation_report(rows, PRED_DIR)

    # -----------------------------
    # Canonical base model (single training pass)
    # -----------------------------
    booster = train_booster(
        train_imp,
        valid_imp,
        feat_cols,
        params,
        num_boost_round=5000,
        early_stopping_rounds=200,
        verbose_eval=200,
    )
    booster.save_model(model_path.as_posix())
    print("Saved model:", model_path)

    pred_base = predict_table(booster, test_imp, feat_cols)
    pred_base.to_parquet(pred_path, index=False)
    print("Saved predictions:", pred_path)

    y = pred_base["is_winner"].astype(int).to_numpy()
    p = np.clip(pred_base["p_win"].astype(float).to_numpy(), 1e-12, 1 - 1e-12)
    print("Test logloss (runner rows):", float(log_loss(y, p)))


if __name__ == "__main__":
    main()