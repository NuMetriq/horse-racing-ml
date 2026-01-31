from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_PATH = Path("data/processed/model_frame.parquet")
OUT_DIR = Path("outputs/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = Path("outputs/reports")
PRED_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR = Path("outputs/features")
FEAT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------
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


def eval_metrics(test_df: pd.DataFrame, p_col: str) -> Dict[str, float]:
    y = test_df["is_winner"].astype(int).to_numpy()
    p = np.clip(test_df[p_col].astype(float).to_numpy(), 1e-12, 1 - 1e-12)
    return {
        "n_races": int(test_df["race_id"].nunique()),
        "logloss": float(log_loss(y, p)),
        "brier": brier(y, p),
        "top1_acc": top1_acc(test_df, p_col),
        "top3_hit": topk_hit(test_df, p_col, 3),
    }

# ------------------------------------------------------------------
# Splitting / normalization
# ------------------------------------------------------------------
def time_split(
    df: pd.DataFrame, train_end: str, valid_end: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = pd.to_datetime(df["race_date"])
    train = df[d <= pd.to_datetime(train_end)].copy()
    valid = df[(d > pd.to_datetime(train_end)) & (d <= pd.to_datetime(valid_end))].copy()
    test = df[d > pd.to_datetime(valid_end)].copy()
    return train, valid, test


def race_normalize(proba: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Normalize per race to sum to 1."""
    out = proba.copy().astype(float)
    tmp = pd.DataFrame({"race_id": race_ids, "p": out})
    sums = tmp.groupby("race_id")["p"].transform("sum").values
    sums = np.where(sums <= 0, 1.0, sums)
    out = out / sums
    return np.clip(out, 1e-12, 1 - 1e-12)

# ------------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------------
def get_feature_cols(df: pd.DataFrame) -> List[str]:
    drop = {
        "race_id", "race_date", "course",
        "horse_id", "jockey_id", "trainer_id",
        "finish_position", "is_winner",
        "odds_implied", "odds_implied_norm",
        "race_date_ord",
    }
    cols: List[str] = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

# ------------------------------------------------------------------
# Modeling
# ------------------------------------------------------------------
def objective(
    trial: optuna.Trial,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feat_cols: List[str],
) -> float:
    params = {
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": -1,
    }

    X_tr = train[feat_cols]
    y_tr = train["is_winner"].astype(int)

    X_va = valid[feat_cols]
    y_va = valid["is_winner"].astype(int).values
    race_va = valid["race_id"].astype(str).values

    model = XGBClassifier(**params, early_stopping_rounds=200)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    p_raw = model.predict_proba(X_va)[:, 1]
    p_norm = race_normalize(p_raw, race_va)
    return float(log_loss(y_va, p_norm))


def fit_best(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feat_cols: List[str],
    best_params: Dict,
) -> XGBClassifier:
    params = dict(best_params)
    params.update({
        "n_estimators": 5000,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": -1,
    })

    X_tr = train[feat_cols]
    y_tr = train["is_winner"].astype(int)

    X_va = valid[feat_cols]
    y_va = valid["is_winner"].astype(int)

    model = XGBClassifier(**params, early_stopping_rounds=200)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def fit_predict_eval(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feat_cols: List[str],
    best_params: Dict,
) -> Dict[str, float]:
    tr, va, te = train.copy(), valid.copy(), test.copy()

    med = tr[feat_cols].median(numeric_only=True)
    tr[feat_cols] = tr[feat_cols].fillna(med)
    va[feat_cols] = va[feat_cols].fillna(med)
    te[feat_cols] = te[feat_cols].fillna(med)

    model = fit_best(tr, va, feat_cols, best_params)

    p_raw = model.predict_proba(te[feat_cols])[:, 1]
    p_norm = race_normalize(p_raw, te["race_id"].astype(str).values)

    pred = te[["race_id", "race_date", "horse_id", "is_winner", "field_size"]].copy()
    pred["p_win"] = p_norm

    return eval_metrics(pred, "p_win")

# ------------------------------------------------------------------
# Ablation report writer
# ------------------------------------------------------------------
def write_ablation_report(rows: List[Dict[str, float]], out_dir: Path) -> None:
    df = pd.DataFrame(rows)
    base = df.loc[df["label"] == "base"].iloc[0]

    for col in ["logloss", "brier", "top1_acc", "top3_hit"]:
        df[f"delta_{col}"] = df[col] - float(base[col])

    df.to_csv(out_dir / "ablation_metrics.csv", index=False)
    df.to_json(out_dir / "ablation_metrics.json", orient="records", indent=2)

    cols = [
        "label", "n_races",
        "logloss", "delta_logloss",
        "brier", "delta_brier",
        "top1_acc", "delta_top1_acc",
        "top3_hit", "delta_top3_hit",
    ]

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]

    for _, r in df[cols].iterrows():
        lines.append(
            "| " + " | ".join([
                r["label"],
                str(int(r["n_races"])),
                f"{r['logloss']:.6f}", f"{r['delta_logloss']:+.6f}",
                f"{r['brier']:.6f}", f"{r['delta_brier']:+.6f}",
                f"{r['top1_acc']:.6f}", f"{r['delta_top1_acc']:+.6f}",
                f"{r['top3_hit']:.6f}", f"{r['delta_top3_hit']:+.6f}",
            ]) + " |"
        )

    (out_dir / "ablation_metrics.md").write_text("\n".join(lines), encoding="utf-8")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python -m hrml.features.build_features` first.")

    df = pd.read_parquet(DATA_PATH)
    df = df[pd.notnull(df["race_date"]) & pd.notnull(df["is_winner"])].copy()

    train, valid, test = time_split(df, "2022-12-31", "2024-12-31")

    feat_cols = get_feature_cols(df)
    print("Feature columns:", len(feat_cols))

    med = train[feat_cols].median(numeric_only=True)
    train[feat_cols] = train[feat_cols].fillna(med)
    valid[feat_cols] = valid[feat_cols].fillna(med)
    test[feat_cols] = test[feat_cols].fillna(med)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train, valid, feat_cols), n_trials=30)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    # ------------------------------
    # Feature ablations
    # ------------------------------
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
        m = fit_predict_eval(train, valid, test, cols, study.best_params)
        m["label"] = label
        rows.append(m)

    write_ablation_report(rows, PRED_DIR)

    # ------------------------------
    # Final model (base features)
    # ------------------------------
    model = fit_best(train, valid, feat_cols, study.best_params)
    model_path = OUT_DIR / "xgb_optuna.json"
    model.save_model(model_path.as_posix())
    print("Saved model:", model_path)

    p_raw = model.predict_proba(test[feat_cols])[:, 1]
    p_norm = race_normalize(p_raw, test["race_id"].astype(str).values)

    keep = ["race_id", "race_date", "horse_id", "finish_position", "is_winner", "field_size"]
    keep = [c for c in keep if c in test.columns]
    pred = test[keep].copy()
    pred["p_win_raw"] = p_raw
    pred["p_win"] = p_norm

    if "odds_implied_norm" in test.columns:
        pred["p_odds"] = test["odds_implied_norm"].values
    if "sp_decimal" in test.columns:
        pred["sp_decimal"] = test["sp_decimal"].values

    test[["race_id", "race_date", "horse_id", "is_winner"] + feat_cols] \
        .to_parquet(FEAT_DIR / "test_features.parquet", index=False)
    train[["race_id", "race_date", "horse_id", "is_winner"] + feat_cols] \
        .to_parquet(FEAT_DIR / "train_features.parquet", index=False)

    pred.to_parquet(PRED_DIR / "pred_test.parquet", index=False)
    print("Test logloss:", float(log_loss(test["is_winner"].values, p_norm)))


if __name__ == "__main__":
    main()