from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

DATA_PATH = Path("data/processed/model_frame.parquet")
OUT_DIR = Path("outputs/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = Path("outputs/reports")
PRED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def time_split(df: pd.DataFrame, train_end: str, valid_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def objective(trial: optuna.Trial, train: pd.DataFrame, valid: pd.DataFrame, feat_cols: List[str]) -> float:
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
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    p_raw = model.predict_proba(X_va)[:, 1]
    p_norm = race_normalize(p_raw, race_va)
    return float(log_loss(y_va, p_norm))


def fit_best(train: pd.DataFrame, valid: pd.DataFrame, feat_cols: List[str], best_params: Dict) -> XGBClassifier:
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
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    return model


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python -m hrml.features.build_features` first.")

    df = pd.read_parquet(DATA_PATH)
    df = df[pd.notnull(df["race_date"]) & pd.notnull(df["is_winner"])].copy()

    train, valid, test = time_split(df, train_end="2022-12-31", valid_end="2024-12-31")

    feat_cols = get_feature_cols(df)
    print("Feature columns:", len(feat_cols))

    # impute from train only
    med = train[feat_cols].median(numeric_only=True)
    train[feat_cols] = train[feat_cols].fillna(med)
    valid[feat_cols] = valid[feat_cols].fillna(med)
    test[feat_cols] = test[feat_cols].fillna(med)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train, valid, feat_cols), n_trials=30)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    model = fit_best(train, valid, feat_cols, study.best_params)

    model_path = OUT_DIR / "xgb_optuna.json"
    model.save_model(model_path.as_posix())
    print("Saved model:", model_path)

    # predict on test
    X_te = test[feat_cols]
    y_te = test["is_winner"].astype(int).values
    race_te = test["race_id"].astype(str).values

    p_raw = model.predict_proba(X_te)[:, 1]
    p_norm = race_normalize(p_raw, race_te)

    keep_cols = ["race_id", "race_date", "horse_id", "finish_position", "is_winner", "field_size"]
    keep_cols = [c for c in keep_cols if c in test.columns]
    pred = test[keep_cols].copy()
    pred["p_win_raw"] = p_raw
    pred["p_win"] = p_norm

    # Carry odds baseline directly to avoid merge issues later
    if "odds_implied_norm" in test.columns:
        pred["p_odds"] = test["odds_implied_norm"].values
    if "sp_decimal" in test.columns:
        pred["sp_decimal"] = test["sp_decimal"].values

    out_pred = PRED_DIR / "pred_test.parquet"
    pred.to_parquet(out_pred, index=False)
    print("Saved predictions:", out_pred)
    print("Test logloss:", float(log_loss(y_te, p_norm)))


if __name__ == "__main__":
    main()