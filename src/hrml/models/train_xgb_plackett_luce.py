from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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

RANDOM_SEED = 42


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class PLConfig:
    top_k: int = 3
    mc_samples: int = 200  # set to 0 to disable expected_rank/place_prob
    place_k: int = 3       # "place" threshold if MC enabled


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


def impute_from_train(
    train_df: pd.DataFrame, other: pd.DataFrame, feat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    med = train_df[feat_cols].median(numeric_only=True)
    out = other.copy()
    out[feat_cols] = out[feat_cols].fillna(med)
    return out, med


def _assert_features_numeric(df: pd.DataFrame, feat_cols: List[str]) -> None:
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


def _ensure_complete_finish_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    PL needs finish positions. XGBoost labels cannot contain NaN/inf.
    Best practice: drop entire races that have any missing/invalid finish_position.
    """
    if "finish_position" not in df.columns:
        raise ValueError("finish_position missing (required for Plackett–Luce training).")

    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    ok_row = fp.notna() & np.isfinite(fp)

    # keep only races where ALL runners have valid finish_position
    ok_race = ok_row.groupby(df["race_id"]).transform("all")
    out = df[ok_race].copy()
    out["finish_position"] = pd.to_numeric(out["finish_position"], errors="raise").astype(float)
    return out


def get_feature_cols(df: pd.DataFrame, *, strict: bool = True) -> List[str]:
    policy = FeatureAllowlist(strict_unknown_numeric=strict)
    cols = policy.select(df)

    forbidden = {"is_winner", "finish_position"}
    overlap = forbidden.intersection(cols)
    if overlap:
        raise AssertionError(f"Forbidden leakage columns present in features: {sorted(overlap)}")

    _assert_features_numeric(df, cols)
    return cols


def make_grouped_dmatrix(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[xgb.DMatrix, pd.DataFrame, np.ndarray]:
    d = df.copy()
    d["race_id"] = d["race_id"].astype(str)
    d = d.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    sizes = d.groupby("race_id", sort=False).size().to_numpy()

    _assert_features_numeric(d, feat_cols)
    X = d[feat_cols].apply(pd.to_numeric, errors="raise")

    # IMPORTANT: label carries finish_position for PL objective
    if "finish_position" not in d.columns:
        raise ValueError("finish_position missing (required for Plackett–Luce training).")
    y = d["finish_position"].astype(float).to_numpy()

    dm = xgb.DMatrix(X, label=y)
    dm.set_group(sizes)
    return dm, d, sizes


# -----------------------------
# Plackett–Luce utilities
# -----------------------------
def _stable_softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    ex = np.exp(z)
    denom = np.sum(ex)
    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(ex) / len(ex)
    return ex / denom


def _pl_topk_indices(finish_pos: np.ndarray, top_k: int) -> List[int]:
    """
    Return the indices (within a race slice) of finish_position == 1,2,...,K where present.
    Skips missing/NaN positions. If a position is duplicated or missing, we stop early.
    """
    out: List[int] = []
    if finish_pos.size == 0:
        return out

    fp = finish_pos
    for k in range(1, top_k + 1):
        idx = np.where(fp == float(k))[0]
        if len(idx) != 1:
            break
        out.append(int(idx[0]))
    return out


def plackett_luce_obj_factory(cfg: PLConfig):
    """
    Custom objective: top-K Plackett–Luce negative log-likelihood.
    Uses dtrain.get_label() to retrieve finish_position per row.
    """

    def pl_obj(preds: np.ndarray, dtrain: xgb.DMatrix):
        group = dtrain.get_group().astype(int)
        finish_pos_all = dtrain.get_label()  # finish_position

        grad = np.zeros_like(preds, dtype=float)
        hess = np.zeros_like(preds, dtype=float)

        start = 0
        for gsize in group:
            end = start + gsize

            s = preds[start:end].astype(float)
            fp = finish_pos_all[start:end].astype(float)

            topk = _pl_topk_indices(fp, cfg.top_k)
            if not topk:
                start = end
                continue

            remaining = np.ones(gsize, dtype=bool)

            for true_local in topk:
                if not remaining[true_local]:
                    break

                rem_idx = np.where(remaining)[0]
                p = _stable_softmax(s[rem_idx])

                # gradient/hessian for this stage (multinomial logit over remaining set)
                grad[start + rem_idx] += p
                grad[start + true_local] -= 1.0
                hess[start + rem_idx] += p * (1.0 - p)

                remaining[true_local] = False

            start = end

        return grad, hess

    return pl_obj


def pl_nll_eval_factory(cfg: PLConfig):
    """
    Custom metric: mean top-K PL negative log-likelihood over races.
    Uses dmat.get_label() (finish_position).
    """

    def pl_metric(preds: np.ndarray, dmat: xgb.DMatrix):
        group = dmat.get_group().astype(int)
        finish_pos_all = dmat.get_label()

        losses: List[float] = []
        start = 0
        for gsize in group:
            end = start + gsize

            s = preds[start:end].astype(float)
            fp = finish_pos_all[start:end].astype(float)

            topk = _pl_topk_indices(fp, cfg.top_k)
            if not topk:
                start = end
                continue

            remaining = np.ones(gsize, dtype=bool)

            for true_local in topk:
                rem_idx = np.where(remaining)[0]
                p = _stable_softmax(s[rem_idx])

                loc = np.where(rem_idx == true_local)[0]
                if loc.size != 1:
                    break

                pt = float(np.clip(p[int(loc[0])], 1e-12, 1.0))
                losses.append(-np.log(pt))
                remaining[true_local] = False

            start = end

        name = f"pl_nll@{cfg.top_k}"
        return name, float(np.mean(losses)) if losses else float("nan")

    return pl_metric


def _race_softmax(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """
    Stage-1 PL win probabilities: softmax over all horses in a race.
    """
    out = np.zeros_like(scores, dtype=float)
    tmp = pd.DataFrame({"race_id": race_ids, "s": scores})
    for _, idx in tmp.groupby("race_id", sort=False).indices.items():
        p = _stable_softmax(scores[idx].astype(float))
        out[idx] = p
    return np.clip(out, 1e-12, 1 - 1e-12)


def _race_softmax_temperature(scores: np.ndarray, race_ids: np.ndarray, temperature: float) -> np.ndarray:
    """
    Temperature-scaled stage-1 softmax (calibration step).
    p_i ∝ exp(score_i / T)
    """
    T = float(temperature)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature}")

    out = np.zeros_like(scores, dtype=float)
    tmp = pd.DataFrame({"race_id": race_ids, "s": scores})
    for _, idx in tmp.groupby("race_id", sort=False).indices.items():
        s = scores[idx].astype(float) / T
        p = _stable_softmax(s)
        out[idx] = p
    return np.clip(out, 1e-12, 1 - 1e-12)


def _pl_mc_expected_rank_and_place(
    scores: np.ndarray,
    race_ids: np.ndarray,
    *,
    mc_samples: int,
    place_k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each race, sample permutations from PL and estimate expected rank and place probability.
    """
    rng = np.random.default_rng(seed)
    exp_rank = np.full_like(scores, np.nan, dtype=float)
    place_prob = np.full_like(scores, np.nan, dtype=float)

    tmp = pd.DataFrame({"race_id": race_ids})
    for _, idx in tmp.groupby("race_id", sort=False).indices.items():
        idx = np.asarray(idx, dtype=int)
        s = scores[idx].astype(float)
        m = len(idx)
        if m == 0:
            continue

        rank_sum = np.zeros(m, dtype=float)
        place_hits = np.zeros(m, dtype=float)

        for _ in range(mc_samples):
            remaining = np.arange(m, dtype=int)
            ranks = np.empty(m, dtype=int)

            for r in range(1, m + 1):
                probs = _stable_softmax(s[remaining])
                choice_local = rng.choice(len(remaining), p=probs)
                chosen = remaining[choice_local]
                ranks[chosen] = r
                remaining = np.delete(remaining, choice_local)

            rank_sum += ranks.astype(float)
            place_hits += (ranks <= place_k).astype(float)

        exp_rank[idx] = rank_sum / float(mc_samples)
        place_prob[idx] = place_hits / float(mc_samples)

    return exp_rank, place_prob


# -----------------------------
# Temperature scaling (calibration)
# -----------------------------
def _winner_logloss_from_scores(scores: np.ndarray, race_ids: np.ndarray, is_winner: np.ndarray, temperature: float) -> float:
    """
    Compute winner logloss using per-race softmax(score/T) vs is_winner labels.
    """
    p = _race_softmax_temperature(scores, race_ids, temperature)
    y = is_winner.astype(int)

    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature_valid(
    booster: xgb.Booster,
    valid_df: pd.DataFrame,
    feat_cols: List[str],
    *,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """
    Fit a single temperature scalar T on the validation set by minimizing winner logloss.

    Uses a robust log-space grid search (no scipy dependency).
    """
    dva, va_s, _ = make_grouped_dmatrix(valid_df, feat_cols)
    scores = booster.predict(dva).astype(float)

    race_ids = va_s["race_id"].astype(str).to_numpy()
    y_win = va_s["is_winner"].astype(int).to_numpy()

    # log-space grid over a reasonable range
    # If T is too small, probabilities get ultra-sharp; too big -> uniform.
    grid = np.exp(np.linspace(np.log(0.3), np.log(5.0), 60))

    best_T = 1.0
    best_loss = float("inf")
    for T in grid:
        loss = _winner_logloss_from_scores(scores, race_ids, y_win, float(T))
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)

    # small local refinement around best_T
    lo = max(0.05, best_T / 1.5)
    hi = best_T * 1.5
    grid2 = np.exp(np.linspace(np.log(lo), np.log(hi), 40))
    for T in grid2:
        loss = _winner_logloss_from_scores(scores, race_ids, y_win, float(T))
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)

    return {"temperature": best_T, "valid_logloss": best_loss}


# -----------------------------
# Train / Predict
# -----------------------------
def train_booster(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feat_cols: List[str],
    params: Dict,
    *,
    num_boost_round: int,
    early_stopping_rounds: int,
    verbose_eval: bool | int,
    cfg: PLConfig,
) -> xgb.Booster:
    dtr, _, _ = make_grouped_dmatrix(train_df, feat_cols)
    dva, _, _ = make_grouped_dmatrix(valid_df, feat_cols)

    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=num_boost_round,
        evals=[(dtr, "train"), (dva, "valid")],
        obj=plackett_luce_obj_factory(cfg),
        custom_metric=pl_nll_eval_factory(cfg),
        maximize=False,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    return booster


def predict_table(
    booster: xgb.Booster,
    test_df: pd.DataFrame,
    feat_cols: List[str],
    cfg: PLConfig,
    *,
    temperature: float | None = None,
) -> pd.DataFrame:
    dte, te_s, _ = make_grouped_dmatrix(test_df, feat_cols)
    score = booster.predict(dte).astype(float)

    race_ids = te_s["race_id"].astype(str).to_numpy()

    # Raw stage-1 win probabilities (uncalibrated)
    p_win_raw = _race_softmax(score, race_ids)

    # Calibrated win probabilities (temperature scaling)
    if temperature is None:
        p_win = p_win_raw
    else:
        p_win = _race_softmax_temperature(score, race_ids, float(temperature))

    keep_cols = ["race_id", "race_date", "horse_id", "finish_position", "is_winner", "field_size"]
    keep_cols = [c for c in keep_cols if c in te_s.columns]
    pred = te_s[keep_cols].copy()
    pred["score"] = score
    pred["p_win_raw"] = p_win_raw
    pred["p_win"] = p_win

    if cfg.mc_samples and cfg.mc_samples > 0:
        er, pp = _pl_mc_expected_rank_and_place(
            score,
            race_ids,
            mc_samples=cfg.mc_samples,
            place_k=cfg.place_k,
            seed=RANDOM_SEED,
        )
        pred["expected_rank"] = er
        pred[f"p_place_le_{cfg.place_k}"] = pp

    return pred


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse canonical model+predictions if present.")
    parser.add_argument("--features-only", action="store_true", help="Only build feature list + manifest then exit.")
    parser.add_argument("--fast-dev", action="store_true", help="Fast dev mode for quick iteration.")
    parser.add_argument("--top-k", type=int, default=3, help="Plackett–Luce top-K used in the loss (default: 3).")
    parser.add_argument("--mc-samples", type=int, default=200, help="MC samples per race for expected_rank/place_prob (0 disables).")
    parser.add_argument("--place-k", type=int, default=3, help="Place threshold for p_place_le_k (default: 3).")
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable temperature scaling on p_win (keeps p_win_raw == p_win).",
    )
    args = parser.parse_args()

    cfg = PLConfig(top_k=int(args.top_k), mc_samples=int(args.mc_samples), place_k=int(args.place_k))

    model_path = OUT_DIR / "xgb_plackett_luce.json"
    pred_path = REPORT_DIR / "pred_test_plackett_luce.parquet"
    temp_path = OUT_DIR / "xgb_plackett_luce_temperature.json"

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
    df = _ensure_complete_finish_positions(df)

    print("PL usable races:", df["race_id"].nunique(), "rows:", len(df))

    feat_cols = get_feature_cols(df, strict=True)
    print("Feature columns:", len(feat_cols))

    manifest = write_feature_manifest(feat_cols, REPORT_DIR / "feature_list_plackett_luce.json")
    print("Feature manifest:", manifest["n_features"], "features | sha256:", manifest["sha256"])

    if args.features_only:
        print("Features-only mode: exiting before training.")
        return

    train, valid, test = time_split(df, train_end="2022-12-31", valid_end="2024-12-31")

    train_imp, _ = impute_from_train(train, train, feat_cols)
    valid_imp, _ = impute_from_train(train, valid, feat_cols)
    test_imp, _ = impute_from_train(train, test, feat_cols)

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

    if args.fast_dev:
        num_boost_round = 1200
        early_stop = 75
        verbose_eval = 100
    else:
        num_boost_round = 5000
        early_stop = 200
        verbose_eval = 200

    booster = train_booster(
        train_imp,
        valid_imp,
        feat_cols,
        params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stop,
        verbose_eval=verbose_eval,
        cfg=cfg,
    )

    booster.save_model(model_path.as_posix())
    print("Saved model:", model_path)

    # -----------------------------
    # Fit temperature on VALID (logloss)
    # -----------------------------
    temperature: float | None = None
    if not args.no_calibrate:
        temp_fit = fit_temperature_valid(booster, valid_imp, feat_cols)
        temperature = float(temp_fit["temperature"])
        temp_path.write_text(json.dumps(temp_fit, indent=2), encoding="utf-8")
        print("Saved temperature:", temp_path)
        print("Temperature fit:", f"T={temperature:.6f}", "valid_logloss=", f"{temp_fit['valid_logloss']:.6f}")
    else:
        print("Calibration disabled (--no-calibrate). Using raw p_win_raw as p_win.")

    pred = predict_table(booster, test_imp, feat_cols, cfg, temperature=temperature)
    pred.to_parquet(pred_path, index=False)
    print("Saved predictions:", pred_path)

    # Quick ranking metrics (race-aware) using score column
    from hrml.eval.ranking import RankingEvalConfig, run_ranking_eval

    out_json = REPORT_DIR / "metrics_ranking_plackett_luce.json"
    out_md = REPORT_DIR / "metrics_ranking_plackett_luce.md"

    rep = run_ranking_eval(
        RankingEvalConfig(
            pred_path=pred_path,
            out_json=out_json,
            out_md=out_md,
            race_col="race_id",
            score_col="score",
            winner_col="is_winner",
            k_values=(3, 5),
        )
    )

    quick = {
        "n_races": rep["aggregate"]["n_races_used"],
        "mean_winner_rank": rep["aggregate"]["mean_winner_rank"],
        "mrr": rep["aggregate"]["mrr"],
        "ndcg@3": rep["aggregate"].get("ndcg@3"),
        "ndcg@5": rep["aggregate"].get("ndcg@5"),
        "pl_top_k": cfg.top_k,
        "mc_samples": cfg.mc_samples,
        "temperature": temperature,
    }
    (REPORT_DIR / "metrics_plackett_luce.json").write_text(json.dumps(quick, indent=2), encoding="utf-8")
    print("Saved quick metrics:", REPORT_DIR / "metrics_plackett_luce.json")


if __name__ == "__main__":
    main()