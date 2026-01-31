from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


PRED_PATH = Path("outputs/reports/pred_test.parquet")
FIG_DIR = Path("outputs/figures")
REP_DIR = Path("outputs/reports")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Core helpers
# -----------------------------
def _clip01(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(float), 1e-12, 1.0 - 1e-12)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((_clip01(p) - y) ** 2))


def ece_from_bins(bin_df: pd.DataFrame) -> float:
    # ECE = sum_k (n_k / N) * |acc_k - conf_k|
    n = bin_df["n"].sum()
    if n <= 0:
        return float("nan")
    w = bin_df["n"] / n
    return float(np.sum(w * np.abs(bin_df["mean_y"] - bin_df["mean_p"])))


def reliability_bins(y: np.ndarray, p: np.ndarray, bins: int = 20) -> pd.DataFrame:
    """
    Equal-mass (quantile) bins for robust reliability estimation.
    """
    df = pd.DataFrame({"y": y.astype(int), "p": _clip01(p)})
    # qcut may drop bins with many duplicate values
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    grp = (
        df.groupby("bin", observed=True)
        .agg(mean_p=("p", "mean"), mean_y=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )
    return grp


def calibration_line_fit(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """
    Fit y ≈ a + b * p via OLS (simple calibration slope/intercept proxy).
    This is not the same as Platt scaling, but it's a useful diagnostic:
      - intercept near 0, slope near 1 is "well calibrated" (in this proxy sense).
    """
    y = y.astype(float)
    p = _clip01(p).astype(float)
    X = np.column_stack([np.ones_like(p), p])
    # OLS: beta = (X'X)^-1 X'y
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    return a, b


def plot_reliability(bin_df: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure()
    plt.plot(bin_df["mean_p"], bin_df["mean_y"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed win rate")
    plt.title(title)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------
# Field-size calibration
# -----------------------------
def add_field_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "field_size" not in out.columns:
        out["field_bin"] = "unknown"
        return out

    bins = [0, 7, 10, 14, 1000]
    labels = ["<=7", "8-10", "11-14", "15+"]
    out["field_bin"] = pd.cut(out["field_size"].astype(float), bins=bins, labels=labels)
    out["field_bin"] = out["field_bin"].astype(str)
    return out


def calibration_by_field_size(df: pd.DataFrame, prob_col: str, bins: int = 15) -> pd.DataFrame:
    """
    Compute ECE and reliability summaries per field-size bucket.
    """
    df2 = add_field_bins(df)
    rows = []
    for fb, g in df2.groupby("field_bin", observed=True):
        g = g.dropna(subset=[prob_col, "is_winner"])
        if g.empty:
            continue
        y = g["is_winner"].astype(int).to_numpy()
        p = _clip01(g[prob_col].astype(float).to_numpy())
        bin_df = reliability_bins(y, p, bins=bins)
        rows.append({
            "field_bin": fb,
            "n_rows": int(len(g)),
            "n_races": int(g["race_id"].nunique()) if "race_id" in g.columns else int(pd.NA),
            "logloss": float(log_loss(y, p)),
            "brier": brier(y, p),
            "ece": ece_from_bins(bin_df),
            "cal_intercept_ols": calibration_line_fit(y, p)[0],
            "cal_slope_ols": calibration_line_fit(y, p)[1],
        })
    return pd.DataFrame(rows).sort_values("field_bin")


def plot_fieldsize_metric(tbl: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    if tbl.empty or metric not in tbl.columns:
        return
    plt.figure()
    plt.plot(tbl["field_bin"], tbl[metric], marker="o")
    plt.xlabel("Field size bin")
    plt.ylabel(metric)
    plt.title(title)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def summarize(pred: pd.DataFrame, prob_col: str, label: str, bins: int = 20) -> Tuple[Dict[str, float], pd.DataFrame]:
    pred = pred.dropna(subset=[prob_col, "is_winner"]).copy()
    y = pred["is_winner"].astype(int).to_numpy()
    p = _clip01(pred[prob_col].astype(float).to_numpy())

    bin_df = reliability_bins(y, p, bins=bins)
    ece = ece_from_bins(bin_df)
    a, b = calibration_line_fit(y, p)

    metrics = {
        "label": label,
        "n_rows": int(len(pred)),
        "n_races": int(pred["race_id"].nunique()) if "race_id" in pred.columns else int(pd.NA),
        "logloss": float(log_loss(y, p)),
        "brier": brier(y, p),
        "ece": float(ece),
        "cal_intercept_ols": float(a),
        "cal_slope_ols": float(b),
        "bins_used": int(len(bin_df)),
    }
    return metrics, bin_df


def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError("Missing outputs/reports/pred_test.parquet (run training first).")

    pred = pd.read_parquet(PRED_PATH).copy()

    required = {"race_id", "is_winner", "p_win"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"pred_test.parquet missing required columns: {missing}")

    summary: Dict[str, object] = {}

    # -----------------------------
    # Model calibration (p_win)
    # -----------------------------
    model_metrics, model_bins = summarize(pred, "p_win", "model", bins=20)
    summary["model"] = model_metrics

    model_bins_path = REP_DIR / "calibration_bins_model.csv"
    model_bins.to_csv(model_bins_path, index=False)
    plot_reliability(model_bins, "Reliability: Model (test)", FIG_DIR / "calibration_reliability_model.png")

    # Field-size calibration table + plot
    fs_model = calibration_by_field_size(pred, "p_win", bins=15)
    fs_model_path = REP_DIR / "calibration_by_field_size_model.csv"
    fs_model.to_csv(fs_model_path, index=False)
    plot_fieldsize_metric(
        fs_model, "ece", "ECE by field size: Model (test)", FIG_DIR / "calibration_fieldsize_model.png"
    )

    # -----------------------------
    # Odds calibration if present
    # -----------------------------
    if "p_odds" in pred.columns:
        m = pred["p_odds"].notna()
        odds_cov = float(m.mean())
        summary["odds_coverage"] = odds_cov

        if m.sum() > 1000:
            pred_odds = pred.loc[m].copy()
            odds_metrics, odds_bins = summarize(pred_odds, "p_odds", "odds", bins=20)
            summary["odds"] = odds_metrics

            odds_bins_path = REP_DIR / "calibration_bins_odds.csv"
            odds_bins.to_csv(odds_bins_path, index=False)
            plot_reliability(odds_bins, "Reliability: Odds baseline (test subset)", FIG_DIR / "calibration_reliability_odds.png")

            fs_odds = calibration_by_field_size(pred_odds, "p_odds", bins=15)
            fs_odds_path = REP_DIR / "calibration_by_field_size_odds.csv"
            fs_odds.to_csv(fs_odds_path, index=False)
            plot_fieldsize_metric(
                fs_odds, "ece", "ECE by field size: Odds baseline (test subset)", FIG_DIR / "calibration_fieldsize_odds.png"
            )
        else:
            summary["odds_note"] = "p_odds present but too sparse for robust calibration analysis."
    else:
        summary["odds_note"] = "No p_odds column found."

    # -----------------------------
    # Save summary artifact
    # -----------------------------
    out_summary = REP_DIR / "calibration_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Console summary
    print("Saved:", out_summary)
    print("Saved:", model_bins_path)
    print("Saved:", fs_model_path)
    print("Saved:", FIG_DIR / "calibration_reliability_model.png")
    print("Saved:", FIG_DIR / "calibration_fieldsize_model.png")
    if "odds" in summary:
        print("Saved:", REP_DIR / "calibration_bins_odds.csv")
        print("Saved:", REP_DIR / "calibration_by_field_size_odds.csv")
        print("Saved:", FIG_DIR / "calibration_reliability_odds.png")
        print("Saved:", FIG_DIR / "calibration_fieldsize_odds.png")

    print("\n== Calibration summary ==")
    m = summary["model"]
    print(f"Model: logloss={m['logloss']:.6f} brier={m['brier']:.6f} ece={m['ece']:.6f} "
          f"ols_intercept={m['cal_intercept_ols']:.4f} ols_slope={m['cal_slope_ols']:.4f}")

    if "odds" in summary:
        o = summary["odds"]
        print(f"Odds : logloss={o['logloss']:.6f} brier={o['brier']:.6f} ece={o['ece']:.6f} "
              f"ols_intercept={o['cal_intercept_ols']:.4f} ols_slope={o['cal_slope_ols']:.4f} "
              f"(coverage={summary.get('odds_coverage', float('nan')):.3f})")


if __name__ == "__main__":
    main()