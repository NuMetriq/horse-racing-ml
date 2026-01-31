from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

PRED_PATH = Path("outputs/reports/pred_test.parquet")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR = Path("outputs/reports")
REP_DIR.mkdir(parents=True, exist_ok=True)


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


def calibration_plot(y: np.ndarray, p: np.ndarray, bins: int = 20, title: str = "Calibration") -> None:
    df = pd.DataFrame({"y": y, "p": p})
    # qcut can fail if many identical values; duplicates="drop" handles that
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    grp = df.groupby("bin", observed=True).agg(mean_p=("p", "mean"), mean_y=("y", "mean"), n=("y", "size")).reset_index()

    plt.figure()
    plt.plot(grp["mean_p"], grp["mean_y"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed win rate")
    plt.title(title)


def field_size_breakdown(pred: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    if "field_size" not in pred.columns:
        return pd.DataFrame()

    df = pred.copy()
    # bins that make sense for UK/Ireland fields
    bins = [0, 7, 10, 14, 1000]
    labels = ["<=7", "8-10", "11-14", "15+"]
    df["field_bin"] = pd.cut(df["field_size"].astype(float), bins=bins, labels=labels)

    rows = []
    for b, g in df.groupby("field_bin", observed=True):
        if g.empty:
            continue
        rows.append({
            "field_bin": str(b),
            "n_races": int(g["race_id"].nunique()),
            "top1_acc": top1_acc(g, prob_col),
            "top3_hit": topk_hit(g, prob_col, 3),
        })
    return pd.DataFrame(rows)


def eval_block(pred: pd.DataFrame, prob_col: str, label: str) -> Dict[str, float]:
    y = pred["is_winner"].astype(int).values
    p = np.clip(pred[prob_col].values.astype(float), 1e-12, 1 - 1e-12)

    return {
        "label": label,
        "n_rows": int(len(pred)),
        "n_races": int(pred["race_id"].nunique()),
        "logloss": float(log_loss(y, p)),
        "brier": brier(y, p),
        "top1_acc": top1_acc(pred, prob_col),
        "top3_hit": topk_hit(pred, prob_col, 3),
    }


def to_markdown_table(rows) -> str:
    # rows: list of dicts with same keys
    cols = ["label", "n_races", "logloss", "brier", "top1_acc", "top3_hit"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows:
        line = "| " + " | ".join([
            str(r.get("label", "")),
            str(r.get("n_races", "")),
            f"{r.get('logloss', float('nan')):.6f}",
            f"{r.get('brier', float('nan')):.6f}",
            f"{r.get('top1_acc', float('nan')):.6f}",
            f"{r.get('top3_hit', float('nan')):.6f}",
        ]) + " |"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError("Run training first: `python -m hrml.models.train_xgb_optuna`")

    pred = pd.read_parquet(PRED_PATH).copy()

    # Ensure required cols
    required = {"race_id", "is_winner", "p_win"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"pred_test.parquet missing required columns: {missing}")

    # Uniform baseline always available if field_size exists
    if "field_size" in pred.columns:
        fs = pred["field_size"].astype(float).clip(lower=1.0).values
        pred["p_unif"] = np.clip(1.0 / fs, 1e-12, 1 - 1e-12)

    blocks = []

    # Model
    blocks.append(eval_block(pred, "p_win", "model"))

    # Uniform baseline
    if "p_unif" in pred.columns:
        blocks.append(eval_block(pred, "p_unif", "uniform (1/field_size)"))

    # Odds baseline (subset where present)
    if "p_odds" in pred.columns:
        m = pred["p_odds"].notnull()
        coverage = float(m.mean())
        print(f"\nOdds coverage in test predictions: {coverage:.3f}")
        if m.sum() > 1000:  # sanity minimum
            pred_odds = pred.loc[m].copy()
            pred_odds["p_odds"] = np.clip(pred_odds["p_odds"].astype(float).values, 1e-12, 1 - 1e-12)
            blocks.append(eval_block(pred_odds, "p_odds", "odds baseline (SP implied)"))

            # calibration for odds subset
            y0 = pred_odds["is_winner"].astype(int).values
            p0 = pred_odds["p_odds"].values.astype(float)
            calibration_plot(y0, p0, bins=20, title="Odds baseline calibration (test subset)")
            fig_path2 = FIG_DIR / "calibration_odds.png"
            plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved:", fig_path2)
        else:
            print("Odds baseline present but too sparse; skipping odds metrics.")
    else:
        print("\nNo p_odds column found in predictions (rerun training after patch).")

    # Print blocks
    print("\n== Metrics ==")
    for r in blocks:
        print(f"{r['label']}")
        print(f"  n_races: {r['n_races']}")
        print(f"  logloss: {r['logloss']:.6f}")
        print(f"   brier : {r['brier']:.6f}")
        print(f" top1_acc: {r['top1_acc']:.6f}")
        print(f" top3_hit: {r['top3_hit']:.6f}")

    # Model calibration plot (full test)
    y = pred["is_winner"].astype(int).values
    p = np.clip(pred["p_win"].values.astype(float), 1e-12, 1 - 1e-12)
    calibration_plot(y, p, bins=20, title="Model calibration (test)")
    fig_path = FIG_DIR / "calibration_model.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved:", fig_path)

    # Field-size breakdown
    fs_tbl = field_size_breakdown(pred, "p_win")
    if not fs_tbl.empty:
        fs_path = REP_DIR / "field_size_breakdown.csv"
        fs_tbl.to_csv(fs_path, index=False)
        print("Saved:", fs_path)

    # Save metrics artifacts
    metrics = {"blocks": blocks}
    metrics_path = REP_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved:", metrics_path)

    md = to_markdown_table(blocks)
    md_path = REP_DIR / "metrics.md"
    md_path.write_text(md, encoding="utf-8")
    print("Saved:", md_path)


if __name__ == "__main__":
    main()