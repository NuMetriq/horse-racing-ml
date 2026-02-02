from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CalibrationStrataConfig:
    pred_path: Path = Path("outputs/reports/pred_test.parquet")
    model_frame_path: Path = Path("data/processed/model_frame.parquet")

    out_json: Path = Path("outputs/reports/calibration_strata.json")
    out_md: Path = Path("outputs/reports/calibration_strata.md")
    fig_dir: Path = Path("outputs/figures")

    prob_col: str = "p_win"
    label_col: str = "is_winner"
    race_col: str = "race_id"
    horse_col: str = "horse_id"

    n_bins: int = 15

    # Strata config
    field_size_bins: Tuple[int, ...] = (6, 8, 10, 12, 14, 16, 999)
    # Distance buckets (in furlongs); adjust edges if you want
    sprint_max_f: float = 8.0
    middle_max_f: float = 12.0

    # If race_class is numeric or string, we bucket it coarsely
    max_class_groups: int = 8  # if too many unique values, we'll reduce


# -----------------------------
# Core calibration utilities
# -----------------------------
def ece_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> Tuple[float, pd.DataFrame]:
    """
    Expected Calibration Error (ECE) with equal-width bins in [0,1].
    Returns (ece, table) where table includes:
      bin_lo, bin_hi, n, p_mean, y_mean, abs_gap
    """
    y = y.astype(float)
    p = np.clip(p.astype(float), 1e-12, 1 - 1e-12)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=False)  # 0..n_bins-1

    rows = []
    ece = 0.0
    n_total = len(p)

    for b in range(n_bins):
        mask = bin_ids == b
        nb = int(mask.sum())
        lo = float(edges[b])
        hi = float(edges[b + 1])
        if nb == 0:
            rows.append(
                {"bin": b, "bin_lo": lo, "bin_hi": hi, "n": 0, "p_mean": np.nan, "y_mean": np.nan, "abs_gap": np.nan}
            )
            continue
        p_mean = float(np.mean(p[mask]))
        y_mean = float(np.mean(y[mask]))
        gap = abs(p_mean - y_mean)
        ece += (nb / n_total) * gap
        rows.append({"bin": b, "bin_lo": lo, "bin_hi": hi, "n": nb, "p_mean": p_mean, "y_mean": y_mean, "abs_gap": gap})

    return float(ece), pd.DataFrame(rows)


def reliability_plot(tbl: pd.DataFrame, title: str, out_path: Path) -> None:
    """
    Reliability plot from an ece_table dataframe (tbl).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Only bins with observations
    t = tbl[tbl["n"] > 0].copy()
    if t.empty:
        return

    x = t["p_mean"].to_numpy()
    y = t["y_mean"].to_numpy()

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")  # perfect calibration
    plt.plot(x, y, marker="o")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical win rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "na"


# -----------------------------
# Strata builders
# -----------------------------
def add_field_size_bucket(df: pd.DataFrame, cfg: CalibrationStrataConfig) -> pd.DataFrame:
    if "field_size" not in df.columns:
        raise ValueError("field_size missing. Cannot build field size strata.")

    edges = [0] + list(cfg.field_size_bins)
    labels = []
    prev = 1
    for e in cfg.field_size_bins:
        labels.append(f"{prev}-{e}")
        prev = e + 1

    out = df.copy()
    out["field_size_bucket"] = pd.cut(
        out["field_size"].astype(float),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype("object")
    return out


def add_distance_bucket(df: pd.DataFrame, cfg: CalibrationStrataConfig) -> pd.DataFrame:
    if "dist_f" not in df.columns:
        raise ValueError("dist_f missing. Join from model_frame.parquet first.")

    d = df["dist_f"].astype(float)
    out = df.copy()
    out["distance_bucket"] = np.where(
        d <= cfg.sprint_max_f,
        "sprint",
        np.where(d <= cfg.middle_max_f, "middle", "staying"),
    )
    return out


def add_race_class_bucket(df: pd.DataFrame, cfg: CalibrationStrataConfig) -> pd.DataFrame:
    if "race_class" not in df.columns:
        raise ValueError("race_class missing. Join from model_frame.parquet first.")

    out = df.copy()
    rc = out["race_class"]

    # If numeric, bucket by quantiles into up to max_class_groups groups
    if pd.api.types.is_numeric_dtype(rc):
        uniq = int(rc.nunique(dropna=True))
        q = min(cfg.max_class_groups, max(2, uniq))
        out["race_class_bucket"] = (
            pd.qcut(rc, q=q, duplicates="drop")
            .astype("object")
            .cat.add_categories(["NA"])
            .fillna("NA")
        )
        return out

    # Normalize string/object race_class:
    # - convert to object
    # - replace empty strings with NA
    rc2 = rc.astype("object")
    rc2 = rc2.replace("", "NA").fillna("NA")

    counts = rc2.value_counts(dropna=False)

    if len(counts) <= cfg.max_class_groups:
        out["race_class_bucket"] = rc2
    else:
        top = set(counts.head(cfg.max_class_groups - 1).index.tolist())
        out["race_class_bucket"] = rc2.apply(lambda x: x if x in top else "OTHER")

    return out


# -----------------------------
# Data join
# -----------------------------
def load_and_join(cfg: CalibrationStrataConfig) -> pd.DataFrame:
    pred = pd.read_parquet(cfg.pred_path).copy()

    needed = {cfg.race_col, cfg.horse_col, cfg.prob_col, cfg.label_col}
    missing = [c for c in needed if c not in pred.columns]
    if missing:
        raise ValueError(f"pred table missing required columns: {missing}")

    # Join dist_f, race_class if not present
    need_join_cols = []
    for c in ["dist_f", "race_class"]:
        if c not in pred.columns:
            need_join_cols.append(c)

    if need_join_cols:
        mf = pd.read_parquet(cfg.model_frame_path)
        join_cols = [cfg.race_col, cfg.horse_col] + need_join_cols
        mf2 = mf[join_cols].drop_duplicates(subset=[cfg.race_col, cfg.horse_col])

        pred = pred.merge(
            mf2,
            on=[cfg.race_col, cfg.horse_col],
            how="left",
            validate="many_to_one",
        )

    return pred


# -----------------------------
# Main runner
# -----------------------------
def run_calibration_strata(cfg: CalibrationStrataConfig) -> dict:
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_md.parent.mkdir(parents=True, exist_ok=True)
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_join(cfg)

    # Ensure numeric y/p
    y_all = df[cfg.label_col].astype(int).to_numpy()
    p_all = np.clip(df[cfg.prob_col].astype(float).to_numpy(), 1e-12, 1 - 1e-12)

    overall_ece, overall_tbl = ece_table(y_all, p_all, cfg.n_bins)
    reliability_plot(
        overall_tbl,
        title=f"Reliability (overall) | ECE={overall_ece:.4f}",
        out_path=cfg.fig_dir / "reliability_overall.png",
    )

    # Build strata
    df = add_field_size_bucket(df, cfg)
    # distance + class require join; if missing, skip gracefully with a note
    strata_defs: List[Tuple[str, str]] = [("field_size", "field_size_bucket")]

    dist_ok = "dist_f" in df.columns and df["dist_f"].notna().any()
    class_ok = "race_class" in df.columns and df["race_class"].notna().any()

    if dist_ok:
        df = add_distance_bucket(df, cfg)
        strata_defs.append(("distance", "distance_bucket"))

    if class_ok:
        df = add_race_class_bucket(df, cfg)
        strata_defs.append(("race_class", "race_class_bucket"))

    results = {
        "paths": {
            "pred_path": str(cfg.pred_path),
            "model_frame_path": str(cfg.model_frame_path),
            "out_json": str(cfg.out_json),
            "out_md": str(cfg.out_md),
            "fig_dir": str(cfg.fig_dir),
        },
        "overall": {
            "ece": float(overall_ece),
            "n_rows": int(len(df)),
            "n_races": int(df[cfg.race_col].nunique()),
            "n_bins": int(cfg.n_bins),
            "reliability_plot": str((cfg.fig_dir / "reliability_overall.png").as_posix()),
        },
        "strata": {},
        "notes": {
            "distance_included": bool(dist_ok),
            "race_class_included": bool(class_ok),
        },
    }

    # Per-stratum ECE + plots
    for strata_name, strata_col in strata_defs:
        block = []
        for key, g in df.groupby(strata_col, dropna=False):
            # key can be Interval for qcut, make it stringy
            key_str = str(key) if key is not None else "NA"
            y = g[cfg.label_col].astype(int).to_numpy()
            p = np.clip(g[cfg.prob_col].astype(float).to_numpy(), 1e-12, 1 - 1e-12)

            ece, tbl = ece_table(y, p, cfg.n_bins)

            fig_name = f"reliability_{strata_name}_{_slug(key_str)}.png"
            reliability_plot(
                tbl,
                title=f"Reliability ({strata_name}={key_str}) | ECE={ece:.4f} | n={len(g)}",
                out_path=cfg.fig_dir / fig_name,
            )

            block.append(
                {
                    "stratum": key_str,
                    "n_rows": int(len(g)),
                    "n_races": int(g[cfg.race_col].nunique()),
                    "ece": float(ece),
                    "plot": str((cfg.fig_dir / fig_name).as_posix()),
                    "bin_table": tbl.to_dict(orient="records"),
                }
            )

        # Sort strata by support
        block = sorted(block, key=lambda r: (-r["n_rows"], r["stratum"]))
        results["strata"][strata_name] = block

    # Write JSON
    cfg.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Write MD (summary tables only; bin tables stay in JSON)
    lines = []
    lines.append("# Calibration Stratification Diagnostics\n")
    lines.append(f"Predictions: `{cfg.pred_path.as_posix()}`\n\n")
    lines.append("This report evaluates calibration (ECE + reliability plots) across race strata.\n\n")

    lines.append("## Overall\n")
    lines.append(f"- rows: {results['overall']['n_rows']}\n")
    lines.append(f"- races: {results['overall']['n_races']}\n")
    lines.append(f"- ECE: **{results['overall']['ece']:.6f}**\n")
    lines.append(f"- plot: `{Path(results['overall']['reliability_plot']).as_posix()}`\n\n")

    def write_strata_table(name: str, rows: List[dict]) -> None:
        lines.append(f"## {name}\n")
        lines.append("| stratum | n_rows | n_races | ece | plot |\n")
        lines.append("|---|---:|---:|---:|---|\n")
        for r in rows:
            lines.append(f"| {r['stratum']} | {r['n_rows']} | {r['n_races']} | {r['ece']:.6f} | `{Path(r['plot']).as_posix()}` |\n")
        lines.append("\n")

        # “Most bins < 0.02” quick check
        ok = sum(1 for r in rows if (not math.isnan(r["ece"])) and r["ece"] < 0.02)
        total = sum(1 for r in rows if not math.isnan(r["ece"]))
        if total > 0:
            lines.append(f"- bins with ECE < 0.02: **{ok}/{total}**\n\n")

    write_strata_table("Field size buckets", results["strata"].get("field_size", []))

    if results["notes"]["distance_included"]:
        write_strata_table("Distance buckets", results["strata"].get("distance", []))
    else:
        lines.append("## Distance buckets\n")
        lines.append("- Skipped (missing `dist_f` join).\n\n")

    if results["notes"]["race_class_included"]:
        write_strata_table("Race class buckets", results["strata"].get("race_class", []))
    else:
        lines.append("## Race class buckets\n")
        lines.append("- Skipped (missing `race_class` join).\n\n")

    cfg.out_md.write_text("".join(lines), encoding="utf-8")
    return results