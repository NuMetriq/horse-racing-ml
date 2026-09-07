from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyConfig:
    pred_path: Path = Path("outputs/reports/pred_test.parquet")
    out_dir: Path = Path("outputs/bets")

    # Column names
    race_col: str = "race_id"
    horse_col: str = "horse_id"
    prob_col: str = "p_win"  # calibrated model win probability
    odds_col: str = "sp_decimal"  # decimal odds

    # Selection rule
    min_edge: float = 0.02  # p_model - p_implied >= min_edge
    min_prob: float = 0.03  # don't bet extreme longshots unless model says so
    max_bets_per_race: int = 1  # 1 = "best bet only" per race

    # Staking (simple fractional Kelly on win market)
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25  # 0.25 = quarter-Kelly
    max_stake_frac: float = 0.02  # cap stake to 2% of bankroll per bet
    min_stake: float = 1.0  # minimum stake (currency units)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x.astype(float), 1e-12, 1.0 - 1e-12)


def _kelly_fraction(p: float, d: float) -> float:
    """
    Kelly fraction for decimal odds d:
      b = d - 1
      f* = (p*b - (1-p)) / b = (p*d - 1) / (d - 1)
    """
    d = float(d)
    p = float(p)
    if not np.isfinite(d) or d <= 1.0 or not np.isfinite(p):
        return 0.0
    b = d - 1.0
    f = (p * d - 1.0) / b
    if not np.isfinite(f):
        return 0.0
    return max(0.0, f)


def build_bets(cfg: PolicyConfig) -> tuple[pd.DataFrame, dict]:
    if not cfg.pred_path.exists():
        raise FileNotFoundError(f"Missing predictions: {cfg.pred_path.as_posix()}")

    pred = pd.read_parquet(cfg.pred_path).copy()

    required = {cfg.race_col, cfg.horse_col, cfg.prob_col, cfg.odds_col}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(
            f"Predictions file missing required columns: {sorted(missing)}"
        )

    df = pred[[cfg.race_col, cfg.horse_col, cfg.prob_col, cfg.odds_col]].copy()

    # Basic cleanup
    df[cfg.prob_col] = _clip01(
        pd.to_numeric(df[cfg.prob_col], errors="coerce").to_numpy()
    )
    df[cfg.odds_col] = pd.to_numeric(df[cfg.odds_col], errors="coerce")
    df = df.dropna(subset=[cfg.prob_col, cfg.odds_col]).copy()
    df = df[np.isfinite(df[cfg.odds_col].to_numpy()) & (df[cfg.odds_col] > 1.0)].copy()

    # Market implied probability (not normalized within race; just 1/decimal)
    df["p_implied"] = _clip01((1.0 / df[cfg.odds_col].astype(float)).to_numpy())
    df["edge"] = df[cfg.prob_col].astype(float) - df["p_implied"].astype(float)
    df["ev_per_1"] = (
        df[cfg.prob_col].astype(float) * df[cfg.odds_col].astype(float) - 1.0
    )  # expected profit per $1

    # Filter candidates
    cand = df[(df["edge"] >= cfg.min_edge) & (df[cfg.prob_col] >= cfg.min_prob)].copy()

    # Pick top bets per race (by edge, tie-breaker ev_per_1)
    if cfg.max_bets_per_race >= 1 and not cand.empty:
        cand = (
            cand.sort_values(
                [cfg.race_col, "edge", "ev_per_1"], ascending=[True, False, False]
            )
            .groupby(cfg.race_col, as_index=False)
            .head(cfg.max_bets_per_race)
            .reset_index(drop=True)
        )

    # Stakes: fractional Kelly with cap
    stakes = []
    for _, r in cand.iterrows():
        p = float(r[cfg.prob_col])
        d = float(r[cfg.odds_col])
        f_star = _kelly_fraction(p, d)
        f = cfg.kelly_fraction * f_star
        f = min(f, cfg.max_stake_frac)
        stake = cfg.bankroll * f
        if stake > 0:
            stake = max(cfg.min_stake, stake)
        stakes.append(stake)

    cand["kelly_f_star"] = cand.apply(
        lambda r: _kelly_fraction(float(r[cfg.prob_col]), float(r[cfg.odds_col])),
        axis=1,
    )
    cand["stake"] = np.array(stakes, dtype=float)

    # Final bets table
    bets = cand.rename(
        columns={
            cfg.race_col: "race_id",
            cfg.horse_col: "horse_id",
            cfg.prob_col: "p_win",
            cfg.odds_col: "sp_decimal",
        }
    )[
        [
            "race_id",
            "horse_id",
            "p_win",
            "sp_decimal",
            "p_implied",
            "edge",
            "ev_per_1",
            "kelly_f_star",
            "stake",
        ]
    ].copy()

    # Summary
    summary = {
        "pred_path": cfg.pred_path.as_posix(),
        "n_rows_pred": int(len(df)),
        "n_bets": int(len(bets)),
        "n_races_with_bet": int(bets["race_id"].nunique()) if not bets.empty else 0,
        "min_edge": float(cfg.min_edge),
        "min_prob": float(cfg.min_prob),
        "max_bets_per_race": int(cfg.max_bets_per_race),
        "bankroll": float(cfg.bankroll),
        "kelly_fraction": float(cfg.kelly_fraction),
        "max_stake_frac": float(cfg.max_stake_frac),
        "total_staked": float(bets["stake"].sum()) if not bets.empty else 0.0,
        "mean_edge": float(bets["edge"].mean()) if not bets.empty else float("nan"),
        "mean_ev_per_1": float(bets["ev_per_1"].mean())
        if not bets.empty
        else float("nan"),
    }

    return bets, summary


def main(
    pred_path: str | Path = Path("outputs/reports/pred_test.parquet"),
    out_dir: str | Path = Path("outputs/bets"),
) -> None:
    cfg = PolicyConfig(pred_path=Path(pred_path), out_dir=Path(out_dir))
    bets, summary = build_bets(cfg)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = cfg.out_dir / "bets.parquet"
    out_csv = cfg.out_dir / "bets.csv"
    out_json = cfg.out_dir / "policy_summary.json"

    bets.to_parquet(out_parquet, index=False)
    bets.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:", out_parquet)
    print("Saved:", out_csv)
    print("Saved:", out_json)
    print(
        "Bets:",
        summary["n_bets"],
        "| races_with_bet:",
        summary["n_races_with_bet"],
        "| total_staked:",
        f"{summary['total_staked']:.2f}",
    )
