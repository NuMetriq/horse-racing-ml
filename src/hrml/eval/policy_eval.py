# src/hrml/eval/policy_eval.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyEvalConfig:
    # Inputs
    bets_path: Path = Path("outputs/bets/bets.parquet")
    pred_path: Path = Path("outputs/reports/pred_test.parquet")
    policy_summary_path: Path = Path("outputs/bets/policy_summary.json")

    # Outputs
    out_json: Path = Path("outputs/reports/policy_metrics.json")
    out_md: Path = Path("outputs/reports/policy_metrics.md")
    fig_dir: Path = Path("outputs/figures")
    scored_bets_path: Path = Path("outputs/reports/policy_bets_scored.parquet")

    # Column names (pred table)
    race_col: str = "race_id"
    horse_col: str = "horse_id"
    winner_col: str = "is_winner"
    odds_col: str = "sp_decimal"
    date_col: Optional[str] = (
        "race_date"  # if missing, we fall back to (race_id, horse_id) sort
    )

    # Column names (bets table)
    stake_col: str = "stake"

    # Bankroll
    bankroll0: Optional[float] = (
        None  # if None, will read from policy_summary.json if present
    )


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x.astype(float), 1e-12, 1.0 - 1e-12)


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _max_drawdown(equity: np.ndarray) -> float:
    """
    Max drawdown fraction: max(peak - trough) / peak over the curve.
    """
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return float("nan")
    peaks = np.maximum.accumulate(eq)
    dd = (peaks - eq) / np.where(peaks > 0, peaks, np.nan)
    return float(np.nanmax(dd))


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        return float("nan")
    # per-bet Sharpe scaled by sqrt(n) (treat each bet as one "period")
    return float(mu / sd * np.sqrt(len(r)))


def _sortino(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    mu = float(np.mean(r))
    downside = r[r < 0]
    if downside.size < 2:
        return float("nan")
    dd = float(np.std(downside, ddof=1))
    if dd <= 0:
        return float("nan")
    return float(mu / dd * np.sqrt(len(r)))


def _load_bankroll0(cfg: PolicyEvalConfig) -> float:
    if cfg.bankroll0 is not None:
        return float(cfg.bankroll0)

    if cfg.policy_summary_path.exists():
        try:
            payload = json.loads(cfg.policy_summary_path.read_text(encoding="utf-8"))
            if "bankroll" in payload:
                return float(payload["bankroll"])
        except Exception:
            pass

    # fallback
    return 1000.0


def score_bets(cfg: PolicyEvalConfig) -> Tuple[pd.DataFrame, dict]:
    if not cfg.bets_path.exists():
        raise FileNotFoundError(f"Missing bets: {cfg.bets_path.as_posix()}")
    if not cfg.pred_path.exists():
        raise FileNotFoundError(f"Missing predictions: {cfg.pred_path.as_posix()}")

    bets = pd.read_parquet(cfg.bets_path).copy()
    pred = pd.read_parquet(cfg.pred_path).copy()

    # Minimal required fields in bets
    bets_need = {cfg.race_col, cfg.horse_col, cfg.stake_col}
    missing_b = [c for c in bets_need if c not in bets.columns]
    if missing_b:
        raise ValueError(f"bets table missing required columns: {missing_b}")

    # Minimal required fields in pred
    pred_need = {cfg.race_col, cfg.horse_col, cfg.winner_col}
    missing_p = [c for c in pred_need if c not in pred.columns]
    if missing_p:
        raise ValueError(f"pred table missing required columns: {missing_p}")

    # Odds for settlement: prefer odds from bets; otherwise join from pred
    if cfg.odds_col not in bets.columns:
        if cfg.odds_col not in pred.columns:
            raise ValueError(f"Need '{cfg.odds_col}' in bets or pred to settle PnL.")
        join_cols = [cfg.race_col, cfg.horse_col, cfg.odds_col]
        pred2 = pred[join_cols].drop_duplicates(subset=[cfg.race_col, cfg.horse_col])
        bets = bets.merge(
            pred2, on=[cfg.race_col, cfg.horse_col], how="left", validate="many_to_one"
        )

    # Join outcome (is_winner) + date (optional)
    join_cols = [cfg.race_col, cfg.horse_col, cfg.winner_col]
    if cfg.date_col and cfg.date_col in pred.columns:
        join_cols.append(cfg.date_col)
    pred3 = pred[join_cols].drop_duplicates(subset=[cfg.race_col, cfg.horse_col])

    scored = bets.merge(
        pred3,
        on=[cfg.race_col, cfg.horse_col],
        how="left",
        validate="many_to_one",
        suffixes=("", "_pred"),
    )

    # Clean types
    scored[cfg.winner_col] = (
        pd.to_numeric(scored[cfg.winner_col], errors="coerce").fillna(0).astype(int)
    )
    scored[cfg.stake_col] = (
        pd.to_numeric(scored[cfg.stake_col], errors="coerce").fillna(0.0).astype(float)
    )
    scored[cfg.odds_col] = pd.to_numeric(scored[cfg.odds_col], errors="coerce")

    # Settlement
    # Profit = stake*(odds-1) if win else -stake
    win = scored[cfg.winner_col].astype(int).to_numpy()
    stake = scored[cfg.stake_col].astype(float).to_numpy()
    odds = scored[cfg.odds_col].astype(float).to_numpy()

    valid = np.isfinite(stake) & (stake > 0) & np.isfinite(odds) & (odds > 1.0)
    scored["is_settle_valid"] = valid.astype(int)

    profit = np.full(len(scored), np.nan, dtype=float)
    profit[valid] = np.where(
        win[valid] == 1, stake[valid] * (odds[valid] - 1.0), -stake[valid]
    )

    scored["profit"] = profit
    scored["return_per_bet"] = scored["profit"] / scored[cfg.stake_col].replace(
        0.0, np.nan
    )

    # Ordering for equity curve
    if cfg.date_col and cfg.date_col in scored.columns:
        scored[cfg.date_col] = pd.to_datetime(scored[cfg.date_col], errors="coerce")
        scored = scored.sort_values(
            [cfg.date_col, cfg.race_col, cfg.horse_col]
        ).reset_index(drop=True)
    else:
        scored = scored.sort_values([cfg.race_col, cfg.horse_col]).reset_index(
            drop=True
        )

    bankroll0 = _load_bankroll0(cfg)
    scored["cum_profit"] = scored["profit"].fillna(0.0).cumsum()
    scored["equity"] = bankroll0 + scored["cum_profit"]

    # Metrics (Issue #13 core)
    settle = scored[scored["is_settle_valid"] == 1].copy()
    total_staked = float(settle[cfg.stake_col].sum()) if not settle.empty else 0.0
    total_profit = float(settle["profit"].sum()) if not settle.empty else 0.0
    roi = float(total_profit / total_staked) if total_staked > 0 else float("nan")

    hit_rate = (
        float((settle[cfg.winner_col] == 1).mean())
        if not settle.empty
        else float("nan")
    )

    # Risk metrics (Issue #14)
    rets = (
        settle["return_per_bet"].to_numpy()
        if not settle.empty
        else np.array([], dtype=float)
    )
    equity = (
        scored["equity"].to_numpy() if not scored.empty else np.array([], dtype=float)
    )

    metrics = {
        "paths": {
            "bets_path": cfg.bets_path.as_posix(),
            "pred_path": cfg.pred_path.as_posix(),
            "policy_summary_path": cfg.policy_summary_path.as_posix(),
            "out_json": cfg.out_json.as_posix(),
            "out_md": cfg.out_md.as_posix(),
            "fig_dir": cfg.fig_dir.as_posix(),
            "scored_bets_path": cfg.scored_bets_path.as_posix(),
        },
        "bankroll0": float(bankroll0),
        "n_bets": int(len(scored)),
        "n_settled": int(len(settle)),
        "n_wins": int((settle[cfg.winner_col] == 1).sum()) if not settle.empty else 0,
        "hit_rate": hit_rate,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "equity_final": float(equity[-1]) if equity.size else float("nan"),
        "max_drawdown": _max_drawdown(equity) if equity.size else float("nan"),
        "sharpe_per_bet": _sharpe(rets),
        "sortino_per_bet": _sortino(rets),
    }

    return scored, metrics


def _write_md(metrics: dict) -> str:
    def f(x) -> str:
        try:
            v = float(x)
            if not np.isfinite(v):
                return "NA"
            return f"{v:.6f}"
        except Exception:
            return "NA"

    lines = []
    lines.append("# Policy Backtest Metrics\n\n")
    lines.append("Realized settlement uses win market at `sp_decimal`:\n")
    lines.append("- win: `profit = stake * (sp_decimal - 1)`\n")
    lines.append("- lose: `profit = -stake`\n\n")

    lines.append("## Summary\n\n")
    keys = [
        "bankroll0",
        "n_bets",
        "n_settled",
        "n_wins",
        "hit_rate",
        "total_staked",
        "total_profit",
        "roi",
        "equity_final",
        "max_drawdown",
        "sharpe_per_bet",
        "sortino_per_bet",
    ]
    lines.append("| metric | value |\n")
    lines.append("|---|---:|\n")
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, int):
            lines.append(f"| {k} | {v} |\n")
        else:
            lines.append(f"| {k} | {f(v)} |\n")

    lines.append("\n## Artifacts\n\n")
    paths = metrics.get("paths", {})
    for k in ["out_json", "scored_bets_path", "fig_dir"]:
        if k in paths:
            lines.append(f"- `{paths[k]}`\n")

    return "".join(lines)


def _plot_equity(scored: pd.DataFrame, cfg: PolicyEvalConfig) -> None:
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    if scored.empty or "equity" not in scored.columns:
        return

    x = np.arange(len(scored))
    y = scored["equity"].astype(float).to_numpy()

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Bet index (ordered)")
    plt.ylabel("Equity")
    plt.title("Equity curve (policy)")
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "policy_equity_curve.png", dpi=160)
    plt.close()

    peaks = np.maximum.accumulate(y)
    dd = (peaks - y) / np.where(peaks > 0, peaks, np.nan)

    plt.figure()
    plt.plot(x, dd)
    plt.xlabel("Bet index (ordered)")
    plt.ylabel("Drawdown (fraction)")
    plt.title("Drawdown curve (policy)")
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "policy_drawdown_curve.png", dpi=160)
    plt.close()


def run_policy_eval(cfg: PolicyEvalConfig) -> dict:
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_md.parent.mkdir(parents=True, exist_ok=True)
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    scored, metrics = score_bets(cfg)

    # Save scored bets (debug-friendly)
    cfg.scored_bets_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(cfg.scored_bets_path, index=False)

    # Plots
    _plot_equity(scored, cfg)

    # Write artifacts
    cfg.out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cfg.out_md.write_text(_write_md(metrics), encoding="utf-8")
    return metrics
