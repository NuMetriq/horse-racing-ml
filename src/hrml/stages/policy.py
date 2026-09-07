# src/hrml/stages/policy.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def main(config_path: str | Path | None = None) -> None:
    """
    Pipeline stage: policy

    Responsibility:
      - convert calibrated probabilities + odds into bet decisions

    YAML config controls:
      - pred_path, out_dir
      - policy thresholds + bankroll/kelly settings (via PolicyConfig)
    """
    from hrml.config import load_yaml
    from hrml.policy.run_policy import PolicyConfig, build_bets

    cfg = load_yaml(config_path)

    pred_path = Path(cfg.get("pred_path", "outputs/reports/pred_test.parquet"))
    out_dir = Path(cfg.get("out_dir", "outputs/bets"))

    # policy parameters (match PolicyConfig fields)
    pcfg = cfg.get("policy", {}) if isinstance(cfg.get("policy", {}), dict) else {}

    policy_cfg = PolicyConfig(
        pred_path=pred_path,
        out_dir=out_dir,
        race_col=str(pcfg.get("race_col", "race_id")),
        horse_col=str(pcfg.get("horse_col", "horse_id")),
        prob_col=str(pcfg.get("prob_col", "p_win")),
        odds_col=str(pcfg.get("odds_col", "sp_decimal")),
        min_edge=float(pcfg.get("min_edge", 0.02)),
        min_prob=float(pcfg.get("min_prob", 0.03)),
        max_bets_per_race=int(pcfg.get("max_bets_per_race", 1)),
        bankroll=float(pcfg.get("bankroll", 1000.0)),
        kelly_fraction=float(pcfg.get("kelly_fraction", 0.25)),
        max_stake_frac=float(pcfg.get("max_stake_frac", 0.02)),
        min_stake=float(pcfg.get("min_stake", 1.0)),
    )

    bets, summary = build_bets(policy_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "bets.parquet"
    out_csv = out_dir / "bets.csv"
    out_json = out_dir / "policy_summary.json"

    bets.to_parquet(out_parquet, index=False)
    bets.to_csv(out_csv, index=False)
    out_json.write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")

    # Stage assertions
    if not out_parquet.exists():
        raise FileNotFoundError(
            f"Expected bets parquet not found: {out_parquet.as_posix()}"
        )
    if not out_json.exists():
        raise FileNotFoundError(
            f"Expected policy summary not found: {out_json.as_posix()}"
        )
