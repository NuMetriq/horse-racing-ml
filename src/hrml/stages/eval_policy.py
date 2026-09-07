# src/hrml/stages/eval_policy.py
from __future__ import annotations

from pathlib import Path


def main() -> None:
    """
    Pipeline stage: eval_policy

    Responsibility:
      - evaluate realized betting performance from outputs/bets

    Inputs:
      - outputs/bets/bets.parquet
      - outputs/reports/pred_test.parquet  (must include is_winner; race_date optional)

    Outputs:
      - outputs/reports/policy_metrics.json
      - outputs/reports/policy_metrics.md
      - outputs/reports/policy_bets_scored.parquet
      - outputs/figures/policy_equity_curve.png
      - outputs/figures/policy_drawdown_curve.png
    """
    from hrml.eval.policy_eval import PolicyEvalConfig, run_policy_eval

    cfg = PolicyEvalConfig(
        bets_path=Path("outputs/bets/bets.parquet"),
        pred_path=Path("outputs/reports/pred_test.parquet"),
        policy_summary_path=Path("outputs/bets/policy_summary.json"),
        out_json=Path("outputs/reports/policy_metrics.json"),
        out_md=Path("outputs/reports/policy_metrics.md"),
        fig_dir=Path("outputs/figures"),
        scored_bets_path=Path("outputs/reports/policy_bets_scored.parquet"),
    )
    rep = run_policy_eval(cfg)

    print("=== Policy evaluation ===")
    print(
        "n_bets=",
        rep["n_bets"],
        "n_settled=",
        rep["n_settled"],
        "hit_rate=",
        f"{rep['hit_rate']:.3f}",
    )
    print("total_profit=", f"{rep['total_profit']:.2f}", "roi=", f"{rep['roi']:.4f}")
    print("max_drawdown=", f"{rep['max_drawdown']:.4f}")
    print("Saved:", cfg.out_json)
    print("Saved:", cfg.out_md)
    print("Saved:", cfg.scored_bets_path)
    print("Saved:", cfg.fig_dir / "policy_equity_curve.png")
    print("Saved:", cfg.fig_dir / "policy_drawdown_curve.png")
