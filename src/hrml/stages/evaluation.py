# src/hrml/stages/evaluation.py
from __future__ import annotations

from pathlib import Path


def main() -> None:
    """
    Pipeline stage: evaluation

    Responsibility:
      - compute evaluation metrics/reports from predictions (and optionally bets)
    """
    from hrml.eval.run_eval import main as eval_main

    eval_main()

    out_json = Path("outputs/reports/metrics.json")
    out_md = Path("outputs/reports/metrics.md")
    if not out_json.exists() or not out_md.exists():
        missing = [p.as_posix() for p in (out_json, out_md) if not p.exists()]
        raise FileNotFoundError(
            "Evaluation stage finished but expected outputs were not found: "
            + ", ".join(missing)
        )
