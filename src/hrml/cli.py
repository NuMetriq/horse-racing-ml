from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Keep imports local to commands where possible to reduce import-time failures
# when users only want --help, etc.

LOGGER = logging.getLogger("hrml")


def _configure_logging(verbosity: int) -> None:
    """
    verbosity: 0=WARNING, 1=INFO, 2+=DEBUG
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    LOGGER.debug("Logging configured (verbosity=%s)", verbosity)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v=INFO, -vv=DEBUG).",
    )


# -----------------------------
# Commands
# -----------------------------
def cmd_eval_ranking(args: argparse.Namespace) -> int:
    """
    Compute race-aware ranking metrics (Issue #1) from an existing pred_test.parquet.
    Writes:
      outputs/reports/metrics_ranking.json
      outputs/reports/metrics_ranking.md
    """
    from hrml.eval.ranking import RankingEvalConfig, run_ranking_eval

    cfg = RankingEvalConfig(
        pred_path=Path(args.pred_path),
        out_json=Path(args.out_json),
        out_md=Path(args.out_md),
        k_values=(args.k1, args.k2) if args.k1 != args.k2 else (args.k1,),
        race_col=args.race_col,
        score_col=args.score_col,
        winner_col=args.winner_col,
    )

    LOGGER.info("Reading predictions: %s", cfg.pred_path)
    report = run_ranking_eval(cfg)

    agg = report["aggregate"]
    # Print a concise summary (useful in CI / logs)
    ndcg_keys = sorted([k for k in agg.keys() if k.startswith("ndcg@")], key=lambda s: int(s.split("@")[1]))
    ndcg_str = ", ".join(f"{k}={agg[k]:.6f}" for k in ndcg_keys)

    print("=== Ranking metrics (race-aware) ===")
    print(f"races_used={agg['n_races_used']} / total={agg['n_races_total']}")
    print(f"mean_winner_rank={agg['mean_winner_rank']:.6f}")
    print(f"mrr={agg['mrr']:.6f}")
    if ndcg_str:
        print(ndcg_str)
    print(f"winners_top3={agg['pct_winner_in_top3']:.6f}")
    print(f"winners_top5={agg['pct_winner_in_top5']:.6f}")
    print(f"wrote_json={cfg.out_json}")
    print(f"wrote_md={cfg.out_md}")
    return 0


def cmd_paths(_: argparse.Namespace) -> int:
    """
    Quick helper to print common project paths (handy when debugging runs).
    """
    cwd = Path.cwd()
    print("cwd:", cwd)
    print("outputs:", cwd / "outputs")
    print("reports:", cwd / "outputs" / "reports")
    print("figures:", cwd / "outputs" / "figures")
    return 0


def cmd_version(_: argparse.Namespace) -> int:
    """
    Print package version if available.
    """
    try:
        from importlib.metadata import version  # py3.8+
        print(version("hrml"))
    except Exception:
        # If packaging metadata isn't available (editable installs can vary),
        # fall back to __version__ if you define it.
        try:
            import hrml  # type: ignore
            print(getattr(hrml, "__version__", "unknown"))
        except Exception:
            print("unknown")
    return 0


# -----------------------------
# Parser / entrypoint
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrml",
        description="HRML: Horse Racing ML utilities (training, evaluation, reporting).",
    )
    _add_common_args(parser)

    sub = parser.add_subparsers(dest="command", required=True)

    # eval-ranking (Issue #1)
    p_rank = sub.add_parser(
        "eval-ranking",
        help="Compute race-aware ranking metrics from existing predictions.",
    )
    p_rank.add_argument(
        "--pred-path",
        default="outputs/pred_test.parquet",
        help="Path to predictions parquet (default: outputs/pred_test.parquet).",
    )
    p_rank.add_argument(
        "--out-json",
        default="outputs/reports/metrics_ranking.json",
        help="Output JSON path (default: outputs/reports/metrics_ranking.json).",
    )
    p_rank.add_argument(
        "--out-md",
        default="outputs/reports/metrics_ranking.md",
        help="Output Markdown path (default: outputs/reports/metrics_ranking.md).",
    )
    p_rank.add_argument(
        "--race-col",
        default=None,
        help="Optional override for race id column name.",
    )
    p_rank.add_argument(
        "--score-col",
        default=None,
        help="Optional override for prediction score column name (higher = better).",
    )
    p_rank.add_argument(
        "--winner-col",
        default=None,
        help="Optional override for winner/label column name (truth).",
    )
    # Allow two K's, defaults match your plan
    p_rank.add_argument("--k1", type=int, default=3, help="First NDCG cutoff k (default: 3).")
    p_rank.add_argument("--k2", type=int, default=5, help="Second NDCG cutoff k (default: 5).")
    _add_common_args(p_rank)
    p_rank.set_defaults(func=cmd_eval_ranking)

    # misc helpers
    p_paths = sub.add_parser("paths", help="Print common project paths.")
    _add_common_args(p_paths)
    p_paths.set_defaults(func=cmd_paths)

    p_ver = sub.add_parser("version", help="Print hrml version (if available).")
    _add_common_args(p_ver)
    p_ver.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Root verbosity applies, but allow subparser verbosity to override if provided.
    verbosity = getattr(args, "verbose", 0) or 0
    _configure_logging(verbosity)

    func = getattr(args, "func", None)
    if not func:
        parser.print_help()
        return 2

    try:
        return int(func(args))
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted.")
        return 130
    except Exception as e:
        LOGGER.exception("Command failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())