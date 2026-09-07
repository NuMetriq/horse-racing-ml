# src/hrml/cli.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger("hrml")


def _configure_logging(verbosity: int) -> None:
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


def _add_stage_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Pipeline config YAML (default: configs/default.yaml).",
    )


# -----------------------------
# Commands: Evaluation
# -----------------------------
def cmd_eval_policy(args: argparse.Namespace) -> int:
    from hrml.eval.policy_eval import PolicyEvalConfig, run_policy_eval

    cfg = PolicyEvalConfig(
        bets_path=Path(args.bets_path),
        pred_path=Path(args.pred_path),
        policy_summary_path=Path(args.policy_summary_path),
        out_json=Path(args.out_json),
        out_md=Path(args.out_md),
        fig_dir=Path(args.fig_dir),
        scored_bets_path=Path(args.scored_bets_path),
        bankroll0=float(args.bankroll0) if args.bankroll0 is not None else None,
    )
    rep = run_policy_eval(cfg)
    print("=== Policy evaluation ===")
    print(
        "roi=",
        f"{rep['roi']:.6f}",
        "profit=",
        f"{rep['total_profit']:.2f}",
        "mdd=",
        f"{rep['max_drawdown']:.6f}",
    )
    return 0


def cmd_eval_ranking(args: argparse.Namespace) -> int:
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

    ndcg_keys = sorted(
        [k for k in agg.keys() if k.startswith("ndcg@")],
        key=lambda s: int(s.split("@")[1]),
    )
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


def cmd_eval_calibration_strata(args: argparse.Namespace) -> int:
    from hrml.eval.calibration_strata import (
        CalibrationStrataConfig,
        run_calibration_strata,
    )

    cfg = CalibrationStrataConfig(
        pred_path=Path(args.pred_path),
        model_frame_path=Path(args.model_frame_path),
        out_json=Path(args.out_json),
        out_md=Path(args.out_md),
        fig_dir=Path(args.fig_dir),
        prob_col=args.prob_col,
        label_col=args.label_col,
        n_bins=args.n_bins,
    )
    rep = run_calibration_strata(cfg)
    print("=== Calibration strata ===")
    print("overall_ece=", f"{rep['overall']['ece']:.6f}")
    print("wrote_json=", cfg.out_json)
    print("wrote_md=", cfg.out_md)
    print("fig_dir=", cfg.fig_dir)
    return 0


# -----------------------------
# Commands: Stages
# -----------------------------
def cmd_stage_ingest(args: argparse.Namespace) -> int:
    from hrml.stages.ingest import main as stage_main

    # stage_main may or may not accept config_path; call safely
    try:
        stage_main(config_path=Path(args.config))
    except TypeError:
        stage_main()
    return 0


def cmd_stage_features(args: argparse.Namespace) -> int:
    from hrml.stages.features import main as stage_main

    try:
        stage_main(config_path=Path(args.config))
    except TypeError:
        stage_main()
    return 0


def cmd_stage_model_softmax(args: argparse.Namespace) -> int:
    from hrml.stages.model import main_softmax as stage_main

    stage_main(
        base_only=bool(args.base_only),
        reuse_existing=bool(args.reuse_existing),
        fast_dev=bool(args.fast_dev),
        features_only=bool(args.features_only),
        config_path=Path(args.config),
    )
    return 0


def cmd_stage_model_pairwise(args: argparse.Namespace) -> int:
    from hrml.stages.model import main_pairwise as stage_main

    stage_main(
        base_only=bool(args.base_only),
        reuse_existing=bool(args.reuse_existing),
        fast_dev=bool(args.fast_dev),
        features_only=bool(args.features_only),
        config_path=Path(args.config),
    )
    return 0


def cmd_stage_model_plackett_luce(args: argparse.Namespace) -> int:
    from hrml.stages.model import main_plackett_luce as stage_main

    stage_main(
        reuse_existing=bool(args.reuse_existing),
        features_only=bool(args.features_only),
        fast_dev=bool(args.fast_dev),
        top_k=int(args.top_k),
        mc_samples=int(args.mc_samples),
        place_k=int(args.place_k),
        no_calibrate=bool(args.no_calibrate),
        base_only=bool(args.base_only),
        config_path=Path(args.config),
    )
    return 0


def cmd_stage_calibrate(args: argparse.Namespace) -> int:
    from hrml.stages.calibration import main as stage_main

    try:
        stage_main(config_path=Path(args.config))
    except TypeError:
        stage_main()
    return 0


def cmd_stage_policy(args: argparse.Namespace) -> int:
    from hrml.stages.policy import main as stage_main

    try:
        stage_main(config_path=Path(args.config))
    except TypeError:
        stage_main()
    return 0


def cmd_stage_evaluate(args: argparse.Namespace) -> int:
    from hrml.stages.evaluation import main as stage_main

    try:
        stage_main(config_path=Path(args.config))
    except TypeError:
        stage_main()
    return 0


# -----------------------------
# Misc helpers
# -----------------------------
def cmd_paths(_: argparse.Namespace) -> int:
    cwd = Path.cwd()
    print("cwd:", cwd)
    print("outputs:", cwd / "outputs")
    print("reports:", cwd / "outputs" / "reports")
    print("figures:", cwd / "outputs" / "figures")
    return 0


def cmd_version(_: argparse.Namespace) -> int:
    try:
        from importlib.metadata import version

        print(version("hrml"))
    except Exception:
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
        description="HRML: Horse Racing ML utilities (pipeline stages, training, evaluation, reporting).",
    )
    _add_common_args(parser)

    sub = parser.add_subparsers(dest="command", required=True)

    # Stages
    p_ing = sub.add_parser(
        "ingest", help="Run ingest stage (raw -> processed races/runners)."
    )
    _add_stage_config_arg(p_ing)
    _add_common_args(p_ing)
    p_ing.set_defaults(func=cmd_stage_ingest)

    p_feat = sub.add_parser(
        "features", help="Run features stage (processed -> model_frame)."
    )
    _add_stage_config_arg(p_feat)
    _add_common_args(p_feat)
    p_feat.set_defaults(func=cmd_stage_features)

    p_m_soft = sub.add_parser(
        "model-softmax", help="Train race-softmax model stage and write predictions."
    )
    _add_stage_config_arg(p_m_soft)
    p_m_soft.add_argument(
        "--base-only",
        action="store_true",
        help="Skip ablations and train base model only.",
    )
    p_m_soft.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse canonical model+predictions if present.",
    )
    p_m_soft.add_argument(
        "--fast-dev",
        action="store_true",
        help="Fast dev mode: fewer rounds (and skips ablations).",
    )
    p_m_soft.add_argument(
        "--features-only",
        action="store_true",
        help="Only build feature list + manifest then exit.",
    )
    _add_common_args(p_m_soft)
    p_m_soft.set_defaults(func=cmd_stage_model_softmax)

    p_m_pair = sub.add_parser(
        "model-pairwise", help="Train pairwise rank model stage and write predictions."
    )
    _add_stage_config_arg(p_m_pair)
    p_m_pair.add_argument(
        "--base-only",
        action="store_true",
        help="No-op for pairwise (kept for CLI consistency).",
    )
    p_m_pair.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse canonical model+predictions if present.",
    )
    p_m_pair.add_argument(
        "--fast-dev",
        action="store_true",
        help="Fast dev mode: fewer rounds for quick iteration.",
    )
    p_m_pair.add_argument(
        "--features-only",
        action="store_true",
        help="Only build feature list + manifest then exit.",
    )
    _add_common_args(p_m_pair)
    p_m_pair.set_defaults(func=cmd_stage_model_pairwise)

    p_m_pl = sub.add_parser(
        "model-pl", help="Train Plackett–Luce model stage and write predictions."
    )
    _add_stage_config_arg(p_m_pl)
    p_m_pl.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse canonical model+predictions if present.",
    )
    p_m_pl.add_argument(
        "--features-only",
        action="store_true",
        help="Only build feature list + manifest then exit.",
    )
    p_m_pl.add_argument(
        "--fast-dev", action="store_true", help="Fast dev mode for quick iteration."
    )
    p_m_pl.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Plackett–Luce top-K used in the loss (default: 3).",
    )
    p_m_pl.add_argument(
        "--mc-samples", type=int, default=200, help="MC samples per race (0 disables)."
    )
    p_m_pl.add_argument(
        "--place-k",
        type=int,
        default=3,
        help="Place threshold for place prob (default: 3).",
    )
    p_m_pl.add_argument(
        "--no-calibrate", action="store_true", help="Disable temperature scaling."
    )
    p_m_pl.add_argument(
        "--base-only",
        action="store_true",
        help="Disable calibration + MC outputs (training unchanged).",
    )
    _add_common_args(p_m_pl)
    p_m_pl.set_defaults(func=cmd_stage_model_plackett_luce)

    p_cal_stage = sub.add_parser(
        "calibrate", help="Run calibration stage (post-process predictions)."
    )
    _add_stage_config_arg(p_cal_stage)
    _add_common_args(p_cal_stage)
    p_cal_stage.set_defaults(func=cmd_stage_calibrate)

    p_pol = sub.add_parser(
        "policy", help="Run policy stage (predictions+odds -> bets)."
    )
    _add_stage_config_arg(p_pol)
    _add_common_args(p_pol)
    p_pol.set_defaults(func=cmd_stage_policy)

    p_eval_stage = sub.add_parser(
        "evaluate", help="Run evaluation stage (bets/preds -> reports)."
    )
    _add_stage_config_arg(p_eval_stage)
    _add_common_args(p_eval_stage)
    p_eval_stage.set_defaults(func=cmd_stage_evaluate)

    # Eval utilities
    p_rank = sub.add_parser(
        "eval-ranking",
        help="Compute race-aware ranking metrics from existing predictions.",
    )
    p_rank.add_argument("--pred-path", default="outputs/reports/pred_test.parquet")
    p_rank.add_argument("--out-json", default="outputs/reports/metrics_ranking.json")
    p_rank.add_argument("--out-md", default="outputs/reports/metrics_ranking.md")
    p_rank.add_argument("--race-col", default=None)
    p_rank.add_argument("--score-col", default=None)
    p_rank.add_argument("--winner-col", default=None)
    p_rank.add_argument("--k1", type=int, default=3)
    p_rank.add_argument("--k2", type=int, default=5)
    _add_common_args(p_rank)
    p_rank.set_defaults(func=cmd_eval_ranking)

    p_cal = sub.add_parser(
        "eval-calibration-strata",
        help="Calibration diagnostics by race strata (ECE + reliability plots).",
    )
    p_cal.add_argument("--pred-path", default="outputs/reports/pred_test.parquet")
    p_cal.add_argument(
        "--model-frame-path", default="data/processed/model_frame.parquet"
    )
    p_cal.add_argument("--out-json", default="outputs/reports/calibration_strata.json")
    p_cal.add_argument("--out-md", default="outputs/reports/calibration_strata.md")
    p_cal.add_argument("--fig-dir", default="outputs/figures")
    p_cal.add_argument("--prob-col", default="p_win")
    p_cal.add_argument("--label-col", default="is_winner")
    p_cal.add_argument("--n-bins", type=int, default=15)
    _add_common_args(p_cal)
    p_cal.set_defaults(func=cmd_eval_calibration_strata)

    p_pol_eval = sub.add_parser(
        "eval-policy", help="Evaluate betting policy performance (ROI, drawdown, etc.)."
    )
    p_pol_eval.add_argument("--bets-path", default="outputs/bets/bets.parquet")
    p_pol_eval.add_argument("--pred-path", default="outputs/reports/pred_test.parquet")
    p_pol_eval.add_argument(
        "--policy-summary-path", default="outputs/bets/policy_summary.json"
    )
    p_pol_eval.add_argument("--out-json", default="outputs/reports/policy_metrics.json")
    p_pol_eval.add_argument("--out-md", default="outputs/reports/policy_metrics.md")
    p_pol_eval.add_argument("--fig-dir", default="outputs/figures")
    p_pol_eval.add_argument(
        "--scored-bets-path", default="outputs/reports/policy_bets_scored.parquet"
    )
    p_pol_eval.add_argument("--bankroll0", type=float, default=None)
    _add_common_args(p_pol_eval)
    p_pol_eval.set_defaults(func=cmd_eval_policy)

    # Misc
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

    verbosity = getattr(args, "verbose", 0) or 0
    _configure_logging(int(verbosity))

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
