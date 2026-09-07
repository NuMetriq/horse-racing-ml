# src/hrml/stages/calibration.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def main(config_path: str | Path | None = None) -> None:
    """
    Pipeline stage: calibration

    Responsibility:
      - run calibration diagnostics / produce calibration artifacts
      - does NOT train models

    YAML config controls:
      - pred_path, report_dir, fig_dir
      - whether to run: basic calibration summary, and/or calibration strata
    """
    from hrml.config import load_yaml

    cfg = load_yaml(config_path)
    pred_path = Path(cfg.get("pred_path", "outputs/reports/pred_test.parquet"))
    rep_dir = Path(cfg.get("report_dir", "outputs/reports"))
    fig_dir = Path(cfg.get("fig_dir", "outputs/figures"))

    run_basic = bool(cfg.get("run_basic", True))
    run_strata = bool(cfg.get("run_strata", True))

    # --------
    # Basic calibration (existing module uses module-level constants)
    # --------
    if run_basic:
        import hrml.eval.calibration as calib_mod

        calib_mod.PRED_PATH = pred_path
        calib_mod.REP_DIR = rep_dir
        calib_mod.FIG_DIR = fig_dir
        calib_mod.FIG_DIR.mkdir(parents=True, exist_ok=True)
        calib_mod.REP_DIR.mkdir(parents=True, exist_ok=True)

        calib_mod.main()

        out_path = rep_dir / "calibration_summary.json"
        if not out_path.exists():
            raise FileNotFoundError(
                f"Calibration stage finished but expected output not found: {out_path.as_posix()}"
            )

    # --------
    # Strata calibration (preferred configurable diagnostic)
    # --------
    if run_strata:
        from hrml.eval.calibration_strata import (
            CalibrationStrataConfig,
            run_calibration_strata,
        )

        strata_cfg = (
            cfg.get("strata", {}) if isinstance(cfg.get("strata", {}), dict) else {}
        )
        cs = CalibrationStrataConfig(
            pred_path=pred_path,
            model_frame_path=Path(
                strata_cfg.get("model_frame_path", "data/processed/model_frame.parquet")
            ),
            out_json=Path(
                strata_cfg.get("out_json", rep_dir / "calibration_strata.json")
            ),
            out_md=Path(strata_cfg.get("out_md", rep_dir / "calibration_strata.md")),
            fig_dir=Path(strata_cfg.get("fig_dir", fig_dir)),
            prob_col=strata_cfg.get("prob_col", "p_win"),
            label_col=strata_cfg.get("label_col", "is_winner"),
            n_bins=int(strata_cfg.get("n_bins", 15)),
        )

        run_calibration_strata(cs)
