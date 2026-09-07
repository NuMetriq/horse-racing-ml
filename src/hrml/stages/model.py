# src/hrml/stages/model.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def main_softmax(
    *,
    base_only: bool,
    reuse_existing: bool,
    fast_dev: bool,
    features_only: bool,
    config_path: str | Path | None = None,
) -> None:
    """
    Pipeline stage: model (race-softmax)
    Delegates to: hrml.models.train_xgb_race_softmax.main()

    Config source (single file):
      cfg["model"]["softmax"]

    Precedence:
      - CLI flags (function args) override YAML
      - YAML provides defaults for any arg not set True by CLI
    """
    import sys

    from hrml.config import load_yaml
    from hrml.models.train_xgb_race_softmax import main as train_main

    cfg = load_yaml(config_path)
    mroot = _as_dict(cfg.get("model"))
    mcfg = _as_dict(mroot.get("softmax"))

    base_only = bool(base_only or _get(mcfg, "base_only", False))
    reuse_existing = bool(reuse_existing or _get(mcfg, "reuse_existing", False))
    fast_dev = bool(fast_dev or _get(mcfg, "fast_dev", False))
    features_only = bool(features_only or _get(mcfg, "features_only", False))

    argv = ["train_xgb_race_softmax"]
    if base_only:
        argv.append("--base-only")
    if reuse_existing:
        argv.append("--reuse-existing")
    if fast_dev:
        argv.append("--fast-dev")
    if features_only:
        argv.append("--features-only")

    old = sys.argv[:]
    try:
        sys.argv = argv
        train_main()
    finally:
        sys.argv = old

    model_path = Path("outputs/models/xgb_race_softmax.json")
    pred_path = Path("outputs/reports/pred_test.parquet")
    if not features_only:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Expected model not found: {model_path.as_posix()}"
            )
        if not (pred_path.exists() or reuse_existing):
            raise FileNotFoundError(
                f"Expected predictions not found: {pred_path.as_posix()}"
            )


def main_pairwise(
    *,
    base_only: bool,
    reuse_existing: bool,
    fast_dev: bool,
    features_only: bool,
    config_path: str | Path | None = None,
) -> None:
    """
    Pipeline stage: model (pairwise ranking)
    Delegates to: hrml.models.train_xgb_rank_pairwise.main()

    Config source:
      cfg["model"]["pairwise"]
    """
    import sys

    from hrml.config import load_yaml
    from hrml.models.train_xgb_rank_pairwise import main as train_main

    cfg = load_yaml(config_path)
    mroot = _as_dict(cfg.get("model"))
    mcfg = _as_dict(mroot.get("pairwise"))

    base_only = bool(base_only or _get(mcfg, "base_only", False))
    reuse_existing = bool(reuse_existing or _get(mcfg, "reuse_existing", False))
    fast_dev = bool(fast_dev or _get(mcfg, "fast_dev", False))
    features_only = bool(features_only or _get(mcfg, "features_only", False))

    argv = ["train_xgb_rank_pairwise"]
    if base_only:
        argv.append("--base-only")
    if reuse_existing:
        argv.append("--reuse-existing")
    if fast_dev:
        argv.append("--fast-dev")
    if features_only:
        argv.append("--features-only")

    old = sys.argv[:]
    try:
        sys.argv = argv
        train_main()
    finally:
        sys.argv = old

    model_path = Path("outputs/models/xgb_rank_pairwise.json")
    pred_path = Path("outputs/reports/pred_test_pairwise.parquet")
    if not features_only:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Expected model not found: {model_path.as_posix()}"
            )
        if not (pred_path.exists() or reuse_existing):
            raise FileNotFoundError(
                f"Expected predictions not found: {pred_path.as_posix()}"
            )


def main_plackett_luce(
    *,
    reuse_existing: bool,
    features_only: bool,
    fast_dev: bool,
    top_k: int,
    mc_samples: int,
    place_k: int,
    no_calibrate: bool,
    base_only: bool,
    config_path: str | Path | None = None,
) -> None:
    """
    Pipeline stage: model (Plackett–Luce)
    Delegates to: hrml.models.train_xgb_plackett_luce.main()

    Config source:
      cfg["model"]["plackett_luce"]
    """
    import sys

    from hrml.config import load_yaml
    from hrml.models.train_xgb_plackett_luce import main as train_main

    cfg = load_yaml(config_path)
    mroot = _as_dict(cfg.get("model"))
    mcfg = _as_dict(mroot.get("plackett_luce"))

    reuse_existing = bool(reuse_existing or _get(mcfg, "reuse_existing", False))
    features_only = bool(features_only or _get(mcfg, "features_only", False))
    fast_dev = bool(fast_dev or _get(mcfg, "fast_dev", False))
    no_calibrate = bool(no_calibrate or _get(mcfg, "no_calibrate", False))
    base_only = bool(base_only or _get(mcfg, "base_only", False))

    # Numeric args: if user passed defaults, allow YAML to override
    if top_k == 3:
        top_k = int(_get(mcfg, "top_k", 3))
    if mc_samples == 200:
        mc_samples = int(_get(mcfg, "mc_samples", 200))
    if place_k == 3:
        place_k = int(_get(mcfg, "place_k", 3))

    argv = ["train_xgb_plackett_luce"]
    if reuse_existing:
        argv.append("--reuse-existing")
    if features_only:
        argv.append("--features-only")
    if fast_dev:
        argv.append("--fast-dev")
    if base_only:
        argv.append("--base-only")
    if no_calibrate:
        argv.append("--no-calibrate")

    argv += ["--top-k", str(int(top_k))]
    argv += ["--mc-samples", str(int(mc_samples))]
    argv += ["--place-k", str(int(place_k))]

    old = sys.argv[:]
    try:
        sys.argv = argv
        train_main()
    finally:
        sys.argv = old

    model_path = Path("outputs/models/xgb_plackett_luce.json")
    pred_path = Path("outputs/reports/pred_test_plackett_luce.parquet")
    if not features_only:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Expected model not found: {model_path.as_posix()}"
            )
        if not (pred_path.exists() or reuse_existing):
            raise FileNotFoundError(
                f"Expected predictions not found: {pred_path.as_posix()}"
            )
