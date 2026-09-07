# src/hrml/stages/features.py
from __future__ import annotations

from pathlib import Path


def main(config_path: str | Path = Path("configs/model.yaml")) -> None:
    """
    Pipeline stage: features

    Responsibility:
      - read canonical processed artifacts (races/runners)
      - build model_frame with engineered, *pre-race* features
      - write: data/processed/model_frame.parquet

    Config:
      - output path defaults to data/processed/model_frame.parquet
      - if you later add cfg.data.model_frame_path, we’ll use it
    """
    from hrml.config import get_nested, load_yaml
    from hrml.features.build_features import main as build_features_main

    cfg = load_yaml(Path(config_path))

    build_features_main()

    out_path = Path(
        get_nested(
            cfg, ["data", "model_frame_path"], "data/processed/model_frame.parquet"
        )
    )
    if not out_path.exists():
        raise FileNotFoundError(
            f"Features stage finished but expected output was not found: {out_path.as_posix()}"
        )
