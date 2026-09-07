# src/hrml/stages/ingest.py
from __future__ import annotations

from pathlib import Path


def main(config_path: str | Path = Path("configs/model.yaml")) -> None:
    """
    Pipeline stage: ingest

    Responsibility:
      - read raw data sources
      - write canonical processed parquet artifacts

    Expected outputs:
      - data/processed/races.parquet
      - data/processed/runners.parquet

    Config:
      - reads outputs from config (data.races_path, data.runners_path)
      - defaults to data/processed/*.parquet if absent
    """
    from hrml.config import get_nested, load_yaml
    from hrml.ingest.normalize import main as normalize_main

    cfg = load_yaml(Path(config_path))

    # run ingest (your ingest module decides how raw paths are discovered)
    normalize_main()

    races_path = Path(
        get_nested(cfg, ["data", "races_path"], "data/processed/races.parquet")
    )
    runners_path = Path(
        get_nested(cfg, ["data", "runners_path"], "data/processed/runners.parquet")
    )

    if not races_path.exists() or not runners_path.exists():
        missing = [p.as_posix() for p in (races_path, runners_path) if not p.exists()]
        raise FileNotFoundError(
            "Ingest stage finished but expected outputs were not found: "
            + ", ".join(missing)
        )
