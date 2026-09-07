# src/hrml/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None
    _YAML_IMPORT_ERR = e


def load_yaml(
    path: str | Path | None, *, default_path: str | Path = "configs/default.yaml"
) -> Dict[str, Any]:
    """
    Load a YAML file into a plain dict.

    - Accepts str | Path | None
    - If None, uses default_path (configs/default.yaml)
    - Minimal on purpose (no pydantic/merging)
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load configs. "
            "Install with: pip install pyyaml\n"
            f"Original import error: {_YAML_IMPORT_ERR}"
        )

    p = Path(default_path) if path is None else Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.as_posix()}")

    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"Config must be a YAML mapping (dict). Got: {type(obj)}")
    return obj


def get_nested(d: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """
    Safe nested dict access.
    Example: get_nested(cfg, ["data","races_path"], "data/processed/races.parquet")
    """
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
