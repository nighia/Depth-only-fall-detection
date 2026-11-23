from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config; fail fast on a missing file or a non-mapping root."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Config not found: {p}')
    with p.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Config root must be a mapping: {p}')
    return cfg
