from __future__ import annotations

import json
from pathlib import Path


def save_config_as_json(save_path: str | Path, config) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config.to_serializable(), f, indent=2)
